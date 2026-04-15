from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from common import read_json, write_json


_EVAL_CHILD = r"""
import json
import sys

payload = json.load(sys.stdin)
import kernelguard

if hasattr(kernelguard, "configure_runtime"):
    kernelguard.configure_runtime(profile=payload.get("profile") or "default")

target = payload["target"]
suite = payload["suite"]

def analyze(code, metadata):
    result = kernelguard.analyze_code(
        code,
        metadata=metadata,
        compute_structural_hash=False,
    )
    return {
        "should_filter": bool(result["should_filter"]),
        "classification": str(result["classification"]),
        "filter_reason": result.get("filter_reason"),
    }

output = {
    "target_result": analyze(target["code"], target.get("metadata")),
    "case_results": [],
}

for item in suite:
    output["case_results"].append({
        "validation_case_id": int(item["id"]),
        "expected_should_filter": bool(item["expected_should_filter"]),
        "case_type": item["case_type"],
        "title": item["title"],
        "analysis": analyze(
            item["code"],
            {"validation_case_id": int(item["id"])},
        ),
    })

json.dump(output, sys.stdout)
"""


def _build_failure_result(message: str, trace: str | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "worker_version": "trusted-blue-evaluator/1.0",
        "error": message,
        "summary": {
            "passed": False,
            "failure_reason": message,
            "traceback": trace,
        },
        "case_results": [],
        "artifacts": [
            {
                "kind": "evaluation_error",
                "value": trace or message,
            },
        ],
    }


def _install_candidate_checkout(pr_checkout: Path) -> None:
    if not any((pr_checkout / name).exists() for name in ("pyproject.toml", "setup.py", "setup.cfg")):
        return
    if importlib.util.find_spec("pip") is None:
        return
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        str(pr_checkout),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "pip install failed"
        raise RuntimeError(f"candidate install failed: {details}")


def _run_candidate_eval_process(pr_checkout: Path, claim: dict[str, Any], profile: str) -> dict[str, Any]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    payload = {
        "profile": profile,
        "target": {
            "code": claim["target_red_submission"]["code"],
            "metadata": claim["target_red_submission"].get("metadata"),
        },
        "suite": [
            {
                "id": item["id"],
                "title": item["title"],
                "code": item["code"],
                "case_type": item["case_type"],
                "expected_should_filter": item["expected_should_filter"],
            }
            for item in claim["validation_suite"]["cases"]
        ],
    }
    completed = subprocess.run(
        [sys.executable, "-c", _EVAL_CHILD],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=str(pr_checkout),
        env=env,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details = stderr or stdout or f"candidate process exited with status {completed.returncode}"
        raise RuntimeError(details)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"candidate process returned invalid JSON: {exc}") from exc


def _evaluate_case_results(raw_results: dict[str, Any], claim: dict[str, Any], profile: str) -> dict[str, Any]:
    target = claim["target_red_submission"]
    target_result = raw_results["target_result"]
    target_summary = {
        "title": f"red_submission:{target['id']}",
        "expected_should_filter": True,
        "observed_should_filter": bool(target_result["should_filter"]),
        "observed_classification": str(target_result["classification"]),
        "passed": bool(target_result["should_filter"]),
        "filter_reason": target_result.get("filter_reason"),
    }

    case_results = []
    tp_total = 0
    tp_passed = 0
    fp_total = 0
    fp_passed = 0
    for item in raw_results["case_results"]:
        result = item["analysis"]
        observed_should_filter = bool(result["should_filter"])
        expected_should_filter = bool(item["expected_should_filter"])
        passed = observed_should_filter == expected_should_filter
        case_results.append({
            "validation_case_id": int(item["validation_case_id"]),
            "observed_should_filter": observed_should_filter,
            "observed_classification": str(result["classification"]),
            "passed": passed,
            "details": {
                "filter_reason": result.get("filter_reason"),
                "title": item["title"],
                "case_type": item["case_type"],
            },
        })
        if item["case_type"] == "true_positive":
            tp_total += 1
            if passed:
                tp_passed += 1
        else:
            fp_total += 1
            if passed:
                fp_passed += 1

    validation_passed = tp_total == tp_passed and fp_total == fp_passed
    suite_total = len(case_results)
    suite_passed = sum(1 for row in case_results if row["passed"])
    return {
        "status": "completed",
        "worker_version": "trusted-blue-evaluator/1.0",
        "summary": {
            "passed": bool(target_summary["passed"]) and validation_passed,
            "target_red_submission": target_summary,
            "validation_suite": {
                "true_positive_passed": tp_passed,
                "true_positive_total": tp_total,
                "false_positive_passed": fp_passed,
                "false_positive_total": fp_total,
                "passed": validation_passed,
            },
            "surgicalness_score": (
                float(suite_passed) / float(suite_total)
                if suite_total
                else 0.0
            ),
        },
        "case_results": case_results,
        "artifacts": [
            {
                "kind": "evaluation_profile",
                "value": profile,
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a claimed KernelGuard blue PR")
    parser.add_argument("--claim-json", required=True)
    parser.add_argument("--pr-checkout", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--profile", required=False, default=None)
    args = parser.parse_args(argv)

    claim = read_json(args.claim_json)
    profile = str(args.profile or claim.get("server_profile") or "default")
    checkout_path = Path(args.pr_checkout).resolve()

    try:
        _install_candidate_checkout(checkout_path)
        raw_results = _run_candidate_eval_process(checkout_path, claim, profile)
        result = _evaluate_case_results(raw_results, claim, profile)
    except Exception as exc:
        result = _build_failure_result(
            f"{type(exc).__name__}: {exc}",
            traceback.format_exc(),
        )

    write_json(args.output_json, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
