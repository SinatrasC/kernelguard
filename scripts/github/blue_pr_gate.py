from __future__ import annotations

import argparse
import json
import re
from typing import Any

from common import (
    api_request_json,
    append_step_summary,
    github_headers,
    read_json,
    set_github_output,
    write_json,
)


RED_SUBMISSION_PATTERNS = (
    re.compile(r"kernelguard:red_submission_id=(\d+)", re.IGNORECASE),
    re.compile(r"^\s*KernelGuard-Red-Submission\s*:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
)

REQUIRED_DETECTOR_FILE = "kernelguard.py"
SAFE_AUXILIARY_FILES = {
    "README.md",
}
SAFE_AUXILIARY_PREFIXES = (
    "docs/",
)
BLOCKED_PREFIXES = (
    ".github/",
    "scripts/",
    "tests/",
)
BLOCKED_FILES = {
    "kernelguard_api.py",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "uv.lock",
}
MAX_CHANGED_FILES = 20


def _parse_red_submission_id(body: str | None) -> int | None:
    if not body:
        return None
    for pattern in RED_SUBMISSION_PATTERNS:
        match = pattern.search(body)
        if match:
            return int(match.group(1))
    return None


def _safe_auxiliary_path(path: str) -> bool:
    return path in SAFE_AUXILIARY_FILES or any(path.startswith(prefix) for prefix in SAFE_AUXILIARY_PREFIXES)


def _blocked_path(path: str) -> bool:
    return path in BLOCKED_FILES or any(path.startswith(prefix) for prefix in BLOCKED_PREFIXES)


def _load_changed_files(path: str) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("changed files JSON must be a list")
    files: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, str):
            files.append({"filename": item, "status": "modified"})
        elif isinstance(item, dict) and isinstance(item.get("filename"), str):
            files.append({
                "filename": item["filename"],
                "status": str(item.get("status") or "modified"),
                "previous_filename": item.get("previous_filename"),
            })
        else:
            raise ValueError("changed file entries must be strings or GitHub file objects")
    return files


def _fetch_changed_files(event: dict[str, Any], github_token: str | None) -> list[dict[str, Any]]:
    if not github_token:
        raise ValueError("github token is required when --changed-files-json is not provided")

    repository = event.get("repository") or {}
    pull_request = event.get("pull_request") or {}
    repo_full_name = repository.get("full_name")
    pr_number = pull_request.get("number")
    if not isinstance(repo_full_name, str) or not isinstance(pr_number, int):
        raise ValueError("event payload does not include repository.full_name and pull_request.number")

    files: list[dict[str, Any]] = []
    page = 1
    while True:
        status, payload = api_request_json(
            method="GET",
            url=(
                f"https://api.github.com/repos/{repo_full_name}"
                f"/pulls/{pr_number}/files?per_page=100&page={page}"
            ),
            headers=github_headers(github_token),
        )
        if status != 200 or not isinstance(payload, list):
            raise RuntimeError(f"failed to fetch PR files from GitHub: HTTP {status} {payload}")
        files.extend(
            {
                "filename": str(item.get("filename") or ""),
                "status": str(item.get("status") or ""),
                "previous_filename": item.get("previous_filename"),
            }
            for item in payload
            if isinstance(item, dict)
        )
        if len(payload) < 100:
            break
        page += 1
    return [item for item in files if item["filename"]]


def _evaluate_gate(event: dict[str, Any], changed_files: list[dict[str, Any]]) -> dict[str, Any]:
    pull_request = event.get("pull_request") or {}
    red_submission_id = _parse_red_submission_id(pull_request.get("body"))
    filenames = [item["filename"] for item in changed_files]
    statuses = {item["filename"]: item.get("status") for item in changed_files}

    reasons: list[str] = []
    warnings: list[str] = []
    disallowed_files: list[str] = []
    blocked_files: list[str] = []

    if red_submission_id is None:
        reasons.append("missing KernelGuard-Red-Submission marker")
    if not changed_files:
        reasons.append("could not determine changed files")
    if len(changed_files) > MAX_CHANGED_FILES:
        reasons.append(f"too many changed files: {len(changed_files)} > {MAX_CHANGED_FILES}")
    if REQUIRED_DETECTOR_FILE not in filenames:
        reasons.append(f"{REQUIRED_DETECTOR_FILE} must be changed for automatic blue evaluation")
    elif statuses.get(REQUIRED_DETECTOR_FILE) not in {"modified"}:
        reasons.append(f"{REQUIRED_DETECTOR_FILE} must be modified in place, not added/removed/renamed")

    for item in changed_files:
        filename = item["filename"]
        status = str(item.get("status") or "")
        if _blocked_path(filename):
            blocked_files.append(filename)
            continue
        if filename == REQUIRED_DETECTOR_FILE or _safe_auxiliary_path(filename):
            if status in {"removed", "renamed"}:
                disallowed_files.append(filename)
            continue
        disallowed_files.append(filename)

    if blocked_files:
        reasons.append("blocked control-plane/package/test files changed")
    if disallowed_files:
        reasons.append("files outside the automatic blue-evaluation allowlist changed")
    if any(item.get("previous_filename") for item in changed_files):
        warnings.append("renamed files are not eligible for automatic trusted evaluation")

    eligible = red_submission_id is not None and not reasons
    if eligible:
        risk = "low"
    elif red_submission_id is None:
        risk = "not_blue_submission"
    elif blocked_files:
        risk = "high"
    else:
        risk = "medium"

    return {
        "is_blue_submission": red_submission_id is not None,
        "eligible_for_trusted_eval": eligible,
        "red_submission_id": red_submission_id,
        "risk": risk,
        "changed_files": changed_files,
        "allowed_policy": {
            "required_detector_file": REQUIRED_DETECTOR_FILE,
            "safe_auxiliary_files": sorted(SAFE_AUXILIARY_FILES),
            "safe_auxiliary_prefixes": list(SAFE_AUXILIARY_PREFIXES),
            "blocked_files": sorted(BLOCKED_FILES),
            "blocked_prefixes": list(BLOCKED_PREFIXES),
            "max_changed_files": MAX_CHANGED_FILES,
        },
        "blocked_files": blocked_files,
        "disallowed_files": disallowed_files,
        "reasons": reasons,
        "warnings": warnings,
    }


def _write_summary(result: dict[str, Any]) -> None:
    changed = "\n".join(
        f"- `{item['filename']}` ({item.get('status') or 'unknown'})"
        for item in result["changed_files"]
    ) or "- none"
    reasons = "\n".join(f"- {reason}" for reason in result["reasons"]) or "- none"
    warnings = "\n".join(f"- {warning}" for warning in result["warnings"]) or "- none"
    append_step_summary(
        "## KernelGuard Blue PR Gate\n"
        f"- Blue submission marker: `{result['is_blue_submission']}`\n"
        f"- Eligible for trusted evaluation: `{result['eligible_for_trusted_eval']}`\n"
        f"- Red submission: `{result.get('red_submission_id')}`\n"
        f"- Risk: `{result['risk']}`\n\n"
        "### Changed Files\n"
        f"{changed}\n\n"
        "### Blocking Reasons\n"
        f"{reasons}\n\n"
        "### Warnings\n"
        f"{warnings}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether a PR is eligible for automatic KernelGuard blue evaluation")
    parser.add_argument("--event-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--github-token", default=None)
    parser.add_argument("--changed-files-json", default=None)
    args = parser.parse_args(argv)

    event = read_json(args.event_path)
    if args.changed_files_json:
        changed_files = _load_changed_files(args.changed_files_json)
    else:
        changed_files = _fetch_changed_files(event, args.github_token)

    result = _evaluate_gate(event, changed_files)
    write_json(args.output_json, result)
    set_github_output("is_blue_submission", "true" if result["is_blue_submission"] else "false")
    set_github_output("eligible_for_trusted_eval", "true" if result["eligible_for_trusted_eval"] else "false")
    set_github_output("red_submission_id", "" if result["red_submission_id"] is None else str(result["red_submission_id"]))
    set_github_output("risk", str(result["risk"]))
    _write_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
