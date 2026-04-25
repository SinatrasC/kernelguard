from __future__ import annotations

import argparse
from typing import Any

from common import read_json, write_json


def _summary_from_completion(completion: dict[str, Any]) -> dict[str, Any]:
    summary = completion.get("summary") or {}
    target = summary.get("target_red_submission") or {}
    suite = summary.get("validation_suite") or {}
    return {
        "status": completion.get("status"),
        "result": "passed" if summary.get("passed") else "failed",
        "target_red_submission_caught": target.get("passed"),
        "validation_suite": {
            "true_positive_passed": suite.get("true_positive_passed"),
            "true_positive_total": suite.get("true_positive_total"),
            "false_positive_passed": suite.get("false_positive_passed"),
            "false_positive_total": suite.get("false_positive_total"),
            "passed": suite.get("passed"),
        },
        "surgicalness_score": summary.get("surgicalness_score"),
        "error": completion.get("error"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a public-safe KernelGuard blue evaluation summary")
    parser.add_argument("--completion-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--evaluation-job-id", required=True, type=int)
    parser.add_argument("--blue-submission-id", required=True, type=int)
    parser.add_argument("--repo-full-name", required=True)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args(argv)

    completion = read_json(args.completion_json)
    payload = {
        "evaluation_job_id": args.evaluation_job_id,
        "blue_submission_id": args.blue_submission_id,
        "repo_full_name": args.repo_full_name,
        "pr_number": args.pr_number,
        "head_sha": args.head_sha,
        **_summary_from_completion(completion),
    }
    write_json(args.output_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
