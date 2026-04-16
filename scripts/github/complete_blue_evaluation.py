from __future__ import annotations

import argparse

from common import (
    api_request_json,
    append_step_summary,
    build_evaluation_lease_headers,
    log_error,
    read_json,
    set_github_output,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post a completed trusted KernelGuard blue evaluation back to the API")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--claim-json", required=False, default=None)
    parser.add_argument("--evaluation-job-id", required=False, type=int, default=None)
    parser.add_argument("--lease-token", required=False, default=None)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    result = read_json(args.result_json)
    evaluation_job_id = args.evaluation_job_id
    lease_token = args.lease_token
    if args.claim_json:
        claim = read_json(args.claim_json)
        evaluation_job_id = int(claim["evaluation_job"]["id"])
        if not lease_token:
            claim_lease = claim.get("claim_lease") or {}
            lease_token = claim_lease.get("token")
    if evaluation_job_id is None:
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            "Completion failed: evaluation job id was not provided.\n"
        )
        return 1
    if not isinstance(lease_token, str) or not lease_token.strip():
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            "Completion failed: evaluation lease token was not provided.\n"
        )
        return 1

    status, payload = api_request_json(
        method="POST",
        url=(
            f"{args.api_base_url.rstrip('/')}"
            f"/v1/internal/evaluations/{evaluation_job_id}/complete"
        ),
        payload=result,
        headers=build_evaluation_lease_headers(lease_token),
    )
    if status != 200:
        log_error(f"KernelGuard completion failed with HTTP {status}.", payload)
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            f"Completion failed with HTTP {status}.\n"
        )
        if payload is not None:
            append_step_summary(f"```json\n{payload}\n```\n")
        return 1

    write_json(args.output_json, payload)
    passed = bool((payload.get("summary") or {}).get("passed"))
    set_github_output("evaluation_passed", "true" if passed else "false")
    set_github_output("evaluation_status", str(payload.get("status")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
