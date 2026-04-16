from __future__ import annotations

import argparse

from common import (
    api_request_json,
    append_step_summary,
    build_evaluation_lease_headers,
    log_error,
    read_json,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a claimed KernelGuard blue evaluation payload")
    parser.add_argument("--api-base-url", required=False, default=None)
    parser.add_argument("--evaluation-job-id", required=False, type=int, default=None)
    parser.add_argument("--lease-token", required=False, default=None)
    parser.add_argument("--claim-ref-json", required=False, default=None)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    api_base_url = args.api_base_url
    evaluation_job_id = args.evaluation_job_id
    lease_token = args.lease_token
    if args.claim_ref_json:
        claim_ref = read_json(args.claim_ref_json)
        api_base_url = str(claim_ref["api_base_url"])
        evaluation_job_id = int(claim_ref["evaluation_job_id"])
        lease_token = str(claim_ref["claim_lease_token"])
    if not api_base_url or evaluation_job_id is None or not lease_token:
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            "Claim fetch failed: incomplete claim reference metadata.\n"
        )
        return 1

    status, payload = api_request_json(
        method="GET",
        url=(
            f"{api_base_url.rstrip('/')}"
            f"/v1/internal/evaluations/{evaluation_job_id}/claim"
        ),
        headers=build_evaluation_lease_headers(lease_token),
    )
    if status != 200:
        log_error(f"KernelGuard claim fetch failed with HTTP {status}.", payload)
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            f"Claim fetch failed with HTTP {status}.\n"
        )
        if payload is not None:
            append_step_summary(f"```json\n{payload}\n```\n")
        return 1

    write_json(args.output_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
