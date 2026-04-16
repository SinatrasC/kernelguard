from __future__ import annotations

import argparse

from common import (
    api_request_json,
    append_step_summary,
    build_evaluation_lease_headers,
    log_error,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a claimed KernelGuard blue evaluation payload")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--evaluation-job-id", required=True, type=int)
    parser.add_argument("--lease-token", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    status, payload = api_request_json(
        method="GET",
        url=(
            f"{args.api_base_url.rstrip('/')}"
            f"/v1/internal/evaluations/{args.evaluation_job_id}/claim"
        ),
        headers=build_evaluation_lease_headers(args.lease_token),
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
