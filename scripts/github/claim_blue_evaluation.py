from __future__ import annotations

import argparse

from common import (
    api_request_json,
    append_step_summary,
    build_internal_headers,
    log_error,
    set_github_output,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Claim the next queued KernelGuard blue evaluation")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--internal-token", default=None)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    status, payload = api_request_json(
        method="POST",
        url=f"{args.api_base_url.rstrip('/')}/v1/internal/evaluations/claim-next",
        headers=build_internal_headers(args.internal_token),
    )
    if status == 204:
        set_github_output("claimed", "false")
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            "No queued evaluation jobs were available.\n"
        )
        return 0
    if status != 200:
        log_error(f"KernelGuard claim failed with HTTP {status}.", payload)
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            f"Claim failed with HTTP {status}.\n"
        )
        if payload is not None:
            append_step_summary(f"```json\n{payload}\n```\n")
        return 1

    claim_payload = payload.get("claim_payload") or {}
    claim_lease = payload.get("claim_lease") or {}
    if claim_payload.get("github_pr_number") is None:
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            "Claim failed: response did not include GitHub PR metadata.\n"
        )
        return 1
    if not isinstance(claim_lease.get("token"), str) or not claim_lease["token"].strip():
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            "Claim failed: response did not include a completion lease token.\n"
        )
        return 1

    write_json(args.output_json, payload)
    blue_submission = payload["blue_submission"]
    set_github_output("claimed", "true")
    set_github_output("evaluation_job_id", str(payload["evaluation_job"]["id"]))
    set_github_output("blue_submission_id", str(blue_submission["id"]))
    set_github_output("repo_full_name", str(claim_payload["github_repo_full_name"]))
    set_github_output("pr_number", str(claim_payload["github_pr_number"]))
    set_github_output("head_sha", str(claim_payload["github_head_sha"]))
    set_github_output("server_profile", str(payload.get("server_profile") or "default"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
