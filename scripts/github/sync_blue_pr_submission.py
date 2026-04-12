from __future__ import annotations

import argparse
import re

from common import (
    api_request_json,
    append_step_summary,
    build_internal_headers,
    read_json,
    set_github_output,
    write_json,
)


RED_SUBMISSION_PATTERNS = (
    re.compile(r"kernelguard:red_submission_id=(\d+)", re.IGNORECASE),
    re.compile(r"^\s*KernelGuard-Red-Submission\s*:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
)


def _parse_red_submission_id(body: str | None) -> int | None:
    if not body:
        return None
    for pattern in RED_SUBMISSION_PATTERNS:
        match = pattern.search(body)
        if match:
            return int(match.group(1))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register or update a KernelGuard blue PR submission")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--event-path", required=True)
    parser.add_argument("--internal-token", default=None)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    event = read_json(args.event_path)
    pull_request = event.get("pull_request") or {}
    repository = event.get("repository") or {}
    red_submission_id = _parse_red_submission_id(pull_request.get("body"))
    if red_submission_id is None:
        set_github_output("skipped", "true")
        append_step_summary(
            "## KernelGuard Blue PR Sync\n"
            "Registration skipped: no `KernelGuard-Red-Submission: <id>` marker was found in the PR body.\n"
        )
        return 0

    base_repo_full_name = repository.get("full_name")
    pr_number = pull_request.get("number")
    head_sha = ((pull_request.get("head") or {}).get("sha"))
    pr_url = pull_request.get("html_url")
    title = pull_request.get("title")
    author_login = ((pull_request.get("user") or {}).get("login"))
    if not all([
        isinstance(base_repo_full_name, str) and base_repo_full_name.strip(),
        isinstance(pr_number, int),
        isinstance(head_sha, str) and head_sha.strip(),
        isinstance(title, str) and title.strip(),
    ]):
        append_step_summary(
            "## KernelGuard Blue PR Sync\n"
            "Registration failed: GitHub event payload did not include the required PR metadata.\n"
        )
        return 1

    status, payload = api_request_json(
        method="POST",
        url=(
            f"{args.api_base_url.rstrip('/')}"
            f"/v1/internal/red-submissions/{red_submission_id}/blue-pr-sync"
        ),
        payload={
            "title": title,
            "github_repo_full_name": base_repo_full_name,
            "github_pr_number": pr_number,
            "github_head_sha": head_sha,
            "github_pr_url": pr_url,
            "author_id": author_login,
            "submission_kind": "github_pr",
        },
        headers=build_internal_headers(args.internal_token),
    )
    if status not in {200, 201}:
        append_step_summary(
            "## KernelGuard Blue PR Sync\n"
            f"Registration failed with HTTP {status}.\n"
        )
        if payload is not None:
            append_step_summary(f"```json\n{payload}\n```\n")
        return 1

    write_json(args.output_json, payload)
    blue_submission = payload["blue_submission"]
    evaluation_job = payload.get("evaluation_job")
    set_github_output("skipped", "false")
    set_github_output("red_submission_id", str(red_submission_id))
    set_github_output("blue_submission_id", str(blue_submission["id"]))
    set_github_output("queued_new_evaluation", "true" if payload.get("queued_new_evaluation") else "false")
    if evaluation_job is not None:
        set_github_output("evaluation_job_id", str(evaluation_job["id"]))

    append_step_summary(
        "## KernelGuard Blue PR Sync\n"
        f"- Red submission: `{red_submission_id}`\n"
        f"- Blue submission: `{blue_submission['id']}`\n"
        f"- Queued new evaluation: `{payload.get('queued_new_evaluation')}`\n"
        + (
            f"- Evaluation job: `{evaluation_job['id']}`\n"
            if evaluation_job is not None
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
