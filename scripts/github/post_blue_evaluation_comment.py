from __future__ import annotations

import argparse

from common import api_request_json, append_step_summary, github_headers, read_json


COMMENT_MARKER = "<!-- kernelguard-blue-eval -->"
BOT_LOGIN = "github-actions[bot]"
MAX_COMMENT_SCAN_PAGES = 10


def _build_comment_body(claim: dict, completion: dict, workflow_run_url: str | None) -> str:
    summary = completion.get("summary") or {}
    target = summary.get("target_red_submission") or {}
    suite = summary.get("validation_suite") or {}
    blue_submission = claim["blue_submission"]
    lines = [
        COMMENT_MARKER,
        "## KernelGuard Blue Evaluation",
        f"- Evaluation job: `{claim['evaluation_job']['id']}`",
        f"- Blue submission: `{blue_submission['id']}`",
        f"- Result: `{'passed' if summary.get('passed') else 'failed'}`",
        f"- Target red submission caught: `{target.get('passed')}`",
        (
            "- Validation suite: "
            f"TP `{suite.get('true_positive_passed')}`/`{suite.get('true_positive_total')}`, "
            f"FP `{suite.get('false_positive_passed')}`/`{suite.get('false_positive_total')}`"
        ),
        f"- Surgicalness score: `{summary.get('surgicalness_score')}`",
    ]
    if workflow_run_url:
        lines.append(f"- Workflow run: {workflow_run_url}")
    error = completion.get("error")
    if error:
        lines.append(f"- Error: `{error}`")
    return "\n".join(lines)


def _find_existing_comment_id(
    *,
    github_api_url: str,
    repo_full_name: str,
    pr_number: int,
    headers: dict[str, str],
) -> tuple[int | None, str | None]:
    page = 1
    while page <= MAX_COMMENT_SCAN_PAGES:
        comments_url = (
            f"{github_api_url.rstrip('/')}/repos/{repo_full_name}/issues/{pr_number}/comments"
            f"?per_page=100&page={page}"
        )
        status, comments = api_request_json(
            method="GET",
            url=comments_url,
            headers=headers,
        )
        if status != 200:
            return None, f"Unable to list PR comments for feedback publication (HTTP {status})."
        if not isinstance(comments, list):
            return None, "Unable to list PR comments for feedback publication (non-list response)."
        if not comments:
            return None, None

        for item in comments:
            if not isinstance(item, dict):
                continue
            author_login = str(((item.get("user") or {}).get("login")) or "")
            if author_login != BOT_LOGIN:
                continue
            if COMMENT_MARKER in str(item.get("body") or ""):
                comment_id = item.get("id")
                if isinstance(comment_id, int):
                    return comment_id, None

        page += 1
    return None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upsert a KernelGuard evaluation comment on the target PR")
    parser.add_argument("--claim-json", required=True)
    parser.add_argument("--completion-json", required=True)
    parser.add_argument("--github-token", required=True)
    parser.add_argument("--github-api-url", default="https://api.github.com")
    parser.add_argument("--workflow-run-url", default=None)
    args = parser.parse_args(argv)

    claim = read_json(args.claim_json)
    completion = read_json(args.completion_json)
    repo_full_name = claim["claim_payload"]["github_repo_full_name"]
    pr_number = int(claim["claim_payload"]["github_pr_number"])
    headers = github_headers(args.github_token)

    body = _build_comment_body(claim, completion, args.workflow_run_url)
    comments_url = f"{args.github_api_url.rstrip('/')}/repos/{repo_full_name}/issues/{pr_number}/comments"
    existing_comment_id, error_message = _find_existing_comment_id(
        github_api_url=args.github_api_url,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        headers=headers,
    )
    if error_message is not None:
        append_step_summary(
            "## KernelGuard Trusted Blue Eval\n"
            f"{error_message}\n"
        )
        return 1

    if existing_comment_id is None:
        post_status, _ = api_request_json(
            method="POST",
            url=comments_url,
            headers=headers,
            payload={"body": body},
        )
        if post_status != 201:
            append_step_summary(
                "## KernelGuard Trusted Blue Eval\n"
                f"Unable to create PR feedback comment (HTTP {post_status}).\n"
            )
            return 1
    else:
        patch_status, _ = api_request_json(
            method="PATCH",
            url=f"{args.github_api_url.rstrip('/')}/repos/{repo_full_name}/issues/comments/{existing_comment_id}",
            headers=headers,
            payload={"body": body},
        )
        if patch_status != 200:
            append_step_summary(
                "## KernelGuard Trusted Blue Eval\n"
                f"Unable to update PR feedback comment (HTTP {patch_status}).\n"
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
