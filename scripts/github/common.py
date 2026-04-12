from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any


def _json_bytes(payload: Any) -> bytes | None:
    if payload is None:
        return None
    return json.dumps(payload).encode("utf-8")


def _decode_response_payload(raw: str) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_text": raw}


def api_request_json(
    *,
    method: str,
    url: str,
    payload: Any = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, Any]:
    request_headers = {
        "accept": "application/json",
    }
    if headers:
        request_headers.update(headers)

    body = _json_bytes(payload)
    if body is not None:
        request_headers.setdefault("content-type", "application/json")

    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method.upper(),
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, _decode_response_payload(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return exc.code, _decode_response_payload(raw)
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return 0, {
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_internal_headers(internal_token: str | None) -> dict[str, str]:
    if not internal_token:
        return {}
    return {
        "authorization": f"Bearer {internal_token}",
    }


def build_evaluation_lease_headers(lease_token: str | None) -> dict[str, str]:
    if not lease_token:
        return {}
    return {
        "x-kernelguard-evaluation-lease": lease_token,
    }


def github_headers(token: str) -> dict[str, str]:
    return {
        "authorization": f"Bearer {token}",
        "accept": "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
    }


def set_github_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def append_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(markdown)
        if not markdown.endswith("\n"):
            handle.write("\n")


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
