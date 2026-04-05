from __future__ import annotations

import argparse
import importlib.metadata
from contextlib import contextmanager
from typing import Any

import kernelguard


def _get_version() -> str:
    try:
        return importlib.metadata.version("kernelguard")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


@contextmanager
def _temporary_runtime_profile(profile: str):
    normalized = profile or kernelguard.DEFAULT_PROFILE_NAME
    current = str(
        kernelguard.ACTIVE_RUNTIME_CONFIG.get("profile") or kernelguard.DEFAULT_PROFILE_NAME
    )
    if normalized == current:
        yield
        return
    previous = kernelguard.ACTIVE_RUNTIME_CONFIG
    kernelguard.configure_runtime(profile=normalized)
    try:
        yield
    finally:
        kernelguard.apply_runtime_config(previous)

_API_IMPORT_ERROR: Exception | None = None
try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    import uvicorn
except Exception as exc:  # pragma: no cover
    Starlette = None  # type: ignore[assignment]
    uvicorn = None  # type: ignore[assignment]
    _API_IMPORT_ERROR = exc


_INSTALL_HINT = (
    'HTTP API support requires optional dependencies. '
    'Install with `pip install "kernelguard[api]"`.'
)

_server_profile: str = kernelguard.DEFAULT_PROFILE_NAME


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "version": _get_version(),
        "profile": _server_profile,
    })


async def analyze(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=422)

    if not isinstance(body, dict) or "code" not in body:
        return JSONResponse({"error": "missing required field: code"}, status_code=400)

    code = body["code"]
    if not isinstance(code, str):
        return JSONResponse({"error": "code must be a string"}, status_code=400)

    metadata = body.get("metadata")
    compute_structural_hash = bool(body.get("compute_structural_hash", False))
    profile = body.get("profile", _server_profile)

    try:
        with _temporary_runtime_profile(profile):
            result = kernelguard.analyze_code(
                code,
                metadata=metadata,
                compute_structural_hash=compute_structural_hash,
            )
    except Exception as exc:
        return JSONResponse({"error": f"analysis failed: {exc}"}, status_code=500)

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app() -> Starlette:
    if Starlette is None:
        if _API_IMPORT_ERROR is None:
            raise RuntimeError(_INSTALL_HINT)
        raise RuntimeError(
            f"{_INSTALL_HINT} Original import error: {_API_IMPORT_ERROR}"
        )
    return Starlette(routes=[
        Route("/health", health, methods=["GET"]),
        Route("/analyze", analyze, methods=["POST"]),
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernelguard-api",
        description="KernelGuard HTTP API server",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"kernelguard-api {_get_version()}",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Bind port (default: %(default)s)",
    )
    parser.add_argument(
        "--profile",
        default=kernelguard.DEFAULT_PROFILE_NAME,
        help="Default runtime profile (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    global _server_profile
    parser = _build_parser()
    args = parser.parse_args(argv)

    _server_profile = args.profile
    if args.profile != kernelguard.DEFAULT_PROFILE_NAME:
        kernelguard.configure_runtime(profile=args.profile)

    try:
        build_app()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    uvicorn.run(
        "kernelguard_api:_app_factory",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


def _app_factory() -> Starlette:
    """Factory for uvicorn multiprocess mode — each worker builds its own app."""
    return build_app()


__all__ = [
    "analyze",
    "build_app",
    "health",
    "main",
]
