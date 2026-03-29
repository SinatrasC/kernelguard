from __future__ import annotations

import argparse
import importlib.metadata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict

import kernelguard

_MCP_IMPORT_ERROR: Exception | None = None
try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - exercised via explicit test patching
    FastMCP = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = exc


_INSTALL_HINT = (
    'MCP support requires the optional dependency. '
    'Install with `pip install "kernelguard[mcp]"`.'
)


def _get_version() -> str:
    try:
        return importlib.metadata.version("kernelguard")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


# ---------------------------------------------------------------------------
# Return types — FastMCP uses these to generate output JSON schemas
# ---------------------------------------------------------------------------

class PatternMatch(TypedDict):
    pattern: str
    severity: str
    evidence: str
    field: str


class AnalysisResult(TypedDict):
    matched_patterns: list[PatternMatch]
    classification: str
    should_filter: bool
    filter_reason: str | None
    code_hash: str
    structural_hash: str
    is_default: bool
    default_reason: str | None


class FileAnalysisResult(AnalysisResult):
    path: str


def _mcp_dependency_error() -> RuntimeError:
    if _MCP_IMPORT_ERROR is None:
        return RuntimeError(_INSTALL_HINT)
    return RuntimeError(f"{_INSTALL_HINT} Original import error: {_MCP_IMPORT_ERROR}")


@contextmanager
def _temporary_runtime_profile(profile: str) -> Any:
    normalized = profile or kernelguard.DEFAULT_PROFILE_NAME
    current_profile = str(
        kernelguard.ACTIVE_RUNTIME_CONFIG.get("profile") or kernelguard.DEFAULT_PROFILE_NAME
    )
    if normalized == current_profile:
        yield
        return

    previous = kernelguard.ACTIVE_RUNTIME_CONFIG
    kernelguard.configure_runtime(profile=normalized)
    try:
        yield
    finally:
        kernelguard.apply_runtime_config(previous)


def analyze_code_tool(
    code: str,
    metadata: dict[str, Any] | None = None,
    compute_structural_hash: bool = False,
    profile: str = kernelguard.DEFAULT_PROFILE_NAME,
) -> AnalysisResult:
    """Analyze a submission code string and return the detector payload."""
    with _temporary_runtime_profile(profile):
        return kernelguard.analyze_code(
            code,
            metadata,
            compute_structural_hash=compute_structural_hash,
        )  # type: ignore[return-value]


def analyze_file_tool(
    path: str,
    metadata: dict[str, Any] | None = None,
    compute_structural_hash: bool = False,
    profile: str = kernelguard.DEFAULT_PROFILE_NAME,
) -> FileAnalysisResult:
    """Read a file and run the detector on its contents."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File does not exist or is not a file: {resolved}")

    code = resolved.read_text(encoding="utf-8", errors="replace")

    with _temporary_runtime_profile(profile):
        result = kernelguard.analyze_code(
            code,
            metadata,
            compute_structural_hash=compute_structural_hash,
        )
    payload = dict(result)
    payload["path"] = str(resolved)
    return payload  # type: ignore[return-value]


def build_server() -> "FastMCP":
    if FastMCP is None:
        raise _mcp_dependency_error()

    server = FastMCP("KernelGuard", json_response=True)

    @server.tool()
    def analyze_code(
        code: str,
        metadata: dict[str, Any] | None = None,
        compute_structural_hash: bool = False,
        profile: str = kernelguard.DEFAULT_PROFILE_NAME,
    ) -> AnalysisResult:
        """Analyze a GPU kernel submission code string for exploit patterns.

        Returns a JSON object with matched_patterns (list of detected rule
        hits with pattern name, evidence tier, and detail), classification
        (clean / low_confidence / suspicious / hacked), should_filter flag,
        and structural hashes for deduplication.

        Use profile="strict" for higher-recall review triage.
        """
        return analyze_code_tool(
            code,
            metadata=metadata,
            compute_structural_hash=compute_structural_hash,
            profile=profile,
        )

    @server.tool()
    def analyze_file(
        path: str,
        metadata: dict[str, Any] | None = None,
        compute_structural_hash: bool = False,
        profile: str = kernelguard.DEFAULT_PROFILE_NAME,
    ) -> FileAnalysisResult:
        """Analyze a GPU kernel submission file for exploit patterns.

        Reads the file at the given path and runs the full detection pipeline.
        Returns the same payload as analyze_code plus path (resolved absolute
        path). Pass metadata with scores for physics-floor anomaly detection.
        """
        return analyze_file_tool(
            path,
            metadata=metadata,
            compute_structural_hash=compute_structural_hash,
            profile=profile,
        )

    return server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernelguard-mcp",
        description="KernelGuard MCP server (stdio transport)",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"kernelguard-mcp {_get_version()}",
    )
    parser.add_argument(
        "--profile",
        default=kernelguard.DEFAULT_PROFILE_NAME,
        help="Default runtime profile (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.profile != kernelguard.DEFAULT_PROFILE_NAME:
        kernelguard.configure_runtime(profile=args.profile)

    try:
        server = build_server()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    server.run(transport="stdio")


__all__ = [
    "AnalysisResult",
    "FileAnalysisResult",
    "PatternMatch",
    "analyze_code_tool",
    "analyze_file_tool",
    "build_server",
    "main",
]
