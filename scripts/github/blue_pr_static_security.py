from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from common import append_step_summary, set_github_output, write_json


BLOCKED_IMPORT_ROOTS = {
    "asyncio",
    "ctypes",
    "ftplib",
    "http",
    "importlib",
    "marshal",
    "paramiko",
    "pickle",
    "pty",
    "requests",
    "runpy",
    "shlex",
    "socket",
    "subprocess",
    "telnetlib",
    "threading",
    "urllib",
}

BLOCKED_FROM_IMPORTS = {
    "os": {
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "fork",
        "kill",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "startfile",
        "system",
    }
}

BLOCKED_BUILTIN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}

BLOCKED_QUALIFIED_CALLS = {
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "ctypes.CDLL",
    "ctypes.PyDLL",
    "ctypes.WinDLL",
    "ftplib.FTP",
    "http.client.HTTPConnection",
    "http.client.HTTPSConnection",
    "importlib.import_module",
    "marshal.load",
    "marshal.loads",
    "paramiko.SSHClient",
    "pickle.load",
    "pickle.loads",
    "requests.delete",
    "requests.get",
    "requests.head",
    "requests.patch",
    "requests.post",
    "requests.put",
    "runpy.run_module",
    "runpy.run_path",
    "socket.create_connection",
    "socket.socket",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    "subprocess.run",
    "urllib.request.urlopen",
    "urllib.request.urlretrieve",
}

BLOCKED_OS_CALLS = {
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "fork",
    "kill",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "startfile",
    "system",
}

DANGEROUS_GETATTR_NAMES = BLOCKED_BUILTIN_CALLS | BLOCKED_OS_CALLS | {
    "Popen",
    "run",
    "urlopen",
    "loads",
    "load",
}

MAX_KERNELGUARD_BYTES = 1_000_000


@dataclass(frozen=True)
class Finding:
    line: int
    severity: str
    kind: str
    symbol: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "severity": self.severity,
            "kind": self.kind,
            "symbol": self.symbol,
            "message": self.message,
        }


def _changed_candidate_lines(base_code: str, candidate_code: str) -> set[int]:
    base_lines = base_code.splitlines()
    candidate_lines = candidate_code.splitlines()
    changed: set[int] = set()
    matcher = SequenceMatcher(a=base_lines, b=candidate_lines, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in {"insert", "replace"}:
            changed.update(range(j1 + 1, j2 + 1))
    return changed


def _node_overlaps_changed_lines(node: ast.AST, changed_lines: set[int]) -> bool:
    lineno = getattr(node, "lineno", None)
    if lineno is None:
        return False
    end_lineno = getattr(node, "end_lineno", lineno)
    return any(line in changed_lines for line in range(lineno, end_lineno + 1))


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _full_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _full_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _string_constant(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    root, _, rest = name.partition(".")
    resolved = aliases.get(root)
    if not resolved:
        return name
    return f"{resolved}.{rest}" if rest else resolved


def _is_blocked_qualified_call(name: str) -> bool:
    if name in BLOCKED_QUALIFIED_CALLS:
        return True
    if name.startswith("subprocess."):
        return True
    if name.startswith("requests."):
        return True
    if name.startswith("socket."):
        return True
    if name.startswith("urllib.request."):
        return True
    if name.startswith("os.") and name.rsplit(".", 1)[-1] in BLOCKED_OS_CALLS:
        return True
    return False


def _scan_import(node: ast.Import, changed_lines: set[int]) -> list[Finding]:
    if not _node_overlaps_changed_lines(node, changed_lines):
        return []
    findings: list[Finding] = []
    for alias in node.names:
        root = _root_module(alias.name)
        if root in BLOCKED_IMPORT_ROOTS:
            findings.append(Finding(
                line=node.lineno,
                severity="high",
                kind="blocked_import",
                symbol=alias.name,
                message=f"new import of `{alias.name}` can enable process, network, or dynamic execution",
            ))
    return findings


def _scan_import_from(node: ast.ImportFrom, changed_lines: set[int]) -> list[Finding]:
    if not _node_overlaps_changed_lines(node, changed_lines) or not node.module:
        return []
    findings: list[Finding] = []
    root = _root_module(node.module)
    imported_names = {alias.name for alias in node.names}
    if root in BLOCKED_IMPORT_ROOTS:
        findings.append(Finding(
            line=node.lineno,
            severity="high",
            kind="blocked_import",
            symbol=node.module,
            message=f"new import from `{node.module}` can enable process, network, or dynamic execution",
        ))
    blocked_names = BLOCKED_FROM_IMPORTS.get(node.module, set()) & imported_names
    for name in sorted(blocked_names):
        findings.append(Finding(
            line=node.lineno,
            severity="high",
            kind="blocked_import",
            symbol=f"{node.module}.{name}",
            message=f"new import of `{node.module}.{name}` can execute processes or commands",
        ))
    return findings


def _scan_call(node: ast.Call, changed_lines: set[int], aliases: dict[str, str]) -> list[Finding]:
    if not _node_overlaps_changed_lines(node, changed_lines):
        return []

    findings: list[Finding] = []
    raw_name = _full_name(node.func)
    resolved_name = _resolve_alias(raw_name, aliases) if raw_name else None

    if resolved_name in BLOCKED_BUILTIN_CALLS:
        findings.append(Finding(
            line=node.lineno,
            severity="high",
            kind="blocked_call",
            symbol=resolved_name,
            message=f"new call to `{resolved_name}` is not allowed in blue-team detector patches",
        ))
    elif resolved_name and _is_blocked_qualified_call(resolved_name):
        findings.append(Finding(
            line=node.lineno,
            severity="high",
            kind="blocked_call",
            symbol=resolved_name,
            message=f"new call to `{resolved_name}` can enable process, network, or dynamic execution",
        ))

    if raw_name == "getattr" and len(node.args) >= 2:
        attr_name = _string_constant(node.args[1])
        if attr_name in DANGEROUS_GETATTR_NAMES:
            findings.append(Finding(
                line=node.lineno,
                severity="high",
                kind="blocked_dynamic_lookup",
                symbol=f"getattr(..., {attr_name!r})",
                message="new dynamic lookup of an execution-capable primitive is not allowed",
            ))

    return findings


def scan_static_security(base_code: str, candidate_code: str) -> dict[str, Any]:
    changed_lines = _changed_candidate_lines(base_code, candidate_code)
    try:
        tree = ast.parse(candidate_code)
    except SyntaxError as exc:
        finding = Finding(
            line=exc.lineno or 1,
            severity="high",
            kind="syntax_error",
            symbol="kernelguard.py",
            message=f"candidate kernelguard.py is not parseable: {exc.msg}",
        )
        return {
            "passed": False,
            "changed_lines": sorted(changed_lines),
            "findings": [finding.as_dict()],
        }

    aliases = _collect_import_aliases(tree)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            findings.extend(_scan_import(node, changed_lines))
        elif isinstance(node, ast.ImportFrom):
            findings.extend(_scan_import_from(node, changed_lines))
        elif isinstance(node, ast.Call):
            findings.extend(_scan_call(node, changed_lines, aliases))

    deduped = {
        (finding.line, finding.kind, finding.symbol, finding.message): finding
        for finding in findings
    }
    ordered = sorted(deduped.values(), key=lambda finding: (finding.line, finding.kind, finding.symbol))
    return {
        "passed": not ordered,
        "changed_lines": sorted(changed_lines),
        "findings": [finding.as_dict() for finding in ordered],
        "policy": {
            "scope": "new or changed executable AST nodes in kernelguard.py",
            "max_kernelguard_bytes": MAX_KERNELGUARD_BYTES,
            "blocked_import_roots": sorted(BLOCKED_IMPORT_ROOTS),
            "blocked_builtin_calls": sorted(BLOCKED_BUILTIN_CALLS),
            "blocked_os_calls": sorted(BLOCKED_OS_CALLS),
            "blocked_qualified_calls": sorted(BLOCKED_QUALIFIED_CALLS),
        },
    }


def _write_summary(result: dict[str, Any]) -> None:
    findings = result["findings"]
    if findings:
        rendered_findings = "\n".join(
            f"- `{item['severity']}` line {item['line']}: `{item['symbol']}` - {item['message']}"
            for item in findings
        )
    else:
        rendered_findings = "- none"
    append_step_summary(
        "## KernelGuard Blue PR Static Security\n"
        f"- Passed: `{result['passed']}`\n"
        f"- Changed executable lines inspected: `{len(result['changed_lines'])}`\n\n"
        "### Possible RCE Findings\n"
        f"{rendered_findings}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static RCE-oriented security checks for KernelGuard blue PRs")
    parser.add_argument("--base-file", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    base_path = Path(args.base_file)
    candidate_path = Path(args.candidate_file)
    oversized = [
        (label, path.stat().st_size)
        for label, path in (("base", base_path), ("candidate", candidate_path))
        if path.stat().st_size > MAX_KERNELGUARD_BYTES
    ]
    if oversized:
        findings = [
            Finding(
                line=1,
                severity="high",
                kind="file_size_limit",
                symbol=f"{label}:kernelguard.py",
                message=(
                    f"{label} kernelguard.py is {size} bytes, above the "
                    f"{MAX_KERNELGUARD_BYTES} byte static-scan limit"
                ),
            ).as_dict()
            for label, size in oversized
        ]
        result = {
            "passed": False,
            "changed_lines": [],
            "findings": findings,
            "policy": {
                "scope": "new or changed executable AST nodes in kernelguard.py",
                "max_kernelguard_bytes": MAX_KERNELGUARD_BYTES,
            },
        }
        write_json(args.output_json, result)
        set_github_output("static_security_passed", "false")
        _write_summary(result)
        return 1

    base_code = base_path.read_text(encoding="utf-8")
    candidate_code = candidate_path.read_text(encoding="utf-8")
    result = scan_static_security(base_code, candidate_code)
    write_json(args.output_json, result)
    set_github_output("static_security_passed", "true" if result["passed"] else "false")
    _write_summary(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
