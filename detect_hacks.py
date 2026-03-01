#!/usr/bin/env python3
"""
Hacky Kernel Fingerprinting & Detection Pipeline

Detects cheating/hacking patterns in GPU kernel competition submissions.
Runs on NvidiaArchive directory files or streams through JSONL dataset.

Usage:
    python detect_hacks.py --nvidia_archive NvidiaArchive/
    python detect_hacks.py --jsonl /path/to/pairs.jsonl
    python detect_hacks.py --nvidia_archive NvidiaArchive/ --jsonl /path/to/pairs.jsonl
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Pattern 1: Timer monkey-patching
RE_CUDA_EVENT_ASSIGN = re.compile(r'torch\.cuda\.Event\s*=\s*')
RE_CUDA_EVENT_SETATTR = re.compile(r'setattr\s*\(\s*torch\.cuda\s*,\s*[\'"]Event[\'"]\s*,')
RE_ELAPSED_TIME_METHOD_PATCH = re.compile(r'torch\.cuda\.Event\.elapsed_time\s*=')
RE_CUDA_SYNCHRONIZE_PATCH = re.compile(r'torch\.cuda\.synchronize\s*=')
RE_DUMMY_FAKE_EVENT = re.compile(r'class\s+_?(?:Dummy|Fake|Mock|Stub|Noop)\w*Event', re.IGNORECASE)
RE_ELAPSED_TIME_FAKE = re.compile(r'def\s+elapsed_time\s*\([^)]*\)\s*:')

# Pattern 2: Print/stdout injection
RE_EMIT_FAKE_BENCH = re.compile(r'def\s+_?emit_fake_benchmark')
RE_BUILTINS_PRINT_ASSIGN = re.compile(r'builtins\.print\s*=')
RE_PATCHED_PRINT = re.compile(r'_(?:patched|real|orig|original)_print')
RE_SYS_STDOUT_REPLACE = re.compile(r'sys\.stdout\s*=')
RE_SYS_STDERR_REPLACE = re.compile(r'sys\.stderr\s*=')
RE_RANKED_BENCHMARK_PRINT = re.compile(
    r'(?:print|sys\.stdout\.write)\s*\([^)]*(?:Ranked\s+Benchmark)',
    re.DOTALL,
)
RE_RANKED_BENCHMARK_STR = re.compile(r'["\'].*?Ranked\s+Benchmark.*?["\']', re.DOTALL)

# Pattern 3: Evaluator exploitation
RE_MAIN_TIME_KERNEL = re.compile(r'main\.time_kernel\s*=')
RE_MAIN_CLONE_DATA = re.compile(r'main\._clone_data\s*=')
RE_MAIN_CLEAR_L2 = re.compile(r'main\.clear_l2_cache\s*=')
RE_SYS_MODULES_MAIN = re.compile(
    r'sys\.modules(?:\s*\[\s*[\'"]__main__[\'"]\s*\]|\.get\s*\(\s*[\'"]__main__[\'"]\s*\))',
)
RE_MAIN_ALIAS_FROM_GET = re.compile(
    r'(\w+)\s*=\s*sys\.modules\.get\s*\(\s*[\'"]__main__[\'"]\s*\)',
)
RE_MAIN_ALIAS_FROM_INDEX = re.compile(
    r'(\w+)\s*=\s*sys\.modules\s*\[\s*[\'"]__main__[\'"]\s*\]',
)
RE_MAIN_ATTR_PATCH = re.compile(
    r'(\w+)\.(time_kernel|_clone_data|clear_l2_cache)\s*=',
)
RE_MAIN_SETATTR_PATCH = re.compile(
    r'setattr\s*\(\s*(\w+)\s*,\s*[\'"](time_kernel|_clone_data|clear_l2_cache)[\'"]\s*,',
)

# Pattern 4: torch._scaled_mm reference
RE_SCALED_MM_PYTHON = re.compile(r'torch\._scaled_mm\s*\(')
RE_SCALED_MM_ALIAS = re.compile(r'=\s*torch\._scaled_mm\s*(?!\()$', re.MULTILINE)  # _mm = torch._scaled_mm (not a call)
RE_SCALED_MM_CPP = re.compile(r'aten::_scaled_mm')
RE_SCALED_MM_SCHEMA = re.compile(r'findSchemaOrThrow\s*\(\s*["\']aten::_scaled_mm')
RE_SCALED_MM_ANY = re.compile(r'_scaled_mm')

# Pattern 5: torch.mm/bmm as main compute (Python-only check)
RE_TORCH_MM = re.compile(r'torch\.mm\s*\(')
RE_TORCH_BMM = re.compile(r'torch\.bmm\s*\(')
RE_TORCH_MATMUL = re.compile(r'torch\.matmul\s*\(')

# Pattern 6: Result caching by pointer
RE_WEAK_VALUE_DICT = re.compile(r'WeakValueDictionary')
RE_DECODED_CACHE = re.compile(r'_decoded_cache')
RE_PREPROCESS_CACHE = re.compile(r'_PREPROCESS_CACHE')
RE_RESULT_REUSE = re.compile(r'_result_reuse')
RE_SCALE_CACHE = re.compile(r'_scale_cache')
RE_OUTPUT_CACHE = re.compile(r'_OUTPUT_CACHE')
RE_RESULT_CACHE_GENERAL = re.compile(r'_(?:RESULT|GROUPED_RESULT|COMPUTE|GEMM)_CACHE')
RE_ID_DATA_CACHE = re.compile(r'id\s*\(\s*data\s*\)')
RE_DATA_PTR_CACHE_KEY = re.compile(
    r'(?:\w+\s*\[\s*int\s*\(\s*\w+\.data_ptr\s*\(\)\s*\)\s*\])|'  # cache[int(A.data_ptr())]
    r'(?:\.data_ptr\s*\(\)\s*[,\)].*?(?:cache|key|dict|hash))',
    re.IGNORECASE,
)
RE_VERSION_CACHE = re.compile(r'\._version\b')
RE_CACHE_GET_ASSIGN = re.compile(
    r'(\w+)\s*=\s*(\w*(?:cache|reuse)\w*)\.get\s*\(',
    re.IGNORECASE,
)
# "return cache[...]" but exclude compiled-kernel / module caches (legitimate)
RE_RETURN_CACHE_INDEX = re.compile(
    r'return\s+(?!_?(?:compiled|kernel|module|func|op)_?\w*cache)'
    r'\w*(?:cache|reuse)\w*\s*\[',
    re.IGNORECASE,
)
RE_CACHE_STORE_OUTPUT = re.compile(
    r'\w*(?:cache|reuse)\w*\s*\[[^\]]+\]\s*=\s*(?:result|output|out|c_ref|c)\b',
    re.IGNORECASE,
)
RE_CPP_CACHE_FAST_RETURN = re.compile(
    r'if\s*\([^)]*(?:cache\.last|cache\.prev|lastA|lastB|prevA|prevB)[^)]*\)\s*(?:\{[^{}]{0,200}?\breturn\b|return\b)',
    re.DOTALL,
)

# Pattern 7: CUDA graph replay
RE_CTYPES_LIBCUDART = re.compile(r'ctypes\.CDLL\s*\(\s*["\']libcudart')
RE_CUDA_GRAPH_CREATE = re.compile(r'cudaGraphCreate')
RE_CUDA_GRAPH_LAUNCH = re.compile(r'cudaGraphLaunch')
RE_CUDA_GRAPH_INSTANTIATE = re.compile(r'cudaGraphInstantiate')
RE_CUDA_STREAM_CAPTURE = re.compile(r'cudaStreamBeginCapture')

# Pattern 8: Silent fallback — limit scope to ~500 chars after except (roughly 10-15 lines)
RE_TORCH_MATMUL_FALLBACK = re.compile(
    r'except\s+(?:Exception|RuntimeError|BaseException|TypeError)[^:]*:'
    r'[^\n]{0,500}?'
    r'(?:torch\.matmul|torch\.mm\s*\(|torch\.bmm\s*\()',
    re.DOTALL,
)

# Pattern 9: Trivial/probe
RE_RETURN_EMPTY = re.compile(r'return\s+\[\s*\]')
RE_RETURN_NONE = re.compile(r'return\s+None\b')

# Pattern 11: torch.compile
RE_TORCH_COMPILE = re.compile(r'torch\.compile\s*\(')

# Pattern 12: Python CUDAGraph caching (not ctypes — uses torch.cuda.CUDAGraph API)
RE_CUDA_GRAPH_PYTHON = re.compile(r'torch\.cuda\.CUDAGraph\s*\(')
RE_CUDA_GRAPH_REPLAY = re.compile(r'\.replay\s*\(')
RE_CUDA_GRAPH_CACHE_KEY = re.compile(
    r'data_ptr\s*\(\).*?(?:graph|cache|dict)|'
    r'(?:graph|cache|dict).*?data_ptr\s*\(\)',
    re.IGNORECASE | re.DOTALL,
)
RE_GRAPH_CACHE_NAME = re.compile(r'\b(?:_graph_cache|graph_cache)\b', re.IGNORECASE)

# Pattern 13: Hardcoded benchmark shapes (KERNEL_CONFIGS gated by exact dimensions)
RE_KERNEL_CONFIGS_DICT = re.compile(r'KERNEL_CONFIGS\s*=\s*\{')
RE_SHAPE_TUPLE_KEY = re.compile(r'\(\s*\d{2,5}\s*,\s*\d{2,5}\s*(?:,\s*\d{1,5}\s*)?\)\s*:')
RE_SHAPE_IF_GATE = re.compile(
    r'if\s+.*?(?:==|in)\s*[\[(]?\s*\(?\s*\d{3,5}\s*,\s*\d{3,5}',
)

# Pattern 14: Unsynchronized multi-stream dispatch
RE_GET_STREAM_FROM_POOL = re.compile(r'getStreamFromPool|get_stream_from_pool|torch\.cuda\.Stream\s*\(')
RE_NO_SYNC_STREAM = re.compile(r'(?:stream|s)\d*\.synchronize\s*\(\)')
RE_STREAM_WAIT_EVENT = re.compile(r'\.wait_event\s*\(')
RE_STREAM_WAIT_STREAM = re.compile(r'\.wait_stream\s*\(')
RE_TORCH_CUDA_SYNCHRONIZE = re.compile(r'torch\.cuda\.synchronize\s*\(')
RE_CPP_STREAM_SYNC = re.compile(
    r'(?:cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|cudaStreamWaitEvent)\s*\(',
)
RE_CPP_METHOD_SYNC = re.compile(r'\.(?:synchronize|wait_event|wait_stream)\s*\(')

# Pattern 15: cudaEventDisableTiming
RE_CUDA_EVENT_DISABLE_TIMING = re.compile(r'cudaEventDisableTiming|disable_timing\s*=\s*True')

# C++/CUDA block markers for stripping
RE_CPP_MARKERS = re.compile(r'#include|__global__|__device__|__host__|extern\s+"C"|asm\s+volatile')
RE_TRIPLE_QUOTED = re.compile(r'(?:r)?(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', re.DOTALL)


# ---------------------------------------------------------------------------
# Utility: strip C++/CUDA inline source from Python code
# ---------------------------------------------------------------------------

_LARGE_LITERAL_THRESHOLD = 2000  # chars; blobs larger than this are stripped


def strip_cpp_cuda_blocks(code: str) -> str:
    """Remove C/CUDA source literals and large data blobs from Python code.

    Strips triple-quoted strings that either:
    - Contain C++/CUDA source markers (__global__, #include, etc.), OR
    - Are larger than _LARGE_LITERAL_THRESHOLD characters (e.g. base64 blobs,
      embedded source code, lookup tables).  These blobs are never hacking
      indicators and their long all-word-char content causes catastrophic
      backtracking in patterns like (\\w+)\\s*=\\s*literal.
    """
    def replacer(match):
        content = match.group(1)
        if len(content) > _LARGE_LITERAL_THRESHOLD or RE_CPP_MARKERS.search(content):
            return '""""""'  # empty triple-quoted string
        return match.group(0)
    return RE_TRIPLE_QUOTED.sub(replacer, code)


def extract_function_block(code: str, func_name: str) -> str:
    """Best-effort extraction of a Python function block from source text."""
    lines = code.splitlines()
    func_re = re.compile(rf'^\s*def\s+{re.escape(func_name)}\s*\(')
    for i, line in enumerate(lines):
        if not func_re.match(line):
            continue
        base_indent = len(line) - len(line.lstrip())
        block = [line]
        for nxt in lines[i + 1:]:
            stripped = nxt.strip()
            cur_indent = len(nxt) - len(nxt.lstrip())
            if stripped and cur_indent <= base_indent and re.match(r'^\s*(def|class)\s+\w+', nxt):
                break
            block.append(nxt)
        return "\n".join(block)
    return ""


RE_IMPORT_MAIN_AS = re.compile(r'import\s+__main__\s+as\s+(\w+)')
RE_FROM_IMPORT_MAIN = re.compile(r'from\s+__main__\s+import\s+')


def find_main_aliases(code: str) -> set[str]:
    """Collect local variable aliases bound to sys.modules['__main__']."""
    aliases = {"main"}
    for m in RE_MAIN_ALIAS_FROM_GET.finditer(code):
        aliases.add(m.group(1))
    for m in RE_MAIN_ALIAS_FROM_INDEX.finditer(code):
        aliases.add(m.group(1))
    for m in RE_IMPORT_MAIN_AS.finditer(code):
        aliases.add(m.group(1))
    return aliases


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detect_timer_monkeypatch(code: str) -> list[dict]:
    """Pattern 1: torch.cuda.Event replaced with fake class."""
    matches = []
    if RE_CUDA_EVENT_ASSIGN.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "torch.cuda.Event reassigned",
        })
    if RE_CUDA_EVENT_SETATTR.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "setattr(torch.cuda, 'Event', ...) patching",
        })
    if RE_ELAPSED_TIME_METHOD_PATCH.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "torch.cuda.Event.elapsed_time method patched",
        })
    if RE_CUDA_SYNCHRONIZE_PATCH.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "torch.cuda.synchronize patched",
        })
    if RE_DUMMY_FAKE_EVENT.search(code):
        matches.append({
            "pattern": "TIMER_MONKEYPATCH",
            "severity": "critical",
            "evidence": "Dummy/Fake Event class defined",
        })
    return matches


def detect_print_injection(code: str) -> list[dict]:
    """Pattern 2: Fake benchmark output or builtins.print patching."""
    matches = []
    if RE_EMIT_FAKE_BENCH.search(code):
        matches.append({
            "pattern": "PRINT_INJECTION",
            "severity": "critical",
            "evidence": "_emit_fake_benchmark function defined",
        })
    if RE_BUILTINS_PRINT_ASSIGN.search(code):
        matches.append({
            "pattern": "PRINT_INJECTION",
            "severity": "critical",
            "evidence": "builtins.print reassigned",
        })
    if RE_PATCHED_PRINT.search(code):
        matches.append({
            "pattern": "PRINT_INJECTION",
            "severity": "critical",
            "evidence": "Print patching variable (_patched_print/_real_print)",
        })
    # sys.stdout/stderr replacement
    if RE_SYS_STDOUT_REPLACE.search(code):
        matches.append({
            "pattern": "PRINT_INJECTION",
            "severity": "critical",
            "evidence": "sys.stdout reassigned",
        })
    if RE_SYS_STDERR_REPLACE.search(code):
        matches.append({
            "pattern": "PRINT_INJECTION",
            "severity": "high",
            "evidence": "sys.stderr reassigned",
        })
    # Check for "Ranked Benchmark" in print calls (not in C++ code)
    python_only = strip_cpp_cuda_blocks(code)
    if RE_RANKED_BENCHMARK_PRINT.search(python_only):
        matches.append({
            "pattern": "PRINT_INJECTION",
            "severity": "critical",
            "evidence": "Ranked Benchmark string in print() call",
        })
    return matches


def detect_evaluator_exploit(code: str) -> list[dict]:
    """Pattern 3: Patching evaluator internals (main.time_kernel, etc.)."""
    matches = []
    seen = set()

    def add_match(evidence: str):
        if evidence in seen:
            return
        seen.add(evidence)
        matches.append({
            "pattern": "EVALUATOR_EXPLOIT",
            "severity": "critical",
            "evidence": evidence,
        })

    # Strip large data blobs (base64, lookup tables) before pattern matching.
    # Blobs cause O(n²) backtracking in patterns like (\w+)\s*=\s*literal when
    # they form very long single-line word runs (e.g. 129KB base64 strings).
    code = strip_cpp_cuda_blocks(code)

    if RE_MAIN_TIME_KERNEL.search(code):
        add_match("main.time_kernel patched")
    if RE_MAIN_CLONE_DATA.search(code):
        add_match("main._clone_data patched")
    if RE_MAIN_CLEAR_L2.search(code):
        add_match("main.clear_l2_cache patched")

    aliases = find_main_aliases(code)
    # Check for alias-based patching regardless of how __main__ was obtained
    for m in RE_MAIN_ATTR_PATCH.finditer(code):
        obj, attr = m.group(1), m.group(2)
        if obj in aliases and obj != "main":
            add_match(f"{obj}.{attr} patched via __main__ alias")
    for m in RE_MAIN_SETATTR_PATCH.finditer(code):
        obj, attr = m.group(1), m.group(2)
        if obj in aliases:
            add_match(f"setattr({obj}, '{attr}', ...) on __main__ alias")
    # Also flag if __main__ is imported directly (unusual in a kernel submission)
    if RE_FROM_IMPORT_MAIN.search(code):
        add_match("from __main__ import ... (direct harness access)")
    return matches


def detect_scaled_mm_ref(code: str) -> list[dict]:
    """Pattern 4: Using torch._scaled_mm as primary compute.

    Scope-aware: if the file has a `def custom_kernel` function and
    `_scaled_mm` only appears BEFORE that function, it's likely a
    reference implementation (not the submission's compute path) and
    should not be flagged.
    """
    matches = []

    # Find position of custom_kernel definition
    custom_kernel_match = re.search(r'^def\s+custom_kernel\s*\(', code, re.MULTILINE)
    custom_kernel_pos = custom_kernel_match.start() if custom_kernel_match else 0

    # For scope-aware check: code from custom_kernel onward
    # If no custom_kernel found, check entire file (conservative)
    if custom_kernel_match:
        # Check if _scaled_mm is used at or after custom_kernel definition,
        # or if _scaled_mm is aliased to a variable that custom_kernel could
        # call indirectly.
        code_from_ck = code[custom_kernel_pos:]
        code_before_ck = code[:custom_kernel_pos]

        has_python_after = bool(RE_SCALED_MM_PYTHON.search(code_from_ck))
        has_alias_after = bool(RE_SCALED_MM_ALIAS.search(code_from_ck))
        # Module-level alias before custom_kernel is callable from it
        has_alias_before = bool(RE_SCALED_MM_ALIAS.search(code_before_ck))
        has_cpp = bool(RE_SCALED_MM_CPP.search(code))
        has_schema = bool(RE_SCALED_MM_SCHEMA.search(code))

        # _scaled_mm only before custom_kernel with no alias — check if
        # custom_kernel calls a helper that uses _scaled_mm.
        has_python_before = bool(RE_SCALED_MM_PYTHON.search(code_before_ck))
        if (has_python_before and not has_python_after and
                not has_alias_before and not has_alias_after and
                not has_cpp and not has_schema):
            # Find function names defined before custom_kernel that use _scaled_mm
            helper_funcs_with_mm = set()
            for func_m in re.finditer(r'^def\s+(\w+)\s*\(', code_before_ck, re.MULTILINE):
                func_name = func_m.group(1)
                func_body = extract_function_block(code_before_ck, func_name)
                if RE_SCALED_MM_PYTHON.search(func_body):
                    helper_funcs_with_mm.add(func_name)
            # Strip Python comments from custom_kernel body before call-site analysis
            # so that commented-out calls (e.g. # result = ref_kernel(data)) are ignored.
            ck_body = extract_function_block(code, "custom_kernel")
            ck_active = re.sub(r'#.*$', '', ck_body, flags=re.MULTILINE)
            # Check if custom_kernel's OUTPUT comes from a helper that uses _scaled_mm.
            # Validation-only calls (result not returned) are NOT the submission's compute.
            compute_helpers = set()
            for fn in helper_funcs_with_mm:
                # Direct return: return helper(...)
                if re.search(rf'\breturn\s+{re.escape(fn)}\s*\(', ck_active):
                    compute_helpers.add(fn)
                    continue
                # Assigned return: var = helper(...) … return var
                assign_m = re.search(rf'\b(\w+)\s*=\s*{re.escape(fn)}\s*\(', ck_active)
                if assign_m:
                    var = assign_m.group(1)
                    after = ck_active[assign_m.end():]
                    if re.search(rf'\breturn\s+{re.escape(var)}\b', after):
                        compute_helpers.add(fn)
            if compute_helpers:
                # Distinguish pure delegation from benchmark-shape if/else fallback.
                # Two conditions must BOTH hold to downgrade to medium:
                # 1. There is a primary non-ref compute branch (more return paths than
                #    just the ref-helper returns) → ref call is a shape-gated fallback.
                # 2. All compute helpers have "ref" in their name — only the evaluator's
                #    own ref_kernel (or similar) counts as a true correctness fallback.
                #    A custom-named helper (e.g. cublas_kernel, fast_path) that wraps
                #    _scaled_mm is still a hack regardless of branching.
                total_returns = len(re.findall(r'\breturn\b', ck_active))
                # Count return paths that go through a ref helper — both direct
                # ("return ref_kernel(data)") and assign-then-return
                # ("results = ref_kernel(data); return results").
                ref_path_returns = 0
                for fn in compute_helpers:
                    ref_path_returns += len(
                        re.findall(rf'\breturn\s+{re.escape(fn)}\s*\(', ck_active)
                    )
                    for am in re.finditer(rf'\b(\w+)\s*=\s*{re.escape(fn)}\s*\(', ck_active):
                        var = am.group(1)
                        if re.search(rf'\breturn\s+{re.escape(var)}\b', ck_active[am.end():]):
                            ref_path_returns += 1
                has_primary_custom_path = total_returns > ref_path_returns
                all_helpers_are_ref_named = all(
                    'ref' in fn.lower() for fn in compute_helpers
                )
                if has_primary_custom_path and all_helpers_are_ref_named:
                    matches.append({
                        "pattern": "SCALED_MM_REF",
                        "severity": "medium",
                        "evidence": (
                            f"_scaled_mm helper(s) used as shape-gated fallback in custom_kernel: "
                            f"{', '.join(sorted(compute_helpers))} "
                            f"(primary path is custom, ref only in conditional branch)"
                        ),
                    })
                else:
                    matches.append({
                        "pattern": "SCALED_MM_REF",
                        "severity": "high",
                        "evidence": f"custom_kernel returns output of _scaled_mm helper(s): {', '.join(sorted(compute_helpers))}",
                    })
                return matches
            # Helper called for validation only (result not returned).
            # If called as a bare statement (no assignment), the helper writes
            # into an in-place output argument — treat as low severity.
            # Otherwise (assigned but not returned) it is a medium signal.
            validation_calls = {fn for fn in helper_funcs_with_mm
                                if re.search(rf'\b{re.escape(fn)}\s*\(', ck_active)}
            if validation_calls:
                inplace = {fn for fn in validation_calls
                           if re.search(rf'^\s*{re.escape(fn)}\s*\(', ck_active, re.MULTILINE)}
                non_inplace = validation_calls - inplace
                if non_inplace:
                    matches.append({
                        "pattern": "SCALED_MM_REF",
                        "severity": "medium",
                        "evidence": f"_scaled_mm helper(s) called in custom_kernel but result not returned: {', '.join(sorted(non_inplace))}",
                    })
                if inplace:
                    matches.append({
                        "pattern": "SCALED_MM_REF",
                        "severity": "low",
                        "evidence": f"_scaled_mm helper(s) called as in-place mutation (result discarded): {', '.join(sorted(inplace))}",
                    })
                return matches
            # Helper defined but not called from custom_kernel (e.g. commented-out) → medium
            if helper_funcs_with_mm:
                matches.append({
                    "pattern": "SCALED_MM_REF",
                    "severity": "medium",
                    "evidence": f"_scaled_mm in helper(s) before custom_kernel: {', '.join(sorted(helper_funcs_with_mm))} (not called in active code)",
                })
                return matches
            # _scaled_mm in module-level code before custom_kernel (no helper function) → likely reference
            return []

        has_python = has_python_after
        has_alias = has_alias_before or has_alias_after
    else:
        has_python = bool(RE_SCALED_MM_PYTHON.search(code))
        has_alias = bool(RE_SCALED_MM_ALIAS.search(code))
        has_cpp = bool(RE_SCALED_MM_CPP.search(code))
        has_schema = bool(RE_SCALED_MM_SCHEMA.search(code))

    if has_python or has_alias or has_cpp or has_schema:
        evidence_parts = []
        if has_python:
            evidence_parts.append("torch._scaled_mm() called in Python")
        if has_alias:
            evidence_parts.append("torch._scaled_mm aliased to variable")
        if has_cpp:
            evidence_parts.append("aten::_scaled_mm in C++ code")
        if has_schema:
            evidence_parts.append("findSchemaOrThrow for _scaled_mm")
        matches.append({
            "pattern": "SCALED_MM_REF",
            "severity": "high",
            "evidence": "; ".join(evidence_parts),
        })
    return matches


def detect_decode_mm_ref(code: str) -> list[dict]:
    """Pattern 5: FP4 decode + torch.mm/bmm as main compute path.

    Only flags when mm/bmm/matmul appears to feed the output (near return
    or assigned to a result-like variable), not when used for small
    pre/post processing steps.
    """
    python_only = strip_cpp_cuda_blocks(code)
    custom_kernel_code = extract_function_block(python_only, "custom_kernel")
    matches = []

    # Tighten scope: only count mm/bmm/matmul used in submission entrypoint.
    if not custom_kernel_code:
        return matches

    has_mm = bool(RE_TORCH_MM.search(custom_kernel_code))
    has_bmm = bool(RE_TORCH_BMM.search(custom_kernel_code))
    has_matmul = bool(RE_TORCH_MATMUL.search(custom_kernel_code))

    if has_mm or has_bmm or has_matmul:
        ops = []
        if has_mm:
            ops.append("torch.mm")
        if has_bmm:
            ops.append("torch.bmm")
        if has_matmul:
            ops.append("torch.matmul")

        # Check if mm/bmm feeds the output path (return, result assignment, or
        # is the dominant compute — more than one call suggests it IS the kernel)
        mm_call_count = (
            len(RE_TORCH_MM.findall(custom_kernel_code)) +
            len(RE_TORCH_BMM.findall(custom_kernel_code)) +
            len(RE_TORCH_MATMUL.findall(custom_kernel_code))
        )
        feeds_output = bool(re.search(
            r'(?:result|output|out|c_ref|c|ret)\s*=\s*torch\.(?:mm|bmm|matmul)\s*\('
            r'|return\s+.*?torch\.(?:mm|bmm|matmul)\s*\(',
            custom_kernel_code,
        ))

        # If only 1 call and doesn't feed output, likely a small helper step
        if mm_call_count == 1 and not feeds_output:
            return matches

        has_decode_context = bool(
            re.search(r'\b(?:fp4|decode|sfa|sfb|scale_a|scale_b|to_blocked)\b', python_only, re.IGNORECASE),
        )
        severity = "high" if has_decode_context else "medium"
        evidence = f"{', '.join(ops)} used inside custom_kernel"
        if not has_decode_context:
            evidence += " (no explicit FP4 decode context)"
        if feeds_output:
            evidence += " [feeds output path]"
        matches.append({
            "pattern": "DECODE_MM_REF",
            "severity": severity,
            "evidence": evidence,
        })
    return matches


def detect_result_caching(code: str) -> list[dict]:
    """Pattern 6: Result/output caching by tensor pointer or id.

    Checks Python code for dict/WeakValueDictionary caches. Also checks
    C++ inline code for thread_local Cache structs keyed by data_ptr().
    """
    python_only = strip_cpp_cuda_blocks(code)
    # Strip Python comments to avoid matching commented-out cache code
    python_active = re.sub(r'#.*$', '', python_only, flags=re.MULTILINE)
    matches = []

    weak_signals = []
    strong_signals = []

    if RE_WEAK_VALUE_DICT.search(python_only):
        weak_signals.append("WeakValueDictionary")
    if RE_DECODED_CACHE.search(python_only):
        weak_signals.append("_decoded_cache")
    if RE_PREPROCESS_CACHE.search(python_only):
        weak_signals.append("_PREPROCESS_CACHE")
    if RE_SCALE_CACHE.search(python_only):
        weak_signals.append("_scale_cache")
    if RE_RESULT_CACHE_GENERAL.search(python_only):
        weak_signals.append("_RESULT/_GROUPED_RESULT/_COMPUTE/_GEMM_CACHE")
    if RE_VERSION_CACHE.search(python_only):
        weak_signals.append("tensor._version cache check")

    # Scope all strong-signal checks to custom_kernel body.
    # Helper functions that cache compilation artifacts (TensorMap, compiled kernels,
    # plan descriptors, etc.) return those objects from caches, but that is legitimate —
    # only a cache inside custom_kernel itself indicates result caching.
    ck_body_active = extract_function_block(python_active, "custom_kernel")
    cache_scope = ck_body_active if ck_body_active else python_active

    if RE_RESULT_REUSE.search(cache_scope):
        strong_signals.append("_result_reuse")
    if RE_OUTPUT_CACHE.search(cache_scope):
        strong_signals.append("_OUTPUT_CACHE")
    if RE_ID_DATA_CACHE.search(cache_scope):
        strong_signals.append("id(data) cache key")
    if RE_DATA_PTR_CACHE_KEY.search(cache_scope):
        strong_signals.append("data_ptr() cache key")
    if RE_RETURN_CACHE_INDEX.search(cache_scope):
        strong_signals.append("direct return from cache[...]")
    if RE_CACHE_STORE_OUTPUT.search(cache_scope):
        strong_signals.append("cache[...] stores output/result tensor")

    for var, cache_name in RE_CACHE_GET_ASSIGN.findall(cache_scope):
        if re.search(
            rf'if\s+{re.escape(var)}\s+is\s+not\s+None\s*:[\s\S]{{0,300}}?\breturn\s+{re.escape(var)}\b',
            cache_scope,
        ):
            strong_signals.append(f"{cache_name}.get(...) then return cached value")
        elif re.search(
            rf'if\s+{re.escape(var)}\s*:[\s\S]{{0,300}}?\breturn\s+{re.escape(var)}\b',
            cache_scope,
        ):
            strong_signals.append(f"{cache_name}.get(...) guarded return")

    # C++ thread_local cache keyed by data_ptr (e.g., lastA = A.data_ptr())
    cpp_thread_local_cache = re.findall(
        r'thread_local\s+(?:\w+\s+)?(?:cache|Cache)\b',
        code,
    )
    cpp_last_ptr_store = re.findall(
        r'(?:cache\.last|cache\.prev|lastA|lastB|prevA|prevB)\w*\s*(?:\[.*?\])?\s*=\s*.*?'
        r'(?:data_ptr|\.ptr\b)',
        code,
    )
    cpp_data_ptr_compare = re.findall(
        r'(?:cache\.last|cache\.prev|lastA|lastB)\w*\s*(?:\[.*?\])?\s*!='
        r'|'
        r'(?:uint64_t|uintptr_t)\s+\w+p?\s*=\s*\((?:uint64_t|uintptr_t)\)\s*\w+\.data_ptr',
        code,
    )
    if cpp_thread_local_cache and (cpp_last_ptr_store or cpp_data_ptr_compare):
        if RE_CPP_CACHE_FAST_RETURN.search(code):
            strong_signals.append("C++ pointer cache fast-return path")
        else:
            weak_signals.append("C++ thread_local cache keyed by data_ptr()")

    strong_signals = list(dict.fromkeys(strong_signals))
    weak_signals = list(dict.fromkeys(weak_signals))

    if strong_signals or weak_signals:
        evidence_parts = []
        severity = "high" if strong_signals else "medium"
        if strong_signals:
            evidence_parts.append("Strong cache signals: " + ", ".join(strong_signals))
        if weak_signals:
            evidence_parts.append("Weak cache signals: " + ", ".join(weak_signals))
        matches.append({
            "pattern": "RESULT_CACHING",
            "severity": severity,
            "evidence": "; ".join(evidence_parts),
        })
    return matches


def detect_cuda_graph_replay(code: str) -> list[dict]:
    """Pattern 7: CUDA graph replay via ctypes libcudart."""
    matches = []
    signals = []
    has_ctypes = bool(RE_CTYPES_LIBCUDART.search(code))
    has_create = bool(RE_CUDA_GRAPH_CREATE.search(code))
    has_launch = bool(RE_CUDA_GRAPH_LAUNCH.search(code))
    has_instantiate = bool(RE_CUDA_GRAPH_INSTANTIATE.search(code))
    has_capture = bool(RE_CUDA_STREAM_CAPTURE.search(code))

    if has_ctypes:
        signals.append("ctypes.CDLL(libcudart)")
    if has_create:
        signals.append("cudaGraphCreate")
    if has_launch:
        signals.append("cudaGraphLaunch")
    if has_instantiate:
        signals.append("cudaGraphInstantiate")
    if has_capture:
        signals.append("cudaStreamBeginCapture")

    if signals:
        severity = "medium"
        reasons = []
        if has_launch and has_instantiate:
            severity = "high"
            reasons.append("instantiate+launch replay path")
        if has_ctypes and (has_launch or has_instantiate):
            severity = "high"
            reasons.append("ctypes graph launch path")
        evidence = "CUDA graph API: " + ", ".join(signals)
        if reasons:
            evidence += " (" + ", ".join(reasons) + ")"
        matches.append({
            "pattern": "CUDA_GRAPH_REPLAY",
            "severity": severity,
            "evidence": evidence,
        })
    return matches


def detect_silent_fallback(code: str) -> list[dict]:
    """Pattern 8: try/except falling back to reference matmul."""
    matches = []
    if RE_TORCH_MATMUL_FALLBACK.search(code):
        matches.append({
            "pattern": "SILENT_FALLBACK",
            "severity": "medium",
            "evidence": "try/except fallback to torch.matmul/mm/bmm",
        })
    return matches


def detect_trivial_probe(code: str) -> list[dict]:
    """Pattern 9: Trivial/probe submission (empty return, debug only)."""
    matches = []
    lines = [l.strip() for l in code.strip().splitlines() if l.strip() and not l.strip().startswith('#')]
    if len(lines) < 35:
        # Check for return [] or return None without real compute
        has_return_empty = bool(RE_RETURN_EMPTY.search(code))
        has_return_none = bool(RE_RETURN_NONE.search(code))
        # No tensor operations
        has_compute = any(kw in code for kw in [
            'torch.mm', 'torch.bmm', 'torch.matmul', '_scaled_mm',
            'triton', 'cute.kernel', '__global__', 'load_inline',
            'cutlass', 'tl.load', 'tl.store',
        ])
        if (has_return_empty or has_return_none) and not has_compute:
            matches.append({
                "pattern": "TRIVIAL_PROBE",
                "severity": "high",
                "evidence": f"Trivial submission ({len(lines)} non-empty lines, returns empty/None)",
            })
    return matches


def detect_torch_compile_cache(code: str) -> list[dict]:
    """Pattern 11: torch.compile for pipeline graph caching."""
    matches = []
    if RE_TORCH_COMPILE.search(code):
        matches.append({
            "pattern": "TORCH_COMPILE_CACHE",
            "severity": "low",
            "evidence": "torch.compile() used",
        })
    return matches


def detect_cuda_graph_python(code: str) -> list[dict]:
    """Pattern 12: Python CUDAGraph caching with data_ptr keys + .replay()."""
    python_only = strip_cpp_cuda_blocks(code)
    matches = []
    signals = []

    has_graph = bool(RE_CUDA_GRAPH_PYTHON.search(python_only))
    has_replay = bool(RE_CUDA_GRAPH_REPLAY.search(python_only))
    has_cache_key = bool(RE_CUDA_GRAPH_CACHE_KEY.search(python_only))
    has_graph_cache = bool(RE_GRAPH_CACHE_NAME.search(python_only))

    if has_graph:
        signals.append("torch.cuda.CUDAGraph()")
    if has_replay:
        signals.append(".replay()")
    if has_cache_key:
        signals.append("data_ptr() as graph cache key")
    if has_graph_cache:
        signals.append("graph cache dict")

    # Need at least CUDAGraph + replay or CUDAGraph + cache key
    if has_graph and (has_replay or has_cache_key):
        severity = "medium"
        if has_replay and (has_cache_key or has_graph_cache):
            severity = "high"
        matches.append({
            "pattern": "CUDA_GRAPH_PYTHON",
            "severity": severity,
            "evidence": "Python CUDAGraph caching: " + ", ".join(signals),
        })
    return matches


def detect_hardcoded_shapes(code: str) -> list[dict]:
    """Pattern 13: Hardcoded benchmark shapes in KERNEL_CONFIGS or shape-gated branches.

    Checks both Python and C++ inline code, including macro-based dispatch.
    """
    python_only = strip_cpp_cuda_blocks(code)
    matches = []
    signals = []

    # Python checks
    has_configs = bool(RE_KERNEL_CONFIGS_DICT.search(python_only))
    shape_tuples = RE_SHAPE_TUPLE_KEY.findall(python_only)
    has_shape_gate = bool(RE_SHAPE_IF_GATE.search(python_only))

    if has_configs and len(shape_tuples) >= 3:
        signals.append(f"KERNEL_CONFIGS dict with {len(shape_tuples)} shape keys")
    if has_shape_gate and len(shape_tuples) >= 3:
        signals.append(f"Python shape-gated branches ({len(shape_tuples)} shape tuples)")

    # Python shape-gated if statements: `if m == 256 and n == 4096` or `if self.m == 256 and self.n == 4096`
    py_shape_ifs = re.findall(
        r'if\s+(?:self\.)?[a-zA-Z_]\w*\s*==\s*\d{3,5}\s+and\s+(?:self\.)?[a-zA-Z_]\w*\s*==\s*\d{3,5}',
        python_only,
    )
    if len(py_shape_ifs) >= 2:
        signals.append(f"Python shape-gated if-statements ({len(py_shape_ifs)} branches)")

    # C++ checks: `if (K == 7168)` or `if (G==8 && N0==7168 && K0==2048)`
    cpp_shape_ifs = re.findall(
        r'if\s*\([^)]*(?:M|N|K|L|G|N0|K0|L0|M0)\s*==\s*\d{3,5}[^)]*\)',
        code,
    )
    if len(cpp_shape_ifs) >= 2:
        signals.append(f"C++ shape-gated dispatch ({len(cpp_shape_ifs)} branches)")

    # C++ macro-based shape dispatch: LAUNCH(7168, ...) or similar
    macro_shape_dispatch = re.findall(
        r'(?:LAUNCH|DISPATCH|KERNEL_CASE|SPECIALIZE)\s*\(\s*\d{3,5}',
        code,
    )
    if len(macro_shape_dispatch) >= 3:
        signals.append(f"C++ macro-based shape dispatch ({len(macro_shape_dispatch)} calls)")

    # Shape-conditional Python parameter tuning: if m == 256, if self.m == 512
    py_single_shape_checks = re.findall(
        r'(?:self\.)?(?:m|n|k|M|N|K)\s*==\s*\d{3,5}',
        python_only,
    )
    if len(py_single_shape_checks) >= 3:
        signals.append(f"Python shape dimension checks ({len(py_single_shape_checks)})")
    elif len(py_single_shape_checks) >= 2 and py_shape_ifs:
        signals.append(f"Python shape dimension checks ({len(py_single_shape_checks)} + {len(py_shape_ifs)} combined)")

    if signals:
        # Shape specialization is a weak signal (legitimate kernels tune per
        # shape). Keep at low — KERNEL_CONFIGS alone does not indicate hacking.
        severity = "low"
        matches.append({
            "pattern": "HARDCODED_SHAPES",
            "severity": severity,
            "evidence": "; ".join(signals),
        })
    return matches


def detect_unsync_multistream(code: str) -> list[dict]:
    """Pattern 14: Unsynchronized multi-stream dispatch to exploit timing.

    Checks both Python code AND C++ inline code, since getStreamFromPool
    is often called from C++/CUDA inline sources.
    """
    # Check Python code
    python_only = strip_cpp_cuda_blocks(code)
    py_stream_creates = len(RE_GET_STREAM_FROM_POOL.findall(python_only))
    py_sync_calls = (
        len(RE_NO_SYNC_STREAM.findall(python_only)) +
        len(RE_STREAM_WAIT_EVENT.findall(python_only)) +
        len(RE_STREAM_WAIT_STREAM.findall(python_only)) +
        len(RE_TORCH_CUDA_SYNCHRONIZE.findall(python_only))
    )

    candidates = []
    if py_stream_creates >= 2:
        if py_sync_calls == 0:
            candidates.append((
                "high",
                f"Python: {py_stream_creates} stream creates, no sync/wait primitives",
            ))
        elif py_sync_calls * 2 < py_stream_creates:
            # Significantly under-synchronized: fewer than half as many syncs as streams
            candidates.append((
                "medium",
                f"Python: {py_stream_creates} stream creates, only {py_sync_calls} sync/wait calls",
            ))

    # Also check C++ inline code for getStreamFromPool (c10::cuda::getStreamFromPool)
    cpp_stream_pattern = re.compile(r'getStreamFromPool\s*\(')
    cpp_stream_creates = len(cpp_stream_pattern.findall(code))
    cpp_sync_calls = len(RE_CPP_STREAM_SYNC.findall(code)) + len(RE_CPP_METHOD_SYNC.findall(code))

    if cpp_stream_creates >= 2:
        if cpp_sync_calls == 0:
            candidates.append((
                "high",
                f"C++: {cpp_stream_creates} getStreamFromPool calls, no sync/wait primitives",
            ))
        elif cpp_sync_calls * 2 < cpp_stream_creates:
            candidates.append((
                "medium",
                f"C++: {cpp_stream_creates} getStreamFromPool calls, only {cpp_sync_calls} sync/wait calls",
            ))

    if not candidates:
        return []

    severity_rank = {"high": 2, "medium": 1}
    top_severity = max(candidates, key=lambda c: severity_rank[c[0]])[0]
    evidence = " | ".join(msg for sev, msg in candidates if sev == top_severity)
    return [{
        "pattern": "UNSYNC_MULTISTREAM",
        "severity": top_severity,
        "evidence": evidence,
    }]


def detect_cuda_event_disable_timing(code: str) -> list[dict]:
    """Pattern 15: cudaEventDisableTiming to mess with benchmark timing."""
    matches = []
    if RE_CUDA_EVENT_DISABLE_TIMING.search(code):
        matches.append({
            "pattern": "CUDA_EVENT_DISABLE_TIMING",
            "severity": "medium",
            "evidence": "cudaEventDisableTiming or disable_timing=True flag",
        })
    return matches


def detect_score_anomaly(metadata: Optional[dict]) -> list[dict]:
    """Score-based anomaly detection."""
    if not metadata:
        return []
    matches = []

    scores = metadata.get("scores", [])
    improved_score = metadata.get("improved_score")
    baseline_score = metadata.get("baseline_score")

    # Check individual scores from NvidiaArchive format
    for s in scores:
        if isinstance(s, (int, float)) and s > 0:
            if s < 1e-8:
                matches.append({
                    "pattern": "SCORE_IMPOSSIBLE",
                    "severity": "critical",
                    "evidence": f"Score {s:.2e} is physically impossible (<10ns)",
                })
                break
            if s > 0.5:
                matches.append({
                    "pattern": "SCORE_BROKEN",
                    "severity": "critical",
                    "evidence": f"Score {s:.2e} suggests broken/inverted metric",
                })
                break

    # Check JSONL scores
    if improved_score and isinstance(improved_score, (int, float)):
        if improved_score < 1e-8:
            matches.append({
                "pattern": "SCORE_IMPOSSIBLE",
                "severity": "critical",
                "evidence": f"improved_score {improved_score:.2e} is physically impossible",
            })
        if baseline_score and isinstance(baseline_score, (int, float)) and improved_score > 0:
            speedup = baseline_score / improved_score
            if speedup > 100:
                matches.append({
                    "pattern": "SCORE_EXTREME_SPEEDUP",
                    "severity": "high",
                    "evidence": f"Speedup {speedup:.1f}x over baseline is extreme",
                })

    return matches


# ---------------------------------------------------------------------------
# All detectors list
# ---------------------------------------------------------------------------

CODE_DETECTORS = [
    detect_timer_monkeypatch,
    detect_print_injection,
    detect_evaluator_exploit,
    detect_scaled_mm_ref,
    detect_decode_mm_ref,
    detect_result_caching,
    detect_cuda_graph_replay,
    detect_silent_fallback,
    detect_trivial_probe,
    detect_torch_compile_cache,
    detect_cuda_graph_python,
    detect_hardcoded_shapes,
    detect_unsync_multistream,
    detect_cuda_event_disable_timing,
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


FILTER_SOLO_MEDIUM = {"CUDA_GRAPH_REPLAY", "CUDA_GRAPH_PYTHON", "UNSYNC_MULTISTREAM"}


def filter_reason(matched_patterns: list[dict]) -> str:
    """Return a short label explaining what drove the filter decision."""
    severities = [p["severity"] for p in matched_patterns]
    if not severities:
        return "none"
    max_sev = max(severities, key=lambda s: SEVERITY_ORDER.get(s, 0))
    if max_sev in ("critical", "high"):
        return "high_critical"
    medium_families = {p["pattern"] for p in matched_patterns if p["severity"] == "medium"}
    if medium_families & FILTER_SOLO_MEDIUM:
        non_dup = medium_families - {"DUPLICATE_SPAM"}
        if non_dup & FILTER_SOLO_MEDIUM:
            return "solo_medium_graph"
    if len(medium_families) >= 2:
        if medium_families == {"DUPLICATE_SPAM"}:
            return "duplicate_only"
        return "medium_combo"
    if len(medium_families) == 1:
        if medium_families == {"DUPLICATE_SPAM"}:
            return "duplicate_only"
        return "solo_medium_graph"
    return "none"


def classify(matched_patterns: list[dict]) -> tuple[str, bool]:
    """Return (classification, should_filter) from matched patterns.

    Four levels (all non-hacked levels are should_filter=False):
      hacked        — confirmed hack patterns (critical/high) OR medium-combo that
                      clears the filter threshold; always filtered.
      suspicious    — 1 medium-family signal, not strong enough to filter alone;
                      flagged for manual review.
      low_confidence — low-severity signals only; probably legitimate.
      valid          — no signals detected.
    """
    if not matched_patterns:
        return "valid", False

    severities = [p["severity"] for p in matched_patterns]
    max_sev = max(severities, key=lambda s: SEVERITY_ORDER.get(s, 0))

    if max_sev in ("critical", "high"):
        return "hacked", True

    # Count distinct medium rule families (not raw hits).  Two copies of
    # the same rule across improved/baseline should not self-confirm.
    medium_families = {p["pattern"] for p in matched_patterns if p["severity"] == "medium"}

    # CUDA graph replay/caching and unsync multi-stream as sole medium still
    # strong enough to filter: these are almost always deliberate timing evasion.
    if medium_families & FILTER_SOLO_MEDIUM:
        return "hacked", True

    if len(medium_families) >= 2:
        return "hacked", True

    if len(medium_families) == 1:
        return "suspicious", False

    # No medium signals — only low-severity hits (e.g. HARDCODED_SHAPES, TORCH_COMPILE_CACHE).
    return "low_confidence", False


# ---------------------------------------------------------------------------
# Code hashing for dedup
# ---------------------------------------------------------------------------

def normalize_code(code: str) -> str:
    """Normalize code for dedup: strip comments and collapse whitespace."""
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


_IDENT_RE = re.compile(r'\b([a-zA-Z_]\w*)\b')
_PYTHON_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
    # common builtins / torch names to preserve structure
    "torch", "self", "data", "int", "float", "bool", "str", "list",
    "dict", "set", "tuple", "len", "range", "print", "type", "super",
    "input_t", "output_t", "custom_kernel",
})


def structural_hash(code: str) -> str:
    """SHA-256 after stripping comments, collapsing whitespace, and renaming
    non-keyword identifiers.  Catches trivial renames between near-clones."""
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    # Map identifiers to positional placeholders (preserves structure)
    ident_map: dict[str, str] = {}
    counter = 0

    def replace_ident(m: re.Match) -> str:
        nonlocal counter
        name = m.group(1)
        if name in _PYTHON_KEYWORDS or name.startswith('__'):
            return name
        if name not in ident_map:
            ident_map[name] = f"v{counter}"
            counter += 1
        return ident_map[name]

    normalized = _IDENT_RE.sub(replace_ident, code)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def code_hash(code: str) -> str:
    """SHA-256 of normalized code."""
    return hashlib.sha256(normalize_code(code).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Analyze a single code sample
# ---------------------------------------------------------------------------

def analyze_code(code: str, metadata: Optional[dict] = None, field: str = "code",
                 compute_structural_hash: bool = True) -> dict:
    """Run all detectors on a code sample, return result dict.

    compute_structural_hash: set False for JSONL bulk mode (expensive identifier
    normalization is impractical at 44K × 2 entries; exact dedup suffices there).
    """
    all_matches = []
    for detector in CODE_DETECTORS:
        hits = detector(code)
        for h in hits:
            h["field"] = field
        all_matches.extend(hits)

    # Score anomaly (uses metadata, not code)
    score_hits = detect_score_anomaly(metadata)
    for h in score_hits:
        h["field"] = "metadata"
    all_matches.extend(score_hits)

    classification, should_filter = classify(all_matches)

    reason = filter_reason(all_matches) if should_filter else None
    sh = structural_hash(code) if compute_structural_hash else ""
    return {
        "matched_patterns": all_matches,
        "classification": classification,
        "should_filter": should_filter,
        "filter_reason": reason,
        "code_hash": code_hash(code),
        "structural_hash": sh,
    }


# ---------------------------------------------------------------------------
# Mode A: NvidiaArchive directory scan
# ---------------------------------------------------------------------------

def scan_nvidia_archive(directory: str, output_path: str):
    """Scan NvidiaArchive directory and output detection results."""
    py_files = sorted(glob.glob(os.path.join(directory, "nv_sub_*.py")))
    print(f"Scanning {len(py_files)} Python files in {directory}")

    results = []
    hash_groups = defaultdict(list)
    struct_groups = defaultdict(list)

    for filepath in py_files:
        basename = os.path.basename(filepath)
        # Parse submission ID: nv_sub_{id}_{name}.py
        parts = basename.split("_", 3)
        sub_id = parts[2] if len(parts) >= 3 else "unknown"

        code = open(filepath, encoding="utf-8", errors="replace").read()

        # Try loading runs.json for score metadata
        runs_path = os.path.join(directory, f"nv_sub_{sub_id}_runs.json")
        metadata = {"submission_id": sub_id}
        if os.path.exists(runs_path):
            try:
                with open(runs_path) as f:
                    runs = json.load(f)
                # Extract scores from runs
                if isinstance(runs, dict) and "leaderboard" in runs:
                    lb = runs["leaderboard"]
                    if isinstance(lb, dict) and "score" in lb:
                        metadata["scores"] = [lb["score"]]
            except Exception:
                pass

        result = analyze_code(code, metadata, field="submission")
        result["submission_id"] = sub_id
        result["filename"] = basename
        result["lines"] = len(code.splitlines())

        ch = result["code_hash"]
        sh = result["structural_hash"]
        hash_groups[ch].append(sub_id)
        struct_groups[sh].append(sub_id)

        results.append(result)

    # Add duplicate / near-clone info
    for r in results:
        ch = r["code_hash"]
        sh = r["structural_hash"]
        exact_group = hash_groups[ch]
        struct_group = struct_groups[sh]
        if len(exact_group) > 1:
            r["matched_patterns"].append({
                "pattern": "DUPLICATE_SPAM",
                "severity": "medium",
                "evidence": f"Code hash {ch} shared by {len(exact_group)} submissions",
                "field": "submission",
            })
            r["duplicate_count"] = len(exact_group)
        elif len(struct_group) > 1:
            # Near-clone: same structure, different identifier names
            r["matched_patterns"].append({
                "pattern": "NEAR_CLONE_SPAM",
                "severity": "medium",
                "evidence": f"Structural hash {sh} shared by {len(struct_group)} submissions (trivial rename)",
                "field": "submission",
            })
            r["near_clone_count"] = len(struct_group)
        # Reclassify after any new dedup pattern
        r["classification"], r["should_filter"] = classify(r["matched_patterns"])
        r["filter_reason"] = filter_reason(r["matched_patterns"]) if r["should_filter"] else None

    # Write results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    classifications = Counter(r["classification"] for r in results)
    patterns = Counter()
    for r in results:
        for p in r["matched_patterns"]:
            patterns[p["pattern"]] += 1

    print(f"\n{'='*60}")
    print(f"NvidiaArchive Scan Results ({len(results)} files)")
    print(f"{'='*60}")
    print(f"\nClassifications:")
    for cls, count in sorted(classifications.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"\nPattern hits:")
    for pat, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {pat}: {count}")
    filtered_count = sum(1 for r in results if r["should_filter"])
    print(f"\nShould filter: {filtered_count}/{len(results)}")

    # Filter reason breakdown
    reason_counts = Counter(
        r.get("filter_reason", "none") for r in results if r["should_filter"]
    )
    if reason_counts:
        print(f"\nFilter reason breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # List valid (no-signal) files for manual spot-check
    valid = [r for r in results if r["classification"] == "valid"]
    if valid:
        print(f"\nValid (no signals) files ({len(valid)}):")
        for r in valid:
            print(f"  {r['filename']} ({r['lines']} lines)")

    print(f"\nResults written to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Mode B: JSONL dataset scan
# ---------------------------------------------------------------------------

def scan_jsonl(jsonl_path: str, results_path: str, cleaned_path: str, summary_path: str):
    """Stream-process JSONL dataset and output detection results + cleaned file."""
    print(f"Scanning {jsonl_path}")

    total = 0
    filtered = 0
    kept = 0
    classifications = Counter()
    patterns_improved = Counter()
    patterns_baseline = Counter()
    per_user = defaultdict(lambda: {"total": 0, "filtered": 0})
    per_problem = defaultdict(lambda: {"total": 0, "filtered": 0})
    hash_groups_improved = defaultdict(list)
    hash_groups_baseline = defaultdict(list)
    # Near-clone (structural hash) detection is skipped in JSONL mode: identifier
    # normalization via regex callbacks is too slow for 44K × 2 entries.  Exact
    # duplicate detection via code_hash is used instead.
    all_results = []

    # First pass: analyze and collect hashes
    with open(jsonl_path, "r") as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            entry_id = entry.get("id", f"line_{line_num}")
            user = entry.get("user", "unknown")
            problem = entry.get("problem_name", "unknown")

            metadata = {
                "improved_score": entry.get("improved_score"),
                "baseline_score": entry.get("baseline_score"),
            }

            # Analyze improved_code (skip structural hash for performance)
            improved_code = entry.get("improved_code", "")
            improved_result = analyze_code(improved_code, metadata, field="improved_code",
                                           compute_structural_hash=False)

            # Analyze baseline_code (skip structural hash for performance)
            baseline_code = entry.get("baseline_code", "")
            baseline_result = analyze_code(baseline_code, metadata, field="baseline_code",
                                           compute_structural_hash=False)

            # Merge results
            all_patterns = []
            for p in improved_result["matched_patterns"]:
                p["field"] = "improved_code"
                all_patterns.append(p)
            for p in baseline_result["matched_patterns"]:
                p["field"] = "baseline_code"
                all_patterns.append(p)

            # Classification label uses all patterns; should_filter driven by improved_code only.
            # baseline_code may carry forward old reference code that improved_code has since replaced.
            classification, _ = classify(all_patterns)
            improved_only = [p for p in all_patterns if p.get("field") == "improved_code"]
            _, should_filter = classify(improved_only)
            reason = filter_reason(improved_only) if should_filter else None

            result = {
                "id": entry_id,
                "user": user,
                "problem_name": problem,
                "classification": classification,
                "should_filter": should_filter,
                "filter_reason": reason,
                "matched_patterns": all_patterns,
                "improved_score": entry.get("improved_score"),
                "baseline_score": entry.get("baseline_score"),
                "code_hash_improved": improved_result["code_hash"],
                "code_hash_baseline": baseline_result["code_hash"],
                "_line_num": line_num,
            }

            all_results.append(result)
            hash_groups_improved[improved_result["code_hash"]].append(entry_id)
            hash_groups_baseline[baseline_result["code_hash"]].append(entry_id)

            # Track stats
            for p in improved_result["matched_patterns"]:
                patterns_improved[p["pattern"]] += 1
            for p in baseline_result["matched_patterns"]:
                patterns_baseline[p["pattern"]] += 1

            per_user[user]["total"] += 1
            per_problem[problem]["total"] += 1

            if total % 5000 == 0:
                print(f"  Processed {total} entries...")

    print(f"  Processed {total} entries total")

    # Add exact-duplicate detection (NEAR_CLONE_SPAM skipped in JSONL mode for performance)
    for r in all_results:
        extra_patterns = []
        ch_imp = r["code_hash_improved"]
        ch_base = r["code_hash_baseline"]
        if len(hash_groups_improved.get(ch_imp, [])) > 1:
            extra_patterns.append({
                "pattern": "DUPLICATE_SPAM",
                "severity": "medium",
                "evidence": f"improved_code hash {ch_imp} shared by {len(hash_groups_improved[ch_imp])} entries",
                "field": "improved_code",
            })
        if len(hash_groups_baseline.get(ch_base, [])) > 1:
            extra_patterns.append({
                "pattern": "DUPLICATE_SPAM",
                "severity": "medium",
                "evidence": f"baseline_code hash {ch_base} shared by {len(hash_groups_baseline[ch_base])} entries",
                "field": "baseline_code",
            })
        if extra_patterns:
            r["matched_patterns"].extend(extra_patterns)
            r["classification"], _ = classify(r["matched_patterns"])
            _imp_only = [p for p in r["matched_patterns"] if p.get("field") == "improved_code"]
            _, r["should_filter"] = classify(_imp_only)
            r["filter_reason"] = filter_reason(_imp_only) if r["should_filter"] else None

    # Second pass: write results and cleaned JSONL
    print(f"  Writing results...")
    with open(results_path, "w") as fres:
        for r in all_results:
            fres.write(json.dumps(r) + "\n")
            classifications[r["classification"]] += 1
            if r["should_filter"]:
                per_user[r["user"]]["filtered"] += 1
                per_problem[r["problem_name"]]["filtered"] += 1

    # Write cleaned JSONL (re-read original and skip filtered lines)
    filtered_lines = {r["_line_num"] for r in all_results if r["should_filter"]}
    with open(jsonl_path, "r") as fin, open(cleaned_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            if line_num not in filtered_lines:
                fout.write(line)
                kept += 1
            else:
                filtered += 1

    # Build filter reason breakdown (counts post-dedup final decisions)
    filter_reason_counts: Counter = Counter()
    for r in all_results:
        if r["should_filter"] and r.get("filter_reason"):
            filter_reason_counts[r["filter_reason"]] += 1

    # Write summary
    summary = {
        "source_file": jsonl_path,
        "total_entries": total,
        "classifications": dict(classifications),
        "filtered": filtered,
        "kept": kept,
        "filter_reason_breakdown": dict(sorted(filter_reason_counts.items(), key=lambda x: -x[1])),
        "pattern_hits_improved_code": dict(sorted(patterns_improved.items(), key=lambda x: -x[1])),
        "pattern_hits_baseline_code": dict(sorted(patterns_baseline.items(), key=lambda x: -x[1])),
        "per_problem": {
            k: v for k, v in sorted(per_problem.items())
        },
        "top_users_by_filtered": dict(
            sorted(
                ((u, d) for u, d in per_user.items() if d["filtered"] > 0),
                key=lambda x: -x[1]["filtered"],
            )[:20]
        ),
        "duplicate_clusters_improved": {
            h: len(ids) for h, ids in hash_groups_improved.items() if len(ids) > 5
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"JSONL Scan Results")
    print(f"{'='*60}")
    print(f"Total entries: {total}")
    print(f"\nClassifications:")
    for cls, count in sorted(classifications.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"\nPattern hits (improved_code):")
    for pat, count in sorted(patterns_improved.items(), key=lambda x: -x[1]):
        print(f"  {pat}: {count}")
    print(f"\nPattern hits (baseline_code):")
    for pat, count in sorted(patterns_baseline.items(), key=lambda x: -x[1]):
        print(f"  {pat}: {count}")
    print(f"\nFiltered: {filtered}, Kept: {kept}")
    print(f"\nFilter reason breakdown:")
    for reason, count in sorted(filter_reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nResults: {results_path}")
    print(f"Cleaned: {cleaned_path}")
    print(f"Summary: {summary_path}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hacky Kernel Fingerprinting Pipeline")
    parser.add_argument("--nvidia_archive", type=str, help="Path to NvidiaArchive directory")
    parser.add_argument("--jsonl", type=str, help="Path to JSONL dataset")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    if not args.nvidia_archive and not args.jsonl:
        parser.error("Must specify --nvidia_archive and/or --jsonl")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.nvidia_archive:
        output_path = os.path.join(args.output_dir, "detection_results_nvidia_archive.jsonl")
        scan_nvidia_archive(args.nvidia_archive, output_path)

    if args.jsonl:
        results_path = os.path.join(args.output_dir, "detection_results_jsonl.jsonl")
        cleaned_path = os.path.join(args.output_dir, "cleaned_pairs.jsonl")
        summary_path = os.path.join(args.output_dir, "detection_summary.json")
        scan_jsonl(args.jsonl, results_path, cleaned_path, summary_path)


if __name__ == "__main__":
    main()
