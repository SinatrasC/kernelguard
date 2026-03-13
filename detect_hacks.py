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
import ast
from dataclasses import dataclass, field
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import sys
import tokenize
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
RE_SCALED_MM_ALIAS_ASSIGN = re.compile(r'^\s*(\w+)\s*=\s*torch\._scaled_mm\s*$', re.MULTILINE)
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
    r'\w*(?:cache|reuse)\w*\s*\[[^\]]+\]\s*=\s*(?:result|output|out|c_ref)\b',
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


def strip_python_comments(code: str) -> str:
    """Remove Python comments without touching string literals."""
    try:
        tokens = [
            tok
            for tok in tokenize.generate_tokens(io.StringIO(code).readline)
            if tok.type != tokenize.COMMENT
        ]
    except tokenize.TokenError:
        return code
    return tokenize.untokenize(tokens)


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


def find_scaled_mm_aliases(code: str) -> set[str]:
    """Collect local aliases bound directly to torch._scaled_mm."""
    aliases = set()
    for m in re.finditer(r'^\s*(\w+)\s*=\s*torch\._scaled_mm\s*$', code, re.MULTILINE):
        aliases.add(m.group(1))
    return aliases


def function_uses_scaled_mm(func_body: str, alias_names: set[str]) -> bool:
    """Return True when a function body calls torch._scaled_mm or one of its aliases."""
    if RE_SCALED_MM_PYTHON.search(func_body):
        return True
    for alias in alias_names:
        if re.search(rf'\b{re.escape(alias)}\s*\(', func_body):
            return True
    return False


@dataclass
class SubmissionFacts:
    """Shared normalized views and AST summaries for one submission."""

    raw_code: str
    python_only: str
    python_active: str
    ast_tree: Optional[ast.AST]
    main_aliases: set[str]
    scaled_mm_aliases: set[str]
    trusted_aliases: dict[str, str]
    custom_kernel_pos: Optional[int]
    code_before_custom_kernel: str
    code_from_custom_kernel: str
    custom_kernel_code: str
    custom_kernel_active: str
    _function_blocks: dict[str, str] = field(default_factory=dict)
    _active_function_blocks: dict[str, str] = field(default_factory=dict)

    def get_function_block(self, func_name: str) -> str:
        block = self._function_blocks.get(func_name)
        if block is None:
            block = extract_function_block(self.raw_code, func_name)
            self._function_blocks[func_name] = block
        return block

    def get_active_function_block(self, func_name: str) -> str:
        block = self._active_function_blocks.get(func_name)
        if block is None:
            block = strip_python_comments(self.get_function_block(func_name))
            self._active_function_blocks[func_name] = block
        return block


def build_submission_facts(code: str) -> SubmissionFacts:
    """Parse and normalize a submission once for reuse across all detectors."""
    python_only = strip_cpp_cuda_blocks(code)
    python_active = strip_python_comments(python_only)
    tree = _safe_ast_parse(code)
    custom_kernel_match = re.search(r'^def\s+custom_kernel\s*\(', code, re.MULTILINE)
    custom_kernel_pos = custom_kernel_match.start() if custom_kernel_match else None
    code_before_custom_kernel = code[:custom_kernel_pos] if custom_kernel_pos is not None else code
    code_from_custom_kernel = code[custom_kernel_pos:] if custom_kernel_pos is not None else code
    custom_kernel_code = extract_function_block(code, "custom_kernel")
    custom_kernel_active = strip_python_comments(custom_kernel_code)
    trusted_aliases = _collect_trusted_aliases(tree) if tree is not None else {}

    facts = SubmissionFacts(
        raw_code=code,
        python_only=python_only,
        python_active=python_active,
        ast_tree=tree,
        main_aliases=find_main_aliases(python_only),
        scaled_mm_aliases=find_scaled_mm_aliases(code_before_custom_kernel),
        trusted_aliases=trusted_aliases,
        custom_kernel_pos=custom_kernel_pos,
        code_before_custom_kernel=code_before_custom_kernel,
        code_from_custom_kernel=code_from_custom_kernel,
        custom_kernel_code=custom_kernel_code,
        custom_kernel_active=custom_kernel_active,
    )
    facts._function_blocks["custom_kernel"] = custom_kernel_code
    facts._active_function_blocks["custom_kernel"] = custom_kernel_active
    return facts


def ensure_submission_facts(code_or_facts: str | SubmissionFacts) -> SubmissionFacts:
    """Accept a raw code string or a pre-built SubmissionFacts object."""
    if isinstance(code_or_facts, SubmissionFacts):
        return code_or_facts
    return build_submission_facts(code_or_facts)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detect_timer_monkeypatch(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 1: torch.cuda.Event replaced with fake class."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
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


def detect_print_injection(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 2: separate real benchmark spoofing from benign stdio handling."""
    facts = ensure_submission_facts(code)
    python_active = facts.python_active
    matches = []
    seen: set[tuple[str, str]] = set()

    def add(pattern: str, severity: str, evidence: str):
        key = (pattern, evidence)
        if key in seen:
            return
        seen.add(key)
        matches.append({
            "pattern": pattern,
            "severity": severity,
            "evidence": evidence,
        })

    fake_emit = False
    if RE_EMIT_FAKE_BENCH.search(python_active):
        fake_emit = True
        add("FAKE_BENCHMARK_EMIT", "critical", "_emit_fake_benchmark function defined")
    if RE_BUILTINS_PRINT_ASSIGN.search(python_active):
        fake_emit = True
        add("FAKE_BENCHMARK_EMIT", "critical", "builtins.print reassigned")
    if RE_RANKED_BENCHMARK_PRINT.search(python_active):
        fake_emit = True
        add("FAKE_BENCHMARK_EMIT", "critical", "Ranked Benchmark string printed directly")
    elif RE_PATCHED_PRINT.search(python_active) and RE_RANKED_BENCHMARK_STR.search(python_active):
        fake_emit = True
        add("FAKE_BENCHMARK_EMIT", "critical", "print patching plus Ranked Benchmark payload")

    # Stdio redirection is common as a compatibility shim around compilation and
    # should not be treated as benchmark spoofing on its own.
    if RE_SYS_STDOUT_REPLACE.search(python_active):
        if re.search(r'if\s+sys\.stdout\s+is\s+None\s*:', python_active):
            add("STDIO_REDIRECT", "low", "guarded sys.stdout compatibility redirect")
        else:
            add("STDIO_REDIRECT", "medium", "sys.stdout reassigned")
    if RE_SYS_STDERR_REPLACE.search(python_active):
        if re.search(r'if\s+sys\.stderr\s+is\s+None\s*:', python_active):
            add("STDIO_REDIRECT", "low", "guarded sys.stderr compatibility redirect")
        else:
            add("STDIO_REDIRECT", "medium", "sys.stderr reassigned")

    # Do not keep the old broad PRINT_INJECTION marker; the split rules carry
    # the action semantics now.
    return matches


def detect_evaluator_exploit(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 3: Patching evaluator internals (main.time_kernel, etc.)."""
    facts = ensure_submission_facts(code)
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
    code = facts.python_only

    if RE_MAIN_TIME_KERNEL.search(code):
        add_match("main.time_kernel patched")
    if RE_MAIN_CLONE_DATA.search(code):
        add_match("main._clone_data patched")
    if RE_MAIN_CLEAR_L2.search(code):
        add_match("main.clear_l2_cache patched")

    aliases = facts.main_aliases
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


def detect_scaled_mm_ref(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 4: Using torch._scaled_mm as primary compute.

    Scope-aware: if the file has a `def custom_kernel` function and
    `_scaled_mm` only appears BEFORE that function, it's likely a
    reference implementation (not the submission's compute path) and
    should not be flagged.
    """
    facts = ensure_submission_facts(code)
    matches = []
    code = facts.raw_code
    custom_kernel_pos = facts.custom_kernel_pos or 0

    # For scope-aware check: code from custom_kernel onward
    # If no custom_kernel found, check entire file (conservative)
    if facts.custom_kernel_pos is not None:
        # Check if _scaled_mm is used at or after custom_kernel definition,
        # or if _scaled_mm is aliased to a variable that custom_kernel could
        # call indirectly.
        code_from_ck = facts.code_from_custom_kernel
        code_before_ck = facts.code_before_custom_kernel

        has_python_after = bool(RE_SCALED_MM_PYTHON.search(code_from_ck))
        has_alias_after = bool(RE_SCALED_MM_ALIAS.search(code_from_ck))
        alias_names_before = facts.scaled_mm_aliases
        has_alias_before = bool(alias_names_before)
        has_cpp = bool(RE_SCALED_MM_CPP.search(code))
        has_schema = bool(RE_SCALED_MM_SCHEMA.search(code))

        # _scaled_mm only before custom_kernel — check if custom_kernel actually
        # reaches that code path through a helper or a direct alias call.
        has_python_before = bool(RE_SCALED_MM_PYTHON.search(code_before_ck))
        if ((has_python_before or has_alias_before) and not has_python_after and
                not has_alias_after and
                not has_cpp and not has_schema):
            # Find function names defined before custom_kernel that use _scaled_mm
            helper_funcs_with_mm = set()
            for func_m in re.finditer(r'^def\s+(\w+)\s*\(', code_before_ck, re.MULTILINE):
                func_name = func_m.group(1)
                func_body = facts.get_function_block(func_name)
                if function_uses_scaled_mm(func_body, alias_names_before):
                    helper_funcs_with_mm.add(func_name)
            # Strip Python comments from custom_kernel body before call-site analysis
            # so that commented-out calls (e.g. # result = ref_kernel(data)) are ignored.
            ck_active = facts.custom_kernel_active
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
            for alias in alias_names_before:
                if re.search(rf'\breturn\s+{re.escape(alias)}\s*\(', ck_active):
                    compute_helpers.add(alias)
                    continue
                assign_m = re.search(rf'\b(\w+)\s*=\s*{re.escape(alias)}\s*\(', ck_active)
                if assign_m:
                    var = assign_m.group(1)
                    after = ck_active[assign_m.end():]
                    if re.search(rf'\breturn\s+{re.escape(var)}\b', after):
                        compute_helpers.add(alias)
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
            validation_calls.update(
                alias for alias in alias_names_before
                if re.search(rf'\b{re.escape(alias)}\s*\(', ck_active)
            )
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
            # _scaled_mm or an alias only appears before custom_kernel and is never
            # reached from it → likely reference/dead code, not the submission path.
            return []

        has_python = has_python_after
        has_alias = has_alias_after
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


def detect_decode_mm_ref(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 5: FP4 decode + torch.mm/bmm as main compute path.

    Only flags when mm/bmm/matmul appears to feed the output (near return
    or assigned to a result-like variable), not when used for small
    pre/post processing steps.
    """
    facts = ensure_submission_facts(code)
    python_only = facts.python_only
    custom_kernel_code = facts.custom_kernel_code
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


def detect_result_caching(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 6: split output replay from benign workspace/preprocess caches."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    python_only = facts.python_only
    python_active = facts.python_active
    matches = []

    output_replay_signals = []
    preprocess_signals = []
    workspace_signals = []
    runner_plan_signals = []

    if RE_WEAK_VALUE_DICT.search(python_only):
        workspace_signals.append("WeakValueDictionary")
    if RE_DECODED_CACHE.search(python_only):
        preprocess_signals.append("_decoded_cache")
    if RE_PREPROCESS_CACHE.search(python_only):
        preprocess_signals.append("_PREPROCESS_CACHE")
    if RE_SCALE_CACHE.search(python_only):
        preprocess_signals.append("_scale_cache")
    if RE_RESULT_CACHE_GENERAL.search(python_only):
        workspace_signals.append("_RESULT/_GROUPED_RESULT/_COMPUTE/_GEMM_CACHE")
    if RE_VERSION_CACHE.search(python_only):
        workspace_signals.append("tensor._version cache check")

    # Scope all strong-signal checks to custom_kernel body.
    # Helper functions that cache compilation artifacts (TensorMap, compiled kernels,
    # plan descriptors, etc.) return those objects from caches, but that is legitimate —
    # only a cache inside custom_kernel itself indicates result caching.
    cache_scope = facts.custom_kernel_active if facts.custom_kernel_active else python_active

    stores_output = bool(RE_CACHE_STORE_OUTPUT.search(cache_scope))
    if RE_RESULT_REUSE.search(cache_scope):
        output_replay_signals.append("_result_reuse")
    if RE_OUTPUT_CACHE.search(cache_scope):
        workspace_signals.append("_OUTPUT_CACHE")
    if RE_ID_DATA_CACHE.search(cache_scope):
        workspace_signals.append("id(data) cache key")
    if RE_DATA_PTR_CACHE_KEY.search(cache_scope):
        workspace_signals.append("data_ptr() cache key")
    if RE_RETURN_CACHE_INDEX.search(cache_scope):
        output_replay_signals.append("direct return from cache[...]")
    if stores_output and output_replay_signals:
        output_replay_signals.append("cache[...] stores output/result tensor")
    elif stores_output:
        workspace_signals.append("cache[...] stores reusable output/result tensor")

    for var, cache_name in RE_CACHE_GET_ASSIGN.findall(cache_scope):
        cache_lower = cache_name.lower()
        if any(token in cache_lower for token in ("plan", "dispatch", "runner", "config")):
            runner_plan_signals.append(f"{cache_name}.get(...) runner/plan cache")
        elif any(token in cache_lower for token in ("decoded", "preprocess", "scale", "sort", "view", "shape", "quant", "meta", "pad")):
            preprocess_signals.append(f"{cache_name}.get(...) preprocess cache")
        elif re.search(
            rf'if\s+{re.escape(var)}\s+is\s+not\s+None\s*:[\s\S]{{0,300}}?\breturn\s+{re.escape(var)}\b',
            cache_scope,
        ):
            output_replay_signals.append(f"{cache_name}.get(...) then return cached value")
        elif re.search(
            rf'if\s+{re.escape(var)}\s*:[\s\S]{{0,300}}?\breturn\s+{re.escape(var)}\b',
            cache_scope,
        ):
            output_replay_signals.append(f"{cache_name}.get(...) guarded return")
        else:
            workspace_signals.append(f"{cache_name}.get(...) workspace cache")

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
            output_replay_signals.append("C++ pointer cache fast-return path")
        else:
            workspace_signals.append("C++ thread_local cache keyed by data_ptr()")

    # CUDA graph replay (CUDAGraph + .replay()) explains the cache-and-return
    # pattern — the graph replays actual GPU computation, not cached results.
    # Downgrade cache-return signals when CUDA graph replay is present.
    has_cuda_graph_replay = (
        RE_CUDA_GRAPH_PYTHON.search(python_only) and RE_CUDA_GRAPH_REPLAY.search(python_only)
    )
    if has_cuda_graph_replay:
        workspace_signals.append("CUDA graph replay cache")
        output_replay_signals = [
            s for s in output_replay_signals
            if s not in {
                "direct return from cache[...]",
                "cache[...] stores output/result tensor",
            }
        ]

    output_replay_signals = list(dict.fromkeys(output_replay_signals))
    preprocess_signals = list(dict.fromkeys(preprocess_signals))
    workspace_signals = list(dict.fromkeys(workspace_signals))
    runner_plan_signals = list(dict.fromkeys(runner_plan_signals))

    if output_replay_signals:
        matches.append({
            "pattern": "OUTPUT_REPLAY_CACHE",
            "severity": "high",
            "evidence": "Output replay signals: " + ", ".join(output_replay_signals),
        })
    if preprocess_signals:
        matches.append({
            "pattern": "PREPROCESS_CACHE",
            "severity": "low",
            "evidence": "Preprocess cache signals: " + ", ".join(preprocess_signals),
        })
    if workspace_signals:
        matches.append({
            "pattern": "WORKSPACE_CACHE",
            "severity": "low",
            "evidence": "Workspace cache signals: " + ", ".join(workspace_signals),
        })
    if runner_plan_signals:
        matches.append({
            "pattern": "RUNNER_PLAN_CACHE",
            "severity": "low",
            "evidence": "Runner/plan cache signals: " + ", ".join(runner_plan_signals),
        })
    return matches


def detect_cuda_graph_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 7: CUDA graph replay via ctypes libcudart."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
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


def detect_silent_fallback(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 8: try/except falling back to reference matmul."""
    facts = ensure_submission_facts(code)
    code = facts.python_active
    matches = []
    if RE_TORCH_MATMUL_FALLBACK.search(code):
        matches.append({
            "pattern": "SILENT_FALLBACK",
            "severity": "medium",
            "evidence": "try/except fallback to torch.matmul/mm/bmm",
        })
    return matches


def detect_trivial_probe(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 9: Trivial/probe submission (empty return, debug only)."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
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


def detect_torch_compile_cache(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 11: torch.compile for pipeline graph caching."""
    facts = ensure_submission_facts(code)
    code = facts.python_only
    matches = []
    if RE_TORCH_COMPILE.search(code):
        matches.append({
            "pattern": "TORCH_COMPILE_CACHE",
            "severity": "low",
            "evidence": "torch.compile() used",
        })
    return matches


def detect_cuda_graph_python(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 12: Python CUDAGraph caching with data_ptr keys + .replay()."""
    facts = ensure_submission_facts(code)
    python_only = facts.python_only
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

    # Need at least CUDAGraph + replay or CUDAGraph + cache key.
    # CUDA graph replay is a legitimate optimization (reduces kernel launch
    # overhead).  Keep at medium — exploits using CUDA graphs would also
    # have MODULE_MUTATION / GLOBALS_MUTATION which are critical/high.
    if has_graph and (has_replay or has_cache_key):
        matches.append({
            "pattern": "CUDA_GRAPH_PYTHON",
            "severity": "medium",
            "evidence": "Python CUDAGraph caching: " + ", ".join(signals),
        })
    return matches


def detect_hardcoded_shapes(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 13: Hardcoded benchmark shapes in KERNEL_CONFIGS or shape-gated branches.

    Checks both Python and C++ inline code, including macro-based dispatch.
    """
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    python_only = facts.python_only
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


def detect_unsync_multistream(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 14: Unsynchronized multi-stream dispatch to exploit timing.

    Checks both Python code AND C++ inline code, since getStreamFromPool
    is often called from C++/CUDA inline sources.
    """
    # Check Python code
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    python_only = facts.python_only
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


def detect_cuda_event_disable_timing(code: str | SubmissionFacts) -> list[dict]:
    """Pattern 15: cudaEventDisableTiming to mess with benchmark timing."""
    facts = ensure_submission_facts(code)
    code = facts.raw_code
    matches = []
    if RE_CUDA_EVENT_DISABLE_TIMING.search(code):
        matches.append({
            "pattern": "CUDA_EVENT_DISABLE_TIMING",
            "severity": "medium",
            "evidence": "cudaEventDisableTiming or disable_timing=True flag",
        })
    return matches


# ---------------------------------------------------------------------------
# AST-based detectors (Layer 2: structural analysis beyond regex)
# ---------------------------------------------------------------------------

# Modules that are trusted by the evaluator and must not be mutated by submissions.
TRUSTED_MODULES = frozenset({
    "reference", "utils", "eval", "__main__",
})

# Evaluator-critical functions that should never be replaced.
TRUSTED_FUNCTIONS = frozenset({
    "generate_input", "ref_kernel", "check_implementation",
    "clear_l2_cache", "clear_l2_cache_large", "time_kernel",
    "_clone_data", "_run_single_benchmark", "_run_single_test",
})


def _safe_ast_parse(code: str):
    """Parse Python code into AST, returning None on SyntaxError.

    Submissions may contain embedded C++ in triple-quoted strings which can
    confuse the parser when the string isn't properly closed.
    """
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _collect_trusted_aliases(tree: ast.AST) -> dict[str, str]:
    """Walk AST and collect local aliases for trusted modules.

    Returns mapping: local_name -> module_name
    e.g. ``import reference as _ref`` → {"_ref": "reference"}
         ``import reference``        → {"reference": "reference"}
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                if mod_root in TRUSTED_MODULES:
                    local = alias.asname or alias.name
                    aliases[local] = mod_root
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in TRUSTED_MODULES:
                mod_root = node.module.split(".")[0]
                for alias in node.names:
                    local = alias.asname or alias.name
                    aliases[local] = mod_root
    return aliases


def detect_trusted_module_import(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 1: import of evaluator-trusted modules.

    Catches: import reference, from reference import generate_input, etc.
    Severity medium — the import alone is suspicious but not conclusive.
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                if mod_root in TRUSTED_MODULES:
                    matches.append({
                        "pattern": "TRUSTED_MODULE_IMPORT",
                        "severity": "medium",
                        "evidence": f"import {alias.name} (trusted evaluator module)",
                    })
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in TRUSTED_MODULES:
                names = ", ".join(a.name for a in node.names)
                matches.append({
                    "pattern": "TRUSTED_MODULE_IMPORT",
                    "severity": "medium",
                    "evidence": f"from {node.module} import {names}",
                })
    return matches


def detect_module_mutation(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 2: attribute assignment to trusted modules.

    Catches:
      reference.generate_input = hooked_fn
      _ref.ref_kernel.__code__ = fake.__code__
      _ref_module.clear_l2_cache = noop
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    aliases = facts.trusted_aliases
    if not aliases:
        return []

    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            # Direct: module.attr = value
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id in aliases:
                    mod = aliases[target.value.id]
                    attr = target.attr
                    sev = "critical" if attr in TRUSTED_FUNCTIONS else "high"
                    matches.append({
                        "pattern": "MODULE_MUTATION",
                        "severity": sev,
                        "evidence": f"{target.value.id}.{attr} = ... (mutates trusted module '{mod}')",
                    })

            # Nested: module.func.__code__ = ...
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Attribute)
                    and isinstance(target.value.value, ast.Name)):
                if target.value.value.id in aliases and target.attr == "__code__":
                    mod = aliases[target.value.value.id]
                    matches.append({
                        "pattern": "CODE_REPLACEMENT",
                        "severity": "critical",
                        "evidence": (
                            f"{target.value.value.id}.{target.value.attr}.__code__ = ... "
                            f"(bytecode replacement on '{mod}')"
                        ),
                    })
    return matches


def detect_globals_mutation(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 3: __globals__ dict mutation.

    Catches:
      fn.__globals__['generate_input'] = hooked_fn
      getattr(obj, '__globals__')[key] = value
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    assigned_lines: set[int] = set()

    # Pass 1: find __globals__[key] = value assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "__globals__"):
                    key_name = None
                    if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                        key_name = target.slice.value
                    sev = "critical" if key_name in TRUSTED_FUNCTIONS else "high"
                    evidence = (
                        f"__globals__['{key_name}'] = ..." if key_name
                        else "__globals__[...] = ..."
                    )
                    matches.append({
                        "pattern": "GLOBALS_MUTATION",
                        "severity": sev,
                        "evidence": evidence + " (mutates function's global namespace)",
                    })
                    assigned_lines.add(getattr(node, "lineno", -1))

    # Pass 2: flag __globals__ reads (potential reconnaissance)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "__globals__":
            lineno = getattr(node, "lineno", -1)
            if lineno not in assigned_lines:
                matches.append({
                    "pattern": "GLOBALS_ACCESS",
                    "severity": "high",
                    "evidence": "__globals__ attribute accessed (potential namespace manipulation)",
                })
                break  # one read finding is enough

    return matches


def detect_introspection_exploit(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 4: split frame-walk access from frame-based mutation."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    seen_patterns: set[str] = set()
    frame_namespace_aliases: set[str] = set()
    saw_frame_access = False
    saw_frame_mutation = False

    for node in ast.walk(tree):
        # Track aliases of frame globals/locals dicts.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                value = node.value
                if (isinstance(value, ast.Attribute) and value.attr in ("f_globals", "f_locals")) or (
                    isinstance(value, ast.Subscript)
                    and any(
                        isinstance(sub, ast.Attribute) and sub.attr in ("f_globals", "f_locals")
                        for sub in ast.walk(value.value)
                    )
                ):
                    frame_namespace_aliases.add(target.id)

        # inspect.currentframe() / inspect.stack()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "inspect"
                    and node.func.attr in ("currentframe", "stack", "getmembers")):
                key = f"inspect.{node.func.attr}"
                if key not in seen_patterns:
                    seen_patterns.add(key)
                    saw_frame_access = True

        # f_back, f_globals, f_locals attribute access
        if isinstance(node, ast.Attribute) and node.attr in ("f_back", "f_globals", "f_locals"):
            key = f".{node.attr}"
            if key not in seen_patterns:
                seen_patterns.add(key)
                saw_frame_access = True

        # Writes through frame globals/locals aliases are much stronger than
        # plain access and should stand on their own.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    direct_frame_write = any(
                        isinstance(sub, ast.Attribute) and sub.attr in ("f_globals", "f_locals")
                        for sub in ast.walk(target.value)
                    )
                    alias_frame_write = (
                        isinstance(target.value, ast.Name)
                        and target.value.id in frame_namespace_aliases
                    )
                    if direct_frame_write or alias_frame_write:
                        saw_frame_mutation = True

        # sys.modules access
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys" and node.attr == "modules"):
            if "sys.modules" not in seen_patterns:
                seen_patterns.add("sys.modules")
                matches.append({
                    "pattern": "SYS_MODULES_ACCESS",
                    "severity": "high",
                    "evidence": "sys.modules accessed (potential module namespace manipulation)",
                })

    if saw_frame_mutation:
        matches.append({
            "pattern": "FRAME_WALK_MUTATION",
            "severity": "critical",
            "evidence": "frame globals/locals accessed and then mutated",
        })
    elif saw_frame_access:
        matches.append({
            "pattern": "FRAME_WALK_ACCESS",
            "severity": "medium",
            "evidence": "frame walking / frame globals access",
        })

    return matches


def detect_code_replacement(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 5: __code__ attribute assignment (bytecode replacement).

    Catches any func.__code__ = other.__code__ regardless of module context.
    The module-specific variant is handled by detect_module_mutation;
    this catches the general case.
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    assigned_lines: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "__code__":
                    # Module-scoped __code__ replacement is already explained by
                    # detect_module_mutation with better source attribution.
                    if (isinstance(target.value, ast.Attribute)
                            and isinstance(target.value.value, ast.Name)
                            and target.value.value.id in facts.trusted_aliases):
                        continue
                    lineno = getattr(node, "lineno", -1)
                    assigned_lines.add(lineno)
                    matches.append({
                        "pattern": "CODE_REPLACEMENT",
                        "severity": "critical",
                        "evidence": f"__code__ assignment (bytecode replacement)",
                    })

    # Flag __code__ reads if no assignment found (extraction for later use)
    if not assigned_lines:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "__code__":
                matches.append({
                    "pattern": "CODE_ACCESS",
                    "severity": "high",
                    "evidence": "__code__ attribute read (potential bytecode extraction)",
                })
                break

    return matches


TRUSTED_HARNESS_NAMES = frozenset({
    "generate_input", "check_implementation", "clear_l2_cache",
    "clear_l2_cache_large", "_clone_data", "ref_kernel",
})


def detect_harness_runtime_patching(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern: dynamic runtime patching of trusted evaluator functions.

    Targets the ooousay-style exploit family that scans `sys.modules` / `gc`
    for live evaluator objects, then rewrites module attrs or function globals
    through helper functions instead of direct `reference.foo = ...` syntax.
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    dynamic_discovery = set()
    trusted_targets: set[str] = set()
    globals_write = False
    setattr_write = False

    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
                and node.attr == "modules"):
            dynamic_discovery.add("sys.modules")
        elif (isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "gc"
              and node.func.attr == "get_objects"):
            dynamic_discovery.add("gc.get_objects")

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in TRUSTED_HARNESS_NAMES:
                trusted_targets.add(node.value)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr":
            if len(node.args) >= 2:
                name_arg = node.args[1]
                if isinstance(name_arg, ast.Constant) and name_arg.value in TRUSTED_HARNESS_NAMES:
                    setattr_write = True
                elif isinstance(name_arg, ast.Name):
                    setattr_write = True

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    if any(
                        isinstance(sub, ast.Attribute) and sub.attr == "__globals__"
                        for sub in ast.walk(target)
                    ):
                        globals_write = True

    if dynamic_discovery and (globals_write or setattr_write) and len(trusted_targets) >= 2:
        channels = ", ".join(sorted(dynamic_discovery))
        targets = ", ".join(sorted(trusted_targets))
        return [{
            "pattern": "HARNESS_RUNTIME_PATCHING",
            "severity": "critical",
            "evidence": f"dynamic harness patching via {channels} targeting {targets}",
        }]
    return []


RE_REFERENCE_GENERATE_INPUT = re.compile(r'from\s+reference\s+import\s+generate_input')
RE_PRECOMPUTE_STATE = re.compile(r'\b(?:_PRECOMPUTE_STATE|_PUBLIC_CASES|_PUBLIC_WARMUP_DUPLICATES)\b')
RE_PRECOMPUTE_APPEND = re.compile(r'outputs\.append\s*\(\s*_compute_output\s*\(')
RE_PRECOMPUTE_REPLAY = re.compile(r'state\.outputs\s*\[\s*state\.cursor\s*\]')


def detect_reference_precompute_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: import trusted reference generator, precompute public cases, replay outputs.

    This targets files that do not mutate evaluator state but still leverage
    trusted `reference.generate_input` to synthesize benchmark cases ahead of
    timing and then return precomputed outputs.
    """
    facts = ensure_submission_facts(code)
    code = facts.python_active
    if not RE_REFERENCE_GENERATE_INPUT.search(code):
        return []
    if not RE_PRECOMPUTE_STATE.search(code):
        return []
    if len(re.findall(r'\bgenerate_input\s*\(', code)) < 2:
        return []
    if not RE_PRECOMPUTE_APPEND.search(code):
        return []
    if not RE_PRECOMPUTE_REPLAY.search(code):
        return []

    return [{
        "pattern": "REFERENCE_PRECOMPUTE_REPLAY",
        "severity": "critical",
        "evidence": "trusted reference.generate_input imported, public cases precomputed, outputs replayed from precompute state",
    }]


def detect_shape_output_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: output replay keyed only by config / shape state."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    active = facts.python_active
    if re.search(r'torch\.zeros', active) and re.search(r'return\s+_Z\s*\[\s*key\s*\]', active):
        return [{
            "pattern": "SHAPE_OUTPUT_REPLAY",
            "severity": "critical",
            "evidence": "shape-keyed zero/output cache returned directly from custom_kernel",
        }]

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "custom_kernel":
            continue

        state_slots: dict[str, tuple[object, ...]] = {}
        for arg, default in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
            if isinstance(default, (ast.List, ast.Tuple)) and len(default.elts) >= 2:
                state_slots[arg.arg] = tuple(range(len(default.elts)))

        shape_state_match = False
        replay_state = None
        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue
            test = child.test
            if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
                continue
            left, right = test.left, test.comparators[0]
            for state_name in state_slots:
                if (isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name) and left.value.id == state_name
                        and isinstance(right, ast.Name)) or (
                    isinstance(right, ast.Subscript) and isinstance(right.value, ast.Name) and right.value.id == state_name
                    and isinstance(left, ast.Name)
                ):
                    returns = [stmt for stmt in child.body if isinstance(stmt, ast.Return)]
                    if returns:
                        replay_state = state_name
                        shape_state_match = True
                        break
            if shape_state_match:
                break

        if shape_state_match and replay_state:
            helper_called = any(
                isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call)
                for stmt in node.body
            )
            for sub in ast.walk(tree):
                if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if replay_state not in {arg.arg for arg in sub.args.args}:
                    continue
                assigns_first = any(
                    isinstance(stmt, ast.Assign)
                    and any(
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == replay_state
                        for t in stmt.targets
                    )
                    for stmt in ast.walk(sub)
                )
                if assigns_first:
                    return [{
                        "pattern": "SHAPE_OUTPUT_REPLAY",
                        "severity": "critical",
                        "evidence": "custom_kernel replays saved output based only on config/shape state",
                    }]
    return []


def detect_timed_input_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: timed-input preprocess/quant replay keyed by shape or pointer."""
    facts = ensure_submission_facts(code)
    active = facts.python_active
    indicators = []

    if re.search(r'_q_fp8_cache', active) and re.search(r'q\.data_ptr\s*\(', active):
        indicators.append("q_fp8 cached by q.data_ptr()")
    if re.search(r"c\['_q_ptr'\]\s*!=\s*q_ptr", active) and re.search(r"c\['_q_fp8'\]", active):
        indicators.append("stateful _q_ptr/_q_fp8 reuse")
    if re.search(r'_q_scale', active) and re.search(r'q_scale\s*=\s*_q_scale', active):
        indicators.append("reused q_scale buffer")
    if re.search(r'_FP8_BUF_CACHE', active) and re.search(r'dynamic_per_tensor_quant', active):
        indicators.append("shape-keyed fp8 quant buffer cache")
    if re.search(r'_s\s*=\s*\[\s*-?1\s*,\s*None\s*\]', active) and re.search(r'return\s+_s\s*\[\s*1\s*\]', active):
        indicators.append("saved output reused from mutable default state")

    # Only promote when the cached object is tied to live input quantization or
    # transformed q tensors, not just generic workspace reuse.
    if indicators and (
        re.search(r'q_fp8', active)
        or re.search(r'dynamic_per_tensor_quant', active)
        or "saved output reused from mutable default state" in indicators
    ):
        return [{
            "pattern": "TIMED_INPUT_REPLAY",
            "severity": "critical",
            "evidence": "timed-input quantization / transformed-Q replay: " + ", ".join(sorted(set(indicators))),
        }]
    return []


def detect_pointer_replay(code: str | SubmissionFacts) -> list[dict]:
    """Pattern: single-slot output replay keyed by input pointer equality."""
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    def _is_data_ptr_call(expr: ast.AST | None) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "data_ptr"
        )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "custom_kernel":
            continue

        pointer_aliases: set[str] = set()
        saved_ptr = None
        saved_out = None

        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                if len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                    if _is_data_ptr_call(child.value):
                        pointer_aliases.add(child.targets[0].id)

        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue
            compare = child.test
            if not (isinstance(compare, ast.Compare) and len(compare.ops) == 1 and isinstance(compare.ops[0], ast.Eq)):
                continue
            left = compare.left
            right = compare.comparators[0]
            pair = None
            if isinstance(left, ast.Name) and left.id in pointer_aliases and isinstance(right, ast.Name):
                pair = (right.id, left.id)
            elif isinstance(right, ast.Name) and right.id in pointer_aliases and isinstance(left, ast.Name):
                pair = (left.id, right.id)
            elif _is_data_ptr_call(left) and isinstance(right, ast.Name):
                pair = (right.id, None)
            elif _is_data_ptr_call(right) and isinstance(left, ast.Name):
                pair = (left.id, None)
            if pair is None:
                continue
            returns = [stmt for stmt in child.body if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name)]
            if not returns:
                continue
            saved_ptr = pair[0]
            saved_out = returns[0].value.id
            break

        if not saved_ptr or not saved_out:
            continue

        stores_ptr = False
        stores_out = False
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == saved_ptr:
                    if (isinstance(child.value, ast.Name) and child.value.id in pointer_aliases) or _is_data_ptr_call(child.value):
                        stores_ptr = True
                elif target.id == saved_out:
                    stores_out = True

        if stores_ptr and stores_out:
            return [{
                "pattern": "POINTER_REPLAY",
                "severity": "critical",
                "evidence": "custom_kernel returns saved output when input data_ptr matches previous pointer",
            }]

    return []


def detect_config_cache_exploit(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 7: config-keyed result caching inside custom_kernel.

    Detects: custom_kernel that looks up a cache on entry, returns the cached
    value WITHOUT calling any GPU kernel, and stores output into the cache
    before the final return.

    Distinguishes from legitimate workspace caching by requiring that the
    early-return path does NOT contain any function calls (a real exploit
    returns the cached tensor directly without computation).
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "custom_kernel":
            continue

        def _root_name(expr: ast.AST | None) -> Optional[str]:
            cur = expr
            while cur is not None:
                if isinstance(cur, ast.Name):
                    return cur.id
                if isinstance(cur, ast.Attribute):
                    cur = cur.value
                    continue
                if isinstance(cur, ast.Subscript):
                    cur = cur.value
                    continue
                break
            return None

        def _lookup_source(expr: ast.AST | None, sources: dict[str, str]) -> Optional[str]:
            root = _root_name(expr)
            if root is None:
                return None
            return sources.get(root, root)

        def _body_has_calls(body: list[ast.stmt]) -> bool:
            for stmt in body:
                for nested in ast.walk(stmt):
                    if isinstance(nested, ast.Call):
                        return True
            return False

        cache_reads: dict[str, str] = {}
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if len(child.targets) != 1 or not isinstance(child.targets[0], ast.Name):
                continue
            source = None
            if (isinstance(child.value, ast.Call)
                    and isinstance(child.value.func, ast.Attribute)
                    and child.value.func.attr == "get"):
                source = _root_name(child.value.func.value)
            elif isinstance(child.value, ast.Subscript):
                source = _root_name(child.value.value)
            if source:
                cache_reads[child.targets[0].id] = source

        returned_roots = {
            _lookup_source(sub.value, cache_reads)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Return)
        } - {None}
        shortcircuit_containers: set[str] = set()
        store_containers: set[str] = set()

        for child in ast.walk(node):
            if isinstance(child, ast.If):
                if not any(isinstance(stmt, ast.Return) for stmt in child.body):
                    continue
                if _body_has_calls(child.body):
                    continue
                return_roots = {
                    _lookup_source(stmt.value, cache_reads)
                    for stmt in child.body
                    if isinstance(stmt, ast.Return)
                } - {None}
                if not return_roots:
                    continue
                test_roots = {
                    _lookup_source(sub, cache_reads)
                    for sub in ast.walk(child.test)
                    if isinstance(sub, (ast.Name, ast.Attribute, ast.Subscript))
                } - {None}
                shortcircuit_containers.update(return_roots & test_roots)

            if isinstance(child, ast.Try):
                if _body_has_calls(child.body):
                    continue
                has_keyerror = any(
                    isinstance(handler.type, ast.Name) and handler.type.id == "KeyError"
                    for handler in child.handlers
                    if handler.type is not None
                )
                if not has_keyerror:
                    continue
                local_sources = dict(cache_reads)
                for stmt in child.body:
                    if not isinstance(stmt, ast.Assign):
                        continue
                    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                        continue
                    source = None
                    if isinstance(stmt.value, ast.Subscript):
                        source = _root_name(stmt.value.value)
                    elif (isinstance(stmt.value, ast.Call)
                          and isinstance(stmt.value.func, ast.Attribute)
                          and stmt.value.func.attr == "get"):
                        source = _root_name(stmt.value.func.value)
                    if source:
                        local_sources[stmt.targets[0].id] = source
                for stmt in child.body:
                    if isinstance(stmt, ast.Return):
                        resolved = _lookup_source(stmt.value, local_sources)
                        if resolved:
                            shortcircuit_containers.add(resolved)

            if isinstance(child, ast.Assign):
                value_root = _lookup_source(child.value, cache_reads)
                if value_root not in returned_roots:
                    continue
                for target in child.targets:
                    if isinstance(target, ast.Subscript):
                        target_root = _root_name(target.value)
                        if target_root:
                            store_containers.add(target_root)

        if shortcircuit_containers & store_containers:
            matches.append({
                "pattern": "CONFIG_CACHE_EXPLOIT",
                "severity": "high",
                "evidence": "custom_kernel short-circuits on cache hit + stores output (config-keyed result caching)",
            })

    return matches


def detect_dynamic_execution(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 6: exec/eval/compile and dynamic import.

    These can hide exploit code that static analysis cannot see.
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            func_name = f"{node.func.value.id}.{node.func.attr}"

        if func_name in ("exec", "eval", "compile", "__import__") and func_name not in seen:
            seen.add(func_name)
            matches.append({
                "pattern": "DYNAMIC_EXECUTION",
                "severity": "high",
                "evidence": f"{func_name}() call (could hide exploit code)",
            })
        elif func_name == "importlib.import_module" and func_name not in seen:
            seen.add(func_name)
            matches.append({
                "pattern": "DYNAMIC_EXECUTION",
                "severity": "high",
                "evidence": "importlib.import_module() (dynamic module loading)",
            })
        elif func_name == "importlib.reload" and func_name not in seen:
            seen.add(func_name)
            matches.append({
                "pattern": "MODULE_RELOAD",
                "severity": "high",
                "evidence": "importlib.reload() (module state reset/manipulation)",
            })

    return matches


def detect_thread_injection(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 8: background thread/process to offload computation.

    The kernel spawns a background thread that performs the real work while
    the main thread returns an empty/placeholder tensor immediately.  By the
    time the correctness check runs the thread has finished.

    Ref: reference exploit note: thread injection
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []
    # Collect all imports to resolve aliases (e.g. `import threading as t`)
    threading_aliases: set[str] = {"threading"}
    mp_aliases: set[str] = {"multiprocessing"}
    futures_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name == "threading":
                    threading_aliases.add(name)
                elif alias.name == "multiprocessing":
                    mp_aliases.add(name)
                elif alias.name == "concurrent.futures":
                    futures_aliases.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "threading":
                for alias in node.names:
                    name = alias.asname or alias.name
                    if alias.name == "Thread":
                        threading_aliases.add(name)
            elif node.module == "multiprocessing":
                for alias in node.names:
                    name = alias.asname or alias.name
                    if alias.name == "Process":
                        mp_aliases.add(name)
            elif node.module and node.module.startswith("concurrent"):
                for alias in node.names:
                    futures_aliases.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # threading.Thread(...) or Thread(...)
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id in threading_aliases
                    and node.func.attr == "Thread"):
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": f"{node.func.value.id}.Thread() — background thread offloads compute outside timing",
                })
            elif (isinstance(node.func.value, ast.Name)
                    and node.func.value.id in mp_aliases
                    and node.func.attr == "Process"):
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": f"{node.func.value.id}.Process() — background process offloads compute",
                })
            elif (isinstance(node.func.value, ast.Name)
                    and node.func.value.id in futures_aliases
                    and node.func.attr in ("ThreadPoolExecutor", "ProcessPoolExecutor")):
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": f"concurrent.futures.{node.func.attr}() — executor pool offloads compute",
                })
        elif isinstance(node.func, ast.Name):
            if node.func.id == "Thread" and "Thread" in threading_aliases:
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": "Thread() — background thread offloads compute outside timing",
                })
            elif node.func.id == "Process" and "Process" in mp_aliases:
                matches.append({
                    "pattern": "THREAD_INJECTION",
                    "severity": "high",
                    "evidence": "Process() — background process offloads compute",
                })

    # Deduplicate
    seen = set()
    deduped = []
    for m in matches:
        key = m["evidence"]
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped


def detect_lazy_tensor(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 9: lazy tensor subclass that defers computation.

    The kernel returns a torch.Tensor subclass whose data is not materialized.
    Computation is deferred to __eq__, __torch_function__, or similar hooks
    that run during the correctness check rather than during timing.

    Ref: reference exploit note: lazy evaluation
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if class inherits from torch.Tensor
        is_tensor_subclass = False
        for base in node.bases:
            if isinstance(base, ast.Attribute):
                if (isinstance(base.value, ast.Name)
                        and base.value.id == "torch" and base.attr == "Tensor"):
                    is_tensor_subclass = True
            elif isinstance(base, ast.Name) and base.id == "Tensor":
                is_tensor_subclass = True

        if not is_tensor_subclass:
            continue

        # Found a torch.Tensor subclass — check for suspicious methods
        suspicious_methods = []
        has_make_subclass = False

        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                if child.name in ("__eq__", "__torch_function__", "__torch_dispatch__"):
                    suspicious_methods.append(child.name)
            # _make_subclass call
            if isinstance(child, ast.Attribute) and child.attr == "_make_subclass":
                has_make_subclass = True

        if suspicious_methods or has_make_subclass:
            evidence_parts = [f"torch.Tensor subclass '{node.name}'"]
            if has_make_subclass:
                evidence_parts.append("_make_subclass()")
            if suspicious_methods:
                evidence_parts.append(f"deferred hooks: {', '.join(suspicious_methods)}")
            matches.append({
                "pattern": "LAZY_TENSOR",
                "severity": "critical",
                "evidence": " + ".join(evidence_parts) + " — computation deferred past timing",
            })
        else:
            # Tensor subclass without known hooks — still suspicious
            matches.append({
                "pattern": "LAZY_TENSOR",
                "severity": "high",
                "evidence": f"torch.Tensor subclass '{node.name}' — may defer computation",
            })

    return matches


def detect_precision_downgrade(code: str | SubmissionFacts) -> list[dict]:
    """AST pattern 10: silent precision downgrade for speed.

    The kernel casts inputs to lower precision (fp16/bf16), computes faster,
    then casts back to the original dtype.  This cheats the benchmark while
    potentially passing allclose with loose tolerance.

    Ref: reference exploit note: precision downgrading
    """
    facts = ensure_submission_facts(code)
    tree = facts.ast_tree
    if tree is None:
        return []

    matches = []

    # Look inside custom_kernel (or the whole file if no custom_kernel found)
    target_funcs = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "custom_kernel"
    ]
    if not target_funcs:
        target_funcs = [tree]

    for func_node in target_funcs:
        has_downcast = False
        has_upcast = False
        downcast_evidence = []

        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue

            # .half(), .bfloat16(), .float16() method calls
            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in ("half", "bfloat16", "float16"):
                    has_downcast = True
                    downcast_evidence.append(f".{attr}()")
                elif attr in ("float", "double", "float32"):
                    has_upcast = True
                # .to(torch.float16), .to(dtype=torch.bfloat16)
                elif attr == "to":
                    for arg in node.args:
                        if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                            if arg.value.id == "torch" and arg.attr in ("float16", "bfloat16", "half"):
                                has_downcast = True
                                downcast_evidence.append(f".to(torch.{arg.attr})")
                            elif arg.value.id == "torch" and arg.attr in ("float32", "float64"):
                                has_upcast = True
                    for kw in node.keywords:
                        if kw.arg == "dtype" and isinstance(kw.value, ast.Attribute):
                            if (isinstance(kw.value.value, ast.Name)
                                    and kw.value.value.id == "torch"
                                    and kw.value.attr in ("float16", "bfloat16", "half")):
                                has_downcast = True
                                downcast_evidence.append(f".to(dtype=torch.{kw.value.attr})")
                            elif (isinstance(kw.value.value, ast.Name)
                                    and kw.value.value.id == "torch"
                                    and kw.value.attr in ("float32", "float64")):
                                has_upcast = True

        if has_downcast and has_upcast:
            evidence = ", ".join(dict.fromkeys(downcast_evidence))
            matches.append({
                "pattern": "PRECISION_DOWNGRADE",
                "severity": "medium",
                "evidence": f"Downcast ({evidence}) then upcast back — may cheat benchmark via lower precision",
            })

    return matches


# ---------------------------------------------------------------------------
# Score anomaly detection
# ---------------------------------------------------------------------------

def detect_score_anomaly(metadata: Optional[dict]) -> list[dict]:
    """Score-based anomaly detection.

    Works across all input modes:
    - parquet: metadata has "score" (float)
    - nvidia_archive: metadata has "scores" (list of floats)
    - jsonl: metadata has "improved_score" / "baseline_score"
    """
    if not metadata:
        return []
    matches = []

    # Normalize: collect all available scores into a unified list
    all_scores: list[float] = []

    # Parquet mode: singular "score"
    score = metadata.get("score")
    if score is not None and isinstance(score, (int, float)):
        all_scores.append(score)

    # NvidiaArchive mode: "scores" list
    for s in metadata.get("scores", []):
        if isinstance(s, (int, float)):
            all_scores.append(s)

    # JSONL mode
    improved_score = metadata.get("improved_score")
    baseline_score = metadata.get("baseline_score")
    if improved_score is not None and isinstance(improved_score, (int, float)):
        all_scores.append(improved_score)

    # Check score anomalies
    for s in all_scores:
        if not isinstance(s, (int, float)) or s <= 0:
            continue
        if s < 1e-8:
            matches.append({
                "pattern": "SCORE_IMPOSSIBLE",
                "severity": "critical",
                "evidence": f"Score {s:.2e} is physically impossible (<10ns)",
            })
            break
        if s < 5e-6:
            matches.append({
                "pattern": "SCORE_SUSPECT_FLOOR",
                "severity": "high",
                "evidence": f"Score {s:.2e}s is at/near HIP/CUDA event timing floor (~4µs)",
            })
            break
        if s > 0.5:
            matches.append({
                "pattern": "SCORE_BROKEN",
                "severity": "critical",
                "evidence": f"Score {s:.2e} suggests broken/inverted metric",
            })
            break

    # Extreme speedup check (JSONL mode)
    if (improved_score and baseline_score
            and isinstance(improved_score, (int, float))
            and isinstance(baseline_score, (int, float))
            and improved_score > 0):
        speedup = baseline_score / improved_score
        if speedup > 100:
            matches.append({
                "pattern": "SCORE_EXTREME_SPEEDUP",
                "severity": "high",
                "evidence": f"Speedup {speedup:.1f}x over baseline is extreme",
            })

    return matches


# ---------------------------------------------------------------------------
# Rule policies and detector registry
# ---------------------------------------------------------------------------

AUTO_FILTER = "auto_filter"
SUSPICIOUS_ONLY = "suspicious"
TELEMETRY_ONLY = "low_confidence"


@dataclass(frozen=True)
class RulePolicy:
    rule_name: str
    technique_family: str
    evidence_tier: str
    max_outcome: str
    requires_companion_patterns: tuple[str, ...]
    mapped_positive_fixtures: tuple[str, ...]
    default_verdict: str


AMD_CODE_REPLACEMENT_FIXTURES = ("amd:514036",)
AMD_MODULE_MUTATION_FIXTURES = (
    "amd:514274", "amd:514288", "amd:514336", "amd:514350", "amd:514361",
    "amd:514378", "amd:514394", "amd:515019", "amd:515037", "amd:515048",
    "amd:515064", "amd:515082", "amd:515102", "amd:515123", "amd:519613",
    "amd:519659", "amd:519693", "amd:521294", "amd:521414",
)
AMD_CONFIG_CACHE_FIXTURES = ("amd:520921", "amd:521087", "amd:521134", "amd:521160")
AMD_MANUAL_HARNESS_FIXTURES = ("manual_review:515117",)
AMD_MANUAL_PRECOMPUTE_FIXTURES = ("manual_review:514553",)
AMD_MANUAL_POINTER_REPLAY_FIXTURES = ("manual_review:515367",)
AMD_MANUAL_FRAME_MUTATION_FIXTURES = ("manual_review:516235",)
AMD_PACKAGE_TIMED_INPUT_REPLAY_FIXTURES = ()
AMD_PACKAGE_SHAPE_REPLAY_FIXTURES = ()
NVIDIA_ARCHIVE_TIMER_FIXTURES = (
    "nvidia_archive:108465", "nvidia_archive:108477", "nvidia_archive:108609", "nvidia_archive:108878",
    "nvidia_archive:121398", "nvidia_archive:121401",
)
NVIDIA_ARCHIVE_PRINT_FIXTURES = (
    "nvidia_archive:121398", "nvidia_archive:121401",
)
NVIDIA_ARCHIVE_EVALUATOR_FIXTURES = (
    "nvidia_archive:108465", "nvidia_archive:108477", "nvidia_archive:108609", "nvidia_archive:108878",
)


RULE_REGISTRY: dict[str, RulePolicy] = {
    "TIMER_MONKEYPATCH": RulePolicy(
        "TIMER_MONKEYPATCH", "timing_monkeypatch", "hard", AUTO_FILTER, (),
        NVIDIA_ARCHIVE_TIMER_FIXTURES, "keep",
    ),
    "FAKE_BENCHMARK_EMIT": RulePolicy(
        "FAKE_BENCHMARK_EMIT", "benchmark_output_injection", "hard", AUTO_FILTER, (),
        NVIDIA_ARCHIVE_PRINT_FIXTURES, "keep",
    ),
    "STDIO_REDIRECT": RulePolicy(
        "STDIO_REDIRECT", "stdio_compat", "telemetry", TELEMETRY_ONLY, (),
        (), "split",
    ),
    "EVALUATOR_EXPLOIT": RulePolicy(
        "EVALUATOR_EXPLOIT", "evaluator_state_mutation", "support", SUSPICIOUS_ONLY, (),
        NVIDIA_ARCHIVE_EVALUATOR_FIXTURES, "keep",
    ),
    "HARNESS_RUNTIME_PATCHING": RulePolicy(
        "HARNESS_RUNTIME_PATCHING", "evaluator_state_mutation", "hard", AUTO_FILTER, (),
        AMD_MANUAL_HARNESS_FIXTURES, "keep",
    ),
    "SCALED_MM_REF": RulePolicy(
        "SCALED_MM_REF", "reference_path_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "split",
    ),
    "DECODE_MM_REF": RulePolicy(
        "DECODE_MM_REF", "reference_path_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "remove",
    ),
    "OUTPUT_REPLAY_CACHE": RulePolicy(
        "OUTPUT_REPLAY_CACHE", "result_reuse", "support", SUSPICIOUS_ONLY, (),
        AMD_CONFIG_CACHE_FIXTURES, "rewrite",
    ),
    "SHAPE_OUTPUT_REPLAY": RulePolicy(
        "SHAPE_OUTPUT_REPLAY", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_PACKAGE_SHAPE_REPLAY_FIXTURES, "keep",
    ),
    "TIMED_INPUT_REPLAY": RulePolicy(
        "TIMED_INPUT_REPLAY", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_PACKAGE_TIMED_INPUT_REPLAY_FIXTURES, "keep",
    ),
    "PREPROCESS_CACHE": RulePolicy(
        "PREPROCESS_CACHE", "preprocess_cache", "telemetry", TELEMETRY_ONLY, (),
        (), "keep",
    ),
    "WORKSPACE_CACHE": RulePolicy(
        "WORKSPACE_CACHE", "workspace_cache", "telemetry", TELEMETRY_ONLY, (),
        (), "keep",
    ),
    "RUNNER_PLAN_CACHE": RulePolicy(
        "RUNNER_PLAN_CACHE", "runner_plan_cache", "telemetry", TELEMETRY_ONLY, (),
        (), "keep",
    ),
    "CUDA_GRAPH_REPLAY": RulePolicy(
        "CUDA_GRAPH_REPLAY", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "SILENT_FALLBACK": RulePolicy(
        "SILENT_FALLBACK", "reference_path_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "remove",
    ),
    "TRIVIAL_PROBE": RulePolicy(
        "TRIVIAL_PROBE", "low_signal", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "TORCH_COMPILE_CACHE": RulePolicy(
        "TORCH_COMPILE_CACHE", "performance_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "CUDA_GRAPH_PYTHON": RulePolicy(
        "CUDA_GRAPH_PYTHON", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "HARDCODED_SHAPES": RulePolicy(
        "HARDCODED_SHAPES", "performance_heuristic", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "UNSYNC_MULTISTREAM": RulePolicy(
        "UNSYNC_MULTISTREAM", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "CUDA_EVENT_DISABLE_TIMING": RulePolicy(
        "CUDA_EVENT_DISABLE_TIMING", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "TRUSTED_MODULE_IMPORT": RulePolicy(
        "TRUSTED_MODULE_IMPORT", "evaluator_state_support", "support", SUSPICIOUS_ONLY,
        ("MODULE_MUTATION", "GLOBALS_MUTATION", "CODE_REPLACEMENT", "EVALUATOR_EXPLOIT"),
        AMD_MODULE_MUTATION_FIXTURES + AMD_CODE_REPLACEMENT_FIXTURES, "downgrade",
    ),
    "MODULE_MUTATION": RulePolicy(
        "MODULE_MUTATION", "evaluator_state_mutation", "hard", AUTO_FILTER, (),
        AMD_MODULE_MUTATION_FIXTURES + AMD_CODE_REPLACEMENT_FIXTURES, "keep",
    ),
    "GLOBALS_MUTATION": RulePolicy(
        "GLOBALS_MUTATION", "evaluator_state_mutation", "hard", AUTO_FILTER, (),
        AMD_MODULE_MUTATION_FIXTURES, "keep",
    ),
    "GLOBALS_ACCESS": RulePolicy(
        "GLOBALS_ACCESS", "evaluator_state_support", "support", SUSPICIOUS_ONLY,
        ("GLOBALS_MUTATION", "MODULE_MUTATION", "EVALUATOR_EXPLOIT"),
        AMD_MODULE_MUTATION_FIXTURES, "downgrade",
    ),
    "FRAME_WALK_ACCESS": RulePolicy(
        "FRAME_WALK_ACCESS", "evaluator_state_support", "telemetry", TELEMETRY_ONLY,
        (), (), "split",
    ),
    "FRAME_WALK_MUTATION": RulePolicy(
        "FRAME_WALK_MUTATION", "evaluator_state_mutation", "hard", AUTO_FILTER,
        (), AMD_MANUAL_FRAME_MUTATION_FIXTURES, "keep",
    ),
    "SYS_MODULES_ACCESS": RulePolicy(
        "SYS_MODULES_ACCESS", "evaluator_state_support", "telemetry", TELEMETRY_ONLY,
        (), AMD_MODULE_MUTATION_FIXTURES, "downgrade",
    ),
    "CODE_REPLACEMENT": RulePolicy(
        "CODE_REPLACEMENT", "evaluator_state_mutation", "hard", AUTO_FILTER, (),
        AMD_CODE_REPLACEMENT_FIXTURES, "keep",
    ),
    "CODE_ACCESS": RulePolicy(
        "CODE_ACCESS", "evaluator_state_support", "telemetry", TELEMETRY_ONLY,
        (), AMD_CODE_REPLACEMENT_FIXTURES, "downgrade",
    ),
    "CONFIG_CACHE_EXPLOIT": RulePolicy(
        "CONFIG_CACHE_EXPLOIT", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_CONFIG_CACHE_FIXTURES, "rewrite",
    ),
    "REFERENCE_PRECOMPUTE_REPLAY": RulePolicy(
        "REFERENCE_PRECOMPUTE_REPLAY", "reference_replay", "hard", AUTO_FILTER, (),
        AMD_MANUAL_PRECOMPUTE_FIXTURES, "keep",
    ),
    "POINTER_REPLAY": RulePolicy(
        "POINTER_REPLAY", "result_reuse", "hard", AUTO_FILTER, (),
        AMD_MANUAL_POINTER_REPLAY_FIXTURES, "keep",
    ),
    "DYNAMIC_EXECUTION": RulePolicy(
        "DYNAMIC_EXECUTION", "dynamic_execution", "telemetry", TELEMETRY_ONLY, (),
        (), "split",
    ),
    "MODULE_RELOAD": RulePolicy(
        "MODULE_RELOAD", "dynamic_execution", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "THREAD_INJECTION": RulePolicy(
        "THREAD_INJECTION", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "LAZY_TENSOR": RulePolicy(
        "LAZY_TENSOR", "timing_manipulation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "PRECISION_DOWNGRADE": RulePolicy(
        "PRECISION_DOWNGRADE", "approximation", "telemetry", TELEMETRY_ONLY, (),
        (), "downgrade",
    ),
    "SCORE_IMPOSSIBLE": RulePolicy(
        "SCORE_IMPOSSIBLE", "score_anomaly", "hard", AUTO_FILTER, (),
        (), "keep",
    ),
    "SCORE_SUSPECT_FLOOR": RulePolicy(
        "SCORE_SUSPECT_FLOOR", "score_anomaly", "support", SUSPICIOUS_ONLY, (),
        (), "keep",
    ),
    "SCORE_BROKEN": RulePolicy(
        "SCORE_BROKEN", "score_anomaly", "support", SUSPICIOUS_ONLY, (),
        (), "keep",
    ),
    "SCORE_EXTREME_SPEEDUP": RulePolicy(
        "SCORE_EXTREME_SPEEDUP", "score_anomaly", "support", SUSPICIOUS_ONLY, (),
        (), "downgrade",
    ),
    "DUPLICATE_SPAM": RulePolicy(
        "DUPLICATE_SPAM", "administrative", "support", SUSPICIOUS_ONLY, (),
        (), "downgrade",
    ),
    "NEAR_CLONE_SPAM": RulePolicy(
        "NEAR_CLONE_SPAM", "administrative", "support", SUSPICIOUS_ONLY, (),
        (), "downgrade",
    ),
}


OUTCOME_ORDER = {
    TELEMETRY_ONLY: 1,
    SUSPICIOUS_ONLY: 2,
    AUTO_FILTER: 3,
}


def get_rule_policy(pattern: str) -> RulePolicy:
    return RULE_REGISTRY.get(
        pattern,
        RulePolicy(pattern, "unclassified", "telemetry", TELEMETRY_ONLY, (), (), "keep"),
    )


def strongest_rule_outcome(matched_patterns: list[dict]) -> str:
    if not matched_patterns:
        return TELEMETRY_ONLY
    return max(
        (get_rule_policy(p["pattern"]).max_outcome for p in matched_patterns),
        key=lambda outcome: OUTCOME_ORDER[outcome],
    )


ADMIN_PATTERNS = {"DUPLICATE_SPAM", "NEAR_CLONE_SPAM"}


def split_match_domains(matched_patterns: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split matches into code, metadata, and administrative domains."""
    code_patterns = []
    metadata_patterns = []
    admin_patterns = []
    for pattern in matched_patterns:
        name = pattern["pattern"]
        if name in ADMIN_PATTERNS:
            admin_patterns.append(pattern)
        elif name.startswith("SCORE_") or pattern.get("field") == "metadata":
            metadata_patterns.append(pattern)
        else:
            code_patterns.append(pattern)
    return code_patterns, metadata_patterns, admin_patterns


def support_only_patterns(matched_patterns: list[dict]) -> bool:
    """Return True when every pattern is only support/telemetry evidence."""
    if not matched_patterns:
        return False
    return all(get_rule_policy(p["pattern"]).evidence_tier != "hard" for p in matched_patterns)


# ---------------------------------------------------------------------------
# All detectors list
# ---------------------------------------------------------------------------

CODE_DETECTORS = [
    # Regex-based detectors (Layer 1)
    detect_timer_monkeypatch,
    detect_print_injection,
    detect_evaluator_exploit,
    detect_scaled_mm_ref,
    detect_decode_mm_ref,
    detect_result_caching,
    detect_shape_output_replay,
    detect_timed_input_replay,
    detect_cuda_graph_replay,
    detect_silent_fallback,
    detect_trivial_probe,
    detect_torch_compile_cache,
    detect_cuda_graph_python,
    detect_hardcoded_shapes,
    detect_unsync_multistream,
    detect_cuda_event_disable_timing,
    # AST-based detectors (Layer 2)
    detect_trusted_module_import,
    detect_module_mutation,
    detect_globals_mutation,
    detect_introspection_exploit,
    detect_code_replacement,
    detect_harness_runtime_patching,
    detect_config_cache_exploit,
    detect_reference_precompute_replay,
    detect_pointer_replay,
    detect_dynamic_execution,
    detect_thread_injection,
    detect_lazy_tensor,
    detect_precision_downgrade,
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


DEFAULT_REFERENCE_CLASS = "default_reference"
DEFAULT_REFERENCE_FILTER_REASON = "default_scaled_mm"
DEFAULT_REFERENCE_ALLOWED_MEDIUM = {"DUPLICATE_SPAM"}


def is_default_reference(matched_patterns: list[dict]) -> bool:
    """Return True when the submission is effectively a default/reference path.

    `_scaled_mm` reference paths are common in correctness fallbacks and
    default submissions. Keep a separate class for these rows, but only allow
    them to filter when the rule policy explicitly permits auto-filtering.
    """
    relevant_patterns = {
        p["pattern"] for p in matched_patterns
        if p["pattern"] not in {"DUPLICATE_SPAM", "NEAR_CLONE_SPAM"}
    }
    if not relevant_patterns:
        return False
    if relevant_patterns != {"SCALED_MM_REF"}:
        return False
    medium_families = {p["pattern"] for p in matched_patterns if p["severity"] == "medium"}
    disqualifying_medium = medium_families - DEFAULT_REFERENCE_ALLOWED_MEDIUM - {"SCALED_MM_REF"}
    return not disqualifying_medium


def filter_reason(matched_patterns: list[dict]) -> str:
    """Return a short label explaining what drove the filter decision."""
    if not matched_patterns:
        return "none"
    code_patterns, metadata_patterns, admin_patterns = split_match_domains(matched_patterns)

    code_auto_filter = [
        p for p in code_patterns if get_rule_policy(p["pattern"]).max_outcome == AUTO_FILTER
    ]
    metadata_auto_filter = [
        p for p in metadata_patterns if get_rule_policy(p["pattern"]).max_outcome == AUTO_FILTER
    ]
    if code_auto_filter:
        return "high_critical"
    if metadata_auto_filter:
        return "score_anomaly"
    if admin_patterns:
        return "admin_review"
    return "none"


def classify(matched_patterns: list[dict]) -> tuple[str, bool]:
    """Return (classification, should_filter) from matched patterns.

    Precision-first policy:
      hacked            — at least one rule whose policy allows auto-filtering.
      default_reference — `_scaled_mm` reference behavior; not filterable in this pass.
      suspicious        — support signals or rewrite-before-trust rules.
      low_confidence    — telemetry-only signals.
      valid             — no signals detected.
    """
    if not matched_patterns:
        return "valid", False

    code_patterns, metadata_patterns, admin_patterns = split_match_domains(matched_patterns)

    if is_default_reference(code_patterns):
        return "low_confidence", False

    code_strongest = strongest_rule_outcome(code_patterns) if code_patterns else TELEMETRY_ONLY
    metadata_strongest = strongest_rule_outcome(metadata_patterns) if metadata_patterns else TELEMETRY_ONLY
    admin_strongest = strongest_rule_outcome(admin_patterns) if admin_patterns else TELEMETRY_ONLY

    if code_strongest == AUTO_FILTER:
        return "hacked", True
    if metadata_strongest == AUTO_FILTER:
        return "hacked", True

    if code_patterns:
        if code_strongest == SUSPICIOUS_ONLY:
            if support_only_patterns(code_patterns):
                return "low_confidence", False
            return "suspicious", False
        return "low_confidence", False

    if metadata_strongest == SUSPICIOUS_ONLY:
        return "suspicious", False
    if admin_strongest == SUSPICIOUS_ONLY:
        return "low_confidence", False
    return "valid", False


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
    facts = build_submission_facts(code)
    all_matches = []
    for detector in CODE_DETECTORS:
        hits = detector(facts)
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
# Top-level worker functions (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _worker_jsonl(args: tuple) -> Optional[dict]:
    """Analyze one JSONL pair entry.  Returns result dict or None on parse error."""
    line_num, line = args
    line = line.strip()
    if not line:
        return None
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None

    entry_id = entry.get("id", f"line_{line_num}")
    user = entry.get("user", "unknown")
    problem = entry.get("problem_name", "unknown")
    metadata = {
        "improved_score": entry.get("improved_score"),
        "baseline_score": entry.get("baseline_score"),
    }

    # In pair mode, code-side detectors should stay attached to their source side.
    # Score anomalies are entry-level metadata signals and must be emitted once.
    r_imp = analyze_code(entry.get("improved_code", ""), None,
                         field="improved_code", compute_structural_hash=False)
    r_base = analyze_code(entry.get("baseline_code", ""), None,
                          field="baseline_code", compute_structural_hash=False)

    all_patterns = []
    for p in r_imp["matched_patterns"]:
        all_patterns.append(dict(p, field="improved_code"))
    for p in r_base["matched_patterns"]:
        all_patterns.append(dict(p, field="baseline_code"))
    for p in detect_score_anomaly(metadata):
        all_patterns.append(dict(p, field="metadata"))

    decision_patterns = [
        p for p in all_patterns if p["field"] in ("improved_code", "metadata")
    ]
    classification, should_filter = classify(decision_patterns)
    reason = filter_reason(decision_patterns) if should_filter else None

    return {
        "id": entry_id,
        "user": user,
        "problem_name": problem,
        "classification": classification,
        "should_filter": should_filter,
        "filter_reason": reason,
        "matched_patterns": all_patterns,
        "improved_score": entry.get("improved_score"),
        "baseline_score": entry.get("baseline_score"),
        "code_hash_improved": r_imp["code_hash"],
        "code_hash_baseline": r_base["code_hash"],
        "_line_num": line_num,
    }


def _worker_parquet(args: tuple) -> dict:
    """Analyze one parquet submission row."""
    sub_id, leaderboard_id, user_id, user_name, problem_name, score, passed, code = args
    metadata = {"score": score, "user": user_name, "problem": problem_name}
    r = analyze_code(code or "", metadata, field="code", compute_structural_hash=False)
    return {
        "submission_id": int(sub_id),
        "leaderboard_id": int(leaderboard_id),
        "user_id": str(user_id),
        "user": str(user_name),
        "problem_name": str(problem_name),
        "score": float(score) if score is not None else None,
        "passed": bool(passed),
        "classification": r["classification"],
        "should_filter": r["should_filter"],
        "filter_reason": r.get("filter_reason"),
        "matched_patterns": r["matched_patterns"],
        "code_hash": r["code_hash"],
    }


# ---------------------------------------------------------------------------
# Precision audit
# ---------------------------------------------------------------------------

AUDIT_RESULT_FILES = (
    ("nvidia_archive", "detection_results_nvidia_archive.jsonl"),
    ("amd", "detection_results_amd_submissions.jsonl"),
    ("nvidia", "detection_results_nvidia_submissions.jsonl"),
)

AUDIT_RULE_ORDER = [
    "EVALUATOR_EXPLOIT", "HARNESS_RUNTIME_PATCHING", "MODULE_MUTATION", "GLOBALS_MUTATION", "CODE_REPLACEMENT",
    "FRAME_WALK_ACCESS", "FRAME_WALK_MUTATION", "SYS_MODULES_ACCESS", "GLOBALS_ACCESS", "CODE_ACCESS",
    "TRUSTED_MODULE_IMPORT",
    "OUTPUT_REPLAY_CACHE", "SHAPE_OUTPUT_REPLAY", "TIMED_INPUT_REPLAY", "CONFIG_CACHE_EXPLOIT", "POINTER_REPLAY", "PREPROCESS_CACHE", "WORKSPACE_CACHE",
    "RUNNER_PLAN_CACHE", "CUDA_GRAPH_PYTHON", "CUDA_GRAPH_REPLAY",
    "TIMER_MONKEYPATCH", "FAKE_BENCHMARK_EMIT", "STDIO_REDIRECT", "UNSYNC_MULTISTREAM", "CUDA_EVENT_DISABLE_TIMING",
    "SCALED_MM_REF", "DECODE_MM_REF", "SILENT_FALLBACK", "REFERENCE_PRECOMPUTE_REPLAY", "TORCH_COMPILE_CACHE",
    "HARDCODED_SHAPES", "TRIVIAL_PROBE",
    "DYNAMIC_EXECUTION", "MODULE_RELOAD", "THREAD_INJECTION", "LAZY_TENSOR",
    "PRECISION_DOWNGRADE", "SCORE_IMPOSSIBLE", "SCORE_SUSPECT_FLOOR",
    "SCORE_BROKEN", "SCORE_EXTREME_SPEEDUP", "DUPLICATE_SPAM", "NEAR_CLONE_SPAM",
]


def _parse_nvidia_archive_submission_id(path: str) -> str:
    basename = os.path.basename(path)
    parts = basename.split("_", 3)
    return parts[2] if len(parts) >= 3 else "unknown"


def _find_nvidia_archive_submission_path(directory: str, submission_id: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(directory, f"nv_sub_{submission_id}_*.py")))
    return candidates[0] if candidates else None


def _find_amd_submission_path(directory: str, submission_id: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(directory, f"amd_mla_sub_{submission_id}_*.py")))
    return candidates[0] if candidates else None


def _safe_read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_classifier_fixture_manifest(
    nvidia_archive_dir: str = "NvidiaArchive",
    amd_dir: str = "amd_fixture_archive",
    filtered_nvidia_archive_path: str = "filtered_nvidia_archive.jsonl",
    manual_judgments_path: str = "manual_review_archive/manual_judgments.json",
) -> dict:
    """Build the precision-audit fixture manifest.

    Hard positives:
      - amd_fixture_archive/ground_truth.json exploit=true
      - source-backed entries in NvidiaArchive/nvidia_hacking_manifest.json
    Hard negatives:
      - amd_fixture_archive/ground_truth.json exploit=false
    Soft review negatives:
      - deduplicated NvidiaArchive sources not in the hard-positive manifest and not
        already present in filtered_nvidia_archive.jsonl
    """
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hard_positives": [],
        "hard_negatives": [],
        "soft_review_negatives": [],
    }
    known_hard_positive_amd_ids: set[str] = set()

    gt_path = os.path.join(amd_dir, "ground_truth.json")
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as f:
            ground_truth = json.load(f)
        for row in ground_truth:
            sid = str(row["submission_id"])
            path = _find_amd_submission_path(amd_dir, sid)
            if not path:
                continue
            code = _safe_read_text(path)
            entry = {
                "fixture_id": f"amd:{sid}",
                "submission_id": sid,
                "label": "hard_positive" if row.get("is_exploit") else "hard_negative",
                "source": "amd_fixture_archive",
                "technique": row.get("technique", "unknown"),
                "note": row.get("note"),
                "file": path,
                "code_hash": code_hash(code),
            }
            bucket = "hard_positives" if row.get("is_exploit") else "hard_negatives"
            manifest[bucket].append(entry)
            if row.get("is_exploit"):
                known_hard_positive_amd_ids.add(sid)

    hacking_manifest_path = os.path.join(nvidia_archive_dir, "nvidia_hacking_manifest.json")
    hard_positive_ids: set[str] = set()
    if os.path.exists(hacking_manifest_path):
        with open(hacking_manifest_path, encoding="utf-8") as f:
            hacking_manifest = json.load(f)
        for row in hacking_manifest:
            sid = str(row["submissionId"])
            path = _find_nvidia_archive_submission_path(nvidia_archive_dir, sid)
            if not path:
                continue
            code = _safe_read_text(path)
            manifest["hard_positives"].append({
                "fixture_id": f"nvidia_archive:{sid}",
                "submission_id": sid,
                "label": "hard_positive",
                "source": "nvidia_archive_hacking_manifest",
                "technique": "source_backed_hacking_manifest",
                "leaderboard": row.get("leaderboard"),
                "submitted_at": row.get("submittedAt"),
                "file": path,
                "code_hash": code_hash(code),
            })
            hard_positive_ids.add(sid)

    if os.path.exists(manual_judgments_path):
        with open(manual_judgments_path, encoding="utf-8") as f:
            manual_rows = json.load(f)
        for row in manual_rows:
            if str(row.get("manual_filter", "")).lower() != "yes":
                continue
            sid = str(row["submission_id"])
            if sid in known_hard_positive_amd_ids:
                continue
            path = row.get("code_path")
            if not path or not os.path.exists(path):
                continue
            code = _safe_read_text(path)
            manifest["hard_positives"].append({
                "fixture_id": f"manual_review:{sid}",
                "submission_id": sid,
                "label": "hard_positive",
                "source": "manual_archive_review",
                "technique": row.get("primary_technique", "manual_review"),
                "problem_name": row.get("problem_name"),
                "manual_judgment": row.get("manual_judgment"),
                "file": path,
                "code_hash": code_hash(code),
            })
            known_hard_positive_amd_ids.add(sid)

    filtered_ids: set[str] = set()
    if os.path.exists(filtered_nvidia_archive_path):
        with open(filtered_nvidia_archive_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                row = json.loads(line)
                filtered_ids.add(str(row.get("submission_id")))

    seen_hashes: set[str] = set()
    for path in sorted(glob.glob(os.path.join(nvidia_archive_dir, "nv_sub_*.py"))):
        sid = _parse_nvidia_archive_submission_id(path)
        if sid in hard_positive_ids or sid in filtered_ids:
            continue
        code = _safe_read_text(path)
        ch = code_hash(code)
        if ch in seen_hashes:
            continue
        seen_hashes.add(ch)
        manifest["soft_review_negatives"].append({
            "fixture_id": f"nvidia_archive_soft:{sid}",
            "submission_id": sid,
            "label": "soft_review_negative",
            "source": "nvidia_archive",
            "technique": "unknown",
            "file": path,
            "code_hash": ch,
        })

    return manifest


def _fixture_pattern_hits(fixtures: list[dict]) -> dict[str, set[str]]:
    hits_by_fixture: dict[str, set[str]] = {}
    for fixture in fixtures:
        code = _safe_read_text(fixture["file"])
        result = analyze_code(code, metadata=None, field="code", compute_structural_hash=False)
        hits_by_fixture[fixture["fixture_id"]] = {p["pattern"] for p in result["matched_patterns"]}
    return hits_by_fixture


def _load_rule_examples_from_results(result_path: str, source_name: str) -> tuple[Counter, dict[str, list[dict]]]:
    sole_counts: Counter = Counter()
    sole_examples: dict[str, list[dict]] = defaultdict(list)
    if not os.path.exists(result_path):
        return sole_counts, sole_examples

    with open(result_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("should_filter"):
                continue
            patterns = sorted({p["pattern"] for p in row.get("matched_patterns", [])})
            if len(patterns) != 1:
                continue
            pattern = patterns[0]
            sole_counts[pattern] += 1
            if len(sole_examples[pattern]) < 20:
                sole_examples[pattern].append({
                    "source": source_name,
                    "submission_id": row.get("submission_id"),
                    "id": row.get("id"),
                    "user": row.get("user"),
                    "problem_name": row.get("problem_name"),
                    "filename": row.get("filename"),
                })
    return sole_counts, sole_examples


def generate_rule_audit_report(manifest: dict) -> dict:
    """Generate a precision-audit report for every registered rule."""
    hard_positive_hits = _fixture_pattern_hits(manifest["hard_positives"])
    hard_negative_hits = _fixture_pattern_hits(manifest["hard_negatives"])
    soft_negative_hits = _fixture_pattern_hits(manifest["soft_review_negatives"])

    sole_counts_by_source: dict[str, Counter] = {}
    sole_examples_by_source: dict[str, dict[str, list[dict]]] = {}
    for source_name, result_path in AUDIT_RESULT_FILES:
        counts, examples = _load_rule_examples_from_results(result_path, source_name)
        sole_counts_by_source[source_name] = counts
        sole_examples_by_source[source_name] = examples

    rules = {}
    hard_negative_total = len(manifest["hard_negatives"])
    for rule_name in AUDIT_RULE_ORDER:
        policy = get_rule_policy(rule_name)
        expected_positive_fixtures = list(policy.mapped_positive_fixtures)
        positive_hits = sorted(
            fixture_id
            for fixture_id in expected_positive_fixtures
            if rule_name in hard_positive_hits.get(fixture_id, set())
        )
        positive_misses = sorted(
            fixture_id
            for fixture_id in expected_positive_fixtures
            if rule_name not in hard_positive_hits.get(fixture_id, set())
        )
        negative_hits = sorted(
            fixture_id for fixture_id, patterns in hard_negative_hits.items() if rule_name in patterns
        )
        soft_hits = sorted(
            fixture_id for fixture_id, patterns in soft_negative_hits.items() if rule_name in patterns
        )
        rules[rule_name] = {
            "rule_name": rule_name,
            "technique_family": policy.technique_family,
            "evidence_tier": policy.evidence_tier,
            "max_outcome": policy.max_outcome,
            "requires_companion_patterns": list(policy.requires_companion_patterns),
            "mapped_positive_fixtures": list(policy.mapped_positive_fixtures),
            "observed_hard_positive_hits": sorted(
                fixture_id for fixture_id, patterns in hard_positive_hits.items() if rule_name in patterns
            ),
            "hard_positive_hits": positive_hits,
            "hard_positive_misses": positive_misses,
            "hard_negative_hits": negative_hits,
            "confusion_matrix": {
                "true_positive": len(positive_hits),
                "false_negative": len(positive_misses),
                "false_positive": len(negative_hits),
                "true_negative": hard_negative_total - len(negative_hits),
            },
            "soft_negative_hit_count": len(soft_hits),
            "soft_negative_hit_samples": soft_hits[:20],
            "sole_hit_frequency": {
                source_name: sole_counts_by_source[source_name].get(rule_name, 0)
                for source_name, _ in AUDIT_RESULT_FILES
            },
            "sole_hit_examples": [
                example
                for source_name, _ in AUDIT_RESULT_FILES
                for example in sole_examples_by_source[source_name].get(rule_name, [])
            ][:20],
            "final_verdict": policy.default_verdict,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_counts": {
            "hard_positives": len(manifest["hard_positives"]),
            "hard_negatives": len(manifest["hard_negatives"]),
            "soft_review_negatives": len(manifest["soft_review_negatives"]),
        },
        "rule_order": AUDIT_RULE_ORDER,
        "rules": rules,
    }


def write_rule_audit_report(output_dir: str, manifest: dict, report: dict) -> tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "classifier_fixture_manifest.json")
    report_json_path = os.path.join(output_dir, "rule_audit_report.json")
    report_md_path = os.path.join(output_dir, "rule_audit_report.md")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Rule Audit Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Hard positives: {report['manifest_counts']['hard_positives']}",
        f"- Hard negatives: {report['manifest_counts']['hard_negatives']}",
        f"- Soft review negatives: {report['manifest_counts']['soft_review_negatives']}",
        "",
    ]
    for rule_name in report["rule_order"]:
        rule = report["rules"][rule_name]
        lines.extend([
            f"## {rule_name}",
            f"- Technique family: `{rule['technique_family']}`",
            f"- Evidence tier: `{rule['evidence_tier']}`",
            f"- Max outcome: `{rule['max_outcome']}`",
            f"- Final verdict: `{rule['final_verdict']}`",
            f"- Hard positive hits: {len(rule['hard_positive_hits'])}",
            f"- Hard negative hits: {len(rule['hard_negative_hits'])}",
            f"- Confusion matrix: {json.dumps(rule['confusion_matrix'], sort_keys=True)}",
            f"- Soft negative hits: {rule['soft_negative_hit_count']}",
            f"- Sole-hit frequency: {json.dumps(rule['sole_hit_frequency'], sort_keys=True)}",
            "",
        ])
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return manifest_path, report_json_path, report_md_path


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

def scan_jsonl(jsonl_path: str, results_path: str, cleaned_path: str, summary_path: str,
               workers: int = 0):
    """Stream-process JSONL dataset and output detection results + cleaned file.

    workers: number of parallel worker processes (0 = os.cpu_count()).
    """
    n_workers = workers if workers > 0 else (os.cpu_count() or 1)
    print(f"Scanning {jsonl_path}  [{n_workers} workers]")

    # Read all lines upfront so we can distribute to the pool.
    # Memory cost: ~300MB for 44K entries; acceptable.
    with open(jsonl_path, "r") as fin:
        raw_lines = list(enumerate(fin, 1))  # [(line_num, line), ...]

    total_lines = len(raw_lines)
    print(f"  Loaded {total_lines} lines, dispatching to pool...")

    all_results: list[dict] = []
    hash_groups_improved: dict = defaultdict(list)
    hash_groups_baseline: dict = defaultdict(list)
    patterns_improved: Counter = Counter()
    patterns_baseline: Counter = Counter()
    patterns_metadata: Counter = Counter()
    per_user = defaultdict(lambda: {"total": 0, "filtered": 0})
    per_problem = defaultdict(lambda: {"total": 0, "filtered": 0})

    chunksize = max(1, total_lines // (n_workers * 8))
    done = 0

    with mp.Pool(n_workers) as pool:
        for result in pool.imap_unordered(_worker_jsonl, raw_lines, chunksize=chunksize):
            if result is None:
                continue
            all_results.append(result)
            hash_groups_improved[result["code_hash_improved"]].append(result["id"])
            hash_groups_baseline[result["code_hash_baseline"]].append(result["id"])
            for p in result["matched_patterns"]:
                if p["field"] == "improved_code":
                    patterns_improved[p["pattern"]] += 1
                elif p["field"] == "baseline_code":
                    patterns_baseline[p["pattern"]] += 1
                else:
                    patterns_metadata[p["pattern"]] += 1
            per_user[result["user"]]["total"] += 1
            per_problem[result["problem_name"]]["total"] += 1
            done += 1
            if done % 5000 == 0:
                print(f"  Processed {done}/{total_lines}...")

    # Sort by original line order for deterministic output
    all_results.sort(key=lambda r: r["_line_num"])
    total = len(all_results)
    print(f"  Processed {total} entries total")

    filtered = 0
    kept = 0
    classifications: Counter = Counter()

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
            decision_patterns = [
                p for p in r["matched_patterns"]
                if p.get("field") in ("improved_code", "metadata")
            ]
            r["classification"], r["should_filter"] = classify(decision_patterns)
            r["filter_reason"] = (
                filter_reason(decision_patterns) if r["should_filter"] else None
            )

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
        "pattern_hits_metadata": dict(sorted(patterns_metadata.items(), key=lambda x: -x[1])),
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
    if patterns_metadata:
        print(f"\nPattern hits (metadata):")
        for pat, count in sorted(patterns_metadata.items(), key=lambda x: -x[1]):
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
# Mode C: Parquet dataset scan
# ---------------------------------------------------------------------------

def _iter_parquet_args(pf, cols: list, batch_size: int):
    """Generator: yield worker arg tuples one row at a time, reading the parquet in
    arrow batches so only one batch is live in the main process at a time."""
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        for r in batch.to_pylist():
            yield (
                int(r["submission_id"]),
                int(r["leaderboard_id"]),
                str(r["user_id"]),
                str(r["user_name"]),
                str(r["problem_name"]),
                float(r["score"]) if r["score"] is not None else None,
                bool(r["passed"]),
                str(r["code"]) if r["code"] else "",
            )


def scan_parquet(parquet_path: str, results_path: str, best_path: str, summary_path: str,
                 workers: int = 0, batch_size: int = 2000):
    """Scan a submission parquet file using parallel workers.

    Streams via a generator so only batch_size rows are loaded at a time in the
    main process, keeping forked worker memory usage low.

    workers: parallel workers (0 = min(8, cpu_count) — capped to avoid OOM).
    batch_size: rows per arrow read batch.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required for --parquet mode: pip install pyarrow")

    # Default to min(8, cpu_count) — 32 workers OOMs on large files
    n_workers = workers if workers > 0 else min(8, os.cpu_count() or 1)
    print(f"Scanning {parquet_path}  [{n_workers} workers, batch_size={batch_size}]")

    pf = pq.ParquetFile(parquet_path)
    cols = ["submission_id", "leaderboard_id", "user_id", "user_name",
            "problem_name", "score", "passed", "code"]
    total = pf.metadata.num_rows
    print(f"  {total:,} rows in file")

    all_results: list[dict] = []
    hash_groups: dict = defaultdict(list)
    done = 0

    # Single imap_unordered over a generator — workers stay saturated, no idle
    # gaps between batches.  chunksize=50 keeps IPC overhead low.
    with mp.Pool(n_workers) as pool:
        for result in pool.imap_unordered(
            _worker_parquet,
            _iter_parquet_args(pf, cols, batch_size),
            chunksize=50,
        ):
            all_results.append(result)
            hash_groups[result["code_hash"]].append(result["submission_id"])
            done += 1
            if done % 10000 == 0:
                print(f"  Processed {done}/{total}...")

    print(f"  Processed {done:,} submissions total")

    # Attach duplicate spam
    for r in all_results:
        ch = r["code_hash"]
        if len(hash_groups.get(ch, [])) > 1:
            r["matched_patterns"].append({
                "pattern": "DUPLICATE_SPAM",
                "severity": "medium",
                "evidence": f"code hash {ch} shared by {len(hash_groups[ch])} submissions",
                "field": "code",
            })
            r["classification"], r["should_filter"] = classify(r["matched_patterns"])
            r["filter_reason"] = filter_reason(r["matched_patterns"]) if r["should_filter"] else None

    # Sort by submission_id for deterministic output
    all_results.sort(key=lambda r: r["submission_id"])

    # Write per-submission results
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Best-per-user-per-problem: highest passing score, else highest score
    best: dict[tuple, dict] = {}
    for r in all_results:
        key = (r["user_id"], r["problem_name"])
        prev = best.get(key)
        sc = r.get("score") or 0.0
        if prev is None:
            best[key] = r
        else:
            prev_sc = prev.get("score") or 0.0
            # Prefer passing; among passing prefer lower score (faster); non-passing prefer lower too
            if (r.get("passed") and not prev.get("passed")) or \
               (r.get("passed") == prev.get("passed") and sc < prev_sc):
                best[key] = r

    with open(best_path, "w") as f:
        for r in sorted(best.values(), key=lambda r: (r["problem_name"], r.get("score") or 0)):
            f.write(json.dumps(r) + "\n")

    # Summary stats
    classifications: Counter = Counter(r["classification"] for r in all_results)
    filtered_count = sum(1 for r in all_results if r["should_filter"])
    patterns_all: Counter = Counter(
        p["pattern"] for r in all_results for p in r["matched_patterns"]
    )
    filter_reason_counts: Counter = Counter(
        r["filter_reason"] for r in all_results if r["should_filter"] and r.get("filter_reason")
    )
    per_problem: Counter = Counter(r["problem_name"] for r in all_results)
    best_classifications: Counter = Counter(r["classification"] for r in best.values())
    best_filtered = sum(1 for r in best.values() if r["should_filter"])

    summary = {
        "source_file": parquet_path,
        "total_submissions": total,
        "classifications": dict(classifications),
        "filtered": filtered_count,
        "filter_reason_breakdown": dict(sorted(filter_reason_counts.items(), key=lambda x: -x[1])),
        "pattern_hits": dict(sorted(patterns_all.items(), key=lambda x: -x[1])),
        "per_problem": dict(per_problem),
        "best_per_user_total": len(best),
        "best_per_user_classifications": dict(best_classifications),
        "best_per_user_filtered": best_filtered,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Parquet Scan Results")
    print(f"{'='*60}")
    print(f"Total submissions: {total}")
    print(f"\nClassifications:")
    for cls, count in sorted(classifications.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"\nPattern hits:")
    for pat, count in sorted(patterns_all.items(), key=lambda x: -x[1])[:15]:
        print(f"  {pat}: {count}")
    print(f"\nFiltered: {filtered_count} ({100*filtered_count/total:.1f}%)")
    print(f"\nFilter reason breakdown:")
    for reason, count in sorted(filter_reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nBest-per-user ({len(best)} entries): {best_filtered} filtered ({100*best_filtered/len(best):.1f}%)")
    print(f"\nResults:  {results_path}")
    print(f"Best:     {best_path}")
    print(f"Summary:  {summary_path}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hacky Kernel Fingerprinting Pipeline")
    parser.add_argument("--nvidia_archive", type=str, help="Path to NvidiaArchive directory")
    parser.add_argument("--jsonl", type=str, help="Path to JSONL dataset")
    parser.add_argument("--parquet", type=str, help="Path to submission parquet file")
    parser.add_argument(
        "--audit-rules",
        action="store_true",
        help="Build the precision-audit fixture manifest and rule audit report",
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel worker processes (default: os.cpu_count())")
    parser.add_argument(
        "--api-mode",
        action="store_true",
        help="Read submission code from stdin, output JSON result to stdout (for sidecar integration)",
    )
    args = parser.parse_args()

    if args.api_mode:
        import sys as _sys
        code = _sys.stdin.read()
        result = analyze_code(code, compute_structural_hash=False)
        print(json.dumps(result))
        return

    if not args.nvidia_archive and not args.jsonl and not args.parquet and not args.audit_rules:
        parser.error("Must specify at least one of --nvidia_archive, --jsonl, --parquet, --audit-rules")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.audit_rules:
        nvidia_archive_dir = args.nvidia_archive or "NvidiaArchive"
        manifest = build_classifier_fixture_manifest(nvidia_archive_dir=nvidia_archive_dir)
        report = generate_rule_audit_report(manifest)
        manifest_path, report_json_path, report_md_path = write_rule_audit_report(
            args.output_dir, manifest, report,
        )
        print("Precision audit complete")
        print(f"  Manifest: {manifest_path}")
        print(f"  Report:   {report_json_path}")
        print(f"  Summary:  {report_md_path}")

    if args.nvidia_archive:
        output_path = os.path.join(args.output_dir, "detection_results_nvidia_archive.jsonl")
        scan_nvidia_archive(args.nvidia_archive, output_path)

    if args.jsonl:
        results_path = os.path.join(args.output_dir, "detection_results_jsonl.jsonl")
        cleaned_path = os.path.join(args.output_dir, "cleaned_pairs.jsonl")
        summary_path = os.path.join(args.output_dir, "detection_summary.json")
        scan_jsonl(args.jsonl, results_path, cleaned_path, summary_path, workers=args.workers)

    if args.parquet:
        stem = os.path.splitext(os.path.basename(args.parquet))[0]
        results_path = os.path.join(args.output_dir, f"detection_results_{stem}.jsonl")
        best_path = os.path.join(args.output_dir, f"detection_results_{stem}_best.jsonl")
        summary_path = os.path.join(args.output_dir, f"detection_summary_{stem}.json")
        scan_parquet(args.parquet, results_path, best_path, summary_path, workers=args.workers)


if __name__ == "__main__":
    main()
