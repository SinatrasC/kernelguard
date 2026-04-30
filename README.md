# KernelGuard

`KernelGuard` is a rule-based kernel hack detector for GPU-kernel competition submissions.

The core module is `kernelguard.py`, which can:

- classify a single submission from stdin
- scan JSONL pair datasets
- scan submission parquet exports
- generate a rule audit from a local review corpus
- load runtime config from built-in profiles, TOML files, and command-line overrides

## Current Layout

- `kernelguard.py`
  Main detector and audit tool.
- `kernelguard_api.py`
  HTTP API for the competition (kernels, validation suites, red/blue submissions).
- `kernelguard_mcp.py`
  MCP server exposing `analyze_code` to MCP clients.

## Install

```bash
pip install kernelguard
```

With extras:

```bash
pip install "kernelguard[parquet]"
pip install "kernelguard[mcp]"
```

### uv

One-shot (no install):

```bash
uvx kernelguard --help
uvx kernelguard --api-mode < submission.py
uvx --from "kernelguard[parquet]" kernelguard --parquet submissions.parquet
uvx --from "kernelguard[mcp]" kernelguard-mcp --version
```

Permanent install:

```bash
uv tool install kernelguard                # core only
uv tool install "kernelguard[parquet]"     # + parquet scanning
uv tool install "kernelguard[mcp]"         # + MCP server
uv tool install "kernelguard[parquet,mcp]" # + parquet + MCP
uv tool install "kernelguard[api]"         # + HTTP API server
```

Entry points after installation: `kernelguard`, `kguard`, `kernelguard-mcp`, `kguard-mcp`, `kernelguard-api`, `kguard-api`.

## Quick Start

Single kernel from stdin:

```bash
kernelguard --api-mode < submission.py
```

JSONL scan:

```bash
kernelguard --jsonl /path/to/pairs.jsonl --output-dir out/
```

Parquet scan:

```bash
kernelguard --parquet /path/to/submissions.parquet --output-dir out/
```

## MCP Server

KernelGuard runs as a local MCP server over stdio.

```bash
kernelguard-mcp                  # default profile
kernelguard-mcp --profile strict # higher-recall review mode
kernelguard-mcp --version
```

Two tools:

- `analyze_code` — analyze a code string, return detection results.
- `analyze_file` — read a file path, return detection results plus resolved path.

Both accept optional `metadata` (for score-based detection) and `profile` parameters.

The server is read-only: no resources, prompts, scan tools, or write operations.

## HTTP API Server

KernelGuard runs as an HTTP API server for integration with services like KernelBot.

```bash
kernelguard-api                           # default: 127.0.0.1:8088
kernelguard-api --host 0.0.0.0 --port 80 # public
kernelguard-api --profile strict          # higher-recall mode
```

Endpoints:

- `GET /health` — returns `{"status": "ok", "version": "...", "profile": "..."}`.
- `POST /analyze` — accepts `{"code": "...", "metadata": null, "profile": "default"}`, returns detection results.

The API server keeps KernelGuard loaded in memory, eliminating cold-start latency from subprocess invocations.

## Runtime Config

The standalone repo already includes the later config-support additions from the newer line of development.

Runtime behavior can be controlled with:

- `--profile`
  Select a built-in profile.
- `--config`
  Load a TOML config file.
- `--set`
  Apply dotted `key=value` overrides from the CLI.
- `--export-config`
  Print or write the resolved config and exit.

Examples:

Export the default resolved config:

```bash
kernelguard --export-config
```

Export the strict profile:

```bash
kernelguard --profile strict --export-config
```

Run with a TOML config file:

```bash
kernelguard --config kernelguard.toml --jsonl /path/to/pairs.jsonl --output-dir out/
```

Apply a one-off override:

```bash
kernelguard --set 'entrypoints.names=["kernel"]' --api-mode < submission.py
```

The config layer covers:

- rule policy overrides
- score thresholds
- duplicate handling
- classification behavior
- entrypoint-name configuration

## Compatibility

The primary public interfaces are:

- `--api-mode` for a single kernel
- `--jsonl` for pair datasets
- `--parquet` for submission exports

The detector also keeps some legacy compatibility for older internal audit/archive workflows so historical corpora can still be reused.

## Audit Behavior

`--audit-rules` is meant to run in a workspace that contains the audit corpora and prior detector outputs.

This is an internal evaluation mode, not a normal first-run path.

If no audit fixtures are discovered, the command exits with a clear error instead of silently producing an empty audit report.

Minimal audit run:

```bash
kernelguard --audit-rules --output-dir audit_out/
```

If you want to drive audit mode from explicit inputs, put them in your config file:

```toml
[audit]
archive_dir = "/path/to/archive"
ground_truth_dir = "/path/to/ground_truth_dir"
manual_review_files = [
  "/path/to/manual_review_1.json",
  "/path/to/manual_review_2.json",
]
filtered_results_path = "/path/to/filtered_results.jsonl"
```

To compare old and new detector outputs explicitly during audit, put them in config too:

```toml
[audit.result_files]
old = "/path/to/old_results.jsonl"
new = "/path/to/new_results.jsonl"
```

Then run:

```bash
kernelguard --config kernelguard.toml --audit-rules --output-dir audit_out/
```

Generated audit artifacts include:

- `classifier_fixture_manifest.json`
- `rule_audit_report.json`
- `rule_audit_report.md`

Generated scan artifacts include:

- `detection_results_*.jsonl`
- `detection_summary_*.json`
- `cleaned_pairs.jsonl`

These generated files are ignored by the repo-level `.gitignore`.
