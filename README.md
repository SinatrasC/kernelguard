# KernelGuard

`KernelGuard` is a rule-based kernel hack detector for GPU-kernel competition submissions.

The current repository centers on `detect_hacks.py`, which can:

- classify a single submission from stdin
- scan JSONL pair datasets
- scan submission parquet exports
- generate a rule audit from a local review corpus
- load runtime config from built-in profiles, TOML files, and command-line overrides

## Current Layout

- `detect_hacks.py`
  Main detector and audit tool.

## Install

Basic install:

```bash
python3 -m pip install -e .
```

With parquet support:

```bash
python3 -m pip install -e .[parquet]
```

Installed entry point:

```bash
kernelguard --help
```

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

Rule audit:

```bash
kernelguard --audit-rules --output-dir audit_out/
```

## Runtime Config

The standalone repo already includes the later config-support additions from the newer `kguard/detect_hacks.py` line of development.

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

If no audit fixtures are discovered, the command exits with a clear error instead of silently producing an empty audit report.

Generated audit artifacts include:

- `classifier_fixture_manifest.json`
- `rule_audit_report.json`
- `rule_audit_report.md`

Generated scan artifacts include:

- `detection_results_*.jsonl`
- `detection_summary_*.json`
- `cleaned_pairs.jsonl`

These generated files are ignored by the repo-level `.gitignore`.
