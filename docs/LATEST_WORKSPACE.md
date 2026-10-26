# Current workspace source — 2026-10-26

This snapshot includes the six daily paper observers, four frozen monthly
observers, historical-data validation tools, and the independent monthly
research chain through experiment 264, plus the unregistered experiment-265
prototype. Research sources retain their `outputs/`
and `work/` paths. Observer sources come from the separate active local runtime;
the research workspace's old observer backups are not the publishing source.

`WORKSPACE_SOURCE_MANIFEST.json` records every exported file's source label,
relative path, mode, source SHA-256, and published SHA-256. The only source
adaptations are the existing private-fixture guards in two historical tests.
`WORK_SOURCE_ALLOWLIST.json` lists reviewed top-level Python sources and eight
excluded local recovery/operational scripts. Neither manifest embeds credentials
or local absolute source-root provenance.

## Current research result

The best retained research candidate is `monthly_local_linear70`. The following
figures cover **2016-08-17–2026-09-24** and use the same fixed candidate:

| Execution convention | CAGR | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| Same close, **zero transaction cost** | **67.67%** | **1.735** | −31.26% |
| Next open, zero transaction cost | 65.74% | 1.700 | −32.44% |
| Next open, 0.05% per side | 64.80% | 1.682 | −32.51% |

The 70% research target refers to the first row. The 67.67% number does **not**
include transaction costs. The other rows are execution-sensitivity diagnostics,
not alternative headline definitions. These are repeatedly researched historical
results using an estimated historical Top32 and an assumed 17-ETF list; they do
not establish a fully point-in-time universe or independent out-of-sample returns.

The preceding experiment, `monthly_risk_clock263`, compared current 63-session risk
normalization with a 63-session window ending 42 sessions earlier. The long
momentum numerator still skips the recent 42 sessions. The current-window control
exactly reproduced the incumbent. The lagged-window candidate changed 55 months,
returned 55.28% same-close zero-cost CAGR (Sharpe 1.534, drawdown −32.23%), and
failed the first-88-month admission gates. Later-34/full-period results were
reported as diagnostics. No candidate was promoted into an observer.

Experiment 263 completed with 120 formal tests, independent accounting checks
for nine execution paths, and a protected-archive check. The local closure receipt
SHA-256 is `0d602a5083d4bde0fae6b010b5cbd5178813059062ea7b2cef787cc306a64b83`.
The private completion receipts, market data, derived arrays, model fits, and
trade exports are excluded from this source snapshot. Publishing this summary
is not a claim that a fresh checkout can independently reproduce those results.

## Completed experiment 264 source

This snapshot includes the nine Python source/test files for
`monthly_gain_survival264` and its registered `PLAN.json`. It explores how long
earlier gains survive before a later closing-price breach, compared with an
endpoint-retention control. The frozen original strategy and observer rules are
unchanged. The source snapshot includes the source-binding and invalid-value
validation fixes made before registration, plus the reconciled saved-result
accounting reader.

The exact current primary source hashes passed 141 synthetic tests; the saved
result reader and its tests passed another 21 synthetic checks. The registration
PLAN SHA-256 is
`fef75a47d4607476c42a1a96e74a44ce7497c707487d2ffe844300d0deaddf84`.
Actual preparation, first-88-month selection, and later-period diagnostics have
finished locally. The continuous-survival candidate returned 61.05% same-close
zero-cost CAGR, compared with 56.48% for the endpoint-retention control and
67.67% for the incumbent. It failed the first-88-month admission gates, so the
incumbent was retained. Saved-reader V4 initially rejected a one-ULP difference
in the incumbent's diagnostic cumulative cost display, although orders,
economic metrics, and both zero-cost exports matched. The additive V5 review
subsequently passed, and the experiment formally closed after all 21,444
protected archive members passed verification. The completion receipt SHA-256 is
`5e22a3ed98dc068912d01a87ac8d36133b26d604738715b35a1366cc2095cd89`.
The V4 failure and correction remain recorded in private provenance. No strategy
is promoted by this source sync; the retained headline result remains 67.67% at
zero transaction cost.

Private contracts, validation receipts, inputs and runtime progress files remain
outside the Git snapshot. The earlier `SYNC_DRAFT264_VALIDATION_20261026.json`
records the prior unregistered snapshot; it is retained as historical evidence,
not the status of the current registered source.

The additional `work/collect_form4_source_probe.py` is a separate experimental
SEC filing-data collector with offline synthetic tests. It requires a configured
compliant access declaration, stops after access/rate-limit failures, and saves
provenance separately from offline replays. Its initial preflight made no HTTP
requests because the required declaration was unavailable. This snapshot does
not contain downloaded Form 4 data or a strategy based on those filings.

### Incremental reader correction and experiment 265 prototype

The additive `monthly_gain_survival264_saved_reader_v4.py` corrects the saved
reader's explicit test-receipt binding to the registered `TESTS_V2.json`.
The original reader and registered strategy files remain unchanged. Its 30
synthetic regression checks passed, including the original 21 reader checks.
The private addendum and receipt are required for an actual local review;
the later V5 review and closure described above provide the completion evidence.

The additive `monthly_gain_survival264_saved_reader_v5.py` addresses that
post-outcome accounting-display discrepancy. It binds each summary fee exactly
to its own chronological period-fee sum, retaining the existing bounded
period-display check and exact checks on other summary fields and exports.
It adds no aggregate tolerance and leaves frozen V4 and strategy files unchanged.
Its 44 synthetic checks passed. Actual V5 review and experiment closure also
passed locally before this source sync; this sync does not rerun those write-once
phases. The V4 failure remains part of local evidence.

The unregistered experiment-265 prototype compares daily positive-gain
concentration with concentration across twelve fixed 21-session blocks. Its
pure kernel has 22 previously passed synthetic tests. This increment adds an
independent mathematical implementation, metadata source binding, and selection
integration with 21, 8, and 41 synthetic checks respectively. The integration
retains the original universe, 52% non-calm core weight, and original support
rules. The latest increment adds the lifecycle host, full independent validator,
saved-result reader, and their tests. Their respective synthetic suites passed
65, 35, and 52 checks on the exact exported source hashes. The host binds the
completed experiment-264 receipt and the metadata source contract. Paper
contracts are still being finalized before registration; they are not published
as admitted experiment evidence. No actual experiment-265 features, strategy
backtest, registration, or promotion are included. Synthetic checks do not
establish a return improvement.

## Observer versions and account continuity

`outputs/paired_stock_monitor/run_pair.py` orchestrates all ten independent
accounts. Use the launcher rather than calling individual `monitor.py` scripts.
The old `stock-monitor` package CLI is retained separately; it is not an alias
for the current multi-account runtime.

The daily accounts use the authorized `causal_replay_v3` execution branch.
Missed sessions are simulated sequentially using only the preceding session's
information for decisions, that session's opening prices and fixed costs for
fills, and closing prices for valuation. Records retain the actual computation
time and historical data vintage. Catch-up days do not count as genuinely
preplanned forward evaluation days. The v2 archive and existing v3 entries are
preserved, and repeated execution must not duplicate fills.

The four monthly observers retain their separately frozen rules. The stability
observer remains the 65.34% historical research rule, not the newer 67.67%
candidate. Account registration is not a fill; missing current pool/price inputs
must block plans. Research and observed account performance are separate.

## Running without Codex

The code runs as ordinary Python. The migrated research runtime is pinned to
Python 3.9.6, NumPy 2.0.2, and pandas 2.3.3; individual frozen experiments also
check exact runtime and input digests. Base package dependencies are in
`pyproject.toml`, with additional research packages in
`requirements-research.txt`. These files alone do not restore the private data
bundle or every historical experiment's dependencies.

From a **prepared local research workspace** with its matching inputs restored:

```sh
./run_local.sh work/monthly_risk_clock263_host.py --help
```

The research wrapper uses `work/venv/bin/python` and macOS `sandbox-exec`, denies
network and CloudStorage access, and protects observer directories. Historical
root strings are translated in memory; frozen source/configuration bytes are
not rewritten. Some archived source and sandbox paths remain machine-specific.
This is a byte-preserving source archive, not a portable deployment installer.
Do not disable hash or sandbox checks to make a replay run.

From the **separate prepared observer runtime**:

```sh
OBSERVER_PYTHON=/absolute/path/to/prepared/venv/bin/python ./run_observer.sh --phase auto
```

The observer wrapper permits public quote downloads but denies CloudStorage
fallback. Its default interpreter path is machine-specific; use the explicit
existing interpreter override when preparing another runtime. Preserve matching
activation/execution-policy files, migration receipts, ledgers, immutable plans,
and account fingerprints outside Git. Their personal runtime copies are not
published here. Do not initialize replacements as if they were existing accounts
or run two schedulers against the same ledgers. No broker is connected.

A scheduler, notification channel, fresh market-data access, and the private
historical input bundle are not installed by cloning this repository. Separate
public-source collection from offline research replay and retain provenance.

## Source-only checks

Use the prepared interpreter and run suites in separate processes, because the
archived scripts use short local module names such as `strategy` and `engine`:

```sh
python -B -m unittest discover -s tests -v
TMPDIR=/private/tmp python -B -m unittest discover -s outputs/paired_stock_monitor -p 'test_*.py' -v
```

On macOS, use the canonical `/private/tmp` temporary path for the observer
fixtures: an unresolved `/var` alias correctly trips the registry alias guard.
The initial default-temp run hit five fixture-path failures; rerunning with the
canonical temporary directory passed all 101 observer tests without changing
frozen source. The base/exporter suite passed 23 tests.

The exporter tests use temporary synthetic fixtures. Full historical parity
suites additionally need the excluded input bundle; missing data must remain an
explicit failure or documented fixture skip, never synthesized inputs.

## Updating this source snapshot

After reviewing changes to the explicit work-source list:

```sh
python -B scripts/sync_workspace.py /absolute/path/to/research/workspace \
  --observer-workspace /absolute/path/to/observer/runtime \
  --include-work-source \
  --work-source-manifest docs/WORK_SOURCE_ALLOWLIST.json \
  --include-local-launchers
```

The exporter scans all selected files and validates source/destination paths
before writing. It excludes raw/vendor data, environments, caches, credentials,
ledgers, saved trade plans, the author's supplied ledger, generated forecasts,
model fits, and unlisted work files. It does not delete unrelated repository
files or promote research candidates. Review the diff, secret-scan findings,
and tests before committing or pushing.
