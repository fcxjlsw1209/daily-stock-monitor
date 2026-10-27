# Current workspace source — 2026-10-27

This snapshot includes the six daily paper observers, four frozen monthly
observers, historical-data validation tools, and the independent monthly
research chain through completed experiment 271. Experiments 266–269 and 271
have completed local replay, saved-evidence review and protected-archive closure.
The separate three-field financial-source prototype described below has only
completed synthetic validation; it is not a new admitted dataset or strategy.
Research sources retain their `outputs/`
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

### Incremental reader correction and registered experiment 265

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

Experiment 265 compares daily positive-gain
concentration with concentration across twelve fixed 21-session blocks. Its
pure kernel has 22 previously passed synthetic tests. This increment adds an
independent mathematical implementation, metadata source binding, and selection
integration with 21, 8, and 41 synthetic checks respectively. The integration
retains the original universe, 52% non-calm core weight, and original support
rules. The latest increment adds the lifecycle host, full independent validator,
saved-result reader, and their tests. Their respective synthetic suites passed
65, 35, and 52 checks on the exact exported source hashes. The host binds the
completed experiment-264 receipt and the metadata source contract. The final
paper and implementation contracts passed review, and 192 formal synthetic
checks passed before registration. The registered PLAN SHA-256 is
`628cd752c6112288cca49fce9d26144e43021a48ca63a5644e4567ed9a6eff48`.
This increment retains that PLAN and the previously published Python sources.
Actual preparation, selection, diagnostics, saved-result review, and final closure
have now completed locally. The first-88-month admission gates retained the
incumbent; no strategy was promoted. The completion receipt SHA-256 is
`c2505d17ed02db8d177419c56a41fee0f687fe0fc2030f1cf244e10710fb97fa`.
Private inputs, contract/review receipts, and generated artifacts remain excluded.

### Registered experiment 266 source

This increment adds the monotone weak-tree prototype, its independent validator,
the first-88-month admission module, a historical-reference comparison helper,
and their synthetic tests. The experiment
compares the existing unconstrained weak-market tree adjustment with the same
model constrained to be nondecreasing in its original long-momentum rank,
holding the other inputs fixed. It retains the original monthly strategy core
and treats the tree as a small scoring adjustment. The constraint does not imply
monotone final portfolio returns or holdings.

The producer, independent-validation, and admission suites passed 48, 31, and
17 synthetic tests respectively on the exported source hashes. Tree refits use
the same pinned learning library; they are not an independently implemented
training algorithm. The separate structure, traversal, timing, selection, and
accounting checks retain their stated scope.

The reference helper passed another 22 synthetic checks. It compares supplied
historical models, forecasts, rankings, and orders, enforcing prefix scope and
bindings to inputs already verified by the caller. It does not load historical
data, authenticate a caller merely from a `passed` flag, or establish actual
historical-control parity on its own.

The lifecycle host and saved-result reader are now included, together with their
synthetic tests. The host requires all ten primary source/test files and both
reader files to be pinned, verifies consumed historical-reference prefixes before
and after each comparison helper, and checks zero-cost outputs against the actual
nested accounting schema. The reader checks saved phase evidence, source bindings,
historical-control mappings and accounting exports. Its 60 synthetic tests and
the host's 35 synthetic tests passed on the exact source hashes published here.
These tests do not constitute an actual historical replay.

Experiment 266 is now registered; this snapshot adds its original `PLAN.json`
(SHA-256 `f9e9643c8e3969d1fdd25b47727adf386e1d2e67e4b3b3c885f5f116d0700407`).
Actual preparation, first-88-month selection, later-period diagnostics, and the
saved-result review passed locally. The constrained candidate and unconstrained
control produced identical monthly holdings and about 66.85% same-close,
zero-cost CAGR, below the retained 67.67% incumbent. The candidate failed the
admission gates. Its protected-archive closure has since completed locally;
publishing the source does not rerun that experiment.

Private contracts, progress records, fixtures, models and result exports remain
outside this snapshot. The earlier
[lifecycle sync validation](SYNC_266_LIFECYCLE_VALIDATION_20261026.json) records
its then-prospective source state; it is not the current experiment status.

### Completed experiment 267: boundary-pair scoring

The full lifecycle now includes the Ridge comparison of the original rank-3
versus rank-4 return gap, with all pairs among the original top five as a
control. The host authenticates original scores, eligibility, features and
adjusted-price endpoints before candidates. It retains the original boundary
selector and does not execute live accounts.

The final source passed 174 formal and 26 saved-reader synthetic checks locally.
The first-88-month gate rejected the candidate. Full-period same-close zero-cost
CAGR was 62.80% for the all-pairs control and 63.90% for rank-3/rank-4, versus
67.67% for the incumbent. The earlier
[prototype validation](SYNC_267_PROTOTYPE_VALIDATION_20261026.json) is retained
as evidence of that earlier snapshot, not the current lifecycle status.

### Completed experiment 268: selector scope

This matched comparison keeps the original Local67 scores, eligibility, model
effects, momentum weights and adjustment budget. It tests removing the top-two
retention rule within the original top-five boundary group, and allowing all
eligible names to compete. Admission for both candidates is evaluated using only
the first 88 months before later-34/full-period and cost diagnostics.

The exact published implementation passed 106 formal and 32 saved-reader
synthetic checks locally. All 123 decision prefixes, four additional physical
prefixes and nine accounting paths passed; 21,444 protected files remained
unchanged. Both candidates changed zero holding sets across all 122 months,
although five months reordered the same equal-weight names. All three strategy arms
retain the same-close, zero-cost headline of 67.67% CAGR; neither candidate was admitted.
These are historical research results, with the data limitations above.

Source, PLANs and a compact aggregate validation receipt are published. Raw
prices, private contracts, source evidence, forecasts, model fits, author
transactions and observer ledgers remain excluded. A clone alone is not a
complete historical replay bundle. See
[the completion-sync validation](SYNC_267_268_COMPLETION_VALIDATION_20261026.json).

### Completed experiment 269: forecast spacing

This matched comparison keeps the original momentum weights, eligible universe,
forecast inputs, adjustment budget and boundary selector. It tests replacing
percentile ranks with an affine map that retains the spacing between raw
forecasts, for the pooled forecast alone and for both forecast components.
The first 88 months determine admission before the later 34 months and full-period
diagnostics are examined.

The exact published primary code passed 111 synthetic checks; the saved-result
reader passed another 38. All 123 cached decision prefixes, four freshly
recomputed physical prefixes, full-prefix reuse and nine accounting paths passed
locally, and all 21,444 protected files
remained unchanged. Both candidates changed only one monthly holding set and
returned 67.33% same-close zero-cost CAGR with Sharpe 1.729, below the retained
67.67% / 1.735 candidate. Neither passed admission. The original registered PLAN
and source are included; generated forecasts, raw inputs and private review
receipts remain outside this snapshot.

### Completed experiment 271: member and joint log growth

This comparison first validates every originally eligible security's 253 adjusted
prices ending 42 sessions before the signal. It keeps at most six positive annual
momentum leaders, then compares the three strongest individual annual trends
with the trio maximizing joint equal-weight log growth over twelve fixed
21-session historical blocks. Exact rational products determine rankings;
displayed logarithms do not determine ties. Deployment remains monthly,
equal-weight and fixed-unit between rebalances. The long-momentum shortlist is
an explicit new domain; this is not an unchanged numerical 52% Local67 core.

The eight formal source/test files passed 112 synthetic tests locally. The
published additive v2 saved-result reader passed nine self-tests and an independent
review; it accepts the recorded root-owner phase metadata while preserving the
original accounting rules. The original reader remains as historical source.
Actual preparation, first-88-month selection, diagnostics, independent saved-result
review and protected closure completed. All 123 cached prefixes, four freshly
computed physical prefixes, full-prefix reuse, nine accounting paths, 45 slices
and 18 training groups were checked in their documented scopes; 21,444 protected
files were unchanged. The saved reader rechecked saved price-window mathematics,
not a new raw-panel reconstruction or return replay.

Neither candidate passed the first-88-month admission gates. The incumbent
remains `monthly_local_linear70`, with the 67.67% same-close zero-cost CAGR above;
the 70% target remains unmet. This publication adds the ten verified Python
files and original registered PLAN. It does not include input windows, choices,
models, return exports, private source evidence or author-supplied ledger files.
The PLAN retains its original registration-time status and dependency digests;
its historical flags are not a claim that the completed run is still prospective.
See [the experiment 271 sync validation](SYNC_271_VALIDATION_20261027.json).

### Source-only prototype 270: same-report financial field scope

`work/roa_field_scope270.py` separately assesses the source requirements for
current annual consolidated net income, opening total assets and closing total
assets. It can identify two specifically named comparison-flow duration holds
as inapplicable to this three-field scope while preserving all original holds.
Missing required fields, conflicting dates, mixed currency declarations and
unknown holds still fail. The function binds the complete event and three cells
to supplied review references without reading files or calculating a ratio.

The module and its tests passed 36 synthetic checks and an independent static
review. A successful pure-function result does not authenticate source files or
review receipts. The caller must still verify those bytes and their substantive
evidence before any actual admission. No real report, cross-report chain or
strategy has been admitted by this prototype, and no return is attributed to it.
The older source gates and frozen observation accounts are unchanged.

The [269/270 source-sync validation](SYNC_269_270_VALIDATION_20261027.json)
records that earlier publication. Later 270 v3 consumer and policy work is still
under review and is excluded from this snapshot. No author-ledger preservation
files or tools are published.

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
frozen source. The base/exporter suite passed 24 tests in the latest lifecycle
source sync; the observer tests above describe the earlier unchanged-source check.

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
