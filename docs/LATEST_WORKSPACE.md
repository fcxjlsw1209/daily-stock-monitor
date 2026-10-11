# Current workspace code — 2026-10-11

The `outputs/` tree preserves the research workspace's relative paths. It now
contains six daily paper accounts, four separately registered monthly observers,
data preparation/audit tools, and the monthly research chain. Source hashes are
listed in `WORKSPACE_SOURCE_MANIFEST.json`; the exporter never uploads market
data, credentials, paper ledgers, saved orders, or the author's supplied ledger.

## Versions

`outputs/paired_stock_monitor/run_pair.py --phase auto` is the current orchestrator.
It runs the six daily accounts and the buffer5, regime, industry, and stability
monthly observers. `stock-monitor` remains the earlier standalone daily CLI;
it is not an alias for the newer multi-account orchestrator.

The stability monthly observer is the registered 65.34% historical CAGR research
rule, with its first prospective signal on 2026-10-30 and first planned execution
on 2026-11-02. The 66.10% candidate in `monthly_state_rank_research` remains
research-only. These figures refer to 2016-08-17–2026-09-24 idealized same-close,
zero-cost historical replay, not promised or prospective returns. Its next-open
replay with 0.05% cost per side was 63.17%. The archived universe is an estimated
historical Top32 plus an assumed 17-ETF list; repeated research is not pristine
out-of-sample evidence.

The subsequent [daily-episode-memory experiment](MONTHLY_EPISODE_RESEARCH.md)
reached 66.36% on the same full-period idealized basis. Its entire incremental
benefit came from one 2016 month, so it also remains research-only. Neither
research result replaces the registered stability account.

## Running without Codex

These are ordinary Python scripts, not an LLM service. Python 3.9+ and the pinned
base dependencies in `pyproject.toml` are required. Some research additionally
uses SciPy, scikit-learn, plotting and data-format packages; see its imports and
the optional research requirements. The current orchestrator uses `fcntl`, so
macOS/Linux is the supported runtime.

From a **prepared workspace with its private data restored**, use:

```sh
python outputs/paired_stock_monitor/run_pair.py --phase auto
```

A clean checkout alone is **not** a complete unattended deployment. In particular:

- Monthly `inputs.py` reads audited local price and membership archives; it does
  not download a fresh Top32 automatically. Stale/missing inputs must block new
  plans. Historical manifests can contain absolute source paths and must be
  supplied/reconciled on migration; never disable their hash checks.
- Preserve the existing SQLite ledgers, immutable plans and account fingerprints
  separately and back them up. Do not initialize replacement ledgers as if they
  were the original accounts or run two schedulers against the same account.
- Use a system scheduler and a separate notification channel. Neither is
  installed by cloning this repository. Local execution needs an awake computer;
  a server needs persistent storage, network access and operational monitoring.
- Existing observer configurations document specific registered accounts and
  dates. They are not generic templates for retrospectively creating trades.
- Research `FROZEN.json` files require excluded input hashes, fitted forecasts,
  feature caches and historical pool evidence. Restore those artifacts before
  replay; do not advertise source-only checkout as independently reproduced.

No broker is connected and no real orders are submitted.

## Validation without private history

Run each suite in a separate Python process because the archived scripts use
short local module names such as `strategy` and `engine`:

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s outputs/paired_stock_monitor -p 'test_*.py' -v
cd outputs/monthly_stability_monitor
python -m unittest test_observe.LedgerTests -v
```

Full historical parity suites additionally require the private data bundle.

## Syncing future research

```sh
python scripts/sync_workspace.py /absolute/path/to/research/workspace
```

The exporter uses a source/metadata allowlist, checks suspicious credential
literals, records hashes, and never deletes unrelated repository files. Review
the diff and run relevant tests before committing. Exporting source does not
promote any research candidate into a paper observer.
