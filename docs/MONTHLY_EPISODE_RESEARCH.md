# Daily episode memory for monthly selection — 2026-10-11

This research is in `outputs/monthly_daily_episodes/`. It is not registered in
the paper observers and changes no existing account, trade plan, or strategy.

The experiment collected 12,501 non-overlapping-within-security pullback anchors
from 90 securities. The trading universe remains the same archived monthly
estimated Top32 plus the assumed 17 ETFs. Each security can contribute its own
earlier price history even before entering the trading pool. Event anchors are
at least 21 sessions apart, chosen without looking at future outcomes. This does
not make events across companies statistically independent.

Labels measure next-open entry to the 5th, 10th or 21st session's close. They are
SPY-relative return, or that return minus half the maximum adverse close-path
excursion. Forecasts only use labels mature at the selection date, within the
previous five years. Same-security/same-signal averages decay with a 24-month
half-life, subtract a one-standard-error dead zone and shrink toward zero.
Insufficient evidence produces exactly zero correction, preserving the original
ranking rather than substituting a new global model.

Only the third position may change, among original ranks 3–5 within 0.02 of the
third score. The first two remain selected. Positions stay fixed within each
month. This is not a daily trading strategy or a new intramonth stop rule.

## Results

50 configurations (48 new plus 2 controls) were fixed before training. The first
88 monthly periods selected and froze the candidate before replaying the last
34. The history has already been researched repeatedly; this is not untouched
out-of-sample evidence.

Period: **2016-08-17 to 2026-09-24**, zero-cost, idealized same-final-close signals
and fills.

| Version | CAGR | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| Registered stability rule | 65.34% | 1.704 | −31.26% |
| Previous research candidate | 66.10% | 1.718 | −31.26% |
| Episode-memory candidate | 66.36% | 1.724 | −31.26% |

The selected change uses **5-session downside-penalized excess return**, exact
signal-type matching, and a 5% support-correction budget. Its next-open replay
was 64.35% CAGR without costs, or 63.40% with 0.05% per side.

**All improvement came from one month:** on 2016-08-31, the third position changed
from XOM to INTC. The September basket return rose from 3.78% to 5.46%. Removing
that single change restores the original 66.10% CAGR. The final 34 periods have
the same holdings and returns; internal sequential-selection diagnostics also
show no incremental benefit. The new number is therefore weak evidence for a
durable improvement, and no automatic promotion was made.

The median same-company/same-signal history at the 70 training boundary cases
increased from 4.5 monthly observations to 13.5 daily-event observations, with
effective sample size 11.17. More samples improved coverage, not broad performance.

## Validation and reproduction

- 18 new tests passed, including future-outcome isolation, next-open labels,
  non-overlapping anchors, exact zero correction, and locked first-two positions.
- Both original controls reproduce all 122 archived selections and scores.
- Independent daily fixed-unit replay agrees with monthly return stitching.
- 536 protected original files retained their hashes.
- The source sync separately passed 8 CLI, 20 multi-account and 9 monthly-ledger
  tests. Archived source formatting was preserved; an unrestricted diff whitespace
  check reports pre-existing trailing whitespace in several legacy research files.

After restoring the private research input bundle, run the phases in separate
processes in order: `initialize`, `prepare`, tests, `train`, `evaluate`, then
`report.py`. Existing frozen artifacts intentionally prevent overwriting a study;
use a new study directory for another experiment. The source repository excludes
the raw price data, feature/forecast caches, fitted artifacts and personal ledgers.
