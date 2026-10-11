# Round 4 pre-run protocol, 2026-10-04
Research only; no changes to five frozen observation accounts.
Same round3 inputs, 2019-01-01 through 2026-09-18, same early/middle/late partitions, same 30% R12-1 + 20% secondary momentum scores, 5 positions and sector cap2, next-open execution.
2 families x 4 variants x 2 costs =16 runs (4 baseline reproductions,12 new variants). No parameter search after seeing results.
Baseline fixed20. Vol variant: entry budget = 20% prior-close equity * min(1,25% / trailing60 adjusted-return annualized volatility). No leverage, no subsequent resizing. This is entry volatility sizing, not a literal reproduction of portfolio-volatility targeting literature. Cash earns zero.
Fast variant: prior adjusted close below SMA10 triggers next-open exit, with fixed20 maximum. New entries must be above SMA10; capacity from early exits is available on the next decision. Combined uses both changes. This is a simple fast trend exit, not a replication of deep-learning change-point papers.
Compare CAGR, max drawdown, win rate, exposure and periods at roundtrip10/20bps. No automatic promotion or optimizing to reach50%. Data has current-universe survivorship bias and repeated exploration; no pristine holdout.
