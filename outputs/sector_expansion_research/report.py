"""Render all research results, without selecting or deploying a live strategy."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parent

def table(df,cols):
 labels=[c.replace('_',' ') for c in cols]
 rows=['| '+' | '.join(labels)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for _,r in df.iterrows():
  vals=[]
  for c in cols:
   x=r[c]
   if pd.isna(x):v='n/a'
   elif c in ['cagr','max_drawdown','trade_win','average_exposure','total_return']:v=f'{100*x:.2f}%'
   elif c in ['closed_trades','cost_bps']:v=str(int(x))
   else:v=str(x)
   vals.append(v)
  rows.append('| '+' | '.join(vals)+' |')
 return '\n'.join(rows)

def main():
 m=pd.read_csv(P/'metrics.csv');a=pd.read_csv(P/'yearly.csv');v=json.loads((P/'validation.json').read_text())
 full=m[m.split.eq('all')];long=full[(~full.shorts)&full.cost_bps.eq(10)]
 out=['# Expanded-universe sector-factor backtest',
 'Completed study: 2019-01-02 through 2026-10-05. USD100,000 per run. 34 runs: 24 long/cash, 6 hypothetical long/short cost scenarios, 4 benchmarks. No existing paper accounts changed.',
 '## Findings',
 'Expanding the selectable pool did not improve these recipes. Sector trend and breadth replacements underperformed their acceleration controls. The activity recipe is promising only in the 77-name intersection and is not robust to universe expansion. No full-period strategy approached 50% CAGR. A high recent-period CAGR is not a stable return expectation.',
 'The original-intersection pool is 77 current candidates overlapping the old100, not the original100 itself. All recipes were recomputed on the same new tape. Do not compare these figures directly with older runs as a controlled experiment: universe membership, data vintage, cutoff and issuer filtering differ.',
 '## Full-period results, 10bps roundtrip',
 table(long,['universe','recipe','cagr','max_drawdown','trade_win','closed_trades','average_exposure']),
 '## Cost stress: 20bps roundtrip',
 table(full[(~full.shorts)&full.cost_bps.eq(20)],['universe','recipe','cagr','max_drawdown','trade_win']),
 '## Period sensitivity',
 'Subperiods slice the same continuing equity path and inherit existing positions; they are not fresh independent accounts. Early=2019-2021, middle=2022-2023, late=2024-2026-10-05.',
 table(m[m.cost_bps.eq(10)&(~m.shorts)&m.recipe.isin(['accel40','activity','QQQ'])&m.split.ne('all')],['universe','recipe','split','cagr','max_drawdown']),
 '## Hypothetical short-enabled control',
 'Uses combined ranking, allows shorts in SPY bear regime, and retains positions for20 sessions. Borrow fees accrued on calendar days; short dividends paid. Missing locates, recalls and broker margin constraints mean this is not executable historical performance. Entry gross exposure capped at equity; subsequent price drift can exceed100%.',
 table(full[full.shorts],['cost_bps','borrow_rate','cagr','max_drawdown','trade_win','closed_trades']),
 'The 5% borrow,10bps case attribution (realized plus unrealized):',
 '```json\n'+json.dumps(json.loads((P/'expanded_combined_10_borrow5_positions.json').read_text())['side_pnl'],indent=2)+'\n```',
 'Short losses are primarily adverse price moves, not just borrowing costs. Closed short positions in the5% case include sharp rebound losses such as OXY in March-April2020. Low momentum plus fixed20 holding is vulnerable to rebound squeezes.',
 '## Exact recipes',
 'Weights are score weights, not capital allocations. All momentum-family weights total50%: R12-1, acceleration, sector60-day return and peer breadth. All variants also have price-trend entry gates.',
 '```json\n'+json.dumps(json.loads((P/'recipes.json').read_text()),indent=2)+'\n```',
 'Jump is negative maximum absolute60-day return divided by sum of absolute60-day returns. Downside is negative standard deviation of clipped-negative60-day returns. CMF uses21 sessions. Breadth is the fraction of eligible same-sector peers aboveSMA100. Activity is the median peer5-day/60-day dollar-volume ratio. Both exclude the target issuer and require10 peers. Current broad GICS sectors are not granular historical industries. Activity measures trading intensity, not net money inflow.',
 'Shared long entries:253 valid consecutive sessions,20-day dollar volume>USD20m, R12-1>0, adjusted close>SMA100, SPY>SMA200, weighted percentile score>=.65. Next-open execution;20% prior-close equity budget, max5 names, max2 per sector, max1 per issuer; fixed20-session exit; no same-day reentry. Cash earns zero. Sector caps can concentrate equity in only3 sectors. Multi-class issuers use trailing20-day liquidity for selection.',
 '## Validation and limits',
 f"Prefix invariance passed for both universes; all34 runs passed accounting identities; {int(full.data_events.sum())} missing-bar/valuation events during simulated holdings. Four execution tests cover next-open timing, fixed holding age, short dividends/borrow, and deferred exits/stale valuation. Raw503-name hashes verified. Results rerun reproducibility is checked separately in completion.json.",
 'Current constituents and current sector labels introduce survivorship and classification look-ahead. Peer statistics share that bias. Historical exits/delistings are incomplete;9 sampled retired symbols had no Yahoo data. No unexamined holdout exists after repeated prior exploration. The study is exploratory evidence, not out-of-sample validation.',
 'Signals quarantine invalid bars and dependent253-session windows. Corporate actions and vendor adjustments have not all been independently confirmed. Yahoo OHLC is split-adjusted; cash dividends are credited/debited separately. Adj Close is used for signal returns. Splits are not applied twice. Benchmarks hold SPY/QQQ from first test open, pay buy fee, receive dividends into zero-interest cash, have no terminal sell fee and no dividend reinvestment. Thus they differ from adjusted-close reinvestment benchmarks in the daily monitor.',
 'No post-result weight search, no new observation account, no frozen strategy changes. Full daily paths, trades, orders, final positions, yearly returns, code and source hashes are in this directory.',
 '## Annual returns: activity candidate versus control and QQQ',
 table(a[a.cost_bps.eq(10)&(~a.shorts)&((a.universe.eq('original_intersection')&a.recipe.isin(['activity','accel40']))|a.recipe.eq('QQQ'))],['universe','recipe','year','total_return','max_drawdown']),
 ]
 (P/'REPORT.md').write_text('\n\n'.join(out)+'\n')
 print('REPORT.md generated')
if __name__=='__main__':main()
