"""Audit a user-supplied simulated holdings ledger; no signal fitting or account writes."""
from pathlib import Path
import sys, json, re, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = Path('/Users/shiweili/.codex/attachments/3c44b23b-18a1-4cba-a5e7-ad6c9b8c6aa4/Pasted text.txt')
sys.path.insert(0, str(ROOT / 'outputs/monthly_path_constraints'))
import study

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def stats(s):
    r = s.pct_change().dropna()
    years = (s.index[-1]-s.index[0]).days/365.2425
    return {'start': str(s.index[0].date()), 'end': str(s.index[-1].date()),
            'daily_returns': len(r), 'ending_multiple': float(s.iloc[-1]/s.iloc[0]),
            'total_return': float(s.iloc[-1]/s.iloc[0]-1),
            'cagr': float((s.iloc[-1]/s.iloc[0])**(1/years)-1),
            'daily_sharpe_rf0_sqrt252_ddof1': float(r.mean()/r.std(ddof=1)*np.sqrt(252)),
            'max_daily_close_drawdown': float((s/s.cummax()-1).min())}

text = SRC.read_text()
num = r'\(?-?\d+(?:\.\d+)?%\)?'
pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}) (\d{4}-\d{2}-\d{2}) (.+?) ('+num+r') ('+num+r') ('+num+r') (COMPLETED PERIOD|OPEN PERIOD)$')
def pct(x):
    return float(x.strip('()%'))/100*(-1 if x.startswith('(') else 1)
rows=[]
for line in text.splitlines():
    match = pattern.match(line)
    if match:
        start, end, held, ret, qqq, excess, status = match.groups()
        rows.append(dict(start=start, end=end, holdings=held.split(' / '), reported_return=pct(ret),
                         reported_qqq=pct(qqq), reported_excess=pct(excess), status=status))
assert len(rows) == 69
assert len({r['start'] for r in rows}) == len(rows)
assert all(len(set(r['holdings'])) == 3 for r in rows)
assert all(a['end'] == b['start'] for a,b in zip(rows, rows[1:]))
px, pools, sources = study.load()
sources[str(SRC)] = sha(SRC)
foreign_manifest = ROOT / 'outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json'
sources[str(foreign_manifest)] = sha(foreign_manifest)
for f in json.loads(foreign_manifest.read_text()):
    if f['symbol'] not in {'BABA','TSM','NVO'}:
        continue
    assert sha(f['file']) == f['sha256']
    px[f['symbol']] = pd.read_csv(f['file'], index_col=0, parse_dates=True)['Adj Close'].reindex(px.index)
    sources[f['file']] = f['sha256']

oldfile = ROOT / 'outputs/monthly_decision_reconstruction/DECISIONS.json'
sources[str(oldfile)] = sha(oldfile)
old = {r['date']:r for r in json.loads(oldfile.read_text())}
alias = lambda s: 'research_equity_FB' if s == 'META' else s
unalias = lambda s: 'META' if s == 'research_equity_FB' else s
nav, pieces, realized = 100000., [], 0.
units, basis = {}, {}
comparison=[]
for i,row in enumerate(rows):
    start,end=pd.Timestamp(row['start']),pd.Timestamp(row['end'])
    cols=[alias(s) for s in row['holdings']]
    prices=px.loc[start:end, cols]
    assert len(prices)>1 and prices.index[0]==start and prices.index[-1]==end
    assert prices.notna().all().all() and prices.gt(0).all().all()
    path=(prices/prices.iloc[0]).mean(axis=1)
    # Equal weights at each start close, fixed adjusted units through next end close.
    row['replayed_return']=float(path.iloc[-1]-1)
    row['replayed_qqq']=float(px.at[end,'QQQ']/px.at[start,'QQQ']-1)
    row['replayed_excess']=row['replayed_return']-row['replayed_qqq']
    for key in ['return','qqq','excess']:
        row[key+'_error_bp']=(row['replayed_'+key]-row['reported_'+key])*10000
        row[key+'_rounding_match']=abs(row['replayed_'+key]-row['reported_'+key])<=.00005+1e-10
    targets={s:nav/3/float(px.at[start,alias(s)]) for s in row['holdings']}
    # Average-cost accounting: sell only excess units, add cost basis on purchases.
    for s in set(units)|set(targets):
        oldunits=units.get(s,0.)
        newunits=targets.get(s,0.)
        p=float(px.at[start,alias(s)])
        if newunits<oldunits:
            avg=basis[s]/oldunits
            realized+=(oldunits-newunits)*(p-avg)
            basis[s]-=(oldunits-newunits)*avg
        elif newunits>oldunits:
            basis[s]=basis.get(s,0.)+(newunits-oldunits)*p
        units[s]=newunits
    local=path*nav
    pieces.append(local if i==0 else local.iloc[1:])
    nav=float(local.iloc[-1])
    if row['start'] in old:
        prev=old[row['start']]
        inferred={unalias(t) for t in prev['target']}
        actual=set(row['holdings'])
        comp={'start':row['start'], 'end':row['end'], 'supplied':sorted(actual),
              'previous_inferred':sorted(inferred), 'exact':actual==inferred,
              'shared_count':len(actual&inferred),
              'outside_previous_universe':sorted(actual-{unalias(t) for t in prev['universe']}),
              'outside_previous_eligible':sorted(actual-{unalias(t) for t in prev['eligible']})}
        comparison.append(comp)

curve=pd.concat(pieces)
assert not curve.index.duplicated().any()
assert curve.index.equals(px.loc[rows[0]['start']:rows[-1]['end']].index)
final_date=pd.Timestamp(rows[-1]['end'])
unrealized=sum(u*float(px.at[final_date,alias(s)])-basis[s] for s,u in units.items())
assert abs(realized+unrealized+100000-nav)<1e-6
qqq=px.loc[curve.index, 'QQQ']/px.loc[curve.index[0],'QQQ']*100000
rounded=np.prod([1+r['reported_return'] for r in rows])*100000
summary={'initial':100000., 'realized':3468384.30, 'unrealized':858080.66, 'ending':4426464.96}
years=(curve.index[-1]-curve.index[0]).days/365.2425
res={
 'classification':'User-supplied simulated replay; observed holdings, not broker trades or recovered selection algorithm',
 'source_sha256':sha(SRC), 'period_count':len(rows), 'completed':sum(r['status']=='COMPLETED PERIOD' for r in rows),
 'unique_tickers':sorted({s for r in rows for s in r['holdings']}),
 'assumptions':['Equal thirds at each supplied start close; fixed adjusted units within each period',
                'Zero transaction costs; fractional adjusted units', 'QQQ adjusted close total-return approximation',
                'META priced via existing validated research_equity_FB alias',
                'Daily Sharpe is zero-risk-free mean / sample standard deviation times sqrt(252)',
                'This reconstructs known holdings after the fact; does not test how they were selected'],
 'return_rounding_matches':sum(r['return_rounding_match'] for r in rows),
 'qqq_rounding_matches':sum(r['qqq_rounding_match'] for r in rows),
 'excess_rounding_matches':sum(r['excess_rounding_match'] for r in rows),
 'max_return_error_bp':max(abs(r['return_error_bp']) for r in rows),
 'supplied_summary':summary,
 'supplied_summary_arithmetic_ok':abs(summary['initial']+summary['realized']+summary['unrealized']-summary['ending'])<1e-8,
 'summary_implied_cagr':(summary['ending']/100000)**(1/years)-1,
 'rounded_periods_ending':float(rounded),
 'replay_accounting':{'ending':nav, 'realized':realized, 'unrealized':unrealized},
 'replay_accounting_minus_supplied':{'ending':nav-summary['ending'], 'realized':realized-summary['realized'], 'unrealized':unrealized-summary['unrealized']},
 'replay':stats(curve), 'qqq':stats(qqq),
 'previous_inferred_holdings':{'comparable_periods':len(comparison), 'exact_matches':sum(r['exact'] for r in comparison),
                              'mean_tickers_shared':float(np.mean([r['shared_count'] for r in comparison])),
                              'periods_outside_proxy_universe':sum(bool(r['outside_previous_universe']) for r in comparison),
                              'periods_outside_previous_eligible':sum(bool(r['outside_previous_eligible']) for r in comparison)},
 'known_limitations':['Supplied ledger begins 2020-12-31, image begins 2016-08-17; statistics are not the same window',
                      'Authors complete 17 ETF list, exact historical cap-ranking method, ranking formula and training process still unknown',
                      'Holding history is not proof of ex ante availability or out-of-sample selection',
                      'Foreign-price caches existed in earlier research; only recent reconstruction loader omitted these issuers']}
annual=[]
for year in sorted(set(curve.index.year)):
    if year==2020: continue
    subset=curve[curve.index.year==year]
    baseline=curve[curve.index<subset.index[0]].iloc[-1]
    qsubset=qqq.loc[subset.index]
    qbase=qqq[qqq.index<subset.index[0]].iloc[-1]
    annual.append({'year':year, 'end':str(subset.index[-1].date()), 'return':float(subset.iloc[-1]/baseline-1), 'qqq':float(qsubset.iloc[-1]/qbase-1)})
res['annual_returns']=annual
assert all(sha(p)==h for p,h in sources.items()), 'Read-only input changed during audit'
(OUT/'SOURCE.txt').write_bytes(SRC.read_bytes())
dump('SOURCE_HASHES.json', sources)
dump('PERIODS.json', rows)
dump('COMPARISON_WITH_INFERRED.json', comparison)
dump('AUDIT.json', res)
dump('DAILY_REPLAY.json', [{'date':str(d.date()),'nav':float(v),'qqq_nav':float(qqq.at[d])} for d,v in curve.items()])
print(json.dumps(res, ensure_ascii=False, indent=2))
