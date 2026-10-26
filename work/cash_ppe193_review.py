"""Read-only interpretation of finalized193 artifacts; no new strategy calculation."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json, hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/monthly_cash_ppe193'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
assert sha(OUT/'COMPLETION.json')=='586821986e6dfc10d7070215c1379f22ddefaaadc42e816b325f972938afa2ef'
closure=read(OUT/'COMPLETION.json');assert closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED'
assert sha(OUT/'PLAN.json')==closure['plan_sha256']
rc=read(OUT/'results/COMPLETION.json');assert sha(OUT/'results/COMPLETION.json')==closure['result_completion_sha256']
for n,h in rc['artifacts_sha256'].items():assert sha(OUT/'results'/n)==h,n
pc=read(OUT/'prepared/COMPLETION.json')
for n,h in pc['artifacts_sha256'].items():assert sha(OUT/'prepared'/n)==h,n
code_review=ROOT/'outputs/monthly_capex193_source/independent/CODE_REVIEW.json'
assert sha(code_review)=='59015d792099924a87d12cbb7cb7717ebb6e9cae8a763380dee9e906d8ec5b25'
review=read(code_review);assert review['passed']
for n,h in {**review['code_sha256'],**review['input_sha256']}.items():assert sha(ROOT/n)==h,n
r=read(OUT/'results/RESULTS.json');orders=read(OUT/'prepared/ORDERS.json')
b='incumbent_local67';c='same_support_cashflow_control';t='cash_after_ppe'
assert not r['selection']['admitted'] and r['selection']['primary']==b
pairs=[]
for i,(base,cand) in enumerate(zip(orders[b],orders[t])):
 if set(base['symbols'])!=set(cand['symbols']):
  pairs.append(dict(index=i,signal_date=base['date'],holding_end=base['end'],base=base['symbols'],candidate=cand['symbols']))
assert [x['index'] for x in pairs]==[45,49]
assert all(x['symbols']==y['symbols'] for x,y in zip(orders[c],orders[t]))
cells=read(OUT/'prepared/FEATURE_AUDIT.json')['cells'];available=[x for x in cells if x['available']]
summary={}
for name,arm in r['arms'].items():
 summary[name]={path:{k:arm[path]['metrics'][k] for k in ['start','end','cagr','sharpe','max_drawdown']} for path in arm}
for name in summary:
 summary[name]['periods']={k:{m:v[m] for m in ['start','end','cagr','sharpe','max_drawdown']} for k,v in r['arms'][name]['same_close_zero']['slices'].items() if k!='full122'}
receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),schema='monthly_cash_ppe193_research_review_v1',
 source_refs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'COMPLETION.json',OUT/'PLAN.json',OUT/'results/COMPLETION.json',OUT/'prepared/COMPLETION.json',OUT/'selected/FROZEN.json',code_review,ROOT/'work/monthly_cash_ppe193_TESTS.json']},
 result='NOT_ADOPTED',best_retained=closure['best_training_selected_candidate'],new_candidate_summary=summary,
 changed_holdings_vs_incumbent=pairs,all122_cash_ppe_holdings_equal_same_support_control=True,
 financial_support=dict(all_cells=len(cells),available_cells=len(available),by_issuer=dict(Counter(x['symbol'] for x in available)),
 first88_by_issuer=dict(Counter(x['symbol'] for x in available if x['signal_index']<88)),later34_by_issuer=dict(Counter(x['symbol'] for x in available if x['signal_index']>=88))),
 interpretation=['Neither CFO-only nor CFO-minus-cash-PPE improves the fixed first88 selection.', 'Cash-PPE changes scores but creates no extra holding differences relative to CFO-only across122 months.', 'Actual changed holdings versus Local67 are two2020 months; later34 holding sets remain unchanged.', 'Training financial support is only AAPL and MSFT; this does not test broad cross-industry cash-PPE effectiveness.'],
 validation=dict(synthetic_tests=72,independent_cells=5978,independent_execution_paths=9,independent_period_slices=45,order_prefixes=123,protected_files=21444),
 reused_historical_sample=True,pristine_oos=False,observers_modified=False,new_factors_computed_in_this_review=False,new_portfolio_returns_computed_in_this_review=False,
 target_reached=False,next_direction='Source-only194: independently verify signed operating cashflow changes in trade receivables, inventories and trade payables before any new financial ratio.')
p=OUT/'RESEARCH_REVIEW.json'
with p.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(status='REVIEW_SAVED',path=str(p),sha256=sha(p),changes=pairs)))
