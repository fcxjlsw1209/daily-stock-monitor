"""Summarize finalized194 artifacts only; no new features or portfolio replay."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, json, hashlib
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/monthly_trade_wc194'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
parser=argparse.ArgumentParser(); parser.add_argument('--completion-sha256', required=True); args=parser.parse_args()
assert sha(OUT/'COMPLETION.json')==args.completion_sha256
closure=read(OUT/'COMPLETION.json'); assert closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED'
plan=read(OUT/'PLAN.json'); assert sha(OUT/'PLAN.json')==closure['plan_sha256']
for folder in ['prepared','results']:
 c=read(OUT/folder/'COMPLETION.json'); assert c['status']=='COMPLETE' and c['plan_sha256']==closure['plan_sha256']
 for name,digest in c['artifacts_sha256'].items(): assert sha(OUT/folder/name)==digest,name
assert sha(OUT/'results/COMPLETION.json')==closure['result_completion_sha256']
code_review=Path(plan['numerical_code_review']['path']); assert sha(code_review)==plan['numerical_code_review']['sha256']
review=read(code_review); assert review['passed'] and not review['unresolved_blockers']
for name,digest in {**review['code_sha256'],**review['input_sha256']}.items(): assert sha(ROOT/name)==digest,name
r=read(OUT/'results/RESULTS.json'); orders=read(OUT/'prepared/ORDERS.json'); frozen=read(OUT/'selected/FROZEN.json')
assert r['selection']==frozen['selection']
b='incumbent_local67'; c='same_support_cashflow_control'; t='cash_before_trade_working_capital'
assert set(orders)=={b,c,t} and all(len(x)==122 for x in orders.values())
def changes(left,right):
 return [dict(index=i,signal_date=a['date'],holding_end=a['end'],reference=a['symbols'],candidate=z['symbols']) for i,(a,z) in enumerate(zip(orders[left],orders[right])) if set(a['symbols'])!=set(z['symbols'])]
summary={}
for name,arm in r['arms'].items():
 summary[name]={path:{k:arm[path]['metrics'][k] for k in ['start','end','cagr','sharpe','max_drawdown']} for path in arm}
 summary[name]['periods']={key:{k:value[k] for k in ['start','end','cagr','sharpe','max_drawdown']} for key,value in arm['same_close_zero']['slices'].items()}
cells=read(OUT/'prepared/FEATURE_AUDIT.json')['cells']; available=[x for x in cells if x['available']]
validation=read(OUT/'prepared/INDEPENDENT_PREPARE.json'); outcome=read(OUT/'results/INDEPENDENT_OUTCOME.json')
assert validation['feature']['passed'] and outcome['passed']
refs=[OUT/'COMPLETION.json',OUT/'PLAN.json',OUT/'results/COMPLETION.json',OUT/'prepared/COMPLETION.json',OUT/'selected/FROZEN.json',code_review,ROOT/'work/monthly_trade_wc194_TESTS.json',Path(__file__)]
receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),schema='monthly_trade_wc194_research_review_v1',source_refs={str(p.relative_to(ROOT)):sha(p) for p in refs},
 result='ADMITTED' if r['selection']['admitted'] else 'NOT_ADOPTED', selection=r['selection'],best_retained=closure['best_training_selected_candidate'],new_candidate_summary=summary,
 candidate_changed_holdings_vs_incumbent=changes(b,t),control_changed_holdings_vs_incumbent=changes(b,c),candidate_changed_holdings_vs_control=changes(c,t),
 all122_candidate_holdings_equal_control=all(set(x['symbols'])==set(y['symbols']) for x,y in zip(orders[c],orders[t])),
 financial_support=dict(all_cells=len(cells),available_cells=len(available),by_issuer=dict(Counter(x['symbol'] for x in available)),first88_by_issuer=dict(Counter(x['symbol'] for x in available if x['signal_index']<88)),later34_by_issuer=dict(Counter(x['symbol'] for x in available if x['signal_index']>=88))),
 validation=dict(synthetic_tests=read(ROOT/'work/monthly_trade_wc194_TESTS.json')['tests'],independent_cells=validation['feature']['checked_cells'],independent_execution_paths=outcome['checked_paths'],independent_period_slices=outcome['checked_slices'],order_prefixes=validation['prefixes'],protected_files=closure['protected_files']),
 limitations=plan['contract']['limitation'],reused_historical_sample=True,pristine_oos=False,observers_modified=False,new_factors_computed_in_this_review=False,new_portfolio_returns_computed_in_this_review=False,target_reached=closure['target_reached'])
p=OUT/'RESEARCH_REVIEW.json'
with p.open('x') as f: json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False); f.write(chr(10))
print(json.dumps(dict(path=str(p),sha256=sha(p),result=receipt['result'],summary=summary,selection=r['selection'],financial_support=receipt['financial_support'],candidate_changes=receipt['candidate_changed_holdings_vs_incumbent'],control_changes=receipt['control_changed_holdings_vs_incumbent'],candidate_vs_control=receipt['candidate_changed_holdings_vs_control'])))
