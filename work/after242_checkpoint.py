"""Record verified completed242 evidence without new strategy calculation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
prefix='outputs/monthly_block_sharpe242/'
central=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
closed=read(prefix+'COMPLETION.json');results=read(prefix+'results/RESULTS.json')
review=read('work/monthly_block_sharpe242_RESULT_REVIEW.json')
assert closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and not closed['target_reached'] and not closed['observers_modified']
assert central['current']=='monthly_block_sharpe242' and not central['registered_queue']
assert len(central['completed'])==238 and sum(x['trials'] for x in central['completed'])==1227
assert review['passed'] is True
orders=read(prefix+'prepared/ORDERS.json');base=orders['incumbent_local67']
changes={a:{name:sum(set(rows[i]['symbols'])!=set(base[i]['symbols']) for i in range(lo,hi)) for name,lo,hi in [('full122',0,122),('first88',0,88),('early53',0,53),('recent35',53,88),('later34',88,122)]} for a,rows in orders.items()}
files=[prefix+x for x in ['PLAN.json','COMPLETION.json','prepared/COMPLETION.json','prepared/FEATURE_AUDIT.json','prepared/DECISION_AUDIT.json','prepared/SOURCE_PARITY.json','prepared/INDEPENDENT_PREPARE.json','selected/FROZEN.json','results/RESULTS.json','results/COMPLETION.json','results/INDEPENDENT_OUTCOME.json']]+['work/monthly_block_sharpe242_context.json','work/monthly_block_sharpe242_TESTS.json','work/monthly_block_sharpe242_CODE_REVIEW.json','work/monthly_block_sharpe242_RESULT_REVIEW.json','work/block_sharpe242_SOURCE_REVIEW.json','work/after241_block_sharpe_coverage.json','work/after242_checkpoint.py']
d=dict(schema='after242_strategy_checkpoint_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='COMPLETED_INDEPENDENTLY_VERIFIED',goal_status='active_unmet',last_goal_turn_classification='progress',
 primary_target_basis='2016-08-17 through2026-09-24 same-close ZERO-cost CAGR>=70%; timing/cost sensitivity separate.',
 selection=results['selection'],best=central['best_training_selected_candidate'],
 metrics={a:{p:x['metrics'] for p,x in paths.items()} for a,paths in results['arms'].items()},
 same_close_slices={a:paths['same_close_zero']['slices'] for a,paths in results['arms'].items()},changed_holdings_vs_incumbent=changes,
 feature_counts=read(prefix+'prepared/FEATURE_AUDIT.json')['counts'],decision_counts=read(prefix+'prepared/DECISION_AUDIT.json')['counts'],listing_bounds=read(prefix+'prepared/SOURCE_PARITY.json')['listing_bounds'],
 validation=dict(synthetic_tests=72,independent_paths=9,independent_slices=45,all123_decision_prefixes=True,physical_prefixes=[0,1,53,88,122],protected_files=21444),completed_round_count=238,trial_count=1227,registered_queue=[],new_trials=3,new_model_fits=0,observers_modified=False,strict_pit=False,reused_sample=True,
 interpretation='Fixed21-session-block Sharpe modestly exceeds dailySharpe overfullhistory but both materially trail incumbent. Candidate fails first88 versus BOTH references, and everyearly/recent/later segment trails incumbent. Not a low-impact boundary tweak:49/122holdingmonths change (32first88). Do not select based on the slightfullperiodcandidate/control reversal.',
 next_action='Retain Local67; do not tune this failed21-session phase/window/blend or promote dailycontrol. Before another trial, choose a distinct guidance-aligned mechanism after bounded coverage review. Original source/pool/PIT limitations remain unresolved and must stay explicit; no pendingexperiment orliveprocess.',evidence_sha256={f:sha(f) for f in files})
p=ROOT/'work/after242_strategy_checkpoint.json';assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),goal_status=d['goal_status'],completed=238,trials=1227)))
