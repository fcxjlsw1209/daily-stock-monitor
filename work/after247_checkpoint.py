"""Summarize only a completed and independently reviewed portfolio experiment."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
prefix='outputs/monthly_portfolio_path247/'
central=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
closed=read(prefix+'COMPLETION.json');results=read(prefix+'results/RESULTS.json')
review=read('work/monthly_portfolio_path247_RESULT_REVIEW.json')
assert closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and not closed['observers_modified']
assert central['current']=='monthly_portfolio_path247' and not central['registered_queue']
assert len(central['completed'])==243 and sum(x['trials'] for x in central['completed'])==1243
assert review['passed'] is True
orders=read(prefix+'prepared/ORDERS.json');base=orders['incumbent_local67']
changes={a:{name:sum(set(rows[i]['symbols'])!=set(base[i]['symbols']) for i in range(lo,hi)) for name,lo,hi in [('full122',0,122),('first88',0,88),('early53',0,53),('recent35',53,88),('later34',88,122)]} for a,rows in orders.items()}
files=[prefix+x for x in ['PLAN.json','COMPLETION.json','prepared/COMPLETION.json','prepared/FEATURES.json','prepared/DECISION_AUDIT.json','prepared/SOURCE_PARITY.json','prepared/INDEPENDENT_PREPARE.json','selected/FROZEN.json','results/RESULTS.json','results/COMPLETION.json','results/INDEPENDENT_OUTCOME.json']]+['work/monthly_portfolio_path247_context.json','work/monthly_portfolio_path247_TESTS.json','work/monthly_portfolio_path247_CODE_REVIEW.json','work/monthly_portfolio_path247_RESULT_REVIEW.json','work/after246_portfolio_definition_coverage.json','work/after246_portfolio_definition_review.json','work/after246_portfolio_source_preflight.json','work/after247_checkpoint.py']
audit=read(prefix+'prepared/DECISION_AUDIT.json')
d=dict(schema='after247_strategy_checkpoint_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='COMPLETED_INDEPENDENTLY_VERIFIED',goal_status='completion_audit_required' if closed['target_reached'] else 'active_unmet',last_goal_turn_classification='progress',
 primary_target_basis='2016-08-17 through2026-09-24 same-close ZERO-cost CAGR>=70%; execution timing and transaction fees reported separately.',selection=results['selection'],best=central['best_training_selected_candidate'],metrics={a:{p:x['metrics'] for p,x in paths.items()} for a,paths in results['arms'].items()},same_close_slices={a:paths['same_close_zero']['slices'] for a,paths in results['arms'].items()},changed_holdings_vs_incumbent=changes,decision_counts=audit['counts'],source_parity=read(prefix+'prepared/SOURCE_PARITY.json'),validation=dict(synthetic_tests=read('work/monthly_portfolio_path247_TESTS.json')['tests'],independent_paths=12,independent_slices=60,training_groups=24,all123_decision_prefixes=True,physical_prefixes=[0,1,53,88,122],protected_files=21444),completed_round_count=243,trial_count=1243,registered_queue=[],new_trials=4,new_predictive_model_fits=0,observers_modified=False,strict_pit=False,reused_sample=True,
 interpretation='Allarms use originalP Top5 andcommon core/finalregret domain. Joint versus diagonal isolates risk decomposition on the same fixed-unit formation return numerator; versus structural control also changes auxiliary funding. New252return statistics end42sessions earlier; originalstate clocks unchanged. Common missing/zerovariance fallback preserves originaldecision. Formation fixed-unit path is descriptive, not a monthly rebalanced outcome. No claims of author replication or forwardprofitability. Fullperiod cannot overridefirst88 rejection.',limitations=read('work/monthly_portfolio_path247_context.json')['limitations'],next_action='Use frozen first88 selection and independently rebuilt outcomes; any further hypothesis needs prospective registration. No observer promotion or post-outcome tuning.',evidence_sha256={f:sha(f) for f in files})
p=ROOT/'work/after247_strategy_checkpoint.json';assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(path=str(p),sha256=sha(str(p.relative_to(ROOT))),goal_status=d['goal_status'],completed=243,trials=1243)))
