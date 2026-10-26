"""Save completed244 audit evidence; no strategy replay or parameter search."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
prefix='outputs/monthly_up_participation244/'
central=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
closed=read(prefix+'COMPLETION.json');results=read(prefix+'results/RESULTS.json')
review=read('work/monthly_up_participation244_RESULT_REVIEW.json')
assert closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and not closed['observers_modified']
assert central['current']=='monthly_up_participation244' and not central['registered_queue']
assert len(central['completed'])==240 and sum(x['trials'] for x in central['completed'])==1233
assert review['passed'] is True
orders=read(prefix+'prepared/ORDERS.json');base=orders['incumbent_local67']
changes={a:{name:sum(set(rows[i]['symbols'])!=set(base[i]['symbols']) for i in range(lo,hi)) for name,lo,hi in [('full122',0,122),('first88',0,88),('early53',0,53),('recent35',53,88),('later34',88,122)]} for a,rows in orders.items()}
files=[prefix+x for x in ['PLAN.json','COMPLETION.json','prepared/COMPLETION.json','prepared/DECISION_AUDIT.json','prepared/FEATURE_AUDIT.json','prepared/SOURCE_PARITY.json','prepared/INDEPENDENT_PREPARE.json','selected/FROZEN.json','results/RESULTS.json','results/COMPLETION.json','results/INDEPENDENT_OUTCOME.json']]+['work/monthly_up_participation244_context.json','work/monthly_up_participation244_TESTS.json','work/monthly_up_participation244_CODE_REVIEW.json','work/monthly_up_participation244_RESULT_REVIEW.json','work/after243_up_participation_coverage.json','work/after243_participation_definition_review.json','work/up_participation244_SOURCE_REVIEW.json','work/after244_checkpoint.py']
d=dict(schema='after244_strategy_checkpoint_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='COMPLETED_INDEPENDENTLY_VERIFIED',goal_status='completion_audit_required' if closed['target_reached'] else 'active_unmet',last_goal_turn_classification='progress',
 primary_target_basis='2016-08-17 through2026-09-24 same-close ZERO-cost CAGR>=70%; timing and cost sensitivity separate.',selection=results['selection'],best=central['best_training_selected_candidate'],metrics={a:{p:x['metrics'] for p,x in paths.items()} for a,paths in results['arms'].items()},same_close_slices={a:paths['same_close_zero']['slices'] for a,paths in results['arms'].items()},changed_holdings_vs_incumbent=changes,feature_counts=read(prefix+'prepared/FEATURE_AUDIT.json')['counts'],decision_counts=read(prefix+'prepared/DECISION_AUDIT.json')['counts'],validation=dict(synthetic_tests=read('work/monthly_up_participation244_TESTS.json')['tests'],independent_paths=9,independent_slices=45,all123_decision_prefixes=True,physical_prefixes=[0,1,53,88,122],protected_files=21444),completed_round_count=240,trial_count=1233,artifact_writer_repair=dict(initial_exit_code=1,reason='Checkpoint-only two-argument read typo corrected before any checkpoint file was written; no strategy phases rerun or frozen artifacts changed.'),registered_queue=[],new_trials=3,new_model_fits=0,observers_modified=False,strict_pit=False,reused_sample=True,
 interpretation='Evaluate only the registered SPY-up conditional relative-win correction against the same-support all-day control and original. Direct winning frequency discards magnitude and can favor high beta; the control does not remove this confounding. No forward performance claim. Full-period metrics may not override the first88 frozen rejection.',next_action='Use frozen selection and independent outcome evidence; no retrospective parameter fitting or observer promotion. New hypotheses require bounded coverage and prospectively fixed controls. Source/pool/PIT limitations remain explicit.',evidence_sha256={f:sha(f) for f in files})
p=ROOT/'work/after244_strategy_checkpoint.json';assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),goal_status=d['goal_status'],completed=240,trials=1233)))
