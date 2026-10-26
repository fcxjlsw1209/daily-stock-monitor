"""Update resumable task metadata only after verified round143 closure."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os
R=Path(__file__).resolve().parents[1]
P=R/'outputs/monthly_paired_target143'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),sha256=sha(p))
central=R/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
central_hash=sha(central);state=json.loads(central.read_text())
completion=json.loads((P/'COMPLETION.json').read_text())
assert completion['status']=='completed_independently_verified'
assert state['current']=='monthly_paired_target143' and not state['registered_queue']
assert len(state['completed'])==143 and sum(r['trials'] for r in state['completed'])==929
assert state['latest_round_receipt']==ref(P/'COMPLETION.json')
assert completion['verification']['protected_files']==21444
p=R/'migration/RESEARCH_RESUME.json';before=p.read_bytes();d=json.loads(before)
now=datetime.now(timezone.utc);checkpoint=R/'migration'/('round143_checkpoint_'+now.strftime('%Y%m%dT%H%M%S%fZ'))
checkpoint.mkdir();(checkpoint/'RESEARCH_RESUME.before.json').write_bytes(before)
files=('PLAN','CACHE','FROZEN','TRAIN_TRIALS','TRAIN_OPEN_TRIALS','RESULTS','INDEPENDENT_PREPARE','INDEPENDENT_OUTCOME','PROTECTED_AFTER','COMPLETION')
d.update(updated_utc=now.isoformat(),round=143,current_experiment='monthly_paired_target143',completed_research_rounds=143,strategy_evaluations=929,
 current_experiment_artifacts={x:ref(P/(x+'.json')) for x in files},formal_round143_completed=True,formal_round143_completion=ref(P/'COMPLETION.json'),
 frozen_primary=completion['primary'],plan_sha256=sha(P/'PLAN.json'),target_70_reached=completion['target_reached'],
 full_protection_check_passed=True,latest_protected_check=ref(P/'PROTECTED_AFTER.json'),
 previous_goal_turn_classification=state['last_goal_turn_classification'],next_mechanism_registered=False,
 next_research_status='round143_complete_next_distinct_hypothesis_pending',next_step=state['next_step'],
 next_mechanism_deduplication=ref(R/'work/after143_stability_novelty.json'),
 do_not_rerun_completed_phases=True,resume_update_checkpoint=str(checkpoint.relative_to(R)),
 completed_model_preparation=dict(actual_pooled_model_fits=completion['new_model_fits'],independent_review=ref(P/'INDEPENDENT_PREPARE.json')))
d.pop('pending_model_preparation',None)
after=(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
assert p.read_bytes()==before and sha(central)==central_hash
q=p.with_name('.RESEARCH_RESUME.round143.tmp')
with q.open('xb') as f:f.write(after);f.flush();os.fsync(f.fileno())
os.replace(q,p)
(checkpoint/'RECEIPT.json').write_text(json.dumps(dict(updated=ref(p),central_unchanged_sha256=central_hash,completion=ref(P/'COMPLETION.json'),observer_modified=False),indent=2)+'\n')
print(json.dumps(dict(checkpoint=str(checkpoint),completed_rounds=143,trials=929,target_reached=completion['target_reached'])))
