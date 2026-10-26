"""Update resumable research metadata only after independently verified closure."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
R=Path(__file__).resolve().parents[1];P=R/'outputs/monthly_same_report145'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path':str(p.relative_to(R)),'sha256':sha(p)}
central=R/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
central_hash=sha(central);state=json.loads(central.read_text())
c=json.loads((P/'COMPLETION.json').read_text())
assert c['status']=='completed_independently_verified'
assert state['current']=='monthly_same_report145' and not state['registered_queue']
assert len(state['completed'])==145 and sum(x['trials'] for x in state['completed'])==935
assert state['latest_round_receipt']==ref(P/'COMPLETION.json')
assert c['verification']['protected_files']==21444
p=R/'migration/RESEARCH_RESUME.json';before=p.read_bytes();d=json.loads(before)
now=datetime.now(timezone.utc);checkpoint=R/'migration'/('round145_checkpoint_'+now.strftime('%Y%m%dT%H%M%S%fZ'))
checkpoint.mkdir();(checkpoint/'RESEARCH_RESUME.before.json').write_bytes(before)
files=('PLAN','CACHE','FROZEN','TRAIN_TRIALS','TRAIN_OPEN_TRIALS','TRAIN_REFERENCE_OPEN','RESULTS','INDEPENDENT_PREPARE','INDEPENDENT_OUTCOME','PROTECTED_AFTER','COMPLETION')
d.update(updated_utc=now.isoformat(),round=145,current_experiment='monthly_same_report145',completed_research_rounds=145,strategy_evaluations=935,
 current_experiment_artifacts={x:ref(P/(x+'.json')) for x in files},formal_round145_completed=True,formal_round145_completion=ref(P/'COMPLETION.json'),
 frozen_primary=c['primary'],plan_sha256=sha(P/'PLAN.json'),target_70_reached=c['target_reached'],
 full_protection_check_passed=True,latest_protected_check=ref(P/'PROTECTED_AFTER.json'),
 previous_goal_turn_classification=state['last_goal_turn_classification'],next_mechanism_registered=False,
 next_research_status='round145_complete_next_distinct_hypothesis_pending',next_step=state['next_step'],
 latest_source_admission=ref(R/'outputs/monthly_same_report145_union/AUDIT.json'),do_not_rerun_completed_phases=True,
 resume_update_checkpoint=str(checkpoint.relative_to(R)),
 completed_same_report_preparation={'new_model_fits':c['new_model_fits'],'independent_review':ref(P/'INDEPENDENT_PREPARE.json')})
d.pop('pending_model_preparation',None)
for key in ('next_source_preflight','next_source_admission','next_source_union'): d.pop(key,None)
d['completed_round145_sources']={'union':ref(R/'outputs/monthly_same_report145_union/AUDIT.json'),'independent_union':ref(R/'work/after144_union145_independent_review.json')}
nextnote=R/'work/after145_scientific_next_step.json'
if nextnote.exists():d['unregistered_next_step_review']=ref(nextnote)
after=(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
assert p.read_bytes()==before and sha(central)==central_hash
q=p.with_name('.RESEARCH_RESUME.round145.tmp')
with q.open('xb') as f:f.write(after);f.flush();os.fsync(f.fileno())
os.replace(q,p)
(checkpoint/'RECEIPT.json').write_text(json.dumps({'updated':ref(p),'central_unchanged_sha256':central_hash,'completion':ref(P/'COMPLETION.json'),'observer_modified':False},indent=2)+'\n')
print(json.dumps({'checkpoint':str(checkpoint),'completed_rounds':145,'trials':935,'target_reached':c['target_reached']}))
