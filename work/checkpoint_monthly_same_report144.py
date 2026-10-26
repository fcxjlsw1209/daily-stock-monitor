"""Update resumable task metadata only after verified round144 closure."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os
R=Path(__file__).resolve().parents[1]
P=R/'outputs/monthly_same_report144'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),sha256=sha(p))
central=R/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
central_hash=sha(central);state=json.loads(central.read_text())
completion=json.loads((P/'COMPLETION.json').read_text())
assert completion['status']=='completed_independently_verified'
assert state['current']=='monthly_same_report144' and not state['registered_queue']
assert len(state['completed'])==144 and sum(r['trials'] for r in state['completed'])==932
assert state['latest_round_receipt']==ref(P/'COMPLETION.json')
assert completion['verification']['protected_files']==21444
p=R/'migration/RESEARCH_RESUME.json';before=p.read_bytes();d=json.loads(before)
now=datetime.now(timezone.utc);checkpoint=R/'migration'/('round144_checkpoint_'+now.strftime('%Y%m%dT%H%M%S%fZ'))
checkpoint.mkdir();(checkpoint/'RESEARCH_RESUME.before.json').write_bytes(before)
files=('PLAN','CACHE','FROZEN','TRAIN_TRIALS','TRAIN_OPEN_TRIALS','RESULTS','INDEPENDENT_PREPARE','INDEPENDENT_OUTCOME','PROTECTED_AFTER','COMPLETION')
d.update(updated_utc=now.isoformat(),round=144,current_experiment='monthly_same_report144',completed_research_rounds=144,strategy_evaluations=932,
 current_experiment_artifacts={x:ref(P/(x+'.json')) for x in files},formal_round144_completed=True,formal_round144_completion=ref(P/'COMPLETION.json'),
 frozen_primary=completion['primary'],plan_sha256=sha(P/'PLAN.json'),target_70_reached=completion['target_reached'],
 full_protection_check_passed=True,latest_protected_check=ref(P/'PROTECTED_AFTER.json'),
 previous_goal_turn_classification=state['last_goal_turn_classification'],next_mechanism_registered=False,
 next_research_status='round144_complete_next_distinct_hypothesis_pending',next_step=state['next_step'],
 latest_source_admission=ref(R/'outputs/monthly_same_report144_admission/AUDIT.json'),
 do_not_rerun_completed_phases=True,resume_update_checkpoint=str(checkpoint.relative_to(R)),
 completed_same_report_preparation=dict(new_model_fits=completion['new_model_fits'],independent_review=ref(P/'INDEPENDENT_PREPARE.json')))
d.pop('pending_model_preparation',None)
source_dir=R/'outputs/monthly_msft_nvda_same_report_preflight_v1'
source_audit=json.loads((source_dir/'AUDIT.json').read_text())
assert source_audit['passed'] and source_audit['counts']['extracted_monetary_cells']==96
assert source_audit['counts']['fact_admissions']==0
assert source_audit['data_sha256']==sha(source_dir/'EVENTS.json')
d['next_source_preflight']={
 'directory':str(source_dir.relative_to(R)),
 'artifacts':{x:ref(source_dir/(x+'.json')) for x in ('PLAN','SCHEMA','EVENTS','AUDIT','MANIFEST','TESTS')},
 'feasibility':ref(R/'work/after144_same_report_coverage_feasibility.json'),
 'status':'source_only_24_events_96_cells_extracted_independent_review_required',
 'event_holds':6,'fact_admissions':0,'new_factor_calculations':0,
 'note':'New MSFT/NVDA same-current-report columns, not old141 separate-original-report factors. Never restart round144 or promote unreviewed data.'}
d['next_research_status']='round144_complete_msft_nvda_same_report_source_review_in_progress'
d['next_step']='Finish independent source-cell/accounting review and conditional admission of MSFT/NVDA same-current-report comparatives, then prospectively register a distinct coverage expansion before any new factor or return computation.'
union_dir=R/'outputs/monthly_same_report145_union'
union= json.loads((union_dir/'AUDIT.json').read_text())
assert union['passed'] and union['counts']=={'events':44,'conditionally_admitted':38,'rejected':6,'raw_values':176}
assert union['data_sha256']==sha(union_dir/'DATA.json')
d['next_source_preflight']['status']='independently_reviewed_then_conditionally_admitted_and_unioned'
d['next_source_preflight']['independent_review']=ref(R/'work/after144_msft_nvda_source_review.json')
d['next_source_admission']=ref(R/'outputs/monthly_msft_nvda_same_report_admission_v1/AUDIT.json')
d['next_source_union']={'artifacts':{x:ref(union_dir/(x+'.json')) for x in ('PLAN','DATA','AUDIT','MANIFEST','TESTS')},'counts':union['counts'],'new_factor_calculations':0}
d['next_research_status']='round144_complete_round145_source_union_ready_prospective_registration_pending'
d['next_step']='Freeze final145 proposal, finish independent145 implementation reviews and relevant tests, register exact source/reference versions, then execute145 prepare, first88 selection and later34 diagnostic. Never rerun completed144 phases.'
after=(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
assert p.read_bytes()==before and sha(central)==central_hash
q=p.with_name('.RESEARCH_RESUME.round144.tmp')
with q.open('xb') as f:f.write(after);f.flush();os.fsync(f.fileno())
os.replace(q,p)
(checkpoint/'RECEIPT.json').write_text(json.dumps(dict(updated=ref(p),central_unchanged_sha256=central_hash,completion=ref(P/'COMPLETION.json'),observer_modified=False),indent=2)+'\n')
print(json.dumps(dict(checkpoint=str(checkpoint),completed_rounds=144,trials=932,target_reached=completion['target_reached'])))
