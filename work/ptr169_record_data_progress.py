"""Record independently verified data progress without claiming strategy success."""
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'outputs/monthly_ptr_pool_source169'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
EXPECTED='d115c1114a57fae3d064a0a70d5ea71917753a863201e33516858b905c17a2d1'
PINS={
 'TARGET_QUOTES_PLAN.json':'7f3f6b6afce4436ee2b37350df6bcd252d79e452be37a886f2216a5adb78a7ea',
 'TARGET_QUOTES.json':'05ce7a74d85f2884c77cf11f0f8dea29a029161d352710e021d7cff600957e74',
 'conditional_pool/PLAN.json':'14fa6e900df478414b2f6fccac2ee36180bb09ae63f3425b0028bac375203b1f',
 'conditional_pool/DATA_COMPARISON.json':'361904b034a24984a79dfda33748b39312c2033f973f63e4b73468c88d86964f',
 'conditional_pool/CONDITIONAL_POOLS.json':'9d6b38754e2732d3aaf31a1fd1f1c3671af4a9b6ecbc2673a44d8339b81fd289',
 'conditional_pool/INDEPENDENT_REVIEW.json':'d907eb381b6fe6dcf375ed24dc37c73a411383552d57c30d80dfd7bcf4e4d471'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def main():
    assert sha(CENTRAL)==EXPECTED,'Concurrent central change'
    state=read(CENTRAL);old=read(CENTRAL)
    for p,h in PINS.items():assert sha(P/p)==h,p
    review=read(P/'conditional_pool/INDEPENDENT_REVIEW.json')
    assert review['verdict']=='PASS_CONDITIONAL_DATA_SENSITIVITY_NOT_PIT_CERTIFICATION'
    for p,h in review['input_sha256'].items():assert sha(ROOT/p)==h,p
    data=read(P/'conditional_pool/DATA_COMPARISON.json')
    assert data['summary']['membership_changes']==33 and data['summary']['later_unchanged']==69
    assert len(state['completed'])==167 and sum(r['trials'] for r in state['completed'])==1016
    now=datetime.now(timezone.utc).isoformat()
    receipt=dict(schema='ptr169_verified_data_progress_v1',created_utc=now,
        status='CONDITIONAL_DATA_AUDIT_COMPLETE_MODEL_REBUILD_REQUIRED',artifacts=PINS,
        summary=data['summary'],strategy_rounds=167,strategy_search_trials=1016,
        new_strategy_trials=0,new_strategy_returns_computed=False,observers_modified=False,
        best_not_recomputed=state['best_training_selected_candidate'],target_reached=False,
        limitations=data['assumptions'],
        workflow_diagnostics=[
          'Initial missing-quote extraction syntax error stopped before output; corrected before registration.',
          'Five rather than six overlapping Yahoo comparison rows: April30 also missing; bounded validation corrected before final audit.',
          'Original Fed bulk audit passed48/53; immutable reconciliation retained5 index/page publication-date differences and selected Feb19 fallback for Feb2019.',
          'Yahoo January7 2019 OHLC issue outside target missing-quote window retained; it is not repaired or admitted by this work.'],
        next_step='Fixed original-control rebuild of event/weekly/monthly ranks, mature labels, forecasts and episode scores. Require original parity before conditional replay; retain all old inputs.',
        prior_central_sha256=EXPECTED,script_sha256=sha(Path(__file__)))
    write(P/'DATA_PROGRESS.json',receipt)
    ref=dict(path=str((P/'DATA_PROGRESS.json').relative_to(ROOT)),sha256=sha(P/'DATA_PROGRESS.json'),
        strategy_trials=0,status=receipt['status'],full_point_in_time_dataset_complete=False)
    state['data_evidence_completions'].append(ref);state['latest_pool_data_progress']=ref
    state['next']='PTR A/H conditional data audit complete:33/53 early membership changes;69 later pools unchanged. No new strategy return yet.'
    state['next_step']=receipt['next_step'];state['last_goal_turn_classification']='progress';state['updated_utc']=now
    with (P/'CENTRAL_BEFORE_DATA_PROGRESS.json').open('xb') as f:f.write(CENTRAL.read_bytes())
    temp=CENTRAL.with_name('CONTINUATION_STATUS.ptr169.tmp');write(temp,state)
    assert sha(CENTRAL)==EXPECTED
    os.replace(temp,CENTRAL)
    after=read(CENTRAL)
    for k in ['completed','best_training_selected_candidate','registered_queue','current','current_status']:
        assert after[k]==old[k],k
    write(P/'CENTRAL_DATA_UPDATE.json',dict(before=EXPECTED,after=sha(CENTRAL),receipt_sha256=ref['sha256'],strategy_registry_preserved=True))
    print(json.dumps(dict(status=receipt['status'],receipt_sha256=ref['sha256'])))
if __name__=='__main__':main()
