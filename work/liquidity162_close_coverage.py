"""Record source feasibility evidence without adding or promoting a strategy."""
import fcntl
import json
import os
from pathlib import Path
import tempfile
import liquidity162_coverage as c

PLAN='8994fcf45cb00c570d2ac2033e7403dba2ea3aeee33714d81325d7277f135553'

def main():
    if c.src.digest(c.OUT/'PLAN.json')!=PLAN:raise ValueError('Coverage plan changed')
    plan=json.loads((c.OUT/'PLAN.json').read_text())
    for p,h in plan['hashes'].items():c.src.pinned(p,h)
    c.source_closure()
    result=json.loads((c.OUT/'RESULTS.json').read_text())
    if (result['plan_sha256']!=PLAN or not result['independent_exact_match']
        or result['recomputed_prefixes']!=123 or len(result['rows'])!=244
        or result['gate']!=c.gate(result['summary'])):
        raise ValueError('Incomplete coverage evidence')
    for key in ('strategy_trials_added','financial_ratios_computed','orders_computed','returns_computed'):
        if result[key]!=0:raise ValueError('Unexpected numerical work')
    state_path=c.src.ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    with (state_path.parent/'.lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=state_path.read_bytes();sha=c.src.digest(state_path);state=json.loads(before)
        if state['current']!='monthly_rank161' or len(state['completed'])!=161 or state['registered_queue']:
            raise ValueError('Concurrent strategy progress')
        if sum(r['trials'] for r in state['completed'])!=998:raise ValueError('Trial count changed')
        completion=dict(status='source_feasibility_completed',decision=result['decision'],
            completed_utc=c.src.now(),source_audit={'path':str(c.src.OUT/'COMPLETION.json'),
                'sha256':c.src.digest(c.src.OUT/'COMPLETION.json')},
            coverage_plan_sha256=PLAN,coverage_result_sha256=c.src.digest(c.OUT/'RESULTS.json'),
            gate=result['gate'],overlap={k:s['both_actionable'] for k,s in result['summary'].items()},
            original_early_nvda_eligible_source_months=result['summary']['early']['issuers']['NVDA']['actionable'],
            original_recent_msft_eligible_source_months=result['summary']['recent']['issuers']['MSFT']['actionable'],
            independent_rows_verified=244,recomputed_prefixes=123,strategy_trials_added=0,
            best_candidate_unchanged=state['best_training_selected_candidate'],
            previous_goal_turn_classification='progress: source audit implementation and scope were prepared',
            this_goal_turn_classification='progress: source audit and independently matched coverage completed; two-issuer rank design excluded before returns',
            remaining_goal='70% full-period CAGR has not been reached or claimed.',
            next_action='Do not tune or register this two-issuer rank factor on later34. A broader source design must be separately preregistered before extraction; alternatively investigate a distinct guidance-aligned hypothesis.',
            finalizer_sha256=c.src.digest(__file__),state_before_sha256=sha)
        c.write('COMPLETION.json',completion)
        backup=state_path.parent/('PRE_LIQUIDITY162_COVERAGE_'+sha+'.json')
        with backup.open('xb') as f:f.write(before)
        state.update(updated_utc=c.src.now(),last_goal_turn_classification='progress',
            latest_source_progress={'path':str(c.OUT/'COMPLETION.json'),
                'sha256':c.src.digest(c.OUT/'COMPLETION.json'),
                'status':'source_feasibility_completed_without_strategy_admission','strategy_trials_added':0},
            next='161 completed strategy rounds / 998 trials. Liquidity source feasibility is complete; the two-issuer rank design has no training overlap and is not admitted. Best CAGR remains67.6661737204%.',
            next_step=completion['next_action'])
        if c.src.digest(state_path)!=sha:raise ValueError('State changed before commit')
        with tempfile.NamedTemporaryFile(dir=state_path.parent,delete=False) as f:
            temp=Path(f.name);f.write((json.dumps(state,sort_keys=True,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
        try:
            if c.src.digest(state_path)!=sha:raise ValueError('Concurrent state commit')
            os.replace(temp,state_path)
        finally:temp.unlink(missing_ok=True)
        print(json.dumps({'completed':completion['status'],'decision':completion['decision'],
                          'overlap':completion['overlap'],'strategy_trials_added':0,
                          'best_cagr':state['best_training_selected_candidate']['cagr']}))

if __name__=='__main__':main()
