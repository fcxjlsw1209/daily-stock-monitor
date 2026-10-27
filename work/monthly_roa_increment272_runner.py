"""272 PLAN-first exact-three source increment, frozen154 control and original episode.

Only the caller's physically shortened original27 arrays reach original pre-score
calculation. Historical154 completion/cache bytes are anchors, not a claim that
obsolete PDF runtimes have been re-certified. No data is loaded at import time.
"""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from copy import deepcopy
import argparse, hashlib, io, json, unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import monthly_forecast_spacing269_host as prior
import monthly_stock_comp149_runner as legacy
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_roa_increment272'
PRIOR=ROOT/'outputs/monthly_joint_growth271'
LEGACY=ROOT/'outputs/monthly_financial_coverage154'
INCREMENT=ROOT/'outputs/monthly_roa_field_scope270_increment_v1'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
REBUILD=prior.REBUILD
CONTEXT=ROOT/'work/monthly_roa_increment272_context.json'
CONTRACT=ROOT/'work/monthly_roa_increment272_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_roa_increment272_TESTS.json'
REVIEW=ROOT/'work/monthly_roa_increment272_CODE_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_joint_growth271_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_roa_increment272_RESULT_REVIEW.json'
PHASE_TERMINALS=OUT/'PHASE_TERMINALS.json'
ARMS=('incumbent_local67','final154_expanded_earnings_control','exact3_roa_increment_candidate')
PREFIXES=(0,1,53,88,122)
PRIOR_COMPLETION_SHA='64fb4bda4307f20f2d98cfc1dd0500646a601ce9245671fcf3c5e700a0c16bd0'
LEGACY_PLAN_SHA='87f65aa25e1c97216bbaddf9dcaabf158c10b38a382ac27cea17dfd7bf9a86df'
LEGACY_CACHE_SHA='6b6f7abfc0a41af46cf2726b4b70985487af8eda156231e40def505026d72418'
LEGACY_COMPLETION_SHA='cbc61cf6bd4501dedb3142be5c672a1b2f196ade55f42f9a3b14df66190fb749'
NUMERICAL_VERIFICATION=dict(score_tolerance=0.,selection_tolerance=0.,original_core_preserved=True,new_model_fits=0,financial_dose=.01,hand_absolute_limit=.5)
code_names=tuple(name for stem in ('runner','factor','independent') for name in ('monthly_roa_increment272_'+stem+'.py','test_monthly_roa_increment272_'+stem+'.py'))
SUPPLEMENTAL=('work/monthly_roa_increment272_saved_reader.py','work/test_monthly_roa_increment272_saved_reader.py')
PREP_FILES={'SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json','DECISION_AUDIT.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json','SOURCE_AUDIT.json','FINANCIAL_INPUTS.npz','OVERLAY_AUDIT.json'}
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}
PRE_WORK_FIELDS=('pre_score_verifications','cells_compared','eligible_delta_cells','hand_bound_cells','new_model_fits')
PRODUCER_FIELDS=('original_pre_score_calls','original_selector_calls','financial_correction_calls','overlay_calls','selector_audit_rank_calls','new_model_fits','new_price_windows_read')
INTEGRITY_FIELDS=('checks','cache_values_compared','reference_values_compared','reference_order_events_compared','metadata_events_compared','financial_values_compared')
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
integer,counts_check,sum_counts=prior.integer,prior.counts_check,prior.sum_counts
exact_array,structured_digest=prior.exact_array,prior.structured_digest
train_metrics=prior.train_metrics
source_closure_check,bind_source_closure=prior.source_closure_check,prior.bind_source_closure
original_selector_source=prior.original_selector_source


def now():return datetime.now(timezone.utc).isoformat()

def files():return [ROOT/'work'/p for p in code_names]

def codes():return {str(p):sha(p) for p in files()}

def code_review():
    r=read(REVIEW)
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['code_sha256']==codes(),'Numerical code review')
    require(r['context_sha256']==sha(CONTEXT) and r['implementation_contract_sha256']==sha(CONTRACT) and r['actual_returns_computed'] is False and type(r['input_sha256']) is dict,'Prospective review scope')
    return r

def select(expected):
    meta,orders=prepared(expected);require(not (OUT/'selected').exists(),'Selection write-once');px,op=ev.prices();engine=ev.engine();metrics={}
    for arm in ARMS:
        metrics[arm]={}
        for path,mode,cost in ev.PATHS[:1]:
            _,eq,_,_=engine.replay(px,op,orders[arm][:88],cost=cost,execution=mode,start=pd.Timestamp(meta['dates'][0]),end=pd.Timestamp(meta['ends'][87]))
            require(str(eq.index[-1].date())==meta['ends'][87],'Training cutoff');metrics[arm][path]=train_metrics(engine,eq,meta)
    pick=select_first88(metrics,orders);gate(expected);dest=OUT/'selected';dest.mkdir();dump(dest/'TRAIN_RESULTS.json',metrics)
    dump(dest/'FROZEN.json',dict(created_utc=now(),plan_sha256=expected,prepared_sha256=sha(OUT/'prepared/COMPLETION.json'),orders_sha256=sha(OUT/'prepared/ORDERS.json'),training_sha256=sha(dest/'TRAIN_RESULTS.json'),selection=pick))
    print(json.dumps(pick),flush=True)

def frozen(expected,orders):
    f=read(OUT/'selected/FROZEN.json');require(f['plan_sha256']==expected and f['prepared_sha256']==sha(OUT/'prepared/COMPLETION.json') and f['orders_sha256']==sha(OUT/'prepared/ORDERS.json') and f['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json'),'Frozen selection binding')
    require(f['selection']==select_first88(read(OUT/'selected/TRAIN_RESULTS.json'),orders),'Frozen choice differs');return f

def outcome_check(check):
    require(check.get('passed') is True and check.get('checked_paths')==9 and check.get('checked_slices')==45 and check.get('checked_training_groups')==18 and check.get('independent_accounts') is True and check.get('independent_of_production_model_and_engine') is True and check.get('terminal_liquidation') is False,'9paths45slices independent accounts')
    require(set(check['paths'])==set(ARMS),'Outcome arms')
    for paths in check['paths'].values():
        require(set(paths)=={p[0] for p in ev.PATHS},'Outcome paths')
        for p in paths.values():require(p['passed'] is True and p['monthly_executions']==122 and set(p['slices'])=={'full122','first88','early53','recent35','later34'},'Account slices')

def phase_terminals_check(expected):
    terminals=read(PHASE_TERMINALS)
    require(terminals.get('plan_sha256')==expected and set(terminals.get('phases',{}))=={'prepare','select','diagnose'},'Owned terminal receipt scope')
    for name,relative in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:
        item=terminals['phases'][name]
        require(type(item) is dict and set(item) in ({'session_id','exit_code','chunk_id','artifact_path','artifact_sha256'},{'session_id','exit_code','chunk_id','artifact_path','artifact_sha256','owner'}),'Complete phase terminal '+name)
        require('owner' not in item or (type(item['owner']) is str and item['owner']=='/root'),'Optional actual terminal owner')
        require(item['session_id'] is None or (type(item['session_id']) is int and item['session_id']>0),'Terminal session identity')
        integer(item['exit_code'],'Terminal exit',0)
        require(type(item['chunk_id']) is str and bool(item['chunk_id'].strip()),'Terminal chunk identity')
        path=Path(item['artifact_path']) if Path(item['artifact_path']).is_absolute() else ROOT/item['artifact_path']
        require(path==OUT/relative and item['artifact_sha256']==sha(OUT/relative),'Terminal artifact binding '+name)
    return terminals

def zero_cost_check(exports,results):
    for arm in ARMS:
        for path in ('same_close_zero','next_open_zero'):
            data=exports[arm][path];r=results[arm][path]
            for summary in (r['metrics'],r['slices']['full122']):require(type(summary['cost_per_side']) in (int,float) and summary['cost_per_side']==0.,'ZERO configured cost exact')
            for row in data['TRADES']:require(type(row['cost']) in (int,float) and row['cost']==0.,'ZERO trade cost exact')
            for row in data['HOLDINGS']:require(type(row['fee']) in (int,float) and row['fee']==0.,'ZERO period fee exact')
            for value in (r['metrics']['fees_paid'],r['slices']['full122']['fees_paid']):require(type(value) in (int,float) and value==0.,'ZERO aggregate fee exact')

def digest_string(value,label):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),label+' final SHA256 required')
    return value

def prepare_recorded(expected):
    try:
        return prepare(expected)
    except Exception as exc:
        # Do not turn a data/control failure into a auxiliary availability fallback.
        failure=OUT/'PREPARE_FAILURE.json'
        plan=OUT/'PLAN.json'
        if plan.is_file() and sha(plan)==expected and not failure.exists():
            dump(failure,dict(status='FAILED',phase='prepare',created_utc=now(),plan_sha256=expected,
                error_type=type(exc).__name__,error=str(exc),source_or_control_failure_not_relaxed=True,
                prepared_completion_present=(OUT/'prepared/COMPLETION.json').exists(),
                candidate_execution_scope='Not certified on failure; inspect error and existing phase artifacts.',
                successful_preparation_claim=False,observers_modified=False))
        raise

def canonical_pins(pins):
    require(type(pins) is dict,'Pin map');out={}
    for name,digest in pins.items():
        require(type(name) is str and name and '..' not in Path(name).parts,'Canonical pin path')
        p=Path(name);p=p if p.is_absolute() else ROOT/p;key=str(p.resolve())
        require(p.resolve()!=CENTRAL.resolve(),'Mutable central pin forbidden');digest_string(digest,'Pin digest')
        require(key not in out or out[key]==digest,'Conflicting canonical pin alias')
        out[key]=digest
    old.verify(out);return out

def train_check(training,audit):
    helper=accounting();require(set(training)==set(ARMS),'Training roles')
    for arm in ARMS:
        require(set(training[arm])=={'same_close_zero'},'Only same-close ZERO selection path')
        for path,groups in training[arm].items():
            require(set(groups)=={'training','early','recent'},'Training segment scope')
            for short,full in [('training','first88'),('early','early53'),('recent','recent35')]:helper.check_metrics(audit['paths'][arm][path]['slices'][full],groups[short],arm+'/'+path+'/'+short)

def diagnose(expected):
    meta,orders=prepared(expected);f=frozen(expected,orders);require(not (OUT/'results').exists(),'Diagnostics write-once')
    px,op=ev.prices();engine=ev.engine();dest=OUT/'results';dest.mkdir();results={};exports={}
    for arm in ARMS:
        results[arm]={};exports[arm]={}
        for path,mode,cost in ev.PATHS:
            r,eq,trades,holds,pos=ev.run_path(engine,px,op,orders[arm],meta,mode,cost);r['slices']=dict(full122=r['metrics'],**ev.split_stats(engine,eq,meta));results[arm][path]=r
            data=dict(DAILY=[dict(date=str(d.date()),equity=float(v)) for d,v in eq.items()],TRADES=trades,HOLDINGS=holds,DAILY_POSITIONS=pos);exports[arm][path]=data
            for kind,value in data.items():dump(dest/(arm+'_'+path+'_'+kind+'.json'),value)
    zero_cost_check(exports,results)
    check=audit_outcomes(px,op,orders,exports,results,meta);outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    dump(dest/'INDEPENDENT_OUTCOME.json',check);dump(dest/'RESULTS.json',dict(selection=f['selection'],arms=results,reused_sample=True,strict_pit=False,zero_cost_selection=True,next_open_zero_diagnostic_only=True,next_open_cost_diagnostic_only=True,observer_promotion=False))
    gate(expected);dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,frozen_sha256=sha(OUT/'selected/FROZEN.json'),paths=9,slices=45,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE',full_cagr={a:results[a]['same_close_zero']['metrics']['cagr'] for a in ARMS},completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def result_review_check(expected):
    phase_terminals_check(expected)
    review=read(RESULT_REVIEW)
    require(review['passed'] is True and review['unresolved_blockers']==[],'Saved evidence review passed')
    for key,path in [('plan_sha256',OUT/'PLAN.json'),('context_sha256',CONTEXT),('prepared_completion_sha256',OUT/'prepared/COMPLETION.json'),('results_completion_sha256',OUT/'results/COMPLETION.json')]:
        require(review[key]==sha(path),'Saved review binding:'+key)
    require(review['plan_sha256']==expected,'Saved review explicit PLAN')
    pins=canonical_pins(review['input_sha256'])
    required=[PHASE_TERMINALS,OUT/'PLAN.json',CONTEXT,CONTRACT,OUT/'prepared/COMPLETION.json',OUT/'selected/FROZEN.json',OUT/'selected/TRAIN_RESULTS.json',OUT/'results/COMPLETION.json']
    required += [OUT/'prepared'/name for name in PREP_FILES]
    required += [OUT/'results'/name for name in RESULT_FILES]
    for path in required:require(pins.get(str(path))==sha(path),'Saved review coverage:'+str(path))
    old.verify(pins)
    require(review.get('implementation_contract_sha256')==sha(CONTRACT),'Saved review implementation contract')
    return review

def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('roa272_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in mods]);stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    require(result.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=result.testsRun,runtime=ev.runtime(),code_sha256=codes()))

def initial_state(state):
    require(len(state['completed'])==266 and sum(x['trials'] for x in state['completed'])==1309,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Immediate closed271 predecessor')

def predecessor_binding(state):
    context,_=contract_check();wanted=context['predecessor'];record=state['completed'][-1]['completion']
    require(Path(record['path'])==PRIOR/'COMPLETION.json' and record['sha256']==sha(PRIOR/'COMPLETION.json')==wanted['completion_sha256']==PRIOR_COMPLETION_SHA,'Predecessor central closure binding')
    done=read(PRIOR/'COMPLETION.json');review=read(PRIOR_REVIEW)
    require(done['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and done['protected_files']==21444 and done['plan_sha256']==sha(PRIOR/'PLAN.json') and done['observers_modified'] is False,'Independently closed271 required')
    require(done['result_completion_sha256']==sha(PRIOR/'results/COMPLETION.json') and done['result_review_sha256']==wanted['result_review_sha256']==sha(PRIOR_REVIEW),'Predecessor result/review binding')
    require(review['passed'] is True and review['unresolved_blockers']==[] and review['plan_sha256']==sha(PRIOR/'PLAN.json') and review['results_completion_sha256']==sha(PRIOR/'results/COMPLETION.json'),'Predecessor saved review')
    require(done['target_reached'] is False and state['best_training_selected_candidate']['cagr']<.7,'Target already reached')

def registration_check(state,expected):
    initial_state(state)
    require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration absent')

def contract_check():
    require(CONTEXT.is_file() and CONTRACT.is_file(),'Final prospective papers required; drafts cannot register')
    c,d=read(CONTEXT),read(CONTRACT)
    for record in (c,d):
        require(record.get('frozen') is True and record.get('registration_authorized') is True and record.get('actual_data_started') is False,'Final prospective272 authorization')
        require(type(record.get('input_sha256')) is dict,'Explicit dependency map')
    require(c['arms']==list(ARMS) and c['registered_trials']==3 and c['only_promotable']==ARMS[2] and c['diagnostic_only']==[ARMS[1]],'Three fixed economic roles')
    require(type(c.get('actual_models_fitted')) is int and c['actual_models_fitted']==0 and c.get('actual_orders_computed') is False and c.get('actual_returns_computed') is False,'No actual272 work before registration')
    require(type(d.get('code_files')) is list and d['code_files']==['work/'+name for name in code_names],'Exact ordered formal6 inventory')
    require(type(d.get('supplemental_input_files')) is list and d['supplemental_input_files']==list(SUPPLEMENTAL),'Exact ordered reader/test inventory')
    for label,record in (('context',c),('contract',d)):
        for relative in ['work/'+name for name in code_names]+list(SUPPLEMENTAL):
            value=record['input_sha256'].get(relative);digest_string(value,label+' source pin '+relative)
            require((ROOT/relative).is_file() and value==sha(ROOT/relative),label+' current source binding '+relative)
    require(d['context_sha256']==sha(CONTEXT),'Contract context binding')
    require(c['predecessor']['directory']==PRIOR.name and c['predecessor']['completion_sha256']==PRIOR_COMPLETION_SHA,'Final271 predecessor')
    digest_string(c['predecessor'].get('result_review_sha256'),'Predecessor review')
    return c,d

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN')
    p=read(OUT/'PLAN.json');c,d=contract_check()
    require(p['context']==c and p['context_sha256']==sha(CONTEXT) and p['implementation_contract']==d and p['implementation_contract_sha256']==sha(CONTRACT) and p['runtime']==ev.runtime(),'Prospective runtime/papers unchanged')
    require(p['numerical_verification']==NUMERICAL_VERIFICATION,'Frozen numerical rule')
    old.verify(p['input_sha256']);require(p['original169_source_closure']==source_closure_check(p['input_sha256']),'Complete169 closure')
    code_review();require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Prestate bytes changed')
    state=read(CENTRAL);registration_check(state,expected);predecessor_binding(state);return p

def accounting():return prior.accounting()

def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Three272 accounting roles')
    remap=lambda value:{a:value[b] for a,b in zip(prior.ARMS,ARMS)}
    proof=prior.audit_outcomes(px,opening,remap(orders),remap(exports),remap(results),meta)
    return dict(proof,paths={b:proof['paths'][a] for a,b in zip(prior.ARMS,ARMS)})

def close(expected):
    meta,orders=prepared(expected);f=frozen(expected,orders);c=read(OUT/'results/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['frozen_sha256']==sha(OUT/'selected/FROZEN.json') and c['paths']==9 and c['slices']==45 and set(c['artifacts_sha256'])==RESULT_FILES,'Complete results')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'results'/name)==h,'Changed result')
    check=read(OUT/'results/INDEPENDENT_OUTCOME.json');outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    result=read(OUT/'results/RESULTS.json');require(result['selection']==f['selection'],'Result selection');result_review_check(expected)
    protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive')
    before=CENTRAL.read_bytes();state=json.loads(before);registration_check(state,expected)
    pick=result['selection']['primary'];arm=result['arms'][pick];m=arm['same_close_zero']['metrics'];best=state['best_training_selected_candidate']
    if result['selection']['admitted'] and m['cagr']>best['cagr']:
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=dict(strict_pit=False,reused_sample=True,global_novelty='UNKNOWN',conditional_three_source_missing_cell_replication_only=True))
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),phase_terminals_sha256=sha(PHASE_TERMINALS),new_trials=3,new_model_fits=0,independent_prepare_sha256=sha(OUT/'prepared/INDEPENDENT_PREPARE.json'),protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==267 and sum(x['trials'] for x in state['completed'])==1312,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review the fixed source-coverage increment; first88 choice immutable and frozen observers unchanged.')
    old.commit_state(before,state,'roa_increment272_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

def selector_inputs(study,P,a,meta,n):
    # This is the frozen adapter's shape normalization, not new source values.
    cw=study.adapter._core_weight(a['core_weight'],a['baseline'].shape,n).copy()
    wanted=np.broadcast_to(np.where(a['state_calm10'][:n],.5,.52)[:,None],P.shape)
    exact_array(cw,wanted,'Original .50/.52 allocation')
    return (P,a['eligible'][:n].copy(),deepcopy(meta['symbols'][:n]),a['effects'][:n].copy(),
            a['baseline'][:n].copy(),a['core'][:n].copy(),cw,
            a['state_calm10'][:n].copy(),a['state_weak_narrow'][:n].copy(),list(meta['dates'][:n]))

def selector_events(inputs,choices,proof):
    """Save actual selector domains for prefix work/ordered-choice verification."""
    P,E,symbols=inputs[:3];dates=inputs[-1];events=[]
    for i in range(len(P)):
        eligible=np.flatnonzero(E[i]).tolist()
        order=sorted(eligible,key=lambda j:(-float(P[i,j]),symbols[i][j]))
        boundary=[] if len(order)<3 else [j for j in order[2:5] if float(P[i,order[2]])-float(P[i,j])<=.02]
        events.append(dict(query_index=i,date=dates[i],eligible_indices=eligible,original_rank=order,boundary_indices=boundary,selected_indices=[int(j) for j in choices[i]]))
    require(selector_event_work(events)==proof['work'],'Actual selector event work equals independent receipt')
    return events

def selector_event_work(events):
    import monthly_selector_scope268_independent as si
    work=dict.fromkeys(si.WORK_FIELDS,0);work['selector_calls']=1
    for i,event in enumerate(events):
        require(type(event) is dict and set(event)=={'query_index','date','eligible_indices','original_rank','boundary_indices','selected_indices'} and event['query_index']==i,'Selector event schema')
        eligible=event['eligible_indices'];order=event['original_rank'];boundary=event['boundary_indices'];selected=event['selected_indices']
        require(type(eligible) is list and len(set(eligible))==len(eligible) and sorted(eligible)==eligible,'Unique eligible indices')
        require(type(order) is list and len(order)==len(eligible) and set(order)==set(eligible),'Original selector rank population')
        require(type(boundary) is list and len(set(boundary))==len(boundary) and all(j in order[2:5] for j in boundary),'Original selector boundary')
        require(type(selected) is list and len(set(selected))==len(selected),'Unique selected indices')
        count=len(eligible)
        if count>=3:
            require(boundary and order[2] in boundary and len(selected)==3 and set(order[:2])<=set(selected) and len(set(selected)&set(boundary))==1,'Frozen top2 plus one boundary winner')
        else:require(boundary==[] and selected==(order if count==2 else []),'Small universe fixed selection')
        work['event_rows']+=1;work['eligible_cells']+=count;work['source_rank_calls']+=1
        work['support_evaluations']+=len(boundary);work['delta_evaluations']+=len(boundary)
        work['selection_rank_calls']+=int(count>=3);work['partial_events']+=int(event['date']=='2016-08-17')
        work['two_eligible_events']+=int(count==2);work['cash_events']+=int(count<2)
    return work

def dependencies():
    names=('monthly_forecast_spacing269_host.py','monthly_selector_scope268_host.py','monthly_selector_scope268_independent.py',
        'monthly_selector_scope268_saved_reader.py','monthly_boundary_pair267_host.py','monthly_monotone_tree266_independent.py',
        'monthly_monotone_tree266_admission.py','monthly_monotone_tree266_saved_reader.py','monthly_boundary_pair267_saved_reader.py',
        'monthly_gain_concentration265_saved_reader.py','ptr169_results_independent.py','ptr169_rebuild_evaluate.py',
        'monthly_ownsector_gate238_host.py','monthly_stock_comp149_runner.py','monthly_same_report145_runner.py',
        'monthly_cash_quality146_factor.py','monthly_forward_guidance132_runner.py','roa_field_scope270_v2.py')
    return [ROOT/'work'/name for name in names]


def source_increment_check(context):
    """Authenticate the accepted three, never relabel the failed44 census."""
    ref=context['source_increment'];require(ref['directory']==INCREMENT.name,'Exact increment directory')
    require(sha(INCREMENT/'MANIFEST.json')==ref['manifest_sha256'] and sha(INCREMENT/'INCREMENT.json')==ref['increment_sha256'],'Increment artifact binding')
    rr=ref['actual_review'];rp=ROOT/rr['path'];require(rp.is_file() and sha(rp)==rr['sha256'],'Independent actual source review')
    manifest,data,review=read(INCREMENT/'MANIFEST.json'),read(INCREMENT/'INCREMENT.json'),read(rp)
    require(manifest['schema']=='roa_field_scope270_increment_manifest_v1' and manifest['increment_accepted'] is True and manifest['full_census_accepted'] is False,'Source increment manifest scope')
    require(manifest['output_sha256']=={'INCREMENT.json':ref['increment_sha256']},'Single increment output')
    require(review['schema']=='roa_field_scope270_increment_actual_independent_review_v1','Actual source review schema')
    for key in ('passed','audit_completed','increment_accepted','source_only'):require(review[key] is True,'Source independent '+key)
    for key in ('strict_pit','full_census_accepted','full_census_passed','old_admissions_recertified','ratio_or_chain_admitted'):require(review[key] is False,'Bounded source scope '+key)
    for key,path in (('manifest',INCREMENT/'MANIFEST.json'),('increment',INCREMENT/'INCREMENT.json')):
        require((ROOT/review[key]['path']).resolve()==path.resolve() and review[key]['sha256']==sha(path),'Actual review member '+key)
    pins=canonical_pins(manifest['input_sha256'])
    for name,digest in canonical_pins(review['input_sha256']).items():old.bind(pins,Path(name),digest)
    require(data['schema']=='roa_field_scope270_increment_v1' and data['source_only'] is True and data['full_census_passed'] is False and data['ratio_or_chain_admitted'] is False,'Bounded increment data')
    import monthly_roa_increment272_factor as factor
    require(data['increment_admitted_event_ids']==list(factor.EVENT_IDS) and [x['event_id'] for x in data['increment_records']]==list(factor.EVENT_IDS),'Exactly predeclared three')
    require(len(data['full_catalog'])==44 and data['counts']==dict(accepted_events=3,accepted_fields=9,retained_catalog_events=44,new_assess_calls=0,ratios_computed=0,features_computed=0,returns_computed=0,NPZ_decodes=0,models_fitted=0),'Source-only counts')
    for path in (INCREMENT/'MANIFEST.json',INCREMENT/'INCREMENT.json',rp):old.bind(pins,path)
    return data,dict(directory=INCREMENT.name,manifest_sha256=ref['manifest_sha256'],increment_sha256=ref['increment_sha256'],actual_review=rr,active_input_sha256=pins,old_runtime_verified=False,full_census_passed=False)


def legacy_binding():
    """Only completed-control evidence and consumed cache; no old runtime traversal."""
    fixed={'PLAN.json':LEGACY_PLAN_SHA,'CACHE.json':LEGACY_CACHE_SHA,'COMPLETION.json':LEGACY_COMPLETION_SHA}
    for name,digest in fixed.items():require(sha(LEGACY/name)==digest,'Frozen154 '+name)
    cache=read(LEGACY/'CACHE.json');require(set(cache)=={'DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json'},'Exact final154 cache')
    for name,digest in cache.items():require(sha(LEGACY/name)==digest,'Frozen154 cache '+name)
    done=read(LEGACY/'COMPLETION.json');prep=read(LEGACY/'INDEPENDENT_PREPARE.json');outcome=read(LEGACY/'INDEPENDENT_OUTCOME.json')
    require(done['directory']==LEGACY.name and done['round']==154 and done['plan_sha256']==LEGACY_PLAN_SHA and done['protected_files']==21444,'Final154 closure')
    require(done['independent_outcome_sha256']==sha(LEGACY/'INDEPENDENT_OUTCOME.json') and outcome['passed'] is True,'Final154 independent outcome')
    for item in (prep,outcome):require(item['plan_sha256']==LEGACY_PLAN_SHA and item['cache_sha256']==LEGACY_CACHE_SHA and item['orders_sha256']==cache['PRETRAIN_ORDERS.json'],'154 independent artifact bindings')
    require(prep['passed'] is True and prep['original27_arrays_byte_identical'] is True and prep['original149_earnings_cache_and_mask_verified'] is True and prep['all_seven_score_matrices_and_ordered_choices_independently_verified'] is True,'154 baseline independent proof')
    session_ref=read(LEGACY/'PLAN.json')['inputs']['sessions'];session_path=Path(session_ref['path'])
    require(session_path.resolve()==(ROOT/'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json').resolve() and sha(session_path)==session_ref['sha256'],'Historical154 calendar reference')
    old_sessions=read(session_path)['sessions'];current_sessions=read(old.CALENDAR)['sessions']
    require(old_sessions and [day for day in current_sessions if old_sessions[0]<=day<=old_sessions[-1]]==old_sessions,'Exact shared154/original session calendar')
    members={name:sha(LEGACY/name) for name in (*fixed,*cache,'INDEPENDENT_PREPARE.json','INDEPENDENT_OUTCOME.json')}
    return dict(directory=LEGACY.name,artifacts_sha256=members,control='expanded_earnings',hand='earnings_expanded',mask='expanded_available',historical_runtime_reverified=False,original149_SBC_common_mask_inherited=True,session_reference=dict(path=str(session_path),sha256=session_ref['sha256']),shared_session_calendar_exact=True)


def register():
    c,d=contract_check();require(not OUT.exists(),'Fresh experiment directory')
    before=CENTRAL.read_bytes();state=json.loads(before);initial_state(state);require(state['registered_queue']==[],'No queued research');predecessor_binding(state)
    t=read(TESTS);require(t['passed'] is True and t['synthetic_only'] is True and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Current formal6 synthetic tests')
    review=code_review();pins=canonical_pins(read(PRIOR/'PLAN.json')['input_sha256']);closure=bind_source_closure(pins)
    _,increment=source_increment_check(c);control=legacy_binding();study=old.local()
    paths=files()+[ROOT/p for p in SUPPLEMENTAL]+dependencies()+list(study.implementations())+list(study.source_files())
    paths += [CONTEXT,CONTRACT,TESTS,REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',PRIOR_REVIEW,old.PROTECTED,LOCAL/'local_linear_ORDERS.json',old.CALENDAR]
    paths += [PRIOR/'prepared'/name for name in ('COMPLETION.json','SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json')]
    paths += [LEGACY/name for name in control['artifacts_sha256']]+[Path(control['session_reference']['path'])]
    for path in paths:old.bind(pins,Path(path))
    for values in (increment['active_input_sha256'],c['input_sha256'],d['input_sha256'],review['input_sha256']):
        for name,digest in canonical_pins(values).items():old.bind(pins,Path(name),digest)
    raw=read(LOCAL/'PLAN.json')['raw_source_hashes'];require(len(raw)==64,'Original64 source map')
    for name,digest in raw.items():old.bind(pins,Path(name),digest)
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    dump(OUT/'PLAN.json',dict(schema='monthly_roa_increment272_v1',created_utc=now(),context=c,context_sha256=sha(CONTEXT),implementation_contract=d,implementation_contract_sha256=sha(CONTRACT),runtime=ev.runtime(),input_sha256=pins,original169_source_closure=closure,raw_source_hashes=raw,calendar_sha256=sha(old.CALENDAR),prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,numerical_verification=NUMERICAL_VERIFICATION,legacy154=control,source_increment=increment,actual_labels_before_registration=0,actual_predictions_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0))
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),last_goal_turn_classification='progress',next_step='Calibrate original and154 before exact3 overlay; first88 same-closeZERO only.')
    old.commit_state(before,state,'roa_increment272_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


def select_first88(metrics,orders=None):
    import monthly_roa_increment272_factor as factor
    require(type(metrics) is dict and set(metrics)==set(ARMS),'Three fixed selection arms')
    rows=[]
    for arm in ARMS:
        require(set(metrics[arm])=={'same_close_zero'},'Only same-closeZERO can select')
        rows.append(dict(name=arm,**metrics[arm]['same_close_zero']))
    return factor.select_first88(rows,execution='close',cost_per_side=0.,training_months=88)


_SOURCE_TOKEN=object()
_ISSUED=[]
class _SourceAuthority:
    __slots__=('reference','arrays','financial','baseline','array_baseline','financial_baseline','binding')
    def __init__(self,reference,arrays,financial,binding,token):
        require(token is _SOURCE_TOKEN,'Source issuer')
        self.reference=reference;self.arrays=arrays;self.financial=financial;self.baseline=deepcopy(reference)
        self.array_baseline={k:v.copy() for k,v in arrays.items()}
        self.financial_baseline={k:financial[k].copy() for k in ('hand','mask')};self.binding=deepcopy(binding)


class _PreScoreCache:
    def __init__(self,authority,values,token):
        require(token is _SOURCE_TOKEN,'P cache issuer');self.authority=authority;self.values=values.copy();self.baseline=values.copy()


def authority_check(authority,reference,arrays,financial,meta,n):
    require(type(authority) is _SourceAuthority and any(authority is x for x in _ISSUED) and authority.reference is reference and authority.arrays is arrays and authority.financial is financial,'Host-issued verified source identity required')
    require(set(arrays)==set(authority.array_baseline) and len(arrays)==27,'Original27 cache members')
    values=0
    for name,a in arrays.items():
        exact_array(a[:n],authority.array_baseline[name][:n],'Original cache '+name);values+=a[:n].size
    require(old.meta_prefix(reference['meta'],n)==old.meta_prefix(meta,n)==old.meta_prefix(authority.baseline['meta'],n),'Full metadata prefix with static fields')
    for role in ('incumbent','control'):
        actual,saved=reference[role],authority.baseline[role]
        require(actual['orders'][:n]==saved['orders'][:n],'Reference ordered choices mutated')
        exact_array(actual['scores'][:n],saved['scores'][:n],'Reference F '+role)
    exact_array(reference['incumbent']['pre_scores'][:n],authority.baseline['incumbent']['pre_scores'][:n],'Reference original P')
    for key in ('hand','mask'):exact_array(financial[key][:n],authority.financial_baseline[key][:n],'154 '+key)
    w=arrays['baseline'].shape[1]
    return dict(checks=1,cache_values_compared=int(values),reference_values_compared=3*n*w,reference_order_events_compared=2*n,metadata_events_compared=n,financial_values_compared=2*n*w)


def physical_prefix(arrays,meta,n):
    """No original computation can see the future tail, including core_weight."""
    integer(n,'Physical prefix');require(type(arrays) is dict and len(arrays)==27,'Original27 arrays')
    total,width=arrays['baseline'].shape;require(n<=total,'Physical prefix bound')
    result={}
    for key,value in arrays.items():
        require(isinstance(value,np.ndarray) and value.ndim>=1 and value.shape[0]==total,'Original row axis '+key)
        result[key]=value[:n].copy()
    mm=old.meta_prefix(meta,n)
    require(len(mm['dates'])==n and len(mm['symbols'])==n,'Physical metadata axes')
    return result,mm


def selector_receipt(proof,n,width):return prior.selector_receipt(proof,n,width)


def mathematical_receipt(proof,kind,n,width):
    import monthly_roa_increment272_factor as factor
    require(type(proof) is dict and proof.get('passed') is True and type(proof.get('events')) is int and proof['events']==n and type(proof.get('width')) is int and proof['width']==width,'Independent '+kind+' scope')
    common={'schema','passed','events','width','source_files_authenticated','computes_returns'}
    require(proof['source_files_authenticated'] is False and proof['computes_returns'] is False,'Independent pure provenance scope')
    if kind=='overlay':
        require(set(proof)==common|{'counts','overlay_verifications','audit_matched','baseline_bytes_preserved','availability_independent_of_E'} and proof['schema']=='monthly_roa_increment272_overlay_independent_v1','Independent overlay schema')
        integer(proof['overlay_verifications'],'One independent overlay verification',1)
        for key in ('audit_matched','baseline_bytes_preserved','availability_independent_of_E'):require(proof[key] is True,'Independent overlay '+key)
        counts_check(proof['counts'],factor.COUNT_FIELDS,'Independent overlay work')
        integer(proof['counts']['events'],'Overlay n',n);integer(proof['counts']['symbols_per_event'],'Overlay width',width)
    else:
        require(kind=='pre_score' and set(proof)==common|{'P_modified_once','eligible_only','adapter_bound_checked','work'} and proof['schema']=='monthly_roa_increment272_pre_score_independent_v1','Independent P schema')
        for key in ('P_modified_once','eligible_only','adapter_bound_checked'):require(proof[key] is True,'Independent P '+key)
        counts_check(proof['work'],PRE_WORK_FIELDS,'Independent P work')
        integer(proof['work']['pre_score_verifications'],'One P verification',1);integer(proof['work']['cells_compared'],'Full P compare',n*width)
        integer(proof['work']['new_model_fits'],'Independent zero fits',0)
        require(proof['work']['eligible_delta_cells']<=proof['work']['hand_bound_cells']<=n*width,'Independent P work domains')
    return proof


def episode(study,P,hand,a,m):
    """Delegate exact149 correction and one existing episode; capture actual P."""
    n,width=P.shape;require(width==legacy.WIDTH,'Frozen149 width')
    captures=[]
    def call(*args,**kwargs):
        captures.append(kwargs['pre_score'].copy())
        return study.adapter.score_and_choose(*args,**kwargs)
    backend=SimpleNamespace(adapter=SimpleNamespace(score_and_choose=call),selector=study.ep.select_boundary)
    policy=dict(budget=.05,ceiling=5,gap=.02)
    if hand is None:
        choices,F=call('incumbent66',0.,np.full(P.shape,np.nan),a,m,n,pre_score=P.copy(),effects=a['effects'],episode_policy=policy,boundary_selector=backend.selector)
    else:
        require(type(hand) is np.ndarray and hand.dtype==np.float64 and hand.shape==P.shape,'Physical financial hand')
        require(not np.isinf(hand).any() and np.all(np.abs(hand[np.isfinite(hand)])<=.5),'Original adapter .5 hand boundary; no clipping')
        choices,F=legacy.corrected(P,hand,a,m,n,backend,policy)
    require(len(captures)==1,'Exactly one original adapter/episode')
    return choices,F,captures[0]


def prepare_one(study,arrays,meta,reference,financial,n,*,authority,cached_pre=None):
    import monthly_roa_increment272_factor as factor
    import monthly_roa_increment272_independent as ind
    import monthly_selector_scope268_independent as si
    integer(n,'Prefix');total,width=arrays['baseline'].shape;require(n<=total,'Prefix source bound')
    checks=[authority_check(authority,reference,arrays,financial,meta,n)]
    ap,mm=physical_prefix(arrays,meta,n);before=structured_digest(ap)
    if cached_pre is None:
        require(study.configs()[4]==dict(id=4,base='incumbent_scope67',mode='local_linear'),'Original config4')
        P=study.modified_pre_score(study.configs()[4],ap,n)
        require(type(P) is np.ndarray and P.shape==(n,width),'Physically shortened original P')
        P=P.copy()
    else:
        require(type(cached_pre) is _PreScoreCache and cached_pre.authority is authority,'Issued original P cache')
        exact_array(cached_pre.values[:n],cached_pre.baseline[:n],'Cached original P changed');P=cached_pre.values[:n].copy()
    exact_array(P,reference['incumbent']['pre_scores'][:n],'Original P before candidate')
    require(before==structured_digest(ap),'Original pre helper mutated source')
    hands={'control_hand':financial['hand'][:n].copy(),'control_mask':financial['mask'][:n].copy()}
    require(hands['control_mask'].dtype==bool and np.array_equal(np.isfinite(hands['control_hand']),hands['control_mask']),'Frozen154 finite-mask equivalence')
    out={};scores={};pres={};audits={};proofs={}
    for arm,hand,refrole in ((ARMS[0],None,'incumbent'),(ARMS[1],hands['control_hand'],'control')):
        cc,F,actualP=episode(study,P,hand,ap,mm)
        orders=study.rp.make_orders(cc,F,mm);expected=reference[refrole]
        require(orders==expected['orders'][:n],'Exact '+refrole+' ordered choices before candidate')
        exact_array(F,expected['scores'][:n],'Exact '+refrole+' F before candidate')
        inputs=selector_inputs(study,actualP,ap,mm,n);snapshot=structured_digest((inputs,cc,F,P,hand))
        sp=selector_receipt(si.verify(*inputs,cc,F,'incumbent_local67'),n,width)
        pp=None if hand is None else mathematical_receipt(ind.verify_pre_score(P,hand,ap['eligible'],actualP),'pre_score',n,width)
        require(snapshot==structured_digest((inputs,cc,F,P,hand)) and before==structured_digest(ap),'Reference verifier mutated source/output')
        out[arm]=orders;scores[arm]=F;pres[arm]=actualP;proofs[arm]=dict(pre_score=pp,selector=sp)
        audits[arm]=dict(selector_events=selector_events(inputs,cc,sp))
        checks.append(authority_check(authority,reference,arrays,financial,meta,n))
    # Both saved-control gates precede any new financial value calculation.
    base_before=structured_digest(hands)
    hand,mask,overlay=factor.overlay_prefix(mm,hands['control_hand'],hands['control_mask'],financial['catalog'],financial['records'],financial['sessions'])
    require(base_before==structured_digest(hands),'Overlay mutated frozen154 prefix')
    output_before=structured_digest((hand,mask,overlay))
    op=mathematical_receipt(ind.verify_overlay(mm,hands['control_hand'],hands['control_mask'],financial['catalog'],financial['records'],financial['sessions'],hand,mask,overlay),'overlay',n,width)
    require(output_before==structured_digest((hand,mask,overlay)) and base_before==structured_digest(hands),'Overlay verifier mutated output/baseline')
    cc,F,actualP=episode(study,P,hand,ap,mm)
    pre_before=structured_digest((P,hand,actualP))
    pp=mathematical_receipt(ind.verify_pre_score(P,hand,ap['eligible'],actualP),'pre_score',n,width)
    require(pre_before==structured_digest((P,hand,actualP)),'Candidate P verifier mutated inputs/output')
    inputs=selector_inputs(study,actualP,ap,mm,n);snapshot=structured_digest((inputs,cc,F))
    sp=selector_receipt(si.verify(*inputs,cc,F,'incumbent_local67'),n,width)
    require(snapshot==structured_digest((inputs,cc,F)) and before==structured_digest(ap),'Candidate verifier mutated source/output')
    arm=ARMS[2];out[arm]=study.rp.make_orders(cc,F,mm);scores[arm]=F;pres[arm]=actualP
    proofs[arm]=dict(pre_score=pp,selector=sp);audits[arm]=dict(selector_events=selector_events(inputs,cc,sp))
    hands.update(candidate_hand=hand,candidate_mask=mask)
    checks.append(authority_check(authority,reference,arrays,financial,meta,n))
    pc=dict.fromkeys(PRODUCER_FIELDS,0);pc.update(original_pre_score_calls=int(cached_pre is None),original_selector_calls=3,financial_correction_calls=2,overlay_calls=1,selector_audit_rank_calls=3*n)
    sa=dict(schema='roa_increment272_source_audit_v1',events=n,width=width,cached_original_pre=cached_pre is not None,source_integrity_work=sum_counts(checks,INTEGRITY_FIELDS),original_pre_scores_exact=True,original_eligibility_exact=True,original_scores_exact=True,original_orders_exact=True,legacy154_scores_exact=True,legacy154_orders_exact=True,both_reference_gates_before_overlay=True,source_files_authenticated=False,new_price_windows_read=False,new_model_fits=0)
    entry=dict(prefix=n,recomputed_original_pre=cached_pre is None,reuses_main_verified_full=False,source_audit=sa,independent=proofs,overlay_proof=op,overlay_counts=deepcopy(overlay['counts']),producer_counts=pc)
    audit=dict(schema='roa_increment272_decision_audit_v1',events=n,width=width,arms=audits)
    entry_check(entry,n,width,cached=cached_pre is not None)
    return (out,scores,pres,audit,hands,overlay),entry,_PreScoreCache(authority,P,_SOURCE_TOKEN)

def exact_prefix(full,part,n):
    for arm in ARMS:
        require(part[0][arm]==full[0][arm][:n],'Ordered prefix '+arm)
        exact_array(part[1][arm],full[1][arm][:n],'F prefix '+arm)
        exact_array(part[2][arm],full[2][arm][:n],'P prefix '+arm)
        require(part[3]['arms'][arm]['selector_events']==full[3]['arms'][arm]['selector_events'][:n],'Selector event prefix')
    for key in full[4]:exact_array(part[4][key],full[4][key][:n],'Financial prefix '+key)
    require(part[5]['cells']==full[5]['cells'][:n*full[3]['width']],'Overlay audit prefix')


def entry_check(entry,n,width,*,cached=False):
    import monthly_roa_increment272_factor as factor
    require(type(entry) is dict and set(entry)=={'prefix','recomputed_original_pre','reuses_main_verified_full','source_audit','independent','overlay_proof','overlay_counts','producer_counts'},'Exact preparation entry')
    integer(entry['prefix'],'Entry prefix',n)
    require(entry['recomputed_original_pre'] is (not cached) and type(entry['reuses_main_verified_full']) is bool,'Entry source mode')
    expected=dict.fromkeys(PRODUCER_FIELDS,0);expected.update(original_pre_score_calls=int(not cached),original_selector_calls=3,financial_correction_calls=2,overlay_calls=1,selector_audit_rank_calls=3*n)
    counts_check(entry['producer_counts'],PRODUCER_FIELDS,'Producer work');require(entry['producer_counts']==expected,'Actual calls')
    require(set(entry['independent'])==set(ARMS),'Independent role scope')
    for arm in ARMS:
        item=entry['independent'][arm];require(set(item)=={'pre_score','selector'},'Independent role proof')
        selector_receipt(item['selector'],n,width)
        if arm==ARMS[0]:require(item['pre_score'] is None,'No financial correction for incumbent')
        else:mathematical_receipt(item['pre_score'],'pre_score',n,width)
    mathematical_receipt(entry['overlay_proof'],'overlay',n,width)
    counts_check(entry['overlay_counts'],factor.COUNT_FIELDS,'Producer overlay work')
    require(entry['overlay_counts']==entry['overlay_proof']['counts'],'Independent financial count equality')
    sa=entry['source_audit'];require(sa['schema']=='roa_increment272_source_audit_v1' and sa['events']==n and sa['width']==width and sa['cached_original_pre'] is cached,'Source audit axes')
    for flag in ('original_pre_scores_exact','original_eligibility_exact','original_scores_exact','original_orders_exact','legacy154_scores_exact','legacy154_orders_exact','both_reference_gates_before_overlay'):require(sa[flag] is True,'Source gate '+flag)
    require(sa['source_files_authenticated'] is False and sa['new_price_windows_read'] is False,'Pure seam does not authenticate files')
    integer(sa['new_model_fits'],'Source fits',0);counts_check(sa['source_integrity_work'],INTEGRITY_FIELDS,'Source integrity')
    work=sa['source_integrity_work'];require(work['checks']==4 and work['reference_values_compared']==12*n*width and work['reference_order_events_compared']==8*n and work['metadata_events_compared']==4*n and work['financial_values_compared']==8*n*width,'Four source checks actual domains')
    return entry


def execution_accounting(evidence):
    import monthly_roa_increment272_factor as factor
    import monthly_roa_increment272_independent as ind
    import monthly_selector_scope268_independent as si
    entries=[evidence['main']]+evidence['cached_prefixes']+[x for x in evidence['physical_prefixes'] if not x['reuses_main_verified_full']]
    return dict(actual_prepare_entries=len(entries),producer=sum_counts([e['producer_counts'] for e in entries],PRODUCER_FIELDS),
        producer_overlay=sum_counts([e['overlay_counts'] for e in entries],factor.COUNT_FIELDS),
        independent_overlay=sum_counts([e['overlay_proof']['counts'] for e in entries],factor.COUNT_FIELDS),
        independent_pre_score=sum_counts([e['independent'][a]['pre_score']['work'] for e in entries for a in ARMS[1:]],PRE_WORK_FIELDS),
        independent_selector=sum_counts([e['independent'][a]['selector']['work'] for e in entries for a in ARMS],si.WORK_FIELDS),
        source_integrity_work=sum_counts([e['source_audit']['source_integrity_work'] for e in entries],INTEGRITY_FIELDS),
        physical_full_reuses_main=True,new_model_fits=0,
        work_definition='Count actual calls and independent receipts; cached means only original P reused. All entries recompute exact3 overlay and two financial corrections, three original episodes. Main122 is reused by physical122, not executed twice. Source hashing/full decoded snapshots and legacy full122 cached rank checks are separately disclosed; no zero-read claim.')


def preparation_evidence(evidence,audit,source,*,total=122,width=49,prefixes=PREFIXES):
    require(evidence['schema']=='roa_increment272_independent_prepare_v1','Prepare schema')
    entry_check(evidence['main'],total,width)
    require(evidence['main']['reuses_main_verified_full'] is False,'Main actual entry')
    cached,physical=evidence['cached_prefixes'],evidence['physical_prefixes']
    require([x['prefix'] for x in cached]==list(range(total+1)),'All cached prefixes')
    for e in cached:entry_check(e,e['prefix'],width,cached=True);require(e['reuses_main_verified_full'] is False,'Cached entry executed')
    require([x['prefix'] for x in physical]==list(prefixes),'Physical prefixes')
    for e in physical:
        entry_check(e,e['prefix'],width)
        if e['prefix']==total:
            target=deepcopy(evidence['main']);target['reuses_main_verified_full']=True;require(e==target,'Full122 exact alias')
        else:require(e['reuses_main_verified_full'] is False,'Fresh physical entry')
    for flag in ('all123_order_prefixes','source_arrays_unchanged','original_reference_unchanged','financial_source_unchanged'):require(evidence[flag] is True,'Preparation invariant '+flag)
    require(evidence['execution_accounting']==execution_accounting(evidence),'Exact actual execution ledger')
    require(audit['schema']=='roa_increment272_decision_audit_v1' and audit['events']==total and audit['width']==width and set(audit['arms'])==set(ARMS),'Full decisions')
    for e in [evidence['main']]+cached+physical:
        for arm in ARMS:require(e['independent'][arm]['selector']['work']==selector_event_work(audit['arms'][arm]['selector_events'][:e['prefix']]),'Exact per-prefix selector work')
    expected=dict(main=evidence['main']['source_audit'],physical=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in physical],cached=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in cached])
    require(source==expected,'Source audit projection')


def load_sources(expected):
    """Only PLAN-authorized actual numeric entrypoint; no data at import/test time."""
    plan=gate(expected);study=old.local();study.verify_source();meta,arrays,px,opening=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 cache')
    previous=read(PRIOR/'prepared/COMPLETION.json');require(previous['status']=='COMPLETE','Complete271 source')
    for name in ('META.json','SCORES.npz','PRE_SCORES.npz','ORDERS.json'):require(previous['artifacts_sha256'][name]==sha(PRIOR/'prepared'/name),'Predecessor prepared member')
    require(read(PRIOR/'prepared/META.json')==meta==read(LEGACY/'META.json'),'Full monthly identity across frozen controls')
    require(legacy_binding()==plan['legacy154'],'Historical154 anchors unchanged')
    saved=ev.load_arrays(LEGACY/'DATA.npz')
    require(set(arrays)<=set(saved),'154 retains original arrays')
    for name in arrays:exact_array(saved[name],arrays[name],'154 original cache '+name)
    reference=dict(meta=meta,incumbent=dict(orders=read(PRIOR/'prepared/ORDERS.json')[ARMS[0]],scores=ev.load_arrays(PRIOR/'prepared/SCORES.npz')[ARMS[0]],pre_scores=ev.load_arrays(PRIOR/'prepared/PRE_SCORES.npz')[ARMS[0]]),
        control=dict(orders=read(LEGACY/'PRETRAIN_ORDERS.json')['expanded_earnings'],scores=ev.load_arrays(LEGACY/'SCORES.npz')['expanded_earnings']))
    require(reference['incumbent']['orders']==read(LOCAL/'local_linear_ORDERS.json'),'Original Local67 ordered reference')
    increment,binding=source_increment_check(read(CONTEXT));require(binding==plan['source_increment'],'Accepted source unchanged')
    financial=dict(hand=saved['earnings_expanded'].copy(),mask=saved['expanded_available'].copy(),catalog=increment['full_catalog'],records=increment['increment_records'],sessions=read(old.CALENDAR)['sessions'])
    require(financial['hand'].dtype==np.float64 and financial['mask'].dtype==bool and financial['hand'].shape==financial['mask'].shape==(122,49),'Frozen154 hand/mask')
    panel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'] and len(raw)==64,'Original64 panel source');old.verify(raw)
    execution=study.verify_panel_parity(panel,px,opening)
    authority=_SourceAuthority(reference,arrays,financial,dict(plan_sha256=expected,legacy154=plan['legacy154'],source_increment=binding),_SOURCE_TOKEN);_ISSUED.append(authority)
    return plan,study,arrays,meta,reference,financial,authority,execution


def source_evidence(value,plan):
    for key in ('original_arrays_unchanged','original_orders_exact','original_scores_exact','original_pre_scores_exact','original_eligibility_exact','legacy154_scores_exact','legacy154_orders_exact','both_reference_gates_before_overlay','whole_original_cache_decoded'):
        require(value[key] is True,'Source truth '+key)
    for key in ('original169_source_closure','raw_source_hashes','calendar_sha256','legacy154','source_increment'):require(value[key]==plan[key],'Source binding '+key)
    for key,path in (('original_training_hash',REBUILD/'TRAIN_INPUTS.npz'),('original_identity_hash',REBUILD/'MONTHLY_META.json'),('saved169_parity_hash',REBUILD/'PARITY.json'),('prior_prepared_sha256',PRIOR/'prepared/COMPLETION.json')):require(value[key]==sha(path),'Source member '+key)
    for key in ('training_arrays_decoded','new_price_windows_read','strict_pit','legacy154_runtime_reverified','full270_census_passed'):require(value[key] is False,'Source limitations '+key)
    integer(value['new_model_fits'],'Source zero fits',0)
    require(len(value['original_array_fingerprints'])==27,'Original27 fingerprints')
    for item in value['original_array_fingerprints'].values():
        require(set(item)=={'dtype','shape','sha256'} and item['shape'][0]==122,'Fingerprint shape');digest_string(item['sha256'],'Original array')


def prepare(expected):
    require(not (OUT/'prepared').exists(),'Preparation write-once')
    plan,study,arrays,meta,reference,financial,authority,execution=load_sources(expected)
    ab=old.array_snapshot(arrays);rb=structured_digest(reference);fb=structured_digest(financial)
    full,main,cache=prepare_one(study,arrays,meta,reference,financial,122,authority=authority)
    cached=[];physical=[]
    for n in range(123):
        got,e,_=prepare_one(study,arrays,meta,reference,financial,n,authority=authority,cached_pre=cache)
        exact_prefix(full,got,n);cached.append(e)
        if n in PREFIXES and n!=122:
            print(json.dumps(dict(stage='272_fresh_physical_prefix',events=n)),flush=True)
            got,e,_=prepare_one(study,arrays,meta,reference,financial,n,authority=authority)
            exact_prefix(full,got,n);physical.append(e)
    reuse=deepcopy(main);reuse['reuses_main_verified_full']=True;physical.append(reuse)
    require(ab==old.array_snapshot(arrays) and rb==structured_digest(reference) and fb==structured_digest(financial),'All source/reference bytes unchanged')
    evidence=dict(schema='roa_increment272_independent_prepare_v1',main=main,cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,source_arrays_unchanged=True,original_reference_unchanged=True,financial_source_unchanged=True)
    evidence['execution_accounting']=execution_accounting(evidence)
    orders,scores,pres,audit,hands,overlay=full
    sa=dict(main=main['source_audit'],physical=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in physical],cached=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in cached])
    parity=dict(original169_source_closure=plan['original169_source_closure'],raw_source_hashes=plan['raw_source_hashes'],original_training_hash=sha(REBUILD/'TRAIN_INPUTS.npz'),original_identity_hash=sha(REBUILD/'MONTHLY_META.json'),saved169_parity_hash=sha(REBUILD/'PARITY.json'),prior_prepared_sha256=sha(PRIOR/'prepared/COMPLETION.json'),calendar_sha256=plan['calendar_sha256'],legacy154=plan['legacy154'],source_increment=plan['source_increment'],execution_price_parity=execution,original_array_fingerprints=old.array_fingerprints(arrays),whole_original_cache_decoded=True,training_arrays_decoded=False,new_price_windows_read=False,new_model_fits=0,legacy154_runtime_reverified=False,full270_census_passed=False,strict_pit=False,
        consumption_scope='Original27 and historical154 cache NPZ decoded once after PLAN. Legacy study.load validates full122 cached local ranks; raw64 execution parity checked, no new price windows or training decode. Full source/reference snapshots at load and complete preparation boundaries; each entry checks consumed prefix four times. Fresh original P only sees physical prefix arrays/meta/core_weight. Cached reuses original P but recomputes overlay/corrections. Historical154 PLAN/cache/completion/independent receipts are byte-bound control anchors; obsolete PDF runtime dependency graph is not re-certified. New accepted exact3 active source pins are all verified; historical574 runtime edges remain historical. Financial NI/average-assets ratios are newly computed only inside accepted intervals after both control gates; no new predictive fits. No zero-read or strict PIT claim.',
        original_arrays_unchanged=True,original_orders_exact=True,original_scores_exact=True,original_pre_scores_exact=True,original_eligibility_exact=True,legacy154_scores_exact=True,legacy154_orders_exact=True,both_reference_gates_before_overlay=True)
    preparation_evidence(evidence,audit,sa);source_evidence(parity,plan);gate(expected)
    dest=OUT/'prepared';dest.mkdir()
    for name,values in (('SCORES',scores),('PRE_SCORES',pres),('FINANCIAL_INPUTS',hands)):np.savez_compressed(dest/(name+'.npz'),**values)
    for name,value in {'DECISION_AUDIT.json':audit,'SOURCE_AUDIT.json':sa,'SOURCE_PARITY.json':parity,'INDEPENDENT_PREPARE.json':evidence,'OVERLAY_AUDIT.json':overlay,'ORDERS.json':orders,'META.json':meta}.items():dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={name:sha(dest/name) for name in PREP_FILES},new_returns_computed=False,new_model_fits=0,execution_accounting_sha256=structured_digest(evidence['execution_accounting'])))
    print(json.dumps(dict(status='PREPARED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    plan=gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['events']==122 and c['new_returns_computed'] is False and type(c['new_model_fits']) is int and c['new_model_fits']==0 and set(c['artifacts_sha256'])==PREP_FILES,'Complete prepared scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Prepared immutable member')
    evidence,audit,source=[read(OUT/'prepared'/name) for name in ('INDEPENDENT_PREPARE.json','DECISION_AUDIT.json','SOURCE_AUDIT.json')]
    preparation_evidence(evidence,audit,source);require(c['execution_accounting_sha256']==structured_digest(evidence['execution_accounting']),'Prepared work ledger')
    source_evidence(read(OUT/'prepared/SOURCE_PARITY.json'),plan)
    return read(OUT/'prepared/META.json'),read(OUT/'prepared/ORDERS.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    elif args.phase=='prepare':require(bool(args.plan_sha256),'PLAN hash required');prepare_recorded(args.plan_sha256)
    else:require(bool(args.plan_sha256),'PLAN hash required');globals()[args.phase](args.plan_sha256)
