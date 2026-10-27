"""268 immutable selector-scope lifecycle; synthetic callers or PLAN-first actual work.

The historical Local67 P/E/effects and source closure stay fixed. Only the
scope of its bounded episode selector changes; no new features/labels/fits.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse, hashlib, io, json, unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import monthly_boundary_pair267_host as prior
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_selector_scope268'
PRIOR=ROOT/'outputs/monthly_boundary_pair267'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
REBUILD=prior.REBUILD
CONTEXT=ROOT/'work/monthly_selector_scope268_context.json'
CONTRACT=ROOT/'work/monthly_selector_scope268_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_selector_scope268_TESTS.json'
REVIEW=ROOT/'work/monthly_selector_scope268_CODE_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_boundary_pair267_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_selector_scope268_RESULT_REVIEW.json'
PHASE_TERMINALS=OUT/'PHASE_TERMINALS.json'
ARMS=('incumbent_local67','unlocked_top5_gap','unlocked_all')
PREFIXES=(0,1,53,88,122)
PRIOR_COMPLETION_SHA='b429eebc44946764bd1731383332ae1c33911b40550f376311cfedd9230e828e'
NUMERICAL_VERIFICATION=dict(score_tolerance=0.,selection_tolerance=0.,original_core_preserved=True,new_model_fits=0)
code_names=tuple(name for stem in ('host','kernel','independent','admission') for name in ('monthly_selector_scope268_'+stem+'.py','test_monthly_selector_scope268_'+stem+'.py'))
SUPPLEMENTAL=('work/monthly_selector_scope268_saved_reader.py','work/test_monthly_selector_scope268_saved_reader.py')
PREP_FILES={'SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json','DECISION_AUDIT.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json','SOURCE_AUDIT.json'}
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}
PRODUCER_FIELDS=('original_pre_score_calls','original_selector_calls','experimental_selector_calls','selector_events','new_model_fits','new_label_values_computed','new_price_values_read')
INTEGRITY_FIELDS=('checks','cache_values_compared','reference_values_compared','reference_order_events_compared','metadata_events_compared')
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
        for path,mode,cost in ev.PATHS[:2]:
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
        require(type(item) is dict and set(item)=={'session_id','exit_code','chunk_id','artifact_path','artifact_sha256'},'Complete phase terminal '+name)
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
        require(set(training[arm])=={'same_close_zero','next_open_zero'},'Only ZERO primary paths')
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
    dump(dest/'INDEPENDENT_OUTCOME.json',check);dump(dest/'RESULTS.json',dict(selection=f['selection'],arms=results,reused_sample=True,strict_pit=False,zero_cost_selection=True,next_open_cost_diagnostic_only=True,observer_promotion=False))
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
    mods=[old.module('scope268_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5,7))]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in mods]);stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    require(result.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=result.testsRun,runtime=ev.runtime(),code_sha256=codes()))


def dependencies():
    names=('monthly_boundary_pair267_host.py','monthly_monotone_tree266_independent.py',
           'monthly_monotone_tree266_admission.py','monthly_monotone_tree266_saved_reader.py',
           'monthly_boundary_pair267_saved_reader.py','monthly_gain_concentration265_saved_reader.py',
           'ptr169_results_independent.py','ptr169_rebuild_evaluate.py','monthly_ownsector_gate238_host.py')
    return [ROOT/'work'/name for name in names]


def initial_state(state):
    require(len(state['completed'])==263 and sum(x['trials'] for x in state['completed'])==1300,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Immediate closed267 predecessor')


def predecessor_binding(state):
    context,_=contract_check();wanted=context['predecessor'];record=state['completed'][-1]['completion']
    require(Path(record['path'])==PRIOR/'COMPLETION.json' and record['sha256']==sha(PRIOR/'COMPLETION.json')==wanted['completion_sha256']==PRIOR_COMPLETION_SHA,'Predecessor central closure binding')
    done=read(PRIOR/'COMPLETION.json');review=read(PRIOR_REVIEW)
    require(done['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and done['protected_files']==21444 and done['plan_sha256']==sha(PRIOR/'PLAN.json') and done['observers_modified'] is False,'Independently closed267 required')
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
        require(record.get('frozen') is True and record.get('registration_authorized') is True and record.get('actual_data_started') is False,'Final prospective268 authorization')
        require(type(record.get('input_sha256')) is dict,'Explicit dependency map')
    require(c['arms']==list(ARMS) and c['registered_trials']==3 and c['promotable_arms']==list(ARMS[1:]) and c['diagnostic_only']==[],'Three fixed economic roles')
    require(type(c.get('actual_models_fitted')) is int and c['actual_models_fitted']==0 and c.get('actual_orders_computed') is False and c.get('actual_returns_computed') is False,'No actual268 work before registration')
    require(type(d.get('code_files')) is list and d['code_files']==['work/'+name for name in code_names],'Exact ordered formal8 inventory')
    require(type(d.get('supplemental_input_files')) is list and d['supplemental_input_files']==list(SUPPLEMENTAL),'Exact ordered reader/test inventory')
    for label,record in (('context',c),('contract',d)):
        for relative in ['work/'+name for name in code_names]+list(SUPPLEMENTAL):
            value=record['input_sha256'].get(relative);digest_string(value,label+' source pin '+relative)
            require((ROOT/relative).is_file() and value==sha(ROOT/relative),label+' current source binding '+relative)
    require(d['context_sha256']==sha(CONTEXT),'Contract context binding')
    require(c['predecessor']['directory']==PRIOR.name and c['predecessor']['completion_sha256']==PRIOR_COMPLETION_SHA,'Final267 predecessor')
    digest_string(c['predecessor'].get('result_review_sha256'),'Predecessor review')
    return c,d


def register():
    c,d=contract_check();require(not OUT.exists(),'Fresh experiment directory')
    before=CENTRAL.read_bytes();state=json.loads(before);initial_state(state);require(state['registered_queue']==[],'No queued research');predecessor_binding(state)
    t=read(TESTS);require(t['passed'] is True and t['synthetic_only'] is True and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Current formal8 synthetic tests')
    review=code_review();pins=canonical_pins(read(PRIOR/'PLAN.json')['input_sha256']);closure=bind_source_closure(pins)
    study=old.local()
    paths=files()+[ROOT/p for p in SUPPLEMENTAL]+dependencies()+list(study.implementations())+list(study.source_files())
    paths += [CONTEXT,CONTRACT,TESTS,REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',PRIOR_REVIEW,old.PROTECTED,LOCAL/'local_linear_ORDERS.json',old.CALENDAR]
    paths += [PRIOR/'prepared'/name for name in ('COMPLETION.json','SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json')]
    for path in paths:old.bind(pins,Path(path))
    for paper in (c,d,review):
        for name,digest in canonical_pins(paper['input_sha256']).items():old.bind(pins,Path(name),digest)
    raw=read(LOCAL/'PLAN.json')['raw_source_hashes'];require(len(raw)==64,'Original64 source map')
    for name,digest in raw.items():old.bind(pins,Path(name),digest)
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    dump(OUT/'PLAN.json',dict(schema='monthly_selector_scope268_v1',created_utc=now(),context=c,context_sha256=sha(CONTEXT),implementation_contract=d,implementation_contract_sha256=sha(CONTRACT),runtime=ev.runtime(),input_sha256=pins,original169_source_closure=closure,raw_source_hashes=raw,calendar_sha256=sha(old.CALENDAR),prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,numerical_verification=NUMERICAL_VERIFICATION,actual_labels_before_registration=0,actual_predictions_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0))
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),last_goal_turn_classification='progress',next_step='Original Local67 parity then fixed selector-scope comparison; first88 choice before diagnostics.')
    old.commit_state(before,state,'selector_scope268_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN')
    p=read(OUT/'PLAN.json');c,d=contract_check()
    require(p['context']==c and p['context_sha256']==sha(CONTEXT) and p['implementation_contract']==d and p['implementation_contract_sha256']==sha(CONTRACT) and p['runtime']==ev.runtime(),'Prospective runtime/papers unchanged')
    require(p['numerical_verification']==NUMERICAL_VERIFICATION,'Frozen numerical rule')
    old.verify(p['input_sha256']);require(p['original169_source_closure']==source_closure_check(p['input_sha256']),'Complete169 closure')
    code_review();require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Prestate bytes changed')
    state=read(CENTRAL);registration_check(state,expected);predecessor_binding(state);return p


def select_first88(metrics,orders):
    import monthly_selector_scope268_admission as admission
    return admission.select_first88(metrics,orders)


def accounting():return prior.accounting()


def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Three268 accounting roles')
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
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=dict(strict_pit=False,reused_sample=True,global_novelty='UNKNOWN',conditional_selector_scope_replication_only=True))
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),phase_terminals_sha256=sha(PHASE_TERMINALS),new_trials=3,new_model_fits=0,independent_prepare_sha256=sha(OUT/'prepared/INDEPENDENT_PREPARE.json'),protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==264 and sum(x['trials'] for x in state['completed'])==1303,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review matched selector scope; first88 choice immutable and frozen observers unchanged.')
    old.commit_state(before,state,'selector_scope268_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)
_SOURCE_TOKEN=object()
_ISSUED=[]
class _SourceAuthority:
    """Host-issued file/source identity; not a hostile-Python security boundary.

    Issued only after the actual PLAN and full historical manifests have passed.
    One disclosed full decoded copy is made; per-prefix checks consume slices.
    """
    __slots__=('reference','arrays','baseline','array_baseline','binding')
    def __init__(self,reference,arrays,binding,token):
        require(token is _SOURCE_TOKEN,'Source issuer')
        object.__setattr__(self,'reference',reference);object.__setattr__(self,'arrays',arrays)
        object.__setattr__(self,'baseline',deepcopy(reference));object.__setattr__(self,'array_baseline',{k:v.copy() for k,v in arrays.items()})
        object.__setattr__(self,'binding',deepcopy(binding))
    def __setattr__(self,*args):raise TypeError('Immutable source authority')


class _PreScoreCache:
    __slots__=('authority','values','baseline')
    def __init__(self,authority,values,token):
        require(token is _SOURCE_TOKEN,'P cache issuer')
        self.authority=authority;self.values=values.copy();self.baseline=values.copy()


def identity(reference,meta,n):
    axes=('dates','ends','symbols','etfs')
    require(old.meta_prefix({k:reference['meta'][k] for k in axes},n)==old.meta_prefix({k:meta[k] for k in axes},n),'Original canonical monthly identity')
    return reference['orders'][:n],reference['scores'][:n],reference['pre_scores'][:n]


def authority_check(authority,reference,arrays,meta,n):
    require(type(authority) is _SourceAuthority and any(authority is x for x in _ISSUED) and authority.reference is reference and authority.arrays is arrays,'Host-issued verified source identity required')
    require(set(arrays)==set(authority.array_baseline),'Original cache members')
    values=0
    for name,a in arrays.items():
        exact_array(a[:n],authority.array_baseline[name][:n],'Original cache '+name)
        values+=a[:n].size
    actual=identity(reference,meta,n);saved=identity(authority.baseline,meta,n)
    require(actual[0]==saved[0],'Original reference orders mutated')
    exact_array(actual[1],saved[1],'Original reference F');exact_array(actual[2],saved[2],'Original reference P')
    exact_array(reference['eligible'][:n],authority.baseline['eligible'][:n],'Original reference E')
    w=arrays['baseline'].shape[1]
    return dict(checks=1,cache_values_compared=int(values),reference_values_compared=3*n*w,reference_order_events_compared=n,metadata_events_compared=n)


def receipt(proof,mode,n,width):
    import monthly_selector_scope268_independent as ind
    require(type(proof) is dict and proof.get('schema')=='selector_scope268_independent_v1' and proof.get('passed') is True,'Complete independent selector proof')
    require(proof.get('mode')==mode and type(proof.get('events')) is int and proof['events']==n and type(proof.get('width')) is int and proof['width']==width,'Independent selector scope')
    require(proof.get('numerical_tolerance')==0. and proof.get('actual_coverage_claim') is False and proof.get('source_provenance_verified') is False,'Independent numerical/provenance scope')
    require(type(proof.get('flags')) is dict and set(proof['flags'])==set(ind.FLAGS) and all(v is True for v in proof['flags'].values()),'Independent exact flags')
    counts_check(proof.get('work'),ind.WORK_FIELDS,'Independent selector work')
    integer(proof['work']['new_model_fits'],'Independent zero fits',0)
    return proof


def selector_inputs(study,P,a,meta,n):
    # This is the frozen adapter's shape normalization, not new source values.
    cw=study.adapter._core_weight(a['core_weight'],a['baseline'].shape,n).copy()
    wanted=np.broadcast_to(np.where(a['state_calm10'][:n],.5,.52)[:,None],P.shape)
    exact_array(cw,wanted,'Original .50/.52 allocation')
    return (P,a['eligible'][:n].copy(),deepcopy(meta['symbols'][:n]),a['effects'][:n].copy(),
            a['baseline'][:n].copy(),a['core'][:n].copy(),cw,
            a['state_calm10'][:n].copy(),a['state_weak_narrow'][:n].copy(),list(meta['dates'][:n]))


def candidate_audit(audit,mode,n,width):
    import monthly_selector_scope268_kernel as kernel
    require(type(audit) is dict and set(audit)==set(kernel.AUDIT_KEYS) and audit.get('schema')=='monthly_selector_scope268_decisions_v1' and audit.get('mode')==mode,'Candidate audit scope')
    counts_check(audit.get('counts'),kernel.COUNT_FIELDS,'Candidate actual selector work')
    c=audit['counts'];require(c['events']==n and c['symbols_per_event']==width and c['selector_calls']==1 and c['new_model_fits']==c['predictive_model_fits']==0,'Candidate count scope')
    require(type(audit.get('events')) is list and len(audit['events'])==n and all(type(x) is dict and set(x)==set(kernel.EVENT_KEYS) for x in audit['events']),'Candidate event schema')
    for flag in ('original_inputs_preserved','long_core_unchanged','original_eligibility_unchanged','existing_effects_only','initial_partial_exact'):require(audit[flag] is True,'Candidate scope '+flag)
    require(audit['computes_returns'] is False,'No returns in selection')


def prepare_one(study,arrays,meta,reference,n,*,authority,cached_pre=None):
    """Pure supplied inputs; real entry requires the issued authenticated source."""
    import monthly_selector_scope268_kernel as kernel
    import monthly_selector_scope268_independent as ind
    integer(n,'Prefix');total,width=arrays['baseline'].shape;require(n<=total,'Prefix within original source')
    checks=[authority_check(authority,reference,arrays,meta,n)]
    ap=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n)
    if cached_pre is None:
        require(study.configs()[4]==dict(id=4,base='incumbent_scope67',mode='local_linear'),'Original Local67 config4')
        P=study.modified_pre_score(study.configs()[4],ap,n)[:n].copy()
    else:
        require(type(cached_pre) is _PreScoreCache and cached_pre.authority is authority,'Issued original P cache')
        exact_array(cached_pre.values[:n],cached_pre.baseline[:n],'Cached original P changed')
        P=cached_pre.values[:n].copy()
    oo,ff,pp=identity(reference,meta,n)
    exact_array(P,pp,'Full original P before candidate selection')
    exact_array(ap['eligible'],reference['eligible'][:n],'Full original E before candidate selection')
    inputs=selector_inputs(study,P,ap,mm,n);before=structured_digest(inputs)
    original_source=dict(baseline=inputs[4],core=inputs[5],core_weight=inputs[6],state_calm10=inputs[7],state_weak_narrow=inputs[8])
    cc,F=study.ep.select_boundary(P,inputs[1],inputs[2],inputs[3],original_source,dict(budget=.05,ceiling=5,gap=.02))
    orders=study.rp.make_orders(cc,F,mm)
    require(orders==oo,'Original ordered choices before candidate selection');exact_array(F,ff,'Original F before candidate selection')
    snap=structured_digest((cc,F))
    proof=receipt(ind.verify(*inputs,cc,F,ARMS[0]),ARMS[0],n,width)
    require(before==structured_digest(inputs) and snap==structured_digest((cc,F)),'Original verifier mutated inputs/outputs')
    checks.append(authority_check(authority,reference,arrays,meta,n))
    out={ARMS[0]:orders};scores={ARMS[0]:F};pres={a:P.copy() for a in ARMS}
    proofs={ARMS[0]:proof};audits={ARMS[0]:dict(source='frozen_original_select_boundary',selector_calls=1,events=n)}
    for mode in ARMS[1:]:
        choices,score,audit=kernel.select(*inputs,mode)
        require(before==structured_digest(inputs),'Candidate selector mutated source')
        output_before=structured_digest((choices,score,audit))
        proofs[mode]=receipt(ind.verify(*inputs,choices,score,mode),mode,n,width)
        require(before==structured_digest(inputs) and output_before==structured_digest((choices,score,audit)),'Candidate verifier mutated source/output')
        candidate_audit(audit,mode,n,width)
        out[mode]=study.rp.make_orders(choices,score,mm);scores[mode]=score;audits[mode]=audit
        checks.append(authority_check(authority,reference,arrays,meta,n))
    pc=dict.fromkeys(PRODUCER_FIELDS,0);pc.update(original_pre_score_calls=int(cached_pre is None),original_selector_calls=1,experimental_selector_calls=2,selector_events=3*n)
    sa=dict(schema='selector_scope268_source_audit_v1',events=n,width=width,cached_original_pre=cached_pre is not None,original_pre_score_calls=pc['original_pre_score_calls'],source_integrity_work=sum_counts(checks,INTEGRITY_FIELDS),original_pre_scores_exact=True,original_eligibility_exact=True,original_scores_exact=True,original_orders_exact=True,original_gate_before_candidates=True,source_files_authenticated=False,new_features_computed=False,new_labels_computed=False,new_price_windows_read=False,new_model_fits=0)
    evidence=dict(prefix=n,recomputed_original_pre=cached_pre is None,reuses_main_verified_full=False,source_audit=sa,independent=proofs,producer_counts=pc)
    audit=dict(schema='selector_scope268_decision_audit_v1',events=n,width=width,arms=audits,producer_counts=pc)
    entry_check(evidence,n,width,cached=cached_pre is not None)
    return (out,scores,pres,audit),evidence,_PreScoreCache(authority,P,_SOURCE_TOKEN)


def exact_prefix(full,part,n):
    for arm in ARMS:
        require(part[0][arm]==full[0][arm][:n],'Ordered prefix '+arm)
        exact_array(part[1][arm],full[1][arm][:n],'F prefix '+arm)
        exact_array(part[2][arm],full[2][arm][:n],'P prefix '+arm)


def independent_work(entry):
    import monthly_selector_scope268_independent as ind
    return sum_counts([entry['independent'][a]['work'] for a in ARMS],ind.WORK_FIELDS)


def entry_check(entry,n,width,*,cached=False):
    require(type(entry) is dict and set(entry)=={'prefix','recomputed_original_pre','reuses_main_verified_full','source_audit','independent','producer_counts'},'Exact preparation entry')
    integer(entry['prefix'],'Entry prefix',n)
    require(entry['recomputed_original_pre'] is (not cached) and type(entry['reuses_main_verified_full']) is bool,'Entry source mode')
    counts_check(entry['producer_counts'],PRODUCER_FIELDS,'Producer work')
    wanted=dict.fromkeys(PRODUCER_FIELDS,0);wanted.update(original_pre_score_calls=int(not cached),original_selector_calls=1,experimental_selector_calls=2,selector_events=3*n)
    require(entry['producer_counts']==wanted,'Actual selector calls')
    require(set(entry['independent'])==set(ARMS),'Independent three roles')
    for a in ARMS:receipt(entry['independent'][a],a,n,width)
    source=entry['source_audit'];require(source.get('schema')=='selector_scope268_source_audit_v1' and source['events']==n and source['width']==width and source['cached_original_pre'] is cached,'Source audit scope')
    require(source['original_pre_score_calls']==int(not cached),'Source P work')
    for flag in ('original_pre_scores_exact','original_eligibility_exact','original_scores_exact','original_orders_exact','original_gate_before_candidates'):require(source[flag] is True,'Source flag '+flag)
    for flag in ('source_files_authenticated','new_features_computed','new_labels_computed','new_price_windows_read'):require(source[flag] is False,'Pure source scope '+flag)
    integer(source['new_model_fits'],'Source zero fits',0)
    counts_check(source['source_integrity_work'],INTEGRITY_FIELDS,'Source integrity work')
    integ=source['source_integrity_work'];require(integ['checks']==4 and integ['reference_values_compared']==12*n*width and integ['reference_order_events_compared']==4*n and integ['metadata_events_compared']==4*n,'Source integrity performed four times')
    return entry


def execution_accounting(evidence):
    import monthly_selector_scope268_independent as ind
    entries=[evidence['main']]+evidence['cached_prefixes']+[x for x in evidence['physical_prefixes'] if not x['reuses_main_verified_full']]
    producer=sum_counts([x['producer_counts'] for x in entries],PRODUCER_FIELDS)
    independent=sum_counts([independent_work(x) for x in entries],ind.WORK_FIELDS)
    integrity=sum_counts([x['source_audit']['source_integrity_work'] for x in entries],INTEGRITY_FIELDS)
    return dict(actual_prepare_entries=len(entries),producer=producer,independent=independent,source_integrity_work=integrity,new_model_fits=0,
        physical_full_reuses_main=True,work_definition='Logical original P/selector calls; independent work from executed receipts. Four prefix integrity checks per executed prepare entry. Full file hashes, full decoded source/reference copies and legacy full122 cached rank checks occur at loading separately; no zero-read claim.')


def preparation_evidence(evidence,audit,source_audit,*,total=122,width=49,prefixes=PREFIXES):
    require(evidence.get('schema')=='selector_scope268_independent_prepare_v1','Preparation evidence schema')
    main=evidence['main'];entry_check(main,total,width)
    require(main['reuses_main_verified_full'] is False,'Main actual work')
    cached=evidence['cached_prefixes'];physical=evidence['physical_prefixes']
    require(len(cached)==total+1 and [x['prefix'] for x in cached]==list(range(total+1)),'All decision prefixes')
    for x in cached:entry_check(x,x['prefix'],width,cached=True);require(x['reuses_main_verified_full'] is False,'Cached actually executed')
    require([x['prefix'] for x in physical]==list(prefixes),'Physical prefix coverage')
    for x in physical:
        entry_check(x,x['prefix'],width)
        if x['prefix']==total:
            expected=deepcopy(main);expected['reuses_main_verified_full']=True
            require(x==expected,'Full physical aliases identical main evidence')
        else:require(x['reuses_main_verified_full'] is False,'Physical prefix actually executed')
    for flag in ('all123_order_prefixes','source_arrays_unchanged','original_reference_unchanged'):require(evidence[flag] is True,'Preparation scope '+flag)
    require(evidence['execution_accounting']==execution_accounting(evidence),'Actual execution ledger')
    require(audit['schema']=='selector_scope268_decision_audit_v1' and audit['events']==total and audit['width']==width and set(audit['arms'])==set(ARMS) and audit['producer_counts']==main['producer_counts'],'Decision audit scope')
    require(audit['arms'][ARMS[0]]==dict(source='frozen_original_select_boundary',selector_calls=1,events=total),'Original decision audit')
    for mode in ARMS[1:]:candidate_audit(audit['arms'][mode],mode,total,width)
    require(source_audit==dict(main=main['source_audit'],physical=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in physical],cached=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in cached]),'Saved source ledger matches receipts')
def load_sources(expected):
    """Actual numeric loading occurs only after immutable PLAN/source verification."""
    plan=gate(expected);study=old.local();study.verify_source();meta,arrays,px,opening=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 cache')
    previous=read(PRIOR/'prepared/COMPLETION.json');require(previous['status']=='COMPLETE' and previous['events']==122,'Complete predecessor source')
    for name in ('META.json','SCORES.npz','PRE_SCORES.npz','ORDERS.json'):require(previous['artifacts_sha256'][name]==sha(PRIOR/'prepared'/name),'Predecessor prepared member')
    require(read(PRIOR/'prepared/META.json')==meta,'Predecessor full monthly metadata')
    reference=dict(meta=meta,orders=read(PRIOR/'prepared/ORDERS.json')[ARMS[0]],scores=ev.load_arrays(PRIOR/'prepared/SCORES.npz')[ARMS[0]],pre_scores=ev.load_arrays(PRIOR/'prepared/PRE_SCORES.npz')[ARMS[0]],eligible=arrays['eligible'].copy())
    require(reference['orders']==read(LOCAL/'local_linear_ORDERS.json'),'Original Local67/predecessor choices')
    panel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'] and len(raw)==64,'Original64 panel source');old.verify(raw)
    execution=study.verify_panel_parity(panel,px,opening)
    binding=dict(plan_sha256=expected,prior_prepared_sha256=sha(PRIOR/'prepared/COMPLETION.json'),source_closure=plan['original169_source_closure'])
    authority=_SourceAuthority(reference,arrays,binding,_SOURCE_TOKEN);_ISSUED.append(authority)
    return plan,study,arrays,meta,reference,authority,execution


def source_evidence(value,plan):
    for field in ('original_arrays_unchanged','original_orders_exact','original_scores_exact','original_pre_scores_exact','original_eligibility_exact','original_gate_before_candidates','whole_original_cache_decoded'):
        require(value.get(field) is True,'Source preservation '+field)
    require(value['original169_source_closure']==plan['original169_source_closure'] and value['raw_source_hashes']==plan['raw_source_hashes'],'Complete source binding')
    for field,path in [('original_training_hash',REBUILD/'TRAIN_INPUTS.npz'),('original_identity_hash',REBUILD/'MONTHLY_META.json'),('saved169_parity_hash',REBUILD/'PARITY.json'),('prior_prepared_sha256',PRIOR/'prepared/COMPLETION.json')]:require(value[field]==sha(path),'Source member '+field)
    require(value['calendar_sha256']==plan['calendar_sha256'],'Original calendar binding')
    for field in ('training_arrays_decoded','new_features_computed','new_labels_computed','new_price_windows_read','strict_pit'):require(value[field] is False,'Source read scope '+field)
    integer(value['new_model_fits'],'Source zero fits',0)
    fps=value['original_array_fingerprints'];require(type(fps) is dict and len(fps)==27,'Complete original array fingerprints')
    for name,item in fps.items():
        require(set(item)=={'dtype','shape','sha256'} and item['shape'][0]==122,'Original cache axis '+name);digest_string(item['sha256'],'Original array '+name)


def prepare(expected):
    require(not (OUT/'prepared').exists(),'Preparation write-once')
    plan,study,arrays,meta,reference,authority,execution=load_sources(expected)
    ab=old.array_snapshot(arrays);rb=structured_digest(reference)
    wanted,main,cache=prepare_one(study,arrays,meta,reference,122,authority=authority)
    cached=[];physical=[]
    for n in range(123):
        got,ce,_=prepare_one(study,arrays,meta,reference,n,authority=authority,cached_pre=cache)
        exact_prefix(wanted,got,n);cached.append(ce)
        if n in PREFIXES and n!=122:
            print(json.dumps(dict(stage='268_fresh_physical_prefix',events=n)),flush=True)
            got,pe,_=prepare_one(study,arrays,meta,reference,n,authority=authority)
            exact_prefix(wanted,got,n);physical.append(pe)
    reuse=deepcopy(main);reuse['reuses_main_verified_full']=True;physical.append(reuse)
    require(ab==old.array_snapshot(arrays) and rb==structured_digest(reference),'All original source/reference bytes unchanged')
    evidence=dict(schema='selector_scope268_independent_prepare_v1',main=main,cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,source_arrays_unchanged=True,original_reference_unchanged=True)
    evidence['execution_accounting']=execution_accounting(evidence)
    orders,scores,pres,audit=wanted
    sa=dict(main=main['source_audit'],physical=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in physical],cached=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in cached])
    parity=dict(original169_source_closure=plan['original169_source_closure'],raw_source_hashes=plan['raw_source_hashes'],original_training_hash=sha(REBUILD/'TRAIN_INPUTS.npz'),original_identity_hash=sha(REBUILD/'MONTHLY_META.json'),saved169_parity_hash=sha(REBUILD/'PARITY.json'),prior_prepared_sha256=sha(PRIOR/'prepared/COMPLETION.json'),calendar_sha256=plan['calendar_sha256'],execution_price_parity=execution,original_array_fingerprints=old.array_fingerprints(arrays),whole_original_cache_decoded=True,training_arrays_decoded=False,new_features_computed=False,new_labels_computed=False,new_price_windows_read=False,new_model_fits=0,consumption_scope='Complete original27 cache and prior reference NPZ decoded once; legacy study.load rechecks full122 cached local forecast ranks. Raw64 panel/execution parity is rechecked at load without new features or labels; original169 training NPZ is hashed only, never decoded. One full decoded source/reference baseline copy plus before/after full fingerprints. Each prepare entry checks consumed prefix four times; physical means fresh original P computation on physical prefix copies, not whole-file redecodes; cached P is verified and reused. No claim of zero source reads.',original_arrays_unchanged=True,original_orders_exact=True,original_scores_exact=True,original_pre_scores_exact=True,original_eligibility_exact=True,original_gate_before_candidates=True,strict_pit=False)
    preparation_evidence(evidence,audit,sa);source_evidence(parity,plan);gate(expected)
    dest=OUT/'prepared';dest.mkdir()
    for name,values in (('SCORES',scores),('PRE_SCORES',pres)):np.savez_compressed(dest/(name+'.npz'),**values)
    for name,value in {'DECISION_AUDIT.json':audit,'SOURCE_AUDIT.json':sa,'SOURCE_PARITY.json':parity,'INDEPENDENT_PREPARE.json':evidence,'ORDERS.json':orders,'META.json':meta}.items():dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={name:sha(dest/name) for name in PREP_FILES},new_returns_computed=False,new_labels_computed=False,new_model_fits=0,execution_accounting_sha256=structured_digest(evidence['execution_accounting'])))
    print(json.dumps(dict(status='PREPARED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    plan=gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['events']==122 and c['new_returns_computed'] is False and c['new_labels_computed'] is False and type(c['new_model_fits']) is int and c['new_model_fits']==0 and set(c['artifacts_sha256'])==PREP_FILES,'Complete prepared scope')
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
