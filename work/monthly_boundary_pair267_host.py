"""267 prospective lifecycle; all actual work requires an immutable final PLAN.

The source adapter authenticates consumed caller values, while this host binds
complete files, original Local67 decisions, annual models and phase receipts.
Synthetic entry points accept only explicit caller fixtures; no import runs data.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse, hashlib, io, json, unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_boundary_pair267'
PRIOR=ROOT/'outputs/monthly_monotone_tree266'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CONTEXT=ROOT/'work/monthly_boundary_pair267_context.json'
CONTRACT=ROOT/'work/monthly_boundary_pair267_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_boundary_pair267_TESTS.json'
REVIEW=ROOT/'work/monthly_boundary_pair267_CODE_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_monotone_tree266_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_boundary_pair267_RESULT_REVIEW.json'
PHASE_TERMINALS=OUT/'PHASE_TERMINALS.json'
ARMS=('incumbent_local67','top5_all_pairs_control','original_p34_pair_candidate')
PREFIXES=(0,1,53,88,122)
PRIOR_COMPLETION_SHA='f4aa5ff70128b2e51c1c8b369f3235f6c7d553cefe61ef60641c4c2623413671'
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,selection_tolerance=0.,independent_numeric_system='Independent augmented SVD beta and query prediction; canonical Gram/rhs exact bytes with disclosed shared NumPy primitives; deployed production-beta effects exact')
code_names=tuple(name for stem in ('host','kernel','independent','source','admission') for name in ('monthly_boundary_pair267_'+stem+'.py','test_monthly_boundary_pair267_'+stem+'.py'))
SUPPLEMENTAL=('work/monthly_boundary_pair267_saved_reader.py','work/test_monthly_boundary_pair267_saved_reader.py')
PREP_FILES={'MODELS.json','EFFECTS.npz','SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json','DECISION_AUDIT.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json','SOURCE_AUDIT.json'}
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}
PRODUCER_FIELDS=('annual_cohort_builds','producer_ridge_solves','control_pair_rows','candidate_pair_rows','endpoint_return_evaluations','query_predictions','effect_adjustments','original_pre_score_calls','original_selector_calls','selector_events','original_local_fits','original_other_target_fits')
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
SOURCE_FILES169={'AUDIT.json','BASE_AUDIT.json','BASE_DATA.npz','DATA.npz','LABEL_AUDIT.json','META.json','MODEL_AUDIT.json','MONTHLY_META.json','ORDERS.json','TRAIN_INPUTS.npz','WEEKLY_INPUTS.npz','WEEKLY_META.json'}
SOURCE_CHAIN169={
 REBUILD.parent/'PLAN.json':'dd6cc6d21c486882423d0172742443814015c38cfd5e79e4442b81c259d58e82',
 REBUILD/'PARITY.json':'1e9b68d1abee96a89c935c9bbe6c0b92fee3cc856bf4021c8152abdd155fb555',
 REBUILD.parent/'evaluation/PLAN.json':'34bf25eec0d0c524becc003f51d7f7333159d32d2455c1648e408b1a74c81d28',
 REBUILD.parent/'evaluation/results/COMPLETION.json':'60bb9afd9cc69c24b362e65f3e3870ae354df13f9792ba93b3e0d57addedc860',
 REBUILD.parent/'independent_review/COMPLETION.json':'c5eaa80dbc580ad8af8a824a0dc3575f1d3e3020471b78b96137f0c5feb4f1f5'}


def source_closure_check(pins):
    """Verify the existing 169 chain and complete cache manifest; never decode NPZ."""
    for path,digest in SOURCE_CHAIN169.items():
        require(pins.get(str(path))==digest and sha(path)==digest,'Original169 chain pin:'+str(path))
    plan=read(REBUILD.parent/'PLAN.json');parity=read(REBUILD/'PARITY.json')
    ep=read(REBUILD.parent/'evaluation/PLAN.json');ec=read(REBUILD.parent/'evaluation/results/COMPLETION.json')
    ic=read(REBUILD.parent/'independent_review/COMPLETION.json')
    require(parity['passed'] is True and parity['arm']=='control' and parity['returns_computed'] is False,'Original169 control parity')
    require(parity['source_plan_sha256']==SOURCE_CHAIN169[REBUILD.parent/'PLAN.json'],'Original169 source plan link')
    artifacts=parity['artifacts_sha256'];require(set(artifacts)==SOURCE_FILES169,'Complete original169 twelve-file cache')
    for name,digest in artifacts.items():require(pins.get(str(REBUILD/name))==digest,'Original169 full cache pin:'+name)
    for name,digest in plan['input_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        require(pins.get(str(path))==digest,'Original169 source input closure:'+name)
    epins={str(Path(k) if Path(k).is_absolute() else ROOT/k):v for k,v in ep['input_sha256'].items()}
    for path in [REBUILD/name for name in SOURCE_FILES169]+[REBUILD/'PARITY.json',REBUILD.parent/'PLAN.json']:
        require(epins.get(str(path))==pins[str(path)],'Original169 evaluation source binding:'+str(path))
    require(ep['rebuild_plan_sha256']==parity['source_plan_sha256'],'Original169 evaluation plan link')
    require(ec['status']=='COMPLETE' and ec['evaluation_plan_sha256']==SOURCE_CHAIN169[REBUILD.parent/'evaluation/PLAN.json'] and ec['rebuild_plan_sha256']==parity['source_plan_sha256'],'Original169 evaluation completion link')
    require(ec['exact_control_rechecked'] is True and ec['source_hashes_verified_before_after'] is True,'Original169 verified source completion')
    require(ic['status']=='PASS' and ic['checked_paths']==6 and ic['checked_slices']==30 and ic['evaluation_completion_sha256']==SOURCE_CHAIN169[REBUILD.parent/'evaluation/results/COMPLETION.json'],'Original169 independent completion link')
    return dict(schema='monthly_boundary_pair267_original169_closure_v1',chain_sha256={str(k):v for k,v in SOURCE_CHAIN169.items()},artifacts_sha256=dict(artifacts),source_input_count=len(plan['input_sha256']),npz_decoded=False,source_cache_rebuilt=False)

def bind_source_closure(pins):
    for path,digest in SOURCE_CHAIN169.items():old.bind(pins,path,digest)
    for name,digest in read(REBUILD/'PARITY.json')['artifacts_sha256'].items():
        require(name in SOURCE_FILES169,'Unexpected original169 cache artifact');old.bind(pins,REBUILD/name,digest)
    for name,digest in read(REBUILD.parent/'PLAN.json')['input_sha256'].items():old.bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    return source_closure_check(pins)

def now():return datetime.now(timezone.utc).isoformat()

def files():return [ROOT/'work'/p for p in code_names]

def codes():return {str(p):sha(p) for p in files()}

def code_review():
    r=read(REVIEW)
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['code_sha256']==codes(),'Numerical code review')
    require(r['context_sha256']==sha(CONTEXT) and r['implementation_contract_sha256']==sha(CONTRACT) and r['actual_returns_computed'] is False and type(r['input_sha256']) is dict,'Prospective review scope')
    return r

def integer(value,label,expected=None):
    require(type(value) is int and value>=0,label+' nonnegative native int')
    if expected is not None:require(value==expected,label+' exact value')

def counts_check(value,fields,label):
    require(type(value) is dict and set(value)==set(fields),label+' exact fields')
    for key,item in value.items():integer(item,label+'/'+key)

def structured_digest(value):
    digest=hashlib.sha256()
    def walk(v):
        digest.update(type(v).__name__.encode()+b':')
        if isinstance(v,np.ndarray):
            digest.update(str(v.dtype).encode());digest.update(str(v.shape).encode());digest.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v):walk(k);walk(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:walk(x)
        else:digest.update(repr(v).encode())
    walk(value);return digest.hexdigest()

def exact_array(actual,expected,label):
    require(isinstance(actual,np.ndarray) and isinstance(expected,np.ndarray) and actual.shape==expected.shape and actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes(),label+' exact shape/dtype/bits')

def train_metrics(engine,equity,meta):
    """Attach explicit evidence for undefined Sharpe without serializing NaN/Inf."""
    answer={}
    for name,lo,hi in [('training',0,87),('early',0,52),('recent',53,87)]:
        part=equity.loc[meta['dates'][lo]:meta['ends'][hi]]
        scaled=part/part.iloc[0]*100000.
        result=engine.stats(scaled)
        sd=float(scaled.pct_change(fill_method=None).dropna().std(ddof=1))
        require(np.isfinite(sd) and sd>=0.,'Finite sample daily return SD:'+name)
        if sd==0.:
            require(result['sharpe'] is None,'ZeroSD Sharpe must be undefined')
            result.update(sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        else:
            require(result['sharpe'] is not None and np.isfinite(result['sharpe']),'Nonfinite nonzeroSD Sharpe')
            result.update(sharpe_status='defined',sharpe_reason=None)
        result['daily_return_sd']=sd
        answer[name]=result
    return answer

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

def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('boundary267_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5,7,9))]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in mods]);stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    require(result.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=result.testsRun,runtime=ev.runtime(),code_sha256=codes()))


def dependencies():
    """Explicit consumed helper sources, in addition to inherited complete pins."""
    names=('monthly_monotone_tree266_host.py','monthly_monotone_tree266_independent.py',
           'monthly_monotone_tree266_admission.py','monthly_monotone_tree266_saved_reader.py',
           'monthly_gain_concentration265_saved_reader.py','ptr169_results_independent.py',
           'ptr169_rebuild_evaluate.py','monthly_ownsector_gate238_host.py')
    return [ROOT/'work'/name for name in names]


def initial_state(state):
    require(len(state['completed'])==262 and sum(x['trials'] for x in state['completed'])==1297,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Immediate closed266 predecessor')


def predecessor_binding(state):
    context,_=contract_check();wanted=context['predecessor'];record=state['completed'][-1]['completion']
    require(Path(record['path'])==PRIOR/'COMPLETION.json' and record['sha256']==sha(PRIOR/'COMPLETION.json')==wanted['completion_sha256']==PRIOR_COMPLETION_SHA,'Predecessor central closure binding')
    prior=read(PRIOR/'COMPLETION.json');review=read(PRIOR_REVIEW)
    require(prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior['protected_files']==21444 and prior['plan_sha256']==sha(PRIOR/'PLAN.json') and prior['observers_modified'] is False,'Independently closed266 required')
    require(prior['result_review_sha256']==wanted['result_review_sha256']==sha(PRIOR_REVIEW),'Predecessor review binding')
    require(review['passed'] is True and review['unresolved_blockers']==[] and review['plan_sha256']==sha(PRIOR/'PLAN.json') and review['results_completion_sha256']==sha(PRIOR/'results/COMPLETION.json'),'Predecessor saved review')
    require(prior['target_reached'] is False and state['best_training_selected_candidate']['cagr']<.7,'Target already reached')


def registration_check(state,expected):
    initial_state(state)
    require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration absent')


def contract_check():
    require(CONTEXT.is_file() and CONTRACT.is_file(),'Final prospective papers required; drafts cannot register')
    c,d=read(CONTEXT),read(CONTRACT)
    for record in (c,d):
        require(record.get('frozen') is True and record.get('registration_authorized') is True and record.get('actual_data_started') is False,'Final prospective267 authorization')
        require(type(record.get('input_sha256')) is dict,'Explicit dependency map')
    require(c['arms']==list(ARMS) and c['registered_trials']==3 and c['only_promotable']==ARMS[2] and c['diagnostic_only']==ARMS[1],'Three fixed economic roles')
    require(type(c.get('actual_models_fitted')) is int and c['actual_models_fitted']==0 and c.get('actual_orders_computed') is False and c.get('actual_returns_computed') is False,'No actual267 work before registration')
    require(type(d.get('code_files')) is list and d['code_files']==['work/'+name for name in code_names],'Exact ordered formal10 inventory')
    require(type(d.get('supplemental_input_files')) is list and d['supplemental_input_files']==list(SUPPLEMENTAL),'Exact ordered reader/test inventory')
    for label,record in (('context',c),('contract',d)):
        for relative in ['work/'+name for name in code_names]+list(SUPPLEMENTAL):
            value=record['input_sha256'].get(relative);digest_string(value,label+' source pin '+relative)
            require((ROOT/relative).is_file() and value==sha(ROOT/relative),label+' current source binding '+relative)
    require(d['context_sha256']==sha(CONTEXT),'Contract context binding')
    require(c['predecessor']['directory']==PRIOR.name and c['predecessor']['completion_sha256']==PRIOR_COMPLETION_SHA,'Final266 predecessor')
    digest_string(c['predecessor'].get('result_review_sha256'),'Predecessor review')
    return c,d


def canonical_pins(pins):
    require(type(pins) is dict,'Pin map');out={}
    for name,digest in pins.items():
        require(type(name) is str and name and '..' not in Path(name).parts,'Canonical pin path')
        p=Path(name);p=p if p.is_absolute() else ROOT/p;key=str(p.resolve())
        require(p.resolve()!=CENTRAL.resolve(),'Mutable central pin forbidden');digest_string(digest,'Pin digest')
        require(key not in out or out[key]==digest,'Conflicting canonical pin alias')
        out[key]=digest
    old.verify(out);return out


def register():
    c,d=contract_check();require(not OUT.exists(),'Fresh experiment directory')
    before=CENTRAL.read_bytes();state=json.loads(before);initial_state(state);require(state['registered_queue']==[],'No queued research');predecessor_binding(state)
    t=read(TESTS);require(t['passed'] is True and t['synthetic_only'] is True and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Current formal10 synthetic tests')
    review=code_review();pins=canonical_pins(read(PRIOR/'PLAN.json')['input_sha256']);closure=bind_source_closure(pins)
    study=old.local()
    paths=files()+[ROOT/p for p in SUPPLEMENTAL]+dependencies()+list(study.implementations())+list(study.source_files())
    paths += [CONTEXT,CONTRACT,TESTS,REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',PRIOR_REVIEW,old.PROTECTED,LOCAL/'local_linear_ORDERS.json']
    paths += [PRIOR/'prepared'/name for name in ('COMPLETION.json','SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json')]
    import monthly_boundary_pair267_source as source
    paths.append(ROOT/source.CALENDAR_PATH)
    for path in paths:old.bind(pins,Path(path))
    for paper in (c,d,review):
        for name,digest in canonical_pins(paper['input_sha256']).items():old.bind(pins,Path(name),digest)
    require(sha(ROOT/source.CALENDAR_PATH)==source.CALENDAR_SHA256,'Fixed official calendar')
    raw=read(LOCAL/'PLAN.json')['raw_source_hashes'];require(len(raw)==64,'Original64 source map')
    for name,digest in raw.items():old.bind(pins,Path(name),digest)
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    dump(OUT/'PLAN.json',dict(schema='monthly_boundary_pair267_v1',created_utc=now(),context=c,context_sha256=sha(CONTEXT),implementation_contract=d,implementation_contract_sha256=sha(CONTRACT),runtime=ev.runtime(),input_sha256=pins,original169_source_closure=closure,raw_source_hashes=raw,calendar_sha256=source.CALENDAR_SHA256,prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,numerical_verification=NUMERICAL_VERIFICATION,actual_labels_before_registration=0,actual_predictions_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0))
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),last_goal_turn_classification='progress',next_step='Original Local67 gate; shared mature original TopK cohort; fixed pair Ridge; immutable first88 dual-reference selection.')
    old.commit_state(before,state,'boundary_pair267_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN')
    p=read(OUT/'PLAN.json');c,d=contract_check()
    require(p['context']==c and p['context_sha256']==sha(CONTEXT) and p['implementation_contract']==d and p['implementation_contract_sha256']==sha(CONTRACT) and p['runtime']==ev.runtime(),'Prospective runtime/papers unchanged')
    require(p['numerical_verification']==NUMERICAL_VERIFICATION,'Frozen numerical rule')
    old.verify(p['input_sha256']);require(p['original169_source_closure']==source_closure_check(p['input_sha256']),'Complete169 closure')
    code_review();require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Prestate bytes changed')
    state=read(CENTRAL);registration_check(state,expected);predecessor_binding(state);return p


def encode_models(value):
    """JSON float64 arrays retain exact binary64, shape and dtype; no coercion."""
    if isinstance(value,np.ndarray):
        require(value.dtype==np.dtype('float64') and np.isfinite(value).all(),'Finite float64 model array')
        return dict(__ndarray__=True,dtype=value.dtype.str,shape=list(value.shape),values_hex=[float(x).hex() for x in value.flat])
    if type(value) is dict:return {k:encode_models(v) for k,v in value.items()}
    if type(value) is list:return [encode_models(v) for v in value]
    require(value is None or type(value) in (str,int,bool,float),'Supported model evidence type')
    if type(value) is float:require(np.isfinite(value),'Finite model scalar')
    return value


def receipt(value,kind,*,year=None,n=None,width=None):
    import monthly_boundary_pair267_independent as ind
    flags={'models':ind.MODEL_FLAGS,'adjustment':ind.ADJUSTMENT_FLAGS,'selection':ind.SELECTION_FLAGS}[kind]
    require(type(value) is dict and value.get('schema')=='monthly_boundary_pair267_independent_'+kind+'_v1' and value.get('passed') is True,'Real independent '+kind+' receipt')
    for flag in flags:require(value.get(flag) is True,'Independent flag '+flag)
    counts_check(value.get('work'),ind.WORK_FIELDS,'Independent actual work')
    require(value.get('actual_coverage_claim') is False,'Caller proof is not actual coverage admission')
    if year is not None:integer(value['query_year'],'Proof year',year)
    if n is not None:integer(value['events'],'Proof events',n)
    if width is not None:integer(value['width'],'Proof width',width)
    if kind=='models':require(value['continuous_atol']==value['continuous_rtol']==2e-10 and value['exact_cohort_bits'] is True,'Model continuous-only tolerance')
    if kind=='selection':require(value['numerical_tolerance']==0. and value['original_episode_calls']==1 and value['new_model_fits']==0,'Exact single selector proof')
    return value


def original_selector_source(arrays,n):
    fields=('baseline','core','core_weight','state_calm10','state_weak_narrow')
    result={k:arrays[k][:n].copy() for k in fields}
    cw=result['core_weight'];calm=result['state_calm10']
    require(calm.dtype==np.dtype(bool) and result['state_weak_narrow'].dtype==np.dtype(bool),'Original state flags')
    wanted=np.where(calm,.5,.52)
    if cw.ndim==2:wanted=np.broadcast_to(wanted[:,None],cw.shape)
    exact_array(cw,wanted,'Original .50/.52 core allocation')
    return result


def model_counts(bundle):
    out=dict.fromkeys(PRODUCER_FIELDS,0);cohort=bundle['cohort']
    out['annual_cohort_builds']=1;out['producer_ridge_solves']=bundle['actual_solve_calls']
    out['control_pair_rows']=len(cohort['arms'][ARMS[1]]['y']);out['candidate_pair_rows']=len(cohort['arms'][ARMS[2]]['y'])
    for row in cohort['months']:
        if row['status']=='common_valid_month':out['endpoint_return_evaluations']+=len(row['selected_indices'])
    return out


def sum_counts(records,fields):
    total=dict.fromkeys(fields,0)
    for record in records:
        counts_check(record,fields,'Execution work')
        for name in fields:total[name]+=record[name]
    return total


def expected_identity(reference,meta,n):
    axes=('dates','ends','symbols','etfs')
    require(old.meta_prefix({k:reference['meta'][k] for k in axes},n)==old.meta_prefix({k:meta[k] for k in axes},n),'Original reference identities')
    return reference['orders'][:n],reference['scores'][:n],reference['pre_scores'][:n]


def exact_original(reference,meta,n,orders,scores,P):
    oo,ff,pp=expected_identity(reference,meta,n);require(orders==oo,'Original ordered choices before new labels')
    exact_array(scores,ff,'Prior original F');exact_array(P,pp,'Prior original P')


def partial_audit(event,arm):
    return dict(signal=event['signal'],arm=arm,status='initial_partial',pair_indices=[],predicted_return_gap=None,q=0.,changed_indices=[],original_selector_calls=0,original_pre_scores_unchanged=True,new_global_score_ranks_computed=False,winner_selected=False)


_SOURCE_TOKEN=object()
_ISSUED=[]
class _SourceAuthority:
    """Current-process source identity; not protection from hostile Python code."""
    __slots__=('reference','baseline','binding')
    def __init__(self,reference,binding,token):
        require(token is _SOURCE_TOKEN,'Source authority issuer')
        object.__setattr__(self,'reference',reference);object.__setattr__(self,'baseline',deepcopy(reference));object.__setattr__(self,'binding',deepcopy(binding))
    def __setattr__(self,*args):raise TypeError('Immutable source authority')


def authority_check(authority,reference,meta,n):
    require(type(authority) is _SourceAuthority and any(authority is x for x in _ISSUED) and authority.reference is reference,'Host-issued verified original reference required')
    require(structured_digest(expected_identity(reference,meta,n))==structured_digest(expected_identity(authority.baseline,meta,n)),'Original reference mutated')
    exact_array(reference['eligible'][:n],authority.baseline['eligible'][:n],'Original reference E unchanged')


def prepare_one(study,join,train,arrays,panel,reference,n,*,authority,cached_source=None,cached_models=None):
    """Original gate, fresh/cached source, model gate, exact effect/selector gates."""
    import monthly_boundary_pair267_source as source
    import monthly_boundary_pair267_kernel as kernel
    import monthly_boundary_pair267_independent as ind
    total,width=join['events'],join['width'];integer(n,'Prefix');require(n<=total,'Prefix within events')
    meta=join['monthly'];authority_check(authority,reference,meta,n)
    before=old.array_snapshot(old.cache_prefix(arrays,n))
    P,E,original_proof=source.original_prefix(study,arrays,meta,reference,n)
    require(original_proof['passed'] is True and original_proof['full_local67_p_exact'] is True and original_proof['original_eligibility_exact'] is True,'Original source gate')
    mm=old.meta_prefix(meta,n);ss=original_selector_source(arrays,n);effects=arrays['effects'][:n].copy()
    choices,F,selector=kernel.select_once(study.ep.select_boundary,P,E,mm['symbols'],effects,ss)
    orders=study.rp.make_orders(choices,F,mm)
    exact_original(reference,meta,n,orders,F,P)
    early=receipt(ind.verify_selection(P,E,mm['symbols'],effects,ss,choices,F),'selection',n=n,width=width)
    authority_check(authority,reference,meta,n)
    require(before==old.array_snapshot(old.cache_prefix(arrays,n)),'Original gate mutated source')
    prepared=source.prepare_inputs(join,train,arrays,P,panel,n) if cached_source is None else source.project_inputs(cached_source,n)
    # Reconstruct source P/E against the independently verified original prefix.
    for i,event in enumerate(prepared['events']):
        exact_array(event['pre_scores'],P[i],'Prepared original P');exact_array(event['eligible'],E[i],'Prepared original E');exact_array(event['effects'],effects[i],'Original effects')
    integrity=dict(checks=0,counts=dict.fromkeys(source.WORK_FIELDS,0))
    def check_inputs():
        check=source.project_inputs(prepared,n)
        integrity['checks']+=1;integrity['counts']=sum_counts([integrity['counts'],check['audit']['counts']],source.WORK_FIELDS)
    models={};proofs={};bindings={};proof_bindings={};work=dict.fromkeys(PRODUCER_FIELDS,0);work['original_pre_score_calls']=1
    work['original_selector_calls']=1;work['selector_events']=n
    years=prepared['audit']['consumed_years']
    for year in years:
        key=str(year)
        if cached_models is None:
            b=kernel.fit_models(prepared['months'],year);snap=structured_digest(b)
            proof=receipt(ind.verify_models(prepared['months'],year,b),'models',year=year)
            require(structured_digest(b)==snap,'Verifier mutated production model')
            check_inputs();models[key]=b;proofs[key]=proof
            work=sum_counts([work,model_counts(b)],PRODUCER_FIELDS)
        else:
            b=cached_models['annual'][key];proof=cached_models['proofs'][key]
            require(structured_digest(b)==cached_models['bindings'][key],'Cached consumed annual model changed')
            require(structured_digest(proof)==cached_models['proof_bindings'][key],'Cached consumed model proof changed')
            receipt(proof,'models',year=year);models[key]=b
        for arm in ARMS[1:]:require(ind.model_fingerprint(b['models'][arm])==proof['model_sha256'][arm],'Verified model identity before deployment')
        bindings[key]=structured_digest(b);proof_bindings[key]=structured_digest(proof)
    result_orders={ARMS[0]:orders};scores={ARMS[0]:F};pres={a:P.copy() for a in ARMS};effect_maps={ARMS[0]:effects.copy()}
    selections={ARMS[0]:early};adjustments={a:[] for a in ARMS[1:]};audit_arms={ARMS[0]:dict(events=[dict(signal=x,status='incumbent_identity') for x in mm['dates']],selector=selector)}
    for arm in ARMS[1:]:
        deployed=effects.copy();events=[]
        for i,event in enumerate(prepared['events']):
            if event['signal']==kernel.INITIAL_PARTIAL:
                events.append(partial_audit(event,arm));continue
            key=event['signal'][:4];model=models[key]['models'][arm]
            proof=(proofs if cached_models is None else cached_models['proofs'])[key]
            require(ind.model_fingerprint(model)==proof['model_sha256'][arm],'Model unchanged before each deployment')
            adjusted,audit=kernel.adjust_effects(event,model)
            check=receipt(ind.verify_adjustment(event,model,adjusted,audit,model_verification=proof),'adjustment')
            require(ind.model_fingerprint(model)==proof['model_sha256'][arm],'Adjustment verifier mutated model')
            require(structured_digest(proof)==proof_bindings[key],'Adjustment verifier mutated model proof')
            if audit['predicted_return_gap'] is not None:
                require(check.get('independent_svd_prediction_checked') is True,'Active query independent SVD prediction required')
                work['query_predictions']+=1
            work['effect_adjustments']+=1;deployed[i]=adjusted;events.append(audit);adjustments[arm].append(check)
        cc,ff,sel=kernel.select_once(study.ep.select_boundary,P,E,mm['symbols'],deployed,ss)
        selections[arm]=receipt(ind.verify_selection(P,E,mm['symbols'],deployed,ss,cc,ff),'selection',n=n,width=width)
        result_orders[arm]=study.rp.make_orders(cc,ff,mm);scores[arm]=ff;effect_maps[arm]=deployed
        work['original_selector_calls']+=1;work['selector_events']+=n
        audit_arms[arm]=dict(events=events,selector=sel)
        if deployed.tobytes()==effects.tobytes():
            require(result_orders[arm]==orders,'Identity effects preserve original orders');exact_array(ff,F,'Identity effects preserve F')
    check_inputs();authority_check(authority,reference,meta,n)
    require(before==old.array_snapshot(old.cache_prefix(arrays,n)),'Prefix source mutation')
    for key,b in models.items():require(structured_digest(b)==bindings[key],'Annual model mutated after decisions')
    evidence=dict(prefix=n,model_recomputed=cached_models is None,reuses_main_verified_full=False,original_source=original_proof,original_selection=early,source_audit=prepared['audit'],source_integrity_work=integrity,model_proofs=proofs,model_bindings=bindings,adjustment_proofs=adjustments,selection_proofs=selections,producer_counts=work)
    audit=dict(schema='monthly_boundary_pair267_decision_audit_v1',events=n,width=width,arms=audit_arms,counts=work)
    package=dict(annual=models,proofs=proofs if cached_models is None else {k:cached_models['proofs'][k] for k in models},bindings=bindings,proof_bindings=proof_bindings)
    return package,(result_orders,scores,pres,effect_maps,audit),evidence,prepared
def exact_prefix(full,part,n):
    require(set(full[0])==set(part[0])==set(ARMS),'Prefix arm scope')
    for arm in ARMS:
        require(part[0][arm]==full[0][arm][:n],'Exact ordered prefix')
        for axis,label in ((1,'F'),(2,'P'),(3,'effects')):exact_array(part[axis][arm],full[axis][arm][:n],label+' prefix')


def annual_prefix(full,part):
    for year,b in part['annual'].items():require(structured_digest(b)==structured_digest(full['annual'][year]),'Fresh annual exact cohort/model prefix')


def independent_work(entry):
    import monthly_boundary_pair267_independent as ind
    # original_selection aliases the incumbent in selection_proofs, not new work.
    receipts=list(entry['model_proofs'].values())+list(entry['selection_proofs'].values())
    receipts += [p for values in entry['adjustment_proofs'].values() for p in values]
    return sum_counts([p['work'] for p in receipts],ind.WORK_FIELDS)


def execution_accounting(e):
    import monthly_boundary_pair267_independent as ind
    import monthly_boundary_pair267_source as source
    groups={'main':[e['main']],'cached':e['cached_prefixes'],'physical_fresh':[p for p in e['physical_prefixes'] if not p['reuses_main_verified_full']]}
    out={}
    for name,entries in groups.items():
        out[name]=dict(executions=len(entries),producer=sum_counts([x['producer_counts'] for x in entries],PRODUCER_FIELDS),independent=sum_counts([independent_work(x) for x in entries],ind.WORK_FIELDS),source=sum_counts([c for x in entries for c in (x['source_audit']['counts'],x['source_integrity_work']['counts'])],source.WORK_FIELDS))
    out['total']={kind:sum_counts([out[g][kind] for g in groups],fields) for kind,fields in [('producer',PRODUCER_FIELDS),('independent',ind.WORK_FIELDS),('source',source.WORK_FIELDS)]}
    out['physical122_reuses_main']=True
    out['count_definition']='Routine and consumed-value counters, not NumPy/hash primitives; original_selection aliases incumbent selection proof; physical122 aliases main; cached models and SVD fits zero; host complete-file hashes/full decode and final source fingerprints separate; original P/E comparison cells reported in original_source outside adapter13 units.'
    return out


def entry_check(entry,n,width,*,cached=False):
    import monthly_boundary_pair267_independent as ind
    import monthly_boundary_pair267_source as source
    require(type(entry) is dict and set(entry)=={'prefix','model_recomputed','reuses_main_verified_full','original_source','original_selection','source_audit','source_integrity_work','model_proofs','model_bindings','adjustment_proofs','selection_proofs','producer_counts'},'Complete prefix evidence')
    integer(entry['prefix'],'Prefix evidence',n);counts_check(entry['producer_counts'],PRODUCER_FIELDS,'Producer actual work')
    c=entry['producer_counts'];require(c['original_local_fits']==c['original_other_target_fits']==0 and c['original_selector_calls']==3 and c['selector_events']==3*n and c['original_pre_score_calls']==1,'Fixed original cache/selector work')
    op=entry['original_source'];require(op['passed'] is True and op['events']==n and op['width']==width and op['full_local67_p_exact'] is True and op['original_eligibility_exact'] is True and op['source_arrays_unchanged'] is True and op['model_fits']==0 and op['source_numeric_cells_compared']==2*n*width,'Full original P/E proof')
    require(set(entry['selection_proofs'])==set(ARMS) and entry['original_selection']==entry['selection_proofs'][ARMS[0]],'Incumbent proof alias')
    for value in entry['selection_proofs'].values():receipt(value,'selection',n=n,width=width)
    require(set(entry['adjustment_proofs'])==set(ARMS[1:]),'Experimental adjustment arms')
    for values in entry['adjustment_proofs'].values():
        for value in values:receipt(value,'adjustment')
    sa=entry['source_audit'];require(sa['events']==n and sa['width']==width and sa['labels_computed'] is False and sa['models_fitted']==0 and sa['orders_computed'] is False and sa['canonical_label_price_source']=="panel['adj']",'Price-only source audit')
    counts_check(sa['counts'],source.WORK_FIELDS,'Consumed source work')
    integrity=entry['source_integrity_work'];require(type(integrity) is dict and set(integrity)=={'checks','counts'},'Explicit source integrity work')
    counts_check(integrity['counts'],source.WORK_FIELDS,'Source integrity work')
    checks=1 if cached else len(entry['model_proofs'])+1
    integer(integrity['checks'],'Source integrity checks',checks)
    expected_integrity=dict.fromkeys(source.WORK_FIELDS,0);expected_integrity.update(projected_event_records_checked=checks*n,projected_month_records_checked=checks*len(sa['consumed_month_indices']))
    require(integrity['counts']==expected_integrity,'Actual source integrity projection counts')
    require(set(entry['model_bindings'])=={str(y) for y in sa['consumed_years']},'Consumed annual model binding scope')
    for digest in entry['model_bindings'].values():digest_string(digest,'Annual model binding')
    if cached:
        require(entry['model_recomputed'] is False and entry['model_proofs']=={} and c['producer_ridge_solves']==c['annual_cohort_builds']==c['endpoint_return_evaluations']==c['control_pair_rows']==c['candidate_pair_rows']==0,'Cached no labels/cohorts/fits')
        require(sa['mode']=='cached_projection','Cached source projection')
    else:
        require(set(entry['model_proofs'])==set(entry['model_bindings']),'All consumed year model proofs')
        for key,value in entry['model_proofs'].items():receipt(value,'models',year=int(key))
        require(c['annual_cohort_builds']==len(entry['model_proofs']) and c['producer_ridge_solves']==2*sum(p['status']=='ready' for p in entry['model_proofs'].values()),'Two ready-year fits; zero cold')
    if n<=1:require(not entry['model_bindings'] and c['producer_ridge_solves']==0 and c['effect_adjustments']==0,'No partial/empty annual numeric work')


def preparation_evidence(e,models,audit,source_audit,*,total=122,width=49,prefixes=PREFIXES):
    require(set(e)=={'schema','main','cached_prefixes','physical_prefixes','all123_order_prefixes','source_arrays_unchanged','source_training_unchanged','source_panel_unchanged','original_reference_unchanged','fit_execution_accounting'},'Complete preparation evidence')
    require(e['schema']=='monthly_boundary_pair267_independent_prepare_v1','Prepare schema')
    entry_check(e['main'],total,width)
    require([x['prefix'] for x in e['cached_prefixes']]==list(range(total+1)),'Every cached prefix')
    for item in e['cached_prefixes']:entry_check(item,item['prefix'],width,cached=True)
    require([x['prefix'] for x in e['physical_prefixes']]==list(prefixes),'All fixed physical prefixes')
    for item in e['physical_prefixes']:
        entry_check(item,item['prefix'],width)
        if item['prefix']==total:
            expected=deepcopy(e['main']);expected.update(reuses_main_verified_full=True,model_recomputed=False)
            require(item==expected,'Full physical evidence aliases verified main')
        else:require(item['model_recomputed'] is True and item['reuses_main_verified_full'] is False,'Fresh physical source/models')
    for field in ('all123_order_prefixes','source_arrays_unchanged','source_training_unchanged','source_panel_unchanged','original_reference_unchanged'):require(e[field] is True,'Preparation '+field)
    require(e['fit_execution_accounting']==execution_accounting(e),'Actual work receipt sum')
    require(models['schema']=='monthly_boundary_pair267_saved_models_v1' and models['annual_years']==e['main']['source_audit']['consumed_years'],'Annual model inventory')
    require([x['query_year'] for x in models['annual']]==models['annual_years'] and models['producer_counts']==e['main']['producer_counts'],'Model year/count binding')
    require(audit['schema']=='monthly_boundary_pair267_decision_audit_v1' and audit['events']==total and audit['width']==width and set(audit['arms'])==set(ARMS) and audit['counts']==e['main']['producer_counts'],'Full decision audit')
    for arm,row in audit['arms'].items():require(len(row['events'])==total and row['selector']['original_selector_calls']==1,'Full decision events/single original selector')
    require(source_audit['main']==e['main']['source_audit'],'Main source audit binding')
    require(source_audit['physical']==[dict(prefix=x['prefix'],audit=x['source_audit']) for x in e['physical_prefixes']] and source_audit['cached']==[dict(prefix=x['prefix'],audit=x['source_audit']) for x in e['cached_prefixes']],'All source prefix audits')


def load_sources(expected):
    """Only actual loading entry; verifies the PLAN before any numerical decode."""
    plan=gate(expected)
    import monthly_boundary_pair267_source as source
    study=old.local();study.verify_source();meta,arrays,px,opening=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 cache')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');combined=read(REBUILD/'MONTHLY_META.json')
    require(combined['monthly']==meta and combined['training']==read(study.arch.P/'META.json')['training'],'Original shared identities')
    prior=read(PRIOR/'prepared/COMPLETION.json');require(prior['status']=='COMPLETE' and prior['events']==122,'Complete predecessor source')
    for name in ('META.json','SCORES.npz','PRE_SCORES.npz','ORDERS.json'):require(prior['artifacts_sha256'][name]==sha(PRIOR/'prepared'/name),'Predecessor prepared member')
    require(read(PRIOR/'prepared/META.json')==meta,'Predecessor monthly identity')
    reference=dict(meta=meta,orders=read(PRIOR/'prepared/ORDERS.json')[ARMS[0]],scores=ev.load_arrays(PRIOR/'prepared/SCORES.npz')[ARMS[0]],pre_scores=ev.load_arrays(PRIOR/'prepared/PRE_SCORES.npz')[ARMS[0]],eligible=arrays['eligible'].copy())
    require(reference['orders']==read(LOCAL/'local_linear_ORDERS.json'),'Original Local67/predecessor choices')
    panel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'] and len(raw)==64,'Original64 panel sources');old.verify(raw)
    execution=study.verify_panel_parity(panel,px,opening)
    require(sha(ROOT/source.CALENDAR_PATH)==plan['calendar_sha256']==source.CALENDAR_SHA256,'Official calendar bytes')
    join=source.build_join(combined,read(ROOT/source.CALENDAR_PATH)['sessions'],expected_overlap=113)
    binding=dict(plan_sha256=expected,prior_prepared_sha256=sha(PRIOR/'prepared/COMPLETION.json'),source_closure=plan['original169_source_closure'])
    authority=_SourceAuthority(reference,binding,_SOURCE_TOKEN);_ISSUED.append(authority)
    return plan,study,join,train,arrays,{'adj':panel['adj']},reference,authority,execution


def panel_digest(panel):
    a=panel['adj'];return structured_digest(dict(index=[str(d) for d in a.index],columns=list(a.columns),values=a.to_numpy()))


def source_evidence(value,plan):
    for field in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact','original_scores_exact','original_pre_scores_exact','original_eligibility_exact','original_gate_before_new_labels','independent_model_before_deployment','whole_files_decoded'):
        require(value.get(field) is True,'Source preservation '+field)
    require(value['original169_source_closure']==plan['original169_source_closure'] and value['raw_source_hashes']==plan['raw_source_hashes'],'Complete source closure binding')
    for field,path in [('original_training_hash',REBUILD/'TRAIN_INPUTS.npz'),('original_identity_hash',REBUILD/'MONTHLY_META.json'),('saved169_parity_hash',REBUILD/'PARITY.json'),('prior_prepared_sha256',PRIOR/'prepared/COMPLETION.json')]:require(value[field]==sha(path),'Source member '+field)
    require(value['calendar_sha256']==plan['calendar_sha256'] and value['canonical_label_price_source']=="panel['adj']" and value['fresh_training_array_reconstruction'] is False and value['original_local_fits']==value['original_other_target_fits']==0,'Original source scope')


def prepare(expected):
    require(not (OUT/'prepared').exists(),'Preparation write-once')
    plan,study,join,train,arrays,panel,reference,authority,execution=load_sources(expected)
    ab=old.array_snapshot(arrays);tb=old.array_snapshot(train);pb=panel_digest(panel);rb=structured_digest(reference)
    package,wanted,main,source=prepare_one(study,join,train,arrays,panel,reference,122,authority=authority)
    cached=[];physical=[]
    for n in range(123):
        _,got,ce,_=prepare_one(study,join,train,arrays,panel,reference,n,authority=authority,cached_source=source,cached_models=package)
        exact_prefix(wanted,got,n);cached.append(ce)
        if n in PREFIXES and n!=122:
            print(json.dumps(dict(stage='267_fresh_physical_prefix',events=n)),flush=True)
            pp,gg,pe,_=prepare_one(study,join,train,arrays,panel,reference,n,authority=authority)
            exact_prefix(wanted,gg,n);annual_prefix(package,pp);physical.append(pe)
    reuse=deepcopy(main);reuse.update(model_recomputed=False,reuses_main_verified_full=True);physical.append(reuse)
    require(ab==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and pb==panel_digest(panel) and rb==structured_digest(reference),'All source/reference bytes unchanged')
    evidence=dict(schema='monthly_boundary_pair267_independent_prepare_v1',main=main,cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,source_arrays_unchanged=True,source_training_unchanged=True,source_panel_unchanged=True,original_reference_unchanged=True)
    evidence['fit_execution_accounting']=execution_accounting(evidence)
    orders,scores,pres,effects,audit=wanted
    models=dict(schema='monthly_boundary_pair267_saved_models_v1',annual_years=main['source_audit']['consumed_years'],annual=[encode_models(package['annual'][str(y)]) for y in main['source_audit']['consumed_years']],producer_counts=main['producer_counts'])
    sa=dict(join=join,main=main['source_audit'],physical=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in physical],cached=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in cached])
    parity=dict(original169_source_closure=plan['original169_source_closure'],raw_source_hashes=plan['raw_source_hashes'],original_training_hash=sha(REBUILD/'TRAIN_INPUTS.npz'),original_identity_hash=sha(REBUILD/'MONTHLY_META.json'),saved169_parity_hash=sha(REBUILD/'PARITY.json'),prior_prepared_sha256=sha(PRIOR/'prepared/COMPLETION.json'),calendar_sha256=plan['calendar_sha256'],execution_price_parity=execution,canonical_label_price_source="panel['adj']",fresh_training_array_reconstruction=False,cached_training_basis='Pinned169 rebuilt arrays; same-date plain10/E compared; no new original target reconstruction',whole_files_decoded=True,consumption_scope='Full NPZ/reference/panel decoded at admitted load; legacy study.load also rechecks original full122 cached local forecast ranks without new267 labels or fits. Full source fingerprints before/after are separate host work. Source-adapter counts only consumed values; physical prefixes fresh consumed unions, not whole-file redecodes; cached projections check issued per-record hashes.',original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True,original_scores_exact=True,original_pre_scores_exact=True,original_eligibility_exact=True,original_gate_before_new_labels=True,independent_model_before_deployment=True,original_local_fits=0,original_other_target_fits=0,strict_pit=False)
    preparation_evidence(evidence,models,audit,sa);source_evidence(parity,plan);gate(expected)
    dest=OUT/'prepared';dest.mkdir()
    for name,values in (('SCORES',scores),('PRE_SCORES',pres),('EFFECTS',effects)):np.savez_compressed(dest/(name+'.npz'),**values)
    values={'MODELS.json':models,'DECISION_AUDIT.json':audit,'SOURCE_AUDIT.json':sa,'SOURCE_PARITY.json':parity,'INDEPENDENT_PREPARE.json':evidence,'ORDERS.json':orders,'META.json':reference['meta']}
    for name,value in values.items():dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={name:sha(dest/name) for name in PREP_FILES},new_returns_computed=False,training_endpoint_returns_computed=True,model_fit_accounting='Actual producer annual normal solves and independent SVD; all123 cached fits0,physical122 aliases main; source/selector units separately summed'))
    print(json.dumps(dict(status='PREPARED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    plan=gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['events']==122 and c['new_returns_computed'] is False and c['training_endpoint_returns_computed'] is True and set(c['artifacts_sha256'])==PREP_FILES,'Complete prepared scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Prepared immutable member')
    preparation_evidence(*(read(OUT/'prepared'/name) for name in ('INDEPENDENT_PREPARE.json','MODELS.json','DECISION_AUDIT.json','SOURCE_AUDIT.json')))
    source_evidence(read(OUT/'prepared/SOURCE_PARITY.json'),plan)
    return read(OUT/'prepared/META.json'),read(OUT/'prepared/ORDERS.json')


def select_first88(metrics,orders):
    import monthly_boundary_pair267_admission as admission
    return admission.select_first88(metrics,orders)


def accounting():
    import monthly_monotone_tree266_independent as previous
    return previous.accounting()


def audit_outcomes(px,opening,orders,exports,results,meta):
    import monthly_monotone_tree266_independent as previous
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Three267 accounting roles')
    rename=lambda value:{oldarm:value[newarm] for newarm,oldarm in zip(ARMS,previous.ARMS)}
    proof=previous.audit_outcomes(px,opening,rename(orders),rename(exports),rename(results),meta)
    return dict(proof,paths={newarm:proof['paths'][oldarm] for newarm,oldarm in zip(ARMS,previous.ARMS)})


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
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=dict(strict_pit=False,reused_sample=True,global_novelty='UNKNOWN',conditional_original_boundary_pair_loss_only=True))
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),phase_terminals_sha256=sha(PHASE_TERMINALS),new_trials=3,model_audit_sha256=sha(OUT/'prepared/MODELS.json'),independent_prepare_sha256=sha(OUT/'prepared/INDEPENDENT_PREPARE.json'),protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==263 and sum(x['trials'] for x in state['completed'])==1300,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review fixed original-boundary pair loss; first88 dual-reference choice immutable and frozen observers unchanged.')
    old.commit_state(before,state,'boundary_pair267_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    elif args.phase=='prepare':require(bool(args.plan_sha256),'PLAN hash required');prepare_recorded(args.plan_sha256)
    else:require(bool(args.plan_sha256),'PLAN hash required');globals()[args.phase](args.plan_sha256)
