"""260: annual intercept/state leader utility, a fixed 10-point long-core blend.

No source/model/outcome work before registration. Only the state arm is promotable.
"""
from pathlib import Path
from datetime import date,datetime,timezone
import argparse,copy,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_leader_utility260'
PRIOR=ROOT/'outputs/monthly_joint_identity259'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CONTEXT=ROOT/'work/monthly_leader_utility260_context.json'
CONTRACT=ROOT/'work/monthly_leader_utility260_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_leader_utility260_TESTS.json'
REVIEW=ROOT/'work/monthly_leader_utility260_CODE_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_joint_identity259_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_leader_utility260_RESULT_REVIEW.json'
ARMS=('incumbent_local67','unconditional_leader_utility','state_leader_utility')
PHASE_TERMINALS=OUT/'PHASE_TERMINALS.json'
SOURCE_REVIEW=ROOT/'work/after259_router_inputs_review.json'
SOURCE_REVIEW_SHA='2716df22d74bd3ffcd4a0ad6092af158fc1a21142ca1659154421a7899a00d90'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
CONTEXT_SHA='5b0e79942aa9dbb002c5e97a33cad77d8a02726b978a8f5fd86364bfa4a89533'
CONTRACT_SHA='88ad58fbd783a574007fe17dcc8f842b0764425d9b8d2e0260bea0f64246a381'
PREFIXES=(0,1,53,88,122)
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.0,selection_tolerance=0.0,independent_numeric_system='augmented_least_squares_SVD_and_canonical_normal_equations')
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=tuple(name for stem in ('host','kernel','independent') for name in ('monthly_leader_utility260_'+stem+'.py','test_monthly_leader_utility260_'+stem+'.py'))
PREP_FILES={'FORECASTS.npz','MARKET_INPUTS.npz','SCORES.npz','PRE_SCORES.npz','MODEL_AUDIT.json','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()


def files():return [ROOT/'work'/p for p in code_names]


def codes():return {str(p):sha(p) for p in files()}


def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('leader_utility260_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in mods]);stream=io.StringIO()
    r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    require(r.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=r.testsRun,runtime=ev.runtime(),code_sha256=codes()))


def code_review():
    r=read(REVIEW)
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['code_sha256']==codes(),'Numerical code review')
    require(r['context_sha256']==sha(CONTEXT) and r['actual_returns_computed'] is False,'Prospective review scope')
    return r


def initial_state(state):
    require(len(state['completed'])==255 and sum(x['trials'] for x in state['completed'])==1276,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Predecessor changed')


def predecessor_binding(state):
    record=state['completed'][-1]['completion']
    require(Path(record['path'])==PRIOR/'COMPLETION.json' and record['sha256']==sha(PRIOR/'COMPLETION.json'),'Predecessor central closure binding')


def registration_check(state,expected):
    initial_state(state)
    require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration absent')


def contract_check():
    require(sha(CONTEXT)==CONTEXT_SHA and sha(CONTRACT)==CONTRACT_SHA,'Frozen prospective contracts')
    c=read(CONTEXT);d=read(CONTRACT)
    require(c['arms']==list(ARMS) and c['registered_trials']==3 and c['only_promotable']==ARMS[2] and c['diagnostic_only']==ARMS[1],'Three fixed arms; state only')
    require(d['context_sha256']==sha(CONTEXT) and d['actual_data_started'] is False,'Prospective contract binding')
    for key in ('actual_new_features_computed','actual_new_labels_computed','actual_new_models_fitted','actual_new_orders_or_returns_computed'):
        require(c[key] is False,'Prospective scope:'+key)
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Independent source interface review')
    return c,d

def register():
    require(not OUT.exists(),'Fresh experiment directory')
    before=CENTRAL.read_bytes();state=json.loads(before);initial_state(state);require(not state['registered_queue'],'No pending experiment')
    predecessor_binding(state);prior=read(PRIOR/'COMPLETION.json')
    require(prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior['protected_files']==21444,'Completed predecessor')
    t=read(TESTS);require(t['passed'] is True and t['synthetic_only'] is True and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Exact formal tests')
    context,contract=contract_check();review=code_review();previous=read(PRIOR_REVIEW)
    require(previous['passed'] is True and previous['unresolved_blockers']==[] and previous['plan_sha256']==sha(PRIOR/'PLAN.json') and previous['results_completion_sha256']==sha(PRIOR/'results/COMPLETION.json'),'Predecessor independent review')
    require(review['implementation_contract_sha256']==sha(CONTRACT),'Review contract binding')
    pins=dict(read(PRIOR/'PLAN.json')['input_sha256'])
    paths=files()+[CONTEXT,CONTRACT,TESTS,REVIEW,SOURCE_REVIEW,CALENDAR,ev.PRICE,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',PRIOR_REVIEW,old.PROTECTED]
    paths += [PRIOR/'prepared'/name for name in ('COMPLETION.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz','META.json')]
    paths += [REBUILD/'ORDERS.json',LOCAL/'local_linear_ORDERS.json']
    paths += [REBUILD/name for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json','PARITY.json')]
    for path in paths:old.bind(pins,path)
    for source in (context,review,read(SOURCE_REVIEW)):
        for k,v in source['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    for k,v in read(CALENDAR)['provenance']['source_hashes'].items():old.bind(pins,Path(k),v)
    import monthly_leader_utility260_independent as ind
    import monthly_leader_utility260_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_leader_utility260_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),implementation_contract=contract,implementation_contract_sha256=sha(CONTRACT),runtime=ev.runtime(),input_sha256=pins,raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,actual_labels_before_registration=0,actual_predictions_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits='Original control cache reused with zero original predictive fits; annual intercept estimates, conditional Ridge solves and independent SVD/canonical checks separately counted.',numerical_verification=NUMERICAL_VERIFICATION)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),last_goal_turn_classification='progress',next_step='Verify original source/decisions before leader labels and annual router fits; only state-conditioned arm can pass first88 ZERO gates against both references.')
    old.commit_state(before,state,'leader_utility260_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');c,d=contract_check()
    require(p['numerical_verification']==NUMERICAL_VERIFICATION,'Prospective numerical verification unchanged')
    require(p['context_sha256']==sha(CONTEXT) and p['context']==c and p['implementation_contract_sha256']==sha(CONTRACT) and p['implementation_contract']==d and p['runtime']==ev.runtime(),'Context/contract/runtime changed')
    old.verify(p['input_sha256']);r=code_review();require(r['implementation_contract_sha256']==sha(CONTRACT),'Reviewed contract')
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed');registration_check(read(CENTRAL),expected)
    return p


def integer(value,label,expected=None):
    require(type(value) is int and value>=0,label+' nonnegative native int')
    if expected is not None:require(value==expected,label+' exact value')


def counts_check(value,fields,label):
    require(type(value) is dict and set(value)==set(fields),label+' exact fields')
    for key,item in value.items():integer(item,label+'/'+key)


def value_digest(value):
    return hashlib.sha256(json.dumps(json_safe(value),sort_keys=True,separators=(',',':'),allow_nan=True).encode()).hexdigest()


def source_mapping_snapshot(value):
    """Already consumed source only; object-array bytes never convert label values."""
    require(type(value) is dict,'Source mapping snapshot')
    arrays={key:item for key,item in value.items() if isinstance(item,np.ndarray)}
    headers={key:item for key,item in value.items() if not isinstance(item,np.ndarray)}
    return old.array_snapshot(arrays),value_digest(headers)



def json_safe(value):
    """Explicit JSON conversion; never silently stringify numeric arrays."""
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,(pd.Timestamp,datetime)):return str(value.date())
    if isinstance(value,date):return str(value)
    return value


def exact_prefix(wanted,got,n):
    oo,ss,pp,_=wanted;orders,scores,pres,_=got
    require(set(orders)==set(scores)==set(pres)==set(ARMS),'Exactly three decision arms')
    for arm in ARMS:
        require(orders[arm]==oo[arm][:n],'Order prefix')
        np.testing.assert_array_equal(scores[arm],ss[arm][:n]);np.testing.assert_array_equal(pres[arm],pp[arm][:n])


def incumbent_callback(ind,arrays,meta,n,expected_orders=None,expected_scores=None,expected_pres=None):
    receipts={}
    if expected_orders is not None:require(type(expected_orders) is list and len(expected_orders)==n,'Complete frozen incumbent prefix')
    def incumbent(orders,scores,pres):
        require(not receipts,'Incumbent gate once')
        if expected_orders is not None:require(orders==expected_orders,'Frozen incumbent before teacher/new fits or choices')
        if expected_scores is not None:np.testing.assert_array_equal(scores,expected_scores[:n])
        if expected_pres is not None:np.testing.assert_array_equal(pres,expected_pres[:n])
        check=ind.verify_incumbent(arrays,meta,orders,scores,pres,n=n)
        receipt(check,n,ind.INCUMBENT_FLAGS,'incumbent')
        receipts['incumbent']=check;return check
    return incumbent,receipts


def source_evidence(source,plan):
    require(source['raw_source_hashes']==plan['raw_source_hashes'],'Saved raw source pins')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        require(source[key]==plan['input_sha256'][str(REBUILD/name)],'Registered cached source:'+name)
    for key,path in [('price_archive_sha256',ev.PRICE),('calendar_sha256',CALENDAR)]:
        require(source[key]==plan['input_sha256'][str(path)],'Registered source:'+key)
    for name in ('COMPLETION.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz','META.json'):
        require(source['incumbent_cache_sha256'][name]==plan['input_sha256'][str(PRIOR/'prepared'/name)],'Frozen prior incumbent cache:'+name)
    require(source['fresh_training_array_reconstruction'] is False and source['cached_training_basis']=='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.','Honest cache provenance scope')
    for key in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact','original_scores_exact','original_pre_scores_exact'):
        require(source[key] is True,'Source preservation:'+key)
    integer(source['original_predictive_models_refitted'],'Reused control zero refits',0)


def frozen_incumbent_cache(meta):
    completion=read(PRIOR/'prepared/COMPLETION.json')
    require(completion['status']=='COMPLETE' and completion['plan_sha256']==sha(PRIOR/'PLAN.json'),'Frozen incumbent preparation chain')
    for name,digest in completion['artifacts_sha256'].items():require(sha(PRIOR/'prepared'/name)==digest,'Frozen incumbent prepared bytes')
    require(read(PRIOR/'prepared/META.json')==meta,'Frozen incumbent metadata exact')
    orders=read(PRIOR/'prepared/ORDERS.json')[ARMS[0]]
    require(orders==read(LOCAL/'local_linear_ORDERS.json')==read(REBUILD/'ORDERS.json'),'Independent rebuilt/original/prior orders exact')
    scores=ev.load_arrays(PRIOR/'prepared/SCORES.npz')[ARMS[0]]
    pres=ev.load_arrays(PRIOR/'prepared/PRE_SCORES.npz')[ARMS[0]]
    require(scores.shape==pres.shape==(122,49),'Frozen full incumbent matrices')
    return orders,scores,pres


def model_counts_check(c,n):
    import monthly_leader_utility260_kernel as k
    counts_check(c,k.MODEL_COUNT_FIELDS,'Router counts')
    require(c['active_query_events']+c['cold_query_events']+c['partial_events']==n,'Router query partition')
    require(c['intercept_estimates']==c['ridge_solves']<=c['years'],'Paired intercept/Ridge fit counts')
    require(c['valid_target_months']<=c['training_months'],'Valid target subset')
    integer(c['original_model_fits'],'No original predictive model refits',0)


def decision_counts_check(c,n,width=None):
    import monthly_leader_utility260_kernel as k
    counts_check(c,k.COUNT_FIELDS,'Decision counts');integer(c['events'],'Decision events',n)
    integer(c['symbols_per_event'],'Decision width',width)
    require(c['symbols_per_event']>0,'Positive decision width')
    require(c['partial_events']+c['active_events']+c['cold_events']==n,'Decision state partition')
    for name in ('incumbent_episode_calls','intercept_episode_calls','state_episode_calls'):integer(c[name],name,1)
    integer(c['new_model_fits'],'Cached decision no fits',0)


def receipt(r,n,flags,kind,*,width=None):
    """Validate the complete independent layer, including honest actual work."""
    import monthly_leader_utility260_independent as ind
    standards={'inputs':ind.INPUT_FLAGS,'targets':ind.TARGET_FLAGS,'source':ind.SOURCE_FLAGS,
               'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS}
    require(kind in standards and type(r) is dict and r.get('passed') is True,'Passed independent layer')
    integer(r.get('checked_events'),'Independent event scope',n)
    for f in set(standards[kind])|set(flags):require(r.get(f) is True,'Independent '+kind+' flag:'+f)
    require(r.get('producer_modules_imported') is False and r.get('computes_returns') is False,'Independent result-free scope')
    for key in ('old_model_solves','conditional_ridge_solves','independent_canonical_solves','independent_svd_solves','original_episode_calls'):integer(r.get(key),'Actual work:'+key)
    integer(r['old_model_solves'],'Original model cache reused',0)
    if kind=='models':
        model_counts_check(r['counts'],n);fits=r['counts']['ridge_solves']
        integer(r['conditional_ridge_solves'],'Oracle canonical conditional solves',fits)
        integer(r['independent_canonical_solves'],'Actual canonical solves',fits)
        integer(r['independent_svd_solves'],'Actual SVD solves',fits)
        integer(r['intercept_estimates'],'Actual independent intercept estimates',fits)
        require(r.get('full_model_audit_supplied') is True,'Full annual model proof')
    else:
        for key in ('conditional_ridge_solves','independent_canonical_solves','independent_svd_solves'):integer(r[key],'Layer performs no model solves:'+key,0)
    integer(r['original_episode_calls'],'Layer episode calls',1 if kind=='incumbent' else 3 if kind=='selection' else 0)
    if kind in ('incumbent','selection'):
        integer(r.get('ordered_choices_checked'),'Ordered choices',n if kind=='incumbent' else 3*n)
        for key in ('full_scores_supplied','full_pre_scores_supplied'):require(r.get(key) is True,'Full matrices:'+key)
        require(r.get('numerical_tolerance')==0.,'Exact selection/control')
        if kind=='incumbent':require(r.get('candidate_selection_computed') is False,'Original gate before candidates')
        else:
            decision_counts_check(r['counts'],n,width)
            require(r.get('full_audit_supplied') is True and r.get('full_bundle_supplied') is True,'Complete decision audit and bundle')
    else:
        require(r.get('exact_discrete') is True,'Exact discrete proof')
        if kind in ('inputs','targets'):require(r.get('numerical_tolerance')==0.,'Exact inputs/teacher proof')
        else:require(r.get('continuous_atol')==2e-10 and r.get('continuous_rtol')==2e-10 and r.get('rank_tolerance')==0.,'Prospective verification bounds')
        if kind=='inputs':require(r.get('full_input_audit_supplied') is True,'Complete original input proof')
        if kind=='inputs':counts_check(r['counts'],('training_rows','training_months','query_rows'),'Input scope')
        if kind=='targets':
            import monthly_leader_utility260_kernel as k
            counts_check(r['counts'],k.TARGET_COUNT_FIELDS,'Teacher counts')
            require(r['counts']['valid_months']+r['counts']['unavailable_months']==r['counts']['months'],'Paired teacher validity partition')
            require(r.get('full_target_audit_supplied') is True,'Complete teacher proof')
        if kind=='source':
            import monthly_leader_utility260_kernel as k
            counts_check(r['counts'],k.SOURCE_COUNT_FIELDS,'Market counts')
            integer(r['counts']['query_events'],'Market queries',n)
            require(r['counts']['state_vectors']==r['counts']['training_events']+n-r['counts']['partial_events'] and r['counts']['price_observations']==507*r['counts']['state_vectors'],'Source observation accounting')
            require(r.get('full_source_audit_supplied') is True,'Complete source proof')
    return r


def bundle_preservation(bundle,n):
    import monthly_leader_utility260_kernel as k
    require(set(bundle)==set(k.ROUTER_KEYS) and bundle['schema']=='monthly_leader_utility260_router_v1','Router bundle schema')
    integer(bundle['n'],'Router events',n)
    model_counts_check(bundle['audit']['counts'],n)
    require(bundle['audit']['original_models_refitted'] is False,'Original cache reuse')
    require(set(bundle['audit'])==set(k.MODEL_AUDIT_KEYS),'Complete model audit')
    require(bundle['audit']['market_columns']==list(k.COLUMNS),'Fixed market feature order')
    require(len(bundle['audit']['annual'])==bundle['audit']['counts']['years'],'Annual evidence count')
    for row in bundle['audit']['annual']:
        require(set(row)==set(k.ANNUAL_KEYS),'Complete annual sufficient-statistics evidence')
    require(bundle['control_gate'].get('passed') is True,'Early incumbent gate attached')
    for key in ('event_dates','active','intercept_prediction','state_prediction','intercept_p','state_p'):require(np.shape(bundle[key])==(n,),'Router vector '+key)
    require(np.asarray(bundle['active']).dtype==np.dtype(bool),'Boolean router activity')
    for key in ('intercept_prediction','state_prediction','intercept_p','state_p'):
        v=np.asarray(bundle[key]);require(v.dtype==np.dtype(float) and np.isfinite(v).all(),'Finite binary64 router '+key)
        if key.endswith('_p'):require(np.all((v>=.25)&(v<=.75)),'Declared mixing bounds')
        np.testing.assert_array_equal(v[~bundle['active']],np.full(int((~bundle['active']).sum()),.5 if key.endswith('_p') else 0.))


def bundle_parity(full,part,n):
    require(part['n']==n and full['n']>=n,'Physical router prefix')
    for key in ('event_dates','active','intercept_prediction','state_prediction','intercept_p','state_p'):
        np.testing.assert_array_equal(part[key],full[key][:n])


def decision_pass(study,arrays,meta,bundle,event_plain,n,expected_orders,expected_scores,expected_pres,*,kernel=None,ind=None):
    if kernel is None:import monthly_leader_utility260_kernel as kernel
    if ind is None:import monthly_leader_utility260_independent as ind
    # Hash only the consumed router/query prefix, never inspect future bodies.
    consumed=lambda:value_digest({key:(bundle[key][:n] if key in ('event_dates','active','intercept_prediction','state_prediction','intercept_p','state_p') else bundle[key]) for key in ('event_dates','active','intercept_prediction','state_prediction','intercept_p','state_p')})
    before=consumed();plain_before=old.array_snapshot({'event_plain':np.asarray(event_plain)[:n*arrays['baseline'].shape[1]]})
    ig,proof=incumbent_callback(ind,arrays,meta,n,expected_orders,expected_scores,expected_pres)
    result=kernel.decisions(study,arrays,meta,bundle,event_plain,n=n,incumbent_validator=ig,expected_incumbent_orders=expected_orders)
    require(set(proof)=={'incumbent'} and result[3]['incumbent_gate']==proof['incumbent'],'Single original gate preserved before candidate scores')
    oo,ss,pp,au=result
    proof['selection']=ind.verify_selection(arrays,meta,bundle,event_plain,oo,scores=ss,pre_scores=pp,audit=au,n=n)
    receipt(proof['selection'],n,ind.SELECTION_FLAGS,'selection',width=arrays['baseline'].shape[1])
    require(proof['selection']['counts']==au['counts'],'Selection count binding')
    require(consumed()==before and plain_before==old.array_snapshot({'event_plain':np.asarray(event_plain)[:n*arrays['baseline'].shape[1]]}),'Consumed router/source inputs unchanged')
    return result,proof


def prepare_one(study,train,metadata,arrays,adj,calendar,n,expected_orders,expected_scores,expected_pres,*,kernel=None,ind=None):
    """Original gate precedes even teacher choices; all new source gates precede fits."""
    if kernel is None:import monthly_leader_utility260_kernel as kernel
    if ind is None:import monthly_leader_utility260_independent as ind
    array_before=old.array_snapshot(arrays)
    meta=metadata['monthly'];early,eg=incumbent_callback(ind,arrays,meta,n,expected_orders,expected_scores,expected_pres)
    oo,ss,pp=kernel.original_decisions(study,arrays,meta,n=n);early(oo,ss,pp)
    proof={'early_incumbent':eg['incumbent']}
    inp=kernel.router_inputs(train,metadata,arrays,n=n)
    proof['inputs']=ind.verify_inputs(train,metadata,arrays,inp,n=n);receipt(proof['inputs'],n,ind.INPUT_FLAGS,'inputs')
    input_before=source_mapping_snapshot(inp)
    target=kernel.build_expert_targets(inp)
    require(source_mapping_snapshot(inp)==input_before,'Teacher cannot rewrite independently verified source input')
    proof['targets']=ind.verify_targets(inp,target);receipt(proof['targets'],n,ind.TARGET_FLAGS,'targets')
    require(proof['targets']['counts']==target['counts'],'Teacher count binding')
    target_before=source_mapping_snapshot(target)
    tx,qx,market=kernel.build_market_inputs(adj,calendar,target['dates'],inp['event_dates'])
    require(source_mapping_snapshot(inp)==input_before and source_mapping_snapshot(target)==target_before,'Market builder cannot rewrite verified input or target')
    proof['source']=ind.verify_market_inputs(adj,calendar,target['dates'],inp['event_dates'],tx,qx,market)
    receipt(proof['source'],n,ind.SOURCE_FLAGS,'source')
    require(proof['source']['source_audit']==market,'Full market-source receipt binding')
    def source_snapshot():
        # Preserve unselected label objects without invoking their numeric/string conversions.
        numeric={key:value for key,value in inp.items() if isinstance(value,np.ndarray)}
        numeric.update({'target_'+key:value for key,value in target.items() if isinstance(value,np.ndarray)})
        numeric.update(train_X=tx,query_X=qx)
        return old.array_snapshot(numeric),value_digest(dict(market=market,target_records=target['records'])),source_mapping_snapshot(inp),source_mapping_snapshot(target)
    source_before=source_snapshot()
    calls=[]
    def control():
        require(not calls,'Fit source callback once');calls.append(True);return proof['early_incumbent']
    bundle=kernel.fit_router(target,tx,qx,inp['event_dates'],control_validator=control)
    require(calls==[True] and bundle['control_gate']==proof['early_incumbent'],'Source gate before first annual router fit')
    bundle_preservation(bundle,n)
    proof['models']=ind.verify_models(target,tx,qx,inp['event_dates'],bundle);receipt(proof['models'],n,ind.MODEL_FLAGS,'models')
    require(proof['models']['counts']==bundle['audit']['counts'],'Actual model counts binding')
    result,dp=decision_pass(study,arrays,meta,bundle,inp['event_plain'],n,expected_orders,expected_scores,expected_pres,kernel=kernel,ind=ind)
    proof.update(dp);proof['model_counts']=bundle['audit']['counts']
    require(array_before==old.array_snapshot(arrays) and source_before==source_snapshot(),'Verified source/arrays unchanged after model/selection')
    return bundle,dict(inputs=inp,target=target,market=market,train_X=tx,query_X=qx),result,proof


def execution_accounting(proof):
    """Sum actual runs. Full 122 reuse is a reference, never a second fit."""
    physical=proof['physical_prefixes'];fresh=[x for x in physical if x['model_recomputed']]
    runs=[proof]+fresh;cached=proof['cached_prefixes']
    keys=('old_model_solves','independent_canonical_solves','independent_svd_solves','original_episode_calls')
    totals={key:0 for key in keys}
    receipts=[]
    for run in runs:receipts.extend(run[k] for k in ('early_incumbent','inputs','targets','source','models','incumbent','selection'))
    for run in cached:receipts.extend(run[k] for k in ('incumbent','selection'))
    for item in receipts:
        for key in keys:totals[key]+=item[key]
    return dict(schema='monthly_leader_utility260_execution_v1',producer_original_model_fits=0,
        producer_intercept_estimates=sum(r['model_counts']['intercept_estimates'] for r in runs),
        producer_conditional_ridge_solves=sum(r['model_counts']['ridge_solves'] for r in runs),
        producer_original_episode_calls=4*len(runs)+3*len(cached),
        independent_intercept_estimates=sum(r['models']['intercept_estimates'] for r in runs),
        independent_actual_work=totals,total_linear_system_solves=sum(r['model_counts']['ridge_solves'] for r in runs)+totals['independent_canonical_solves']+totals['independent_svd_solves'],
        fresh_model_invocations=1+len(fresh),fresh_physical_prefixes=[x['prefix'] for x in fresh],
        cached_decision_invocations=len(cached),physical122_reuses_main=True,
        producer_teacher_month_evaluations=sum(r['targets']['counts']['months'] for r in runs),
        independent_teacher_month_evaluations=sum(r['targets']['counts']['months'] for r in runs),
        producer_market_state_vectors=sum(r['source']['counts']['state_vectors'] for r in runs),
        independent_market_state_vectors=sum(r['source']['counts']['state_vectors'] for r in runs),
        counts_include_n0_function_invocations=True,returns_computed=False)


def preparation_evidence(proof,model_audit,decision_audit):
    import monthly_leader_utility260_independent as ind
    require(proof.get('all123_order_prefixes') is True and proof.get('original_arrays_unchanged') is True and proof.get('original_training_unchanged') is True,'Full source/prefix preservation')
    require([p['prefix'] for p in proof['cached_prefixes']]==list(range(123)),'All 123 cached prefixes')
    require([p['prefix'] for p in proof['physical_prefixes']]==list(PREFIXES),'Five physical boundaries')
    def run_check(run,n):
        for kind,flag in [('early_incumbent',ind.INCUMBENT_FLAGS),('inputs',ind.INPUT_FLAGS),('targets',ind.TARGET_FLAGS),('source',ind.SOURCE_FLAGS),('models',ind.MODEL_FLAGS),('incumbent',ind.INCUMBENT_FLAGS),('selection',ind.SELECTION_FLAGS)]:
            receipt(run[kind],n,flag,'incumbent' if kind=='early_incumbent' else kind,width=49)
        model_counts_check(run['model_counts'],n)
        require(run['model_counts']==run['models']['counts'],'Model count proof bound')
    run_check(proof,122)
    require(proof['model_counts']==model_audit['model']['counts'],'Saved model audit counts')
    require(proof['saved_model_audit_sha256']==value_digest(model_audit),'Full saved model/source/teacher audit binding')
    require(proof['targets']['counts']==model_audit['target']['counts'] and proof['source']['source_audit']==model_audit['market'],'Saved teacher/source audits')
    require(proof['selection']['counts']==decision_audit['counts'] and proof['incumbent']==decision_audit['incumbent_gate'],'Saved decision proof')
    for item in proof['cached_prefixes']:
        require(item['exact'] is True,'Exact cached prefix')
        receipt(item['incumbent'],item['prefix'],ind.INCUMBENT_FLAGS,'incumbent')
        receipt(item['selection'],item['prefix'],ind.SELECTION_FLAGS,'selection',width=49)
    for item in proof['physical_prefixes']:
        n=item['prefix'];require(item['exact'] is True,'Exact physical prefix')
        require(item['model_recomputed'] is (n!=122),'Only full122 reuses main')
        if n==122:
            require(item.get('reuses_main_verified_full') is True,'Full reuse disclosed')
            for key in ('early_incumbent','inputs','targets','source','models','incumbent','selection','model_counts'):require(item[key]==proof[key],'Reused main proof:'+key)
        else:run_check(item,n)
    require(proof['fit_execution_accounting']==execution_accounting(proof),'Actual work ledger recomputed')


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_leader_utility260_kernel as kernel
    import monthly_leader_utility260_independent as ind
    study=old.local();m,arrays,_,_=study.load()
    require(len(m['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 arrays')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');metadata=read(REBUILD/'MONTHLY_META.json')
    require(metadata['monthly']==m and metadata['training']==read(study.arch.P/'META.json')['training'],'Original identities')
    before=old.array_snapshot(arrays);train_before=old.array_snapshot(train)
    original_orders,original_scores,original_pres=frozen_incumbent_cache(m)
    adj,_=ev.prices();calendar=read(CALENDAR)['sessions']
    bundle,artifacts,wanted,proof=prepare_one(study,train,metadata,arrays,adj,calendar,122,original_orders,original_scores,original_pres,kernel=kernel,ind=ind)
    orders,scores,pres,audit=wanted;cached=[];physical=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(m,n)
        got,cp=decision_pass(study,aa,mm,bundle,artifacts['inputs']['event_plain'],n,original_orders[:n],original_scores,original_pres,kernel=kernel,ind=ind)
        exact_prefix(wanted,got,n);cached.append(dict(prefix=n,exact=True,**cp))
        if n in PREFIXES and n!=122:
            print(json.dumps(dict(stage='260_physical_router_prefix',events=n)),flush=True)
            # Source metadata remains complete; numeric training is sliced by router_inputs.
            # Physically truncate price and event rows, never reconstruct an old model.
            smallmeta=dict(training=metadata['training'],monthly=mm)
            tt={key:(value[:n*49].copy() if key.startswith('event_') and np.shape(value)[:1]==(122*49,) else value) for key,value in train.items()}
            frame=adj.iloc[:0].copy() if n==0 else adj.loc[:pd.Timestamp(mm['dates'][-1])].copy()
            small,sa,sg,sp=prepare_one(study,tt,smallmeta,aa,frame,calendar,n,original_orders[:n],original_scores,original_pres,kernel=kernel,ind=ind)
            exact_prefix(wanted,sg,n);bundle_parity(bundle,small,n)
            physical.append(dict(prefix=n,exact=True,model_recomputed=True,**sp))
    physical.append(dict(prefix=122,exact=True,model_recomputed=False,reuses_main_verified_full=True,**proof))
    require(before==old.array_snapshot(arrays) and train_before==old.array_snapshot(train),'Original arrays/training changed')
    proof.update(cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True)
    proof['fit_execution_accounting']=execution_accounting(proof)
    model_audit=dict(model=bundle['audit'],inputs={key:json_safe(value) for key,value in artifacts['inputs'].items() if key not in ('plain','event_plain','labels')},target={key:json_safe(value) for key,value in artifacts['target'].items() if key not in ('values','valid','dates','ends')},market=artifacts['market'])
    proof['saved_model_audit_sha256']=value_digest(model_audit)
    source=dict(raw_source_hashes=plan['raw_source_hashes'],original_training_hash=sha(REBUILD/'TRAIN_INPUTS.npz'),original_identity_hash=sha(REBUILD/'MONTHLY_META.json'),saved169_parity_hash=sha(REBUILD/'PARITY.json'),price_archive_sha256=sha(ev.PRICE),calendar_sha256=sha(CALENDAR),incumbent_cache_sha256={name:sha(PRIOR/'prepared'/name) for name in ('COMPLETION.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz','META.json')},cached_training_basis='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.',fresh_training_array_reconstruction=False,original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True,original_scores_exact=True,original_pre_scores_exact=True,original_predictive_models_refitted=0)
    preparation_evidence(proof,model_audit,audit);source_evidence(source,plan);gate(expected)
    dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{k:bundle[k] for k in ('event_dates','active','intercept_prediction','state_prediction','intercept_p','state_p')})
    np.savez_compressed(dest/'MARKET_INPUTS.npz',train_X=artifacts['train_X'],query_X=artifacts['query_X'],**{'target_'+key:artifacts['target'][key] for key in ('dates','ends','valid','values')})
    for name,value in [('SCORES',scores),('PRE_SCORES',pres)]:np.savez_compressed(dest/(name+'.npz'),**value)
    for name,value in {'MODEL_AUDIT.json':model_audit,'DECISION_AUDIT.json':audit,'ORDERS.json':orders,'META.json':m,'SOURCE_PARITY.json':source,'INDEPENDENT_PREPARE.json':proof}.items():dump(dest/name,json_safe(value))
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,model_fit_accounting='Original model cache reuse; main plus physical0/1/53/88 new router fits; physical122 main reuse; 123 cached choices with zero new fits. Canonical and SVD verification solves counted separately.'))
    print(json.dumps(dict(status='PREPARED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['events']==122 and c['new_returns_computed'] is False and set(c['artifacts_sha256'])==PREP_FILES,'Prepared scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    preparation_evidence(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),read(OUT/'prepared/MODEL_AUDIT.json'),read(OUT/'prepared/DECISION_AUDIT.json'))
    source_evidence(read(OUT/'prepared/SOURCE_PARITY.json'),read(OUT/'PLAN.json'))
    return read(OUT/'prepared/META.json'),read(OUT/'prepared/ORDERS.json')


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


def required_metrics(metrics,incumbent):
    require(set(metrics)=={'same_close_zero','next_open_zero'},'Exactly two zero-cost training paths')
    undefined=[]
    def finite(value,label):
        require(isinstance(value,(float,int,np.floating,np.integer)) and not isinstance(value,(bool,np.bool_)) and np.isfinite(value),'Finite required metric:'+label)
    for path,groups in metrics.items():
        require(set(groups)=={'training','early','recent'},'Only first88 training groups')
        for group in groups:finite(groups[group]['cagr'],path+'/'+group+'/cagr')
        m=groups['training'];finite(m['max_drawdown'],path+'/training/max_drawdown')
        if m['sharpe'] is None:
            finite(m.get('daily_return_sd'),path+'/training/daily_return_sd')
            require(not incumbent and m.get('sharpe_status')=='undefined_sharpe' and m.get('sharpe_reason')=='zero_daily_return_sd' and m.get('daily_return_sd')==0.,'Undefined Sharpe requires experimental zeroSD evidence')
            undefined.append(path)
        else:
            finite(m['sharpe'],path+'/training/sharpe')
            if 'daily_return_sd' in m:
                finite(m['daily_return_sd'],path+'/training/daily_return_sd')
                require(m['daily_return_sd']>0. and m.get('sharpe_status')=='defined' and m.get('sharpe_reason') is None,'Defined Sharpe evidence')
    return undefined


def select_first88(metrics,orders):
    require(set(metrics)==set(orders)==set(ARMS),'Three fixed arms')
    invalid={a:required_metrics(metrics[a],a==ARMS[0]) for a in ARMS}
    require(all(len(orders[a])>=88 for a in ARMS),'First88 ordered holdings required')
    candidate=ARMS[2];guards={};changes={}
    for reference in (ARMS[0],ARMS[1]):
        changed=[i for i in range(88) if set(orders[candidate][i]['symbols'])!=set(orders[reference][i]['symbols'])]
        changes[reference]=changed
        close=not invalid[candidate] and not invalid[reference] and old.guard(metrics[candidate]['same_close_zero'],metrics[reference]['same_close_zero'],.005 if reference==ARMS[0] else 0.)
        opening=not invalid[candidate] and not invalid[reference] and old.guard(metrics[candidate]['next_open_zero'],metrics[reference]['next_open_zero'],0.)
        support=(len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed)) if reference==ARMS[0] else len(changed)>=1
        guards[reference]=dict(close=bool(close),open=bool(opening),support=bool(support))
    admitted=all(all(g.values()) for g in guards.values())
    return dict(primary=candidate if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months_by_reference=changes,
       undefined_sharpe=invalid,first88_only=True,later34_used=False,cost_used=False,diagnostic_arm_never_promoted=True)

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

def train_check(training,audit):
    import monthly_leader_utility260_independent as ind
    helper=ind.accounting();require(set(training)==set(ARMS),'Training arms')
    for arm in ARMS:
        require(set(training[arm])=={'same_close_zero','next_open_zero'},'Only zero-cost selection')
        for path,groups in training[arm].items():
            require(set(groups)=={'training','early','recent'},'Training split scope')
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
    import monthly_leader_utility260_independent as ind
    zero_cost_check(exports,results)
    check=ind.audit_outcomes(px,op,orders,exports,results,meta);outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
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
    pins={str(Path(k) if Path(k).is_absolute() else ROOT/k):v for k,v in review['input_sha256'].items()}
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
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=read(CONTEXT)['limitations'])
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),phase_terminals_sha256=sha(PHASE_TERMINALS),new_trials=3,model_audit_sha256=sha(OUT/'prepared/MODEL_AUDIT.json'),independent_prepare_sha256=sha(OUT/'prepared/INDEPENDENT_PREPARE.json'),protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==256 and sum(x['trials'] for x in state['completed'])==1279,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review annual state-versus-intercept clipped leader utility; first88 primary remains immutable; target70 subject to independent goal audit; frozen observers unchanged.')
    old.commit_state(before,state,'leader_utility260_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

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
            for row in data['TRADES']:require(type(row['cost']) in (float,int) and row['cost']==0.,'ZERO trade cost exact')
            for row in data['HOLDINGS']:require(type(row['fee']) in (float,int) and row['fee']==0.,'ZERO period fee exact')
            for value in (r['metrics']['fees_paid'],r['slices']['full122']['fees_paid']):
                require(type(value) in (float,int) and value==0.,'ZERO aggregate fee exact')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    elif a.phase=='prepare':require(bool(a.plan_sha256),'PLAN hash required');prepare_recorded(a.plan_sha256)
    else:require(bool(a.plan_sha256),'PLAN hash required');globals()[a.phase](a.plan_sha256)
