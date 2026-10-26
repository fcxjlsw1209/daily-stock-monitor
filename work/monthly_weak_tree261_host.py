"""261 fixed shallow tree replaces the weak pooled three-quarter slot only.

Actual sources/models/outcomes require registration; matched Ridge is calibration.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_weak_tree261'
PRIOR=ROOT/'outputs/monthly_leader_utility260'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CONTEXT=ROOT/'work/monthly_weak_tree261_context.json'
CONTRACT=ROOT/'work/monthly_weak_tree261_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_weak_tree261_TESTS.json'
REVIEW=ROOT/'work/monthly_weak_tree261_CODE_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_leader_utility260_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_weak_tree261_RESULT_REVIEW.json'
ARMS=('incumbent_local67','matched_pooled_ridge','weak_pooled_tree')
ROLES=('control','candidate')
PHASE_TERMINALS=OUT/'PHASE_TERMINALS.json'
SOURCE_REVIEW=ROOT/'work/after260_gbdt_slot_source_review.json'
SOURCE_REVIEW_SHA='35d2ed160d0a4e1c248492961c26364abb80ea22441c67fa7186656b689adcb5'
CONTEXT_SHA='e60db38b1d0fe73fa2453cf17691b007bbb89c03fef29b4711e4e6595dfdd37c'
CONTRACT_SHA='c955aba02bb469c0b35e0499ae80f1b229d638306f72f7d1d0bfde36d9f5c4d6'
RUNTIME_PINS=ROOT/'work/monthly_weak_tree261_RUNTIME_PINS.json'
RUNTIME_REVIEW=ROOT/'work/after260_gbdt_runtime_review.json'
PREFIXES=(0,1,53,88,122)
WORK_FIELDS=('independent_canonical_solves','independent_svd_solves','independent_tree_refits','manual_tree_traversals','original_episode_calls','original_other_target_fits','original_local_fits')
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.0,selection_tolerance=0.0,independent_numeric_system='independent canonical Ridge plus augmented SVD; shared pinned sklearn refit and independent manual tree traversal')
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=tuple(name for stem in ('host','kernel','independent') for name in ('monthly_weak_tree261_'+stem+'.py','test_monthly_weak_tree261_'+stem+'.py'))
PREP_FILES={'FORECASTS.npz','LABELS.npz','SCORES.npz','PRE_SCORES.npz','MODEL_AUDIT.json','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

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
    return dict(schema='monthly_weak_tree261_original169_closure_v1',chain_sha256={str(k):v for k,v in SOURCE_CHAIN169.items()},artifacts_sha256=dict(artifacts),source_input_count=len(plan['input_sha256']),npz_decoded=False,source_cache_rebuilt=False)


def bind_source_closure(pins):
    for path,digest in SOURCE_CHAIN169.items():old.bind(pins,path,digest)
    for name,digest in read(REBUILD/'PARITY.json')['artifacts_sha256'].items():
        require(name in SOURCE_FILES169,'Unexpected original169 cache artifact');old.bind(pins,REBUILD/name,digest)
    for name,digest in read(REBUILD.parent/'PLAN.json')['input_sha256'].items():old.bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    return source_closure_check(pins)


def now():return datetime.now(timezone.utc).isoformat()


def files():return [ROOT/'work'/p for p in code_names]


def codes():return {str(p):sha(p) for p in files()}


def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('weak_tree261_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==256 and sum(x['trials'] for x in state['completed'])==1279,'Predecessor counters')
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
    require(c['arms']==list(ARMS) and c['registered_trials']==3 and c['only_promotable']==ARMS[2] and c['diagnostic_only']==ARMS[1],'Three fixed arms; tree only')
    require(d['context_sha256']==sha(CONTEXT) and d['actual_data_started'] is False,'Prospective contract binding')
    require(c['scope']['actual_data_loaded'] is False and c['scope']['actual_models_fitted']==0 and c['scope']['actual_returns_computed'] is False,'Prospective source scope')
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Independent source review')
    require(c['input_sha256'][str(RUNTIME_PINS.relative_to(ROOT))]==sha(RUNTIME_PINS),'Runtime manifest binding')
    require(c['input_sha256'][str(RUNTIME_REVIEW.relative_to(ROOT))]==sha(RUNTIME_REVIEW),'Runtime note binding')
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
    pins=dict(read(PRIOR/'PLAN.json')['input_sha256']);source_closure=bind_source_closure(pins)
    paths=files()+[CONTEXT,CONTRACT,TESTS,REVIEW,SOURCE_REVIEW,RUNTIME_PINS,RUNTIME_REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',PRIOR_REVIEW,old.PROTECTED]
    paths += [REBUILD/name for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json','PARITY.json')]
    paths += [PRIOR/'prepared'/name for name in ('COMPLETION.json','SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json')]
    for path in paths:old.bind(pins,path)
    for source in (context,contract,review,read(SOURCE_REVIEW),read(RUNTIME_PINS),read(RUNTIME_REVIEW)):
        for k,v in source['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_weak_tree261_independent as ind
    import monthly_weak_tree261_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_weak_tree261_v1',original169_source_closure=source_closure,created_utc=now(),context=context,context_sha256=sha(CONTEXT),implementation_contract=contract,implementation_contract_sha256=sha(CONTRACT),runtime=ev.runtime(),input_sha256=pins,raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,actual_labels_before_registration=0,actual_predictions_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits='Matched pooled log calibration Ridge and fixed tree fits; cached other-target/local models not refitted; independent Ridge/SVD/tree and manual traversal work separately',numerical_verification=NUMERICAL_VERIFICATION)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),last_goal_turn_classification='progress',next_step='Verify all annual matched Ridge and exact source strategy before any tree fit; weak tree only can pass first88 ZERO admission.')
    old.commit_state(before,state,'weak_tree261_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');c,d=contract_check()
    require(p['numerical_verification']==NUMERICAL_VERIFICATION,'Prospective numerical verification unchanged')
    require(p['context_sha256']==sha(CONTEXT) and p['context']==c and p['implementation_contract_sha256']==sha(CONTRACT) and p['implementation_contract']==d and p['runtime']==ev.runtime(),'Context/contract/runtime changed')
    old.verify(p['input_sha256']);require(p['original169_source_closure']==source_closure_check(p['input_sha256']),'Original169 complete source chain');r=code_review();require(r['implementation_contract_sha256']==sha(CONTRACT),'Reviewed contract')
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed');registration_check(read(CENTRAL),expected)
    return p


def integer(value,label,expected=None):
    require(type(value) is int and value>=0,label+' nonnegative native int')
    if expected is not None:require(value==expected,label+' exact value')


def counts_check(value,fields,label):
    require(type(value) is dict and set(value)==set(fields),label+' exact fields')
    for key,item in value.items():integer(item,label+'/'+key)








def model_counts_check(value,n,control_only=False):
    import monthly_weak_tree261_independent as ind
    counts_check(value,ind.MODEL_COUNT_FIELDS,'Model counts')
    integer(value['events'],'Model events',n)
    require(value['valid_training_rows']<=value['training_rows'],'Cohort count')
    require(value['ordinary_years']==value['supported_years']+value['cold_years'],'Annual statuses')
    require(value['supported_years']==value['matched_ridge_fits']+value['no_prediction_years'],'Maturity versus actual fit')
    require(value['tree_fits']==(0 if control_only else value['matched_ridge_fits']),'Paired fits or pre-tree gate')
    require(value['actual_model_fits']==value['matched_ridge_fits']+value['tree_fits'],'Actual calibration and tree fits')
    require(value['original_other_target_fits']==value['original_local_fits']==0,'Cached other-target/local reused')
    require(value['trees']==value['boosting_iterations']==100*value['tree_fits'],'Fixed iterations')
    require(value['manual_tree_traversals']==(0 if control_only else 100*value['prediction_rows']),'Producer actual traversal count')
    require(value['tree_nodes']>=value['trees'],'At least root per tree')
    if control_only:require(value['tree_nodes']==0,'No tree nodes before gate')
    require(value['matched_ridge_fits']<=value['prediction_rows'],'Fits require query U')
    if n==0 or value['ordinary_years']==0:require(all(v==0 for k,v in value.items() if k!='events'),'Empty/partial-only no training numeric or fits')


def model_preservation(model,labels,n,control_only=False):
    import monthly_weak_tree261_independent as ind
    require(model['schema']=='monthly_weak_tree261_model_audit_v1','Model audit schema')
    c=model['counts'];model_counts_check(c,n,control_only)
    runtime=model.get('execution_runtime');require(type(runtime) is list and len(runtime)==c['tree_fits'],'Actual tree threadpool evidence')
    years=[]
    for record in runtime:
        require(type(record) is dict and set(record)=={'year','fit_threadpools','predict_threadpools'},'Tree runtime record')
        integer(record['year'],'Runtime year');years.append(record['year'])
        for phase in ('fit_threadpools','predict_threadpools'):
            pools=record[phase];require(type(pools) is list and bool(pools),'Loaded native pools')
            for pool in pools:
                require(type(pool) is dict and type(pool.get('filepath')) is str and bool(pool['filepath']),'Native pool identity')
                integer(pool.get('num_threads'),'Effective native threads',1)
    require(years==sorted(set(years)),'Unique chronological runtime years')
    for key in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','local_models_reused','other_targets_not_fitted','weak_only'):
        require(model.get(key) is True,'Model preservation:'+key)
    require(model.get('computes_returns') is False,'Models before outcomes')
    parity=model['original_cache_parity']
    require(set(parity)=={'passed','events','original_raw_exact','original_availability_exact','original_ranks_exact','candidate_models_computed'},'Original cache proof')
    integer(parity['events'],'Original parity events',n)
    for key in ('passed','original_raw_exact','original_availability_exact','original_ranks_exact'):require(parity[key] is True,'Exact original proof')
    require(parity['candidate_models_computed'] is False,'Original parity before candidate')
    keys={'schema','rows','valid_rows','target_keys','target_sha256','common_good_sha256','labels_unchanged','common_good_unchanged','original_row_order_preserved','recentered','raw_target_reconstructed'}
    require(type(labels) is dict and set(labels)==keys and labels['schema']=='monthly_weak_tree261_labels_preserved_v1','Label proof schema')
    integer(labels['rows'],'Consumed label rows',c['training_rows']);integer(labels['valid_rows'],'Original common good rows')
    require(c['valid_training_rows']<=labels['valid_rows']<=labels['rows'],'Good versus complete usable cohort')
    require(labels['target_keys']==list(ind.TARGETS) and set(labels['target_sha256'])==set(ind.TARGETS),'All four labels')
    for digest in list(labels['target_sha256'].values())+[labels['common_good_sha256']]:
        require(type(digest) is str and len(digest)==64 and all(x in '0123456789abcdef' for x in digest),'Target/good digest')
    for key in ('labels_unchanged','common_good_unchanged','original_row_order_preserved'):require(labels[key] is True,'Label preservation')
    for key in ('recentered','raw_target_reconstructed'):require(labels[key] is False,'No new labels')


def bundle_preservation(bundle,n,control_only=False):
    model_preservation(bundle['audit'],bundle['label_preservation'],n,control_only)
    integer(bundle['n'],'Bundle events',n);integer(bundle['width'],'Bundle width')
    require(bundle['width']>0,'Positive width')
    if control_only:
        require(bundle['candidate'] is None and bundle['control_gate'] is None,'Control-only pre-tree view')
    else:
        a,b=bundle['control']['available'],bundle['candidate']['available']
        require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'Exact copied original availability')


def receipt(check,n,flags,kind,*,width=None):
    import monthly_weak_tree261_independent as ind
    require(type(check) is dict,'Independent receipt object')
    kind={'model':'models'}.get(kind,kind)
    standard={'control':ind.CONTROL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS}
    require(kind in standard,'Known receipt kind');integer(n,'Requested events');integer(check.get('checked_events'),'Checked events',n)
    require(check.get('passed') is True and all(check.get(k) is True for k in tuple(flags)+tuple(standard[kind])),'Required independent flags:'+kind)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent scope')
    for k in WORK_FIELDS:integer(check.get(k),'Actual independent work:'+k)
    require(check['original_other_target_fits']==check['original_local_fits']==0,'Original model cache reuse')
    def flag(k):require(check.get(k) is True,'Full proof:'+k)
    def tol(k,v):require(type(check.get(k)) is float and np.isfinite(check[k]) and check[k]==v,'Prospective tolerance:'+k)
    if kind in ('control','models'):
        for k in ('continuous_atol','continuous_rtol','rank_tolerance'):tol(k,NUMERICAL_VERIFICATION[k])
        for k in ('exact_discrete','full_bundle_supplied','full_control_supplied','full_label_preservation_supplied','full_model_audit_supplied'):flag(k)
        require(check['independent_numeric_system']==NUMERICAL_VERIFICATION['independent_numeric_system'],'Declared numerical systems')
        require(check.get('shared_tree_trainer') is True and check.get('independent_tree_training_algorithm') is False,'Shared trainer disclosed')
        c=check['counts'];model_counts_check(c,n,kind=='control')
        counts_check(check.get('producer_equivalent_counts'),ind.MODEL_COUNT_FIELDS,'Producer-equivalent counts')
        require(check['producer_equivalent_counts']==c,'Producer-equivalent binding')
        require(check['independent_canonical_solves']==check['independent_svd_solves']==c['matched_ridge_fits'],'Actual independent Ridge work')
        require(check['independent_tree_refits']==c['tree_fits'],'Actual independent tree work')
        require(check['manual_tree_traversals']==(0 if kind=='control' else 2*c['manual_tree_traversals']),'Own and supplied model traversals')
        require(check['original_episode_calls']==0,'Model checks no selector')
        require(check.get('candidate_models_computed') is (kind=='models') and check.get('full_candidate_supplied') is (kind=='models'),'Control precedes any trees')
    else:
        for k in ('full_scores_supplied','full_pre_scores_supplied'):flag(k)
        tol('numerical_tolerance',0.);integer(check.get('ordered_choices_checked'),'Ordered choices',n if kind=='incumbent' else 3*n)
        require(all(check[k]==0 for k in WORK_FIELDS if k!='original_episode_calls'),'Decision checks no fits/traversal')
        integer(check['original_episode_calls'],'Independent episode calls',1 if kind=='incumbent' else 3)
        if kind=='incumbent':require(check.get('candidate_selection_computed') is False,'Original before candidate')
        else:
            for k in ('full_bundle_supplied','full_audit_supplied'):flag(k)
            c=check['counts'];counts_check(c,ind.COUNT_FIELDS,'Decision counts')
            integer(c['events'],'Decision events',n);require(c['symbols_per_event']>0 and c['initial_partial_events']+c['formal_events']==n,'Decision scope')
            if width is not None:integer(c['symbols_per_event'],'Actual width',width)
            require(c['incumbent_episode_calls']==c['matched_episode_calls']==c['tree_episode_calls']==1,'One episode per arm including n0')
            require(c['new_model_fits']==c['predictive_model_fits']==c['matched_changed_holdings_events']==0,'Decision calibration and no fit')
            require(c['tree_changed_holdings_events']==c['tree_vs_matched_changed_holdings_events'],'Exact matched holdings')


def fit_binding(control,models,model_audit,n):
    model_counts_check(model_audit['counts'],n)
    require(models['counts']==model_audit['counts'],'Independent/producer full counts')
    after=models['counts'];before=control['counts']
    for key in after:
        expected=0 if key in ('tree_fits','boosting_iterations','trees','tree_nodes','manual_tree_traversals') else after['matched_ridge_fits'] if key=='actual_model_fits' else after[key]
        integer(before[key],'Pre-tree count:'+key,expected)


def execution_accounting(e):
    invocations=[('main',122,e)]
    invocations += [('physical',v['prefix'],v) for v in e['physical_prefixes'] if v['prefix']!=122]
    entries=[]
    for scope,n,v in invocations:
        model=v['models'] if scope=='main' else v['model'];audit=v['main_model_counts'] if scope=='main' else v['model_counts']
        fit_binding(v['control'],model,audit,n);c=audit['counts']
        checks=[v['early_incumbent'],v['matched_strategy']['independent'],v['control'],model,v['incumbent'],v['selection']]
        item=dict(scope=scope,events=n,producer_matched_ridge_fits=c['matched_ridge_fits'],producer_tree_fits=c['tree_fits'],producer_boosting_iterations=c['boosting_iterations'],producer_trees=c['trees'],producer_tree_nodes=c['tree_nodes'],producer_manual_tree_traversals=c['manual_tree_traversals'],producer_episode_calls=5)
        item.update({k:sum(q[k] for q in checks) for k in WORK_FIELDS})
        item['total_model_fit_calls']=item['producer_matched_ridge_fits']+item['producer_tree_fits']+item['independent_canonical_solves']+item['independent_svd_solves']+item['independent_tree_refits']
        entries.append(item)
    cached=e['cached_prefixes'];checks=[q[k] for q in cached for k in ('incumbent','selection')]
    cached_work=dict(invocations=len(cached),producer_episode_calls=3*len(cached),model_fit_calls=0,**{k:sum(q[k] for q in checks) for k in WORK_FIELDS})
    keys=[k for k in entries[0] if k not in ('scope','events')]
    totals={k:sum(v[k] for v in entries) for k in keys}
    totals['producer_episode_calls']+=cached_work['producer_episode_calls']
    for k in WORK_FIELDS:totals[k]+=cached_work[k]
    return dict(schema='monthly_weak_tree261_execution_accounting_v1',model_invocations=len(entries),physical122_reuses_main=True,entries=entries,cached_decisions=cached_work,totals=totals,original_complete_model_refits=0,matched_ridge_is_actual_calibration_refit=True,manual_traversal_is_model_fit=False)


def preparation_evidence(e,model_audit,decision_audit):
    import monthly_weak_tree261_independent as ind
    kinds=[('control',ind.CONTROL_FLAGS),('models',ind.MODEL_FLAGS),('incumbent',ind.INCUMBENT_FLAGS),('selection',ind.SELECTION_FLAGS)]
    for key,flags in kinds:receipt(e[key],122,flags,key,width=49)
    require(e['all123_order_prefixes'] is True and e['original_arrays_unchanged'] is True and e['original_training_unchanged'] is True,'Preservation proofs')
    require(e['main_model_counts']==model_audit['model'],'Saved main model audit binding')
    model_preservation(model_audit['model'],model_audit['labels'],122)
    fit_binding(e['control'],e['models'],model_audit['model'],122)
    receipt(e['early_incumbent'],122,ind.INCUMBENT_FLAGS,'incumbent');matched_strategy_check(e['matched_strategy'],122)
    require(decision_audit['schema']=='monthly_weak_tree261_decisions_v1' and len(decision_audit['events'])==122 and decision_audit['counts']==e['selection']['counts'],'Saved decision audit scope')
    require(decision_audit['incumbent_gate']==e['incumbent'],'Saved pre-candidate incumbent receipt')
    for key in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact'):
        require(decision_audit[key] is True,'Decision preservation:'+key)
    require(decision_audit['computes_returns'] is False,'Decisions before outcomes')
    cached=e['cached_prefixes'];physical=e['physical_prefixes']
    require(type(cached) is list and len(cached)==123,'All cached prefixes')
    for n,v in enumerate(cached):
        integer(v['prefix'],'Cached prefix',n);require(v['exact'] is True,'Exact cached prefix')
        receipt(v['incumbent'],n,ind.INCUMBENT_FLAGS,'incumbent')
        receipt(v['selection'],n,ind.SELECTION_FLAGS,'selection',width=49)
    require(type(physical) is list and len(physical)==len(PREFIXES),'All model prefixes')
    for n,v in zip(PREFIXES,physical):
        integer(v['prefix'],'Physical prefix',n);require(v['exact'] is True,'Exact physical prefix')
        for key,flags in kinds:receipt(v['model' if key=='models' else key],n,flags,key,width=49)
        fit_binding(v['control'],v['model'],v['model_counts'],n)
        receipt(v['early_incumbent'],n,ind.INCUMBENT_FLAGS,'incumbent');matched_strategy_check(v['matched_strategy'],n)
        if n==122:
            require(v['model_recomputed'] is False and v.get('reuses_main_verified_full') is True,'Physical122 reuse, no new fit')
            for key,_ in kinds:require(v['model' if key=='models' else key]==e[key],'Physical122 main receipt reuse')
            require(v['model_counts']==e['main_model_counts'] and v['early_incumbent']==e['early_incumbent'] and v['matched_strategy']==e['matched_strategy'],'Physical122 main evidence reuse')
        else:
            require(v['model_recomputed'] is True and v.get('reuses_main_verified_full',False) is False,'Physically recomputed prefix')
            physical_scope_check(v['input_scope'],n,v['model_counts']['counts'])
    require(e['fit_execution_accounting']==execution_accounting(e),'Actual solve accounting with122 excluded from repeat count')


def source_evidence(source,plan):
    require(source['raw_source_hashes']==plan['raw_source_hashes'],'Saved raw source pins')
    require(source['original169_source_closure']==plan['original169_source_closure'],'Saved original169 full source closure')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        require(source[key]==plan['input_sha256'][str(REBUILD/name)],'Registered cached source:'+name)
    require(source['fresh_training_array_reconstruction'] is False and source['cached_training_basis']=='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.','Honest cache provenance')
    for key in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact','original_scores_exact','matched_ridge_calibration_refits_counted'):require(source[key] is True,'Source proof:'+key)
    require(source['original_other_target_fits']==source['original_local_fits']==0,'Other original models reused')


def model_callbacks(ind,train,metadata,arrays,n,*,study,original,expected_orders):
    import monthly_weak_tree261_kernel as kernel
    receipts={}
    def control(view):
        require(not receipts,'Control gate once')
        before=old.array_snapshot(arrays);tb=old.array_snapshot(train);vb=structured_digest(view)
        check=ind.verify_control_models(train,metadata,arrays,view,n=n);receipt(check,n,ind.CONTROL_FLAGS,'control')
        bundle_preservation(view,n,True)
        require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and vb==structured_digest(view),'Control gate mutated source/cohort')
        # The independently verified matched rank equals the original rank everywhere.
        # Therefore its declared .0375 rank difference is exactly zero; replay only
        # the original selector on this identical P before releasing any tree fit.
        matched=kernel.matched_decisions(study,arrays,metadata['monthly'],view,n=n)
        exact_single(original,matched,'Matched pre-tree zero-delta strategy')
        require(matched[0]==expected_orders,'Matched frozen orders')
        strategy=ind.verify_incumbent_selection(arrays,old.meta_prefix(metadata['monthly'],n),*matched,n=n)
        receipt(strategy,n,ind.INCUMBENT_FLAGS,'incumbent')
        require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and vb==structured_digest(view),'Matched gate mutated source')
        receipts.update(control=check,matched_strategy=dict(passed=True,events=n,matched_ranks_exact=True,delta_exact_zero=True,pre_scores_exact=True,final_scores_exact=True,ordered_choices_exact=True,producer_episode_calls=1,independent=strategy))
        return check
    return control,receipts


def incumbent_callback(ind,arrays,meta,n,expected_orders=None):
    receipts={}
    if expected_orders is not None:require(type(expected_orders) is list and len(expected_orders)==n,'Complete frozen incumbent prefix')
    def incumbent(orders,scores,pres):
        require(not receipts,'Incumbent gate once')
        if expected_orders is not None:require(orders==expected_orders,'Frozen incumbent before new ranks/selection')
        check=ind.verify_incumbent_selection(arrays,meta,orders,scores,pres,n=n);receipt(check,n,ind.INCUMBENT_FLAGS,'incumbent')
        receipts['incumbent']=check;return check
    return incumbent,receipts

def exact_prefix(wanted,got,n):
    oo,ss,pp,_=wanted;orders,scores,pres,_=got
    require(set(orders)==set(scores)==set(pres)==set(ARMS),'Exactly three decision arms')
    for arm in ARMS:
        require(orders[arm]==oo[arm][:n],'Order prefix')
        np.testing.assert_array_equal(scores[arm],ss[arm][:n]);np.testing.assert_array_equal(pres[arm],pp[arm][:n])


def bundle_parity(full,part,n):
    q=n*full['width']
    for role in ROLES:
        for key in ('raw','available'):np.testing.assert_array_equal(part[role][key],full[role][key][:q])
        np.testing.assert_array_equal(part[role]['rank'],full[role]['rank'][:n])
    wanted={int(row):value for row,value in zip(full['training_row_ids'],full['label_values'])}
    np.testing.assert_array_equal(part['label_values'],np.array([wanted[int(row)] for row in part['training_row_ids']],dtype=full['label_values'].dtype))

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

def exact_single(expected,actual,label):
    require(actual[0]==expected[0],label+' ordered choices')
    for a,b in zip(actual[1:],expected[1:]):
        require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label+' exact P/F bytes')

def matched_strategy_check(value,n):
    import monthly_weak_tree261_independent as ind
    integer(value['events'],'Matched strategy events',n)
    for k in ('passed','matched_ranks_exact','delta_exact_zero','pre_scores_exact','final_scores_exact','ordered_choices_exact'):require(value[k] is True,'Pre-tree matched strategy:'+k)
    integer(value['producer_episode_calls'],'Pre-tree matched episode',1)
    receipt(value['independent'],n,ind.INCUMBENT_FLAGS,'incumbent')

def prepare_one(study,train,metadata,arrays,ind,n,expected_orders,expected_original=None):
    import monthly_weak_tree261_kernel as kernel
    before=old.array_snapshot(arrays);tb=old.array_snapshot(train)
    original=kernel.original_decisions(study,arrays,metadata['monthly'],n=n)
    require(original[0]==expected_orders,'Frozen original before any models')
    if expected_original is not None:exact_single(expected_original,original,'Frozen original source')
    early=ind.verify_incumbent_selection(arrays,old.meta_prefix(metadata['monthly'],n),*original,n=n);receipt(early,n,ind.INCUMBENT_FLAGS,'incumbent')
    require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train),'Early source mutation')
    cb,proof=model_callbacks(ind,train,metadata,arrays,n,study=study,original=original,expected_orders=expected_orders)
    bundle=kernel.fit_models(train,metadata,arrays,n=n,control_validator=cb)
    require(set(proof)=={'control','matched_strategy'} and bundle['control_gate']==proof['control'],'Both pre-tree gates required')
    digest=structured_digest(bundle)
    model=ind.verify_models(train,metadata,arrays,bundle,n=n);receipt(model,n,ind.MODEL_FLAGS,'models')
    require(digest==structured_digest(bundle),'Model verifier mutated supplied model')
    bundle_preservation(bundle,n)
    aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(metadata['monthly'],n)
    ig,ip=incumbent_callback(ind,aa,mm,n,expected_orders)
    got=kernel.decisions(study,aa,mm,bundle,n=n,incumbent_validator=ig,expected_incumbent_orders=expected_orders)
    require(set(ip)=={'incumbent'},'Decision incumbent gate')
    oo,ss,pp,au=got
    selection=ind.verify_selection(aa,mm,bundle,oo,scores=ss,pre_scores=pp,audit=au,n=n);receipt(selection,n,ind.SELECTION_FLAGS,'selection',width=arrays['baseline'].shape[1])
    exact_single(original,(oo[ARMS[0]],ss[ARMS[0]],pp[ARMS[0]]),'Original returned')
    exact_single(original,(oo[ARMS[1]],ss[ARMS[1]],pp[ARMS[1]]),'Matched returned')
    require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and digest==structured_digest(bundle),'Prepare immutable source/models')
    evidence=dict(early_incumbent=early,matched_strategy=proof['matched_strategy'],control=proof['control'],models=model,incumbent=ip['incumbent'],selection=selection,main_model_counts=bundle['audit'])
    return bundle,got,evidence



def physical_inputs(train,metadata,arrays,n):
    """Actually shorten monthly axes and whole mature training months, from row zero.

    No training body is accessed for n0/partial-only. Keeping a prefix starting
    at zero preserves the original training-row identifiers without rebasing.
    """
    aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(metadata['monthly'],n)
    w=aa['baseline'].shape[1];ordinary=[d for d in mm['dates'] if d!='2016-08-17']
    empty=dict(dates=[],ends=[],symbols=[])
    if not ordinary:
        scope=dict(events=n,training_months=0,training_rows=0,cutoff=None,global_row_ids_start=0,global_row_ids_preserved=True,training_numeric_consumed=False,physically_shortened=True)
        return {},dict(monthly=mm,training=empty),aa,scope
    cutoff='%04d-12-31'%(int(ordinary[-1][:4])-1);tm=metadata['training']
    ends=np.asarray(tm['ends'],dtype='datetime64[D]')
    require(ends.ndim==1 and not np.isnat(ends).any() and len(tm['dates'])==len(ends)==len(tm['symbols']),'Physical training date headers')
    mask=ends<=np.datetime64(cutoff);require(not np.any(mask[1:]&~mask[:-1]),'Physical mature whole-month prefix')
    count=int(mask.sum());offsets=np.asarray(train['offsets'])
    require(offsets.shape==(len(ends)+1,) and offsets.dtype.kind in 'iu' and offsets[0]==0 and np.all(np.diff(offsets)>0),'Physical offset headers')
    stop=int(offsets[count]);rows=int(offsets[-1]);total=arrays['baseline'].shape[0]
    shapes={'dates':(rows,),'ends':(rows,),'train_plain':(rows,10),'common_good':(rows,),'event_dates':(total*w,),'event_plain':(total*w,10)}
    shapes.update({'target_'+k:(rows,) for k in ('simple_excess','log_excess','vol_scaled','loss_utility')})
    # Read only array structure before slicing; no future numeric conversion.
    small={}
    for key,shape in shapes.items():
        raw=train[key];require(isinstance(raw,np.ndarray) and raw.shape==shape,'Physical source static shape:'+key)
        small[key]=raw[:(n*w if key.startswith('event_') else stop)].copy()
    small['offsets']=offsets[:count+1].copy()
    mt={key:tm[key][:count] for key in ('dates','ends','symbols')}
    scope=dict(events=n,training_months=count,training_rows=stop,cutoff=cutoff,global_row_ids_start=0,global_row_ids_preserved=True,training_numeric_consumed=bool(stop),physically_shortened=True)
    return small,dict(monthly=mm,training=mt),aa,scope


def physical_scope_check(scope,n,model_counts):
    require(type(scope) is dict and set(scope)=={'events','training_months','training_rows','cutoff','global_row_ids_start','global_row_ids_preserved','training_numeric_consumed','physically_shortened'},'Physical input scope fields')
    integer(scope['events'],'Physical input events',n);integer(scope['global_row_ids_start'],'Original row origin',0)
    integer(scope['training_months'],'Physical training months');integer(scope['training_rows'],'Physical training rows',model_counts['training_rows'])
    require(scope['global_row_ids_preserved'] is True and scope['physically_shortened'] is True,'Physical source restriction and original IDs')
    require(scope['training_numeric_consumed'] is bool(scope['training_rows']),'Physical training consumption')
    if n<=1:require(scope['cutoff'] is None and scope['training_months']==scope['training_rows']==0,'No partial training consumption')
    else:require(type(scope['cutoff']) is str and len(scope['cutoff'])==10 and scope['cutoff'].endswith('-12-31'),'Physical prior-year boundary')

def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_weak_tree261_kernel as kernel
    import monthly_weak_tree261_independent as ind
    study=old.local();m,arrays,px,op=study.load()
    require(len(m['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 arrays')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');metadata=read(REBUILD/'MONTHLY_META.json')
    require(metadata['monthly']==m and metadata['training']==read(study.arch.P/'META.json')['training'],'Original identities')
    prior=read(PRIOR/'prepared/COMPLETION.json')
    for name in ('SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json'):
        require(prior['artifacts_sha256'][name]==sha(PRIOR/'prepared'/name),'Prior prepared binding')
    require(read(PRIOR/'prepared/META.json')==m,'Prior identity')
    original_orders=read(LOCAL/'local_linear_ORDERS.json')
    require(read(PRIOR/'prepared/ORDERS.json')[ARMS[0]]==original_orders,'Prior/source ordered choices')
    original_scores=ev.load_arrays(PRIOR/'prepared/SCORES.npz')[ARMS[0]]
    original_pres=ev.load_arrays(PRIOR/'prepared/PRE_SCORES.npz')[ARMS[0]]
    before=old.array_snapshot(arrays);tb=old.array_snapshot(train)
    bundle,wanted,e=prepare_one(study,train,metadata,arrays,ind,122,original_orders,(original_orders,original_scores,original_pres))
    cached=[];physical=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(m,n)
        cb,cr=incumbent_callback(ind,aa,mm,n,original_orders[:n])
        snapshot=structured_digest(bundle)
        got=kernel.decisions(study,aa,mm,bundle,n=n,incumbent_validator=cb,expected_incumbent_orders=original_orders[:n])
        exact_prefix(wanted,got,n);require(set(cr)=={'incumbent'},'Cached incumbent gate')
        oo,ss,pp,au=got
        ci=ind.verify_selection(aa,mm,bundle,oo,scores=ss,pre_scores=pp,audit=au,n=n);receipt(ci,n,ind.SELECTION_FLAGS,'selection',width=49)
        require(snapshot==structured_digest(bundle),'Cached model mutation')
        cached.append(dict(prefix=n,exact=True,incumbent=cr['incumbent'],selection=ci))
        if n in PREFIXES and n!=122:
            print(json.dumps(dict(stage='261_physical_model_prefix',events=n)),flush=True)
            pt,pm,pa,scope=physical_inputs(train,metadata,arrays,n)
            small,gg,pe=prepare_one(study,pt,pm,pa,ind,n,original_orders[:n],(original_orders[:n],original_scores[:n],original_pres[:n]))
            bundle_parity(bundle,small,n);exact_prefix(wanted,gg,n)
            pe.update(prefix=n,exact=True,model_recomputed=True,input_scope=scope,model=pe.pop('models'),model_counts=pe.pop('main_model_counts'));physical.append(pe)
    reuse={k:v for k,v in e.items()};reuse.update(prefix=122,exact=True,model_recomputed=False,reuses_main_verified_full=True,model=reuse.pop('models'),model_counts=reuse.pop('main_model_counts'));physical.append(reuse)
    e.update(cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True)
    e['fit_execution_accounting']=execution_accounting(e)
    require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train),'Original arrays/training changed')
    orders,scores,pres,audit=wanted
    values={'MODEL_AUDIT.json':dict(model=bundle['audit'],labels=bundle['label_preservation'],control=bundle['control']['audit'],candidate=bundle['candidate']['audit']),
      'DECISION_AUDIT.json':audit,'ORDERS.json':orders,'META.json':m,
      'SOURCE_PARITY.json':dict(original169_source_closure=plan['original169_source_closure'],raw_source_hashes=plan['raw_source_hashes'],original_training_hash=sha(REBUILD/'TRAIN_INPUTS.npz'),original_identity_hash=sha(REBUILD/'MONTHLY_META.json'),saved169_parity_hash=sha(REBUILD/'PARITY.json'),cached_training_basis='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.',fresh_training_array_reconstruction=False,original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True,original_scores_exact=True,original_other_target_fits=0,original_local_fits=0,matched_ridge_calibration_refits_counted=True),
      'INDEPENDENT_PREPARE.json':e}
    preparation_evidence(e,values['MODEL_AUDIT.json'],audit);source_evidence(values['SOURCE_PARITY.json'],plan)
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{role+'_'+key:bundle[role][key] for role in ROLES for key in ('raw','available','rank')})
    np.savez_compressed(dest/'LABELS.npz',training_row_ids=bundle['training_row_ids'],label_values=bundle['label_values'])
    for name,value in [('SCORES',scores),('PRE_SCORES',pres)]:np.savez_compressed(dest/(name+'.npz'),**value)
    for name,value in values.items():dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,model_fit_accounting='Calibration Ridge and tree fits plus independent normal/SVD/refits/traversals; physical122 reuses main;123 cached prefixes fit zero models.'))
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
    invalid={a:required_metrics(metrics[a],a!=ARMS[2]) for a in ARMS}
    require(all(len(orders[a])>=88 for a in ARMS),'First88 holdings required')
    require(metrics[ARMS[1]]==metrics[ARMS[0]] and orders[ARMS[1]][:88]==orders[ARMS[0]][:88],'Matched calibration exact incumbent')
    candidate=ARMS[2];reference=ARMS[0]
    changed=[i for i in range(88) if set(orders[candidate][i]['symbols'])!=set(orders[reference][i]['symbols'])]
    close=not invalid[candidate] and old.guard(metrics[candidate]['same_close_zero'],metrics[reference]['same_close_zero'],.005)
    opening=not invalid[candidate] and old.guard(metrics[candidate]['next_open_zero'],metrics[reference]['next_open_zero'],0.)
    support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed)
    guards=dict(close=bool(close),open=bool(opening),support=bool(support));admitted=all(guards.values())
    return dict(primary=candidate if admitted else reference,admitted=admitted,guards=guards,changed_months=changed,matched_exact_incumbent=True,
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
    import monthly_weak_tree261_independent as ind
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
    import monthly_weak_tree261_independent as ind
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
    require(len(state['completed'])==257 and sum(x['trials'] for x in state['completed'])==1282,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review weak pooled-tree replication; matched is exact-source calibration; first88 choice immutable and frozen observers unchanged.')
    old.commit_state(before,state,'weak_tree261_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

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
