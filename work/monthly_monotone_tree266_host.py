"""266 lifecycle: exact261 control before conditional-monotone candidate.

All actual source/model/outcome work requires the final prospective PLAN.
Synthetic helpers operate only on caller-supplied fixtures.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_monotone_tree266'
PRIOR=ROOT/'outputs/monthly_gain_concentration265'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CONTEXT=ROOT/'work/monthly_monotone_tree266_context.json'
CONTRACT=ROOT/'work/monthly_monotone_tree266_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_monotone_tree266_TESTS.json'
REVIEW=ROOT/'work/monthly_monotone_tree266_CODE_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_gain_concentration265_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_monotone_tree266_RESULT_REVIEW.json'
ARMS=('incumbent_local67','unconstrained_weak_tree261_control','monotone_raw_weak_tree_candidate')
ROLES=('calibration_ridge','unconstrained_control','monotone_candidate')
PHASE_TERMINALS=OUT/'PHASE_TERMINALS.json'
SOURCE_REVIEW=ROOT/'work/after260_gbdt_slot_source_review.json'
SOURCE_REVIEW_SHA='35d2ed160d0a4e1c248492961c26364abb80ea22441c67fa7186656b689adcb5'
RUNTIME_PINS=ROOT/'work/monthly_weak_tree261_RUNTIME_PINS.json'
RUNTIME_REVIEW=ROOT/'work/after260_gbdt_runtime_review.json'
REFERENCE=ROOT/'outputs/monthly_weak_tree261'
REFERENCE_SHA={ROOT/k:v for k,v in {'outputs/monthly_weak_tree261/COMPLETION.json': 'fd84f8bd94b161e8ef393592bdcd3ba42e615363dda2f3877c479dff1feaed68', 'outputs/monthly_weak_tree261/PLAN.json': 'bb603fc8dd82aa52b159d7dd4c15b02f91dc2d210152267751577be2dadb3bd5', 'outputs/monthly_weak_tree261/prepared/COMPLETION.json': '844b2a0f012f8ad5cd7974aadeb1a15f19682b97a5ecbfafef2dbb3fc83c070b', 'outputs/monthly_weak_tree261/prepared/DECISION_AUDIT.json': 'acfad4b1a6328bf75a33d9d644438dcca2fb0b89c7a8723501bf8dfed8f6b958', 'outputs/monthly_weak_tree261/prepared/FORECASTS.npz': 'd1c7e8a33b703ad8a18046a829e865feb4206c58fb3c1feff474567f2521d689', 'outputs/monthly_weak_tree261/prepared/INDEPENDENT_PREPARE.json': '55225eff1111ab304fc02918326a762d75f5ea466b082060c20031c9e434c3ae', 'outputs/monthly_weak_tree261/prepared/LABELS.npz': '56a78fa25ddb68107860b59c99bc9cfc6025464dec142866eb0f21c3d3a77893', 'outputs/monthly_weak_tree261/prepared/META.json': '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4', 'outputs/monthly_weak_tree261/prepared/MODEL_AUDIT.json': '76bc5b072dae496ecbb17d35f3d0441deb88bd6704bf038be849a0ea04ad7957', 'outputs/monthly_weak_tree261/prepared/ORDERS.json': '887ccb6abc09e72a11fb4f5f4a7be4506a090047ac96d44100a1d811e04d328b', 'outputs/monthly_weak_tree261/prepared/PRE_SCORES.npz': 'ee644f6c443cb6dac973508603ece4d615e4b2a15362640d9b756ca328a108b4', 'outputs/monthly_weak_tree261/prepared/SCORES.npz': '047f721e4dd3f22e204848608cc28d2efbbb75e88c8812c9499aa05e1fc20f1e', 'outputs/monthly_weak_tree261/prepared/SOURCE_PARITY.json': '39c523d10a591aa3d0c6927f875b92a98a0a8ba117bf7ef1cbe6e553b2d41e7a'}.items()}
PREFIXES=(0,1,53,88,122)
WORK_FIELDS=('independent_canonical_solves', 'independent_svd_solves', 'independent_tree_refits', 'unconstrained_tree_refits', 'monotone_tree_refits', 'manual_tree_traversals', 'unconstrained_manual_tree_traversals', 'monotone_manual_tree_traversals', 'shape_certificates_checked', 'constrained_nodes_checked', 'original_episode_calls', 'original_other_target_fits', 'original_local_fits')
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.0,selection_tolerance=0.0,independent_numeric_system='independent canonical Ridge plus augmented SVD; shared pinned sklearn dual-tree refit, independent manual traversal and structural monotonicity certificate')
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=tuple(name for stem in ('host','kernel','independent','admission','reference') for name in ('monthly_monotone_tree266_'+stem+'.py','test_monthly_monotone_tree266_'+stem+'.py'))
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
    return dict(schema='monthly_monotone_tree266_original169_closure_v1',chain_sha256={str(k):v for k,v in SOURCE_CHAIN169.items()},artifacts_sha256=dict(artifacts),source_input_count=len(plan['input_sha256']),npz_decoded=False,source_cache_rebuilt=False)


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
    mods=[old.module('monotone_tree266_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5,7,9))]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in mods]);stream=io.StringIO()
    r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    require(r.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=r.testsRun,runtime=ev.runtime(),code_sha256=codes()))


def code_review():
    r=read(REVIEW)
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['code_sha256']==codes(),'Numerical code review')
    require(r['context_sha256']==sha(CONTEXT) and r['implementation_contract_sha256']==sha(CONTRACT) and r['actual_returns_computed'] is False and type(r['input_sha256']) is dict,'Prospective review scope')
    return r


def initial_state(state):
    require(len(state['completed'])==261 and sum(x['trials'] for x in state['completed'])==1294,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Predecessor changed')


def predecessor_binding(state):
    context,_=contract_check();wanted=context['predecessor'];record=state['completed'][-1]['completion']
    require(Path(record['path'])==PRIOR/'COMPLETION.json' and record['sha256']==sha(PRIOR/'COMPLETION.json')==wanted['completion_sha256'],'Predecessor central closure binding')
    prior=read(PRIOR/'COMPLETION.json');review=read(PRIOR_REVIEW)
    require(prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior['protected_files']==21444 and prior['plan_sha256']==sha(PRIOR/'PLAN.json'),'Independently closed265 required')
    require(prior['result_review_sha256']==wanted['result_review_sha256']==sha(PRIOR_REVIEW),'Final predecessor review binding')
    require(review['passed'] is True and review['unresolved_blockers']==[] and review['plan_sha256']==sha(PRIOR/'PLAN.json') and review['results_completion_sha256']==sha(PRIOR/'results/COMPLETION.json'),'Predecessor independent saved review')
    require(prior.get('target_reached') is False and state['best_training_selected_candidate']['cagr']<.7,'Goal already reached; successor not permitted')


def registration_check(state,expected):
    initial_state(state)
    require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration absent')


def contract_check():
    require(CONTEXT.is_file() and CONTRACT.is_file(),'Final prospective context and contract required; drafts cannot register')
    c=read(CONTEXT);d=read(CONTRACT)
    for record in (c,d):
        require(record.get('frozen') is True and record.get('registration_authorized') is True and record.get('actual_data_started') is False,'Final approved prospective266 contracts')
        require(type(record.get('input_sha256')) is dict,'Explicit prospective dependency pins')
    require(c['arms']==list(ARMS) and c['registered_trials']==3 and c['only_promotable']==ARMS[2] and c['diagnostic_only']==ARMS[1],'Three economic arms; internal Ridge is not control')
    require(c.get('actual_models_fitted')==0 and type(c.get('actual_models_fitted')) is int and c.get('actual_orders_computed') is False and c.get('actual_returns_computed') is False,'Prospective actual266 scope')
    formal=['work/'+name for name in code_names]
    supplemental=['work/monthly_monotone_tree266_saved_reader.py','work/test_monthly_monotone_tree266_saved_reader.py']
    require(type(d.get('code_files')) is list and d['code_files']==formal,'Exact ordered formal10 code inventory')
    require(type(d.get('supplemental_input_files')) is list and d['supplemental_input_files']==supplemental,'Exact ordered reader/test supplemental inventory')
    for label,record in (('context',c),('contract',d)):
        for relative in supplemental:
            value=record['input_sha256'].get(relative);digest_string(value,label+' required saved reader pin:'+relative)
            path=ROOT/relative;require(path.is_file() and value==sha(path),label+' current saved reader source binding:'+relative)
    require(d['context_sha256']==sha(CONTEXT),'Contract exact context binding')
    prior=c['predecessor'];require(prior['directory']==PRIOR.name,'Immediate predecessor265')
    for key in ('completion_sha256','result_review_sha256'):digest_string(prior.get(key),'Predecessor '+key)
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Inherited independent source review')
    for path in (RUNTIME_PINS,RUNTIME_REVIEW):
        require(c['input_sha256'].get(str(path.relative_to(ROOT)))==sha(path),'Runtime dependency binding')
    return c,d


def register():
    context,contract=contract_check()
    require(not OUT.exists(),'Fresh experiment directory')
    before=CENTRAL.read_bytes();state=json.loads(before);initial_state(state);require(not state['registered_queue'],'No pending experiment')
    predecessor_binding(state)
    t=read(TESTS);require(t['passed'] is True and t['synthetic_only'] is True and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Exact formal10-code tests')
    review=code_review();require(review['implementation_contract_sha256']==sha(CONTRACT),'Review contract binding')
    pins=dict(read(PRIOR/'PLAN.json')['input_sha256']);source_closure=bind_source_closure(pins)
    for path,h in REFERENCE_SHA.items():old.bind(pins,path,h)
    historical=reference_closure_check(pins)
    paths=files()+[CONTEXT,CONTRACT,TESTS,REVIEW,SOURCE_REVIEW,RUNTIME_PINS,RUNTIME_REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',PRIOR_REVIEW,old.PROTECTED]
    paths += [PRIOR/'prepared'/name for name in ('COMPLETION.json','SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json')]
    for path in paths:old.bind(pins,path)
    for source in (context,contract,review,read(SOURCE_REVIEW),read(RUNTIME_PINS),read(RUNTIME_REVIEW)):
        for k,v in source['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_monotone_tree266_independent as ind
    import monthly_monotone_tree266_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_monotone_tree266_v1',original169_source_closure=source_closure,historical261_reference=historical,created_utc=now(),context=context,context_sha256=sha(CONTEXT),implementation_contract=contract,implementation_contract_sha256=sha(CONTRACT),runtime=ev.runtime(),input_sha256=pins,raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,actual_labels_before_registration=0,actual_predictions_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits='Actual calibration Ridge and two fixed tree fits; independent canonical/SVD and per-stage tree refits/traversals/shape work separately counted; original local/other targets reused',numerical_verification=NUMERICAL_VERIFICATION)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),last_goal_turn_classification='progress',next_step='Exact original Ridge and frozen261 unconstrained control before monotone fit; first88 dual-reference ZERO admission only.')
    old.commit_state(before,state,'monotone_tree266_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');c,d=contract_check()
    require(p['numerical_verification']==NUMERICAL_VERIFICATION,'Prospective numerical verification unchanged')
    require(p['context_sha256']==sha(CONTEXT) and p['context']==c and p['implementation_contract_sha256']==sha(CONTRACT) and p['implementation_contract']==d and p['runtime']==ev.runtime(),'Context/contract/runtime changed')
    old.verify(p['input_sha256']);require(p['original169_source_closure']==source_closure_check(p['input_sha256']),'Original169 complete source chain')
    require(p['historical261_reference']==reference_closure_check(p['input_sha256']),'Complete frozen261 comparator chain')
    r=code_review();require(r['implementation_contract_sha256']==sha(CONTRACT),'Reviewed contract')
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed');state=read(CENTRAL);registration_check(state,expected);predecessor_binding(state)
    return p


def integer(value,label,expected=None):
    require(type(value) is int and value>=0,label+' nonnegative native int')
    if expected is not None:require(value==expected,label+' exact value')


def counts_check(value,fields,label):
    require(type(value) is dict and set(value)==set(fields),label+' exact fields')
    for key,item in value.items():integer(item,label+'/'+key)








def model_counts_check(value,n,stage='full'):
    import monthly_monotone_tree266_independent as ind
    require(stage in ('calibration','control','full'),'Explicit model stage');counts_check(value,ind.MODEL_COUNT_FIELDS,'Model counts');integer(value['events'],'Model events',n)
    require(value['valid_training_rows']<=value['training_rows'],'Cohort count')
    require(value['ordinary_years']==value['supported_years']+value['cold_years'] and value['supported_years']==value['matched_ridge_fits']+value['no_prediction_years'],'Annual fit statuses')
    for prefix,active in (('unconstrained',stage!='calibration'),('monotone',stage=='full')):
        fits=value[prefix+'_tree_fits'];require(fits==(value['matched_ridge_fits'] if active else 0),'Stage-specific actual fits')
        require(value[prefix+'_trees']==value[prefix+'_boosting_iterations']==100*fits,'Fixed100 iterations per fit')
        require(value[prefix+'_manual_tree_traversals']==(100*value['prediction_rows'] if active else 0),'Producer actual query-tree walks')
        require(value[prefix+'_trees']<=value[prefix+'_tree_nodes']<=7*value[prefix+'_trees'],'Depth2 actual node bounds')
    require(value['actual_model_fits']==value['matched_ridge_fits']+value['unconstrained_tree_fits']+value['monotone_tree_fits'],'Actual producer calibration and two trees')
    require(value['original_other_target_fits']==value['original_local_fits']==0,'Original other-target/local cache reused')
    require(value['matched_ridge_fits']<=value['prediction_rows'],'Fits require query U')
    if n==0 or value['ordinary_years']==0:require(all(v==0 for k,v in value.items() if k!='events'),'Empty/partial no training numeric or fits')


def model_preservation(model,labels,n,stage='full'):
    import monthly_monotone_tree266_independent as ind
    require(model['schema']=='monthly_monotone_tree266_model_audit_v1','Model audit schema');c=model['counts'];model_counts_check(c,n,stage)
    runtime=model.get('execution_runtime');require(type(runtime) is list and len(runtime)==c['unconstrained_tree_fits']+c['monotone_tree_fits'],'Actual per-role native runtime evidence')
    seen=[]
    for item in runtime:
        require(type(item) is dict and set(item)=={'role','year','fit_threadpools','predict_threadpools'} and item['role'] in ROLES[1:],'Runtime role record');integer(item['year'],'Runtime year');seen.append((item['role'],item['year']))
        for phase in ('fit_threadpools','predict_threadpools'):
            require(type(item[phase]) is list and bool(item[phase]),'Loaded native pools')
            for pool in item[phase]:
                require(type(pool) is dict and type(pool.get('filepath')) is str and bool(pool['filepath']),'Native pool identity');integer(pool.get('num_threads'),'Native threads',1)
    expected=[]
    for role,prefix in zip(ROLES[1:],('unconstrained','monotone')):
        years=[y for a,y in seen if a==role];require(years==sorted(set(years)) and len(years)==c[prefix+'_tree_fits'],'Chronological per-role runtime years');expected.extend((role,y) for y in years)
    require(seen==expected,'Unconstrained before monotone runtime order')
    for key in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','local_models_reused','other_targets_not_fitted','weak_only'):require(model.get(key) is True,'Model preservation:'+key)
    require(model.get('computes_returns') is False,'No model outcomes')
    parity=model['original_cache_parity'];require(set(parity)=={'passed','events','original_raw_exact','original_availability_exact','original_ranks_exact','candidate_models_computed'},'Original cache proof')
    integer(parity['events'],'Original events',n)
    for key in ('passed','original_raw_exact','original_availability_exact','original_ranks_exact'):require(parity[key] is True,'Original exact cache')
    require(parity['candidate_models_computed'] is False,'Original calibration before candidate')
    fields={'schema','rows','valid_rows','target_keys','target_sha256','common_good_sha256','labels_unchanged','common_good_unchanged','original_row_order_preserved','recentered','raw_target_reconstructed'}
    require(type(labels) is dict and set(labels)==fields and labels['schema']=='monthly_monotone_tree266_labels_preserved_v1','Labels schema')
    integer(labels['rows'],'Consumed training rows',c['training_rows']);integer(labels['valid_rows'],'Good rows');require(c['valid_training_rows']<=labels['valid_rows']<=labels['rows'],'Usable/good rows')
    require(labels['target_keys']==list(ind.TARGETS) and set(labels['target_sha256'])==set(ind.TARGETS),'All4 source targets')
    for h in list(labels['target_sha256'].values())+[labels['common_good_sha256']]:digest_string(h,'Target/common good')
    for key in ('labels_unchanged','common_good_unchanged','original_row_order_preserved'):require(labels[key] is True,'Preserved labels/source')
    require(labels['recentered'] is False and labels['raw_target_reconstructed'] is False,'No new label reconstruction')


def bundle_preservation(bundle,n,stage='full'):
    import monthly_monotone_tree266_independent as ind
    require(set(bundle)==set(ind.BUNDLE_KEYS),'Complete model bundle');model_preservation(bundle['audit'],bundle['label_preservation'],n,stage)
    integer(bundle['n'],'Bundle events',n);integer(bundle['width'],'Bundle width');require(bundle['width']>0,'Positive width')
    if stage=='calibration':require(bundle['unconstrained_control'] is bundle['monotone_candidate'] is bundle['calibration_gate'] is bundle['control_gate'] is None,'Ridge-only pre-control stage')
    elif stage=='control':require(bundle['monotone_candidate'] is None and bundle['control_gate'] is None,'Unconstrained pre-candidate stage')
    for role in ROLES:
        if bundle[role] is not None:
            a,b=bundle['calibration_ridge']['available'],bundle[role]['available'];require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'Exact original availability:'+role)


def receipt(check,n,flags,kind,*,width=None):
    import monthly_monotone_tree266_independent as ind
    kind={'model':'models'}.get(kind,kind)
    standard={'calibration':ind.CALIBRATION_FLAGS,'control':ind.CONTROL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'control_selection':ind.CONTROL_SELECTION_FLAGS,'selection':ind.SELECTION_FLAGS}
    require(kind in standard and type(check) is dict,'Known complete receipt');integer(check.get('checked_events'),'Checked events',n)
    require(check.get('passed') is True and all(check.get(k) is True for k in tuple(flags)+tuple(standard[kind])),'Required independent flags:'+kind)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent scope')
    for key in WORK_FIELDS:integer(check.get(key),'Actual work:'+key)
    require(check['original_other_target_fits']==check['original_local_fits']==0,'Cached local/other models')
    def flag(key):require(check.get(key) is True,'Full proof:'+key)
    def tol(key,value):require(type(check.get(key)) is float and np.isfinite(check[key]) and check[key]==value,'Exact tolerance:'+key)
    if kind in ('calibration','control','models'):
        stage='full' if kind=='models' else kind;c=check['counts'];model_counts_check(c,n,stage)
        for key in ('continuous_atol','continuous_rtol','rank_tolerance'):tol(key,NUMERICAL_VERIFICATION[key])
        for key in ('exact_discrete','full_bundle_supplied','full_calibration_supplied','full_label_preservation_supplied','full_model_audit_supplied'):flag(key)
        require(check['full_control_supplied'] is (stage!='calibration') and check['full_candidate_supplied'] is (stage=='full') and check['candidate_models_computed'] is (stage=='full'),'Exact pre-candidate stages')
        require(check['shared_tree_trainer'] is True and check['independent_tree_training_algorithm'] is False and check['independent_numeric_system']==NUMERICAL_VERIFICATION['independent_numeric_system'],'Honest shared optimizer scope')
        counts_check(check['producer_equivalent_counts'],ind.MODEL_COUNT_FIELDS,'Producer equivalent counts');require(check['producer_equivalent_counts']==c,'Full counts binding')
        require(check['independent_canonical_solves']==check['independent_svd_solves']==c['matched_ridge_fits'],'Executed independent Ridge/SVD')
        for prefix in ('unconstrained','monotone'):
            require(check[prefix+'_tree_refits']==c[prefix+'_tree_fits'] and check[prefix+'_manual_tree_traversals']==2*c[prefix+'_manual_tree_traversals'],'Own and supplied tree work')
        require(check['independent_tree_refits']==check['unconstrained_tree_refits']+check['monotone_tree_refits'],'All tree refits counted')
        require(check['manual_tree_traversals']==check['unconstrained_manual_tree_traversals']+check['monotone_manual_tree_traversals'],'All walks counted')
        require(check['shape_certificates_checked']==3*check['independent_tree_refits'],'Serialize/own/supplied shape checks')
        if stage!='full':integer(check['constrained_nodes_checked'],'No monotone nodes before candidate',0)
        integer(check['original_episode_calls'],'Model gate no selection',0)
    else:
        for key in ('full_scores_supplied','full_pre_scores_supplied'):flag(key)
        tol('numerical_tolerance',0.);calls=3 if kind=='selection' else 1;integer(check['ordered_choices_checked'],'Ordered choices',calls*n);integer(check['original_episode_calls'],'Actual selector invocations',calls)
        require(all(check[k]==0 for k in WORK_FIELDS if k!='original_episode_calls'),'Selection does not fit or traverse models')
        if kind!='selection':require(check['candidate_selection_computed'] is False,'Candidate selection not computed')
        if kind=='control_selection':flag('full_control_supplied')
        if kind=='selection':
            flag('full_bundle_supplied');flag('full_audit_supplied');c=check['counts'];counts_check(c,ind.COUNT_FIELDS,'Decision counts');integer(c['events'],'Decision events',n)
            require(c['symbols_per_event']>0 and c['initial_partial_events']+c['formal_events']==n,'Decision scope')
            if width is not None:integer(c['symbols_per_event'],'Actual width',width)
            require(c['incumbent_episode_calls']==c['control_episode_calls']==c['candidate_episode_calls']==1,'One original selector per arm including n0')
            require(c['new_model_fits']==c['predictive_model_fits']==0,'Selection no model fits')


def fit_binding(calibration,control,models,model_audit,n):
    after=model_audit['counts'];model_counts_check(after,n);require(models['counts']==after,'Independent full producer count binding')
    for prior,stage in ((calibration,'calibration'),(control,'control')):
        model_counts_check(prior['counts'],n,stage)
        for key,value in after.items():
            inactive=(key.startswith('unconstrained_') and stage=='calibration') or key.startswith('monotone_')
            expected=0 if inactive else value
            if key=='actual_model_fits':expected=after['matched_ridge_fits']*(1 if stage=='calibration' else 2)
            integer(prior['counts'][key],'Stage count:'+key,expected)


def execution_accounting(e):
    invocations=[('main',122,e)]+[('physical',v['prefix'],v) for v in e['physical_prefixes'] if v['prefix']!=122]
    entries=[]
    producer_fields=('matched_ridge_fits','unconstrained_tree_fits','unconstrained_boosting_iterations','unconstrained_trees','unconstrained_tree_nodes','unconstrained_manual_tree_traversals','monotone_tree_fits','monotone_boosting_iterations','monotone_trees','monotone_tree_nodes','monotone_manual_tree_traversals')
    for scope,n,v in invocations:
        models=v['models'] if scope=='main' else v['model'];audit=v['main_model_counts'] if scope=='main' else v['model_counts'];fit_binding(v['calibration'],v['control'],models,audit,n);c=audit['counts']
        checks=[v['early_incumbent'],v['calibration'],v['calibration_strategy']['independent'],v['control'],v['control_strategy']['independent'],models,v['incumbent'],v['selection']]
        item=dict(scope=scope,events=n,**{'producer_'+k:c[k] for k in producer_fields})
        item['producer_episode_calls']=1+v['calibration_strategy']['producer_episode_calls']+v['control_strategy']['producer_episode_calls']+sum(v['selection']['counts'][k] for k in ('incumbent_episode_calls','control_episode_calls','candidate_episode_calls'))
        item.update({k:sum(q[k] for q in checks) for k in WORK_FIELDS})
        item['total_model_fit_calls']=sum(item['producer_'+k] for k in ('matched_ridge_fits','unconstrained_tree_fits','monotone_tree_fits'))+item['independent_canonical_solves']+item['independent_svd_solves']+item['independent_tree_refits']
        entries.append(item)
    cached=e['cached_prefixes'];checks=[q[k] for q in cached for k in ('incumbent','selection')]
    cached_work=dict(invocations=len(cached),producer_episode_calls=sum(sum(q['selection']['counts'][k] for k in ('incumbent_episode_calls','control_episode_calls','candidate_episode_calls')) for q in cached),model_fit_calls=0,**{k:sum(q[k] for q in checks) for k in WORK_FIELDS})
    keys=[k for k in entries[0] if k not in ('scope','events')];totals={k:sum(v[k] for v in entries) for k in keys}
    totals['producer_episode_calls']+=cached_work['producer_episode_calls']
    for key in WORK_FIELDS:totals[key]+=cached_work[key]
    import monthly_monotone_tree266_reference as ref
    reference_work={k:sum(v['reference261_'+stage]['comparison']['counts'][k] for _,_,v in invocations for stage in ('calibration','control')) for k in ref.COUNT_FIELDS}
    return dict(schema='monthly_monotone_tree266_execution_accounting_v1',model_invocations=len(entries),physical122_reuses_main=True,entries=entries,cached_decisions=cached_work,totals=totals,historical_reference_comparison_work=reference_work,original_complete_model_refits=0,matched_ridge_is_actual_calibration_refit=True,manual_traversal_is_model_fit=False,source_load_scope='Full pinned reference JSON/NPZ decoded once; opaque full input hashes and immutable snapshots are not model fits or zero-read claims')


def preparation_evidence(e,model_audit,decision_audit):
    import monthly_monotone_tree266_independent as ind
    require(e['historical_reference_binding']==reference_binding(),'Exact all13 saved historical pins')
    kinds=[('calibration',ind.CALIBRATION_FLAGS),('control',ind.CONTROL_FLAGS),('models',ind.MODEL_FLAGS),('incumbent',ind.INCUMBENT_FLAGS),('selection',ind.SELECTION_FLAGS)]
    def one(v,n,physical=False):
        model_key='model' if physical else 'models';audit_key='model_counts' if physical else 'main_model_counts'
        for key,flags in kinds:receipt(v[model_key if key=='models' else key],n,flags,key,width=49)
        fit_binding(v['calibration'],v['control'],v[model_key],v[audit_key],n);receipt(v['early_incumbent'],n,ind.INCUMBENT_FLAGS,'incumbent')
        for stage in ('calibration','control'):
            strategy_check(v[stage+'_strategy'],n,stage=='calibration');reference_receipt(v['reference261_'+stage],n,49,stage,e['historical_reference_binding'])
        require(v['historical_reference_binding']==e['historical_reference_binding'],'Every physical historical chain')
        original=v['original_reference'];integer(original['events'],'Early original events',n)
        for key in ('passed','source_incumbent_orders_exact','prior_incumbent_exact','reference261_incumbent_exact'):require(original[key] is True,'Early original binding')
    one(e,122)
    require(e['all123_order_prefixes'] is True and e['original_arrays_unchanged'] is True and e['original_training_unchanged'] is True and e['historical_reference_unchanged'] is True,'Preparation preservation')
    require(set(model_audit)=={'model','labels','calibration_ridge','unconstrained_control','monotone_candidate'} and e['main_model_counts']==model_audit['model'],'Saved full model audit')
    model_preservation(model_audit['model'],model_audit['labels'],122)
    require(decision_audit['schema']=='monthly_monotone_tree266_decisions_v1' and len(decision_audit['events'])==122 and decision_audit['counts']==e['selection']['counts'] and decision_audit['incumbent_gate']==e['incumbent'],'Saved decision audit')
    for key in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact'):require(decision_audit[key] is True,'Decision preservation')
    require(decision_audit['computes_returns'] is False,'Preparation no outcomes')
    cached=e['cached_prefixes'];require(type(cached) is list and len(cached)==123,'All123 cached prefixes')
    for n,item in enumerate(cached):
        integer(item['prefix'],'Cached prefix',n);require(item['exact'] is True,'Exact cached prefix');receipt(item['incumbent'],n,ind.INCUMBENT_FLAGS,'incumbent');receipt(item['selection'],n,ind.SELECTION_FLAGS,'selection',width=49)
    physical=e['physical_prefixes'];require(type(physical) is list and len(physical)==5,'Physical prefix coverage')
    for n,item in zip(PREFIXES,physical):
        integer(item['prefix'],'Physical prefix',n);require(item['exact'] is True,'Physical exact prefix');one(item,n,True)
        if n==122:
            require(item['model_recomputed'] is False and item.get('reuses_main_verified_full') is True,'Full122 reuse')
            for key in ('early_incumbent','original_reference','calibration','calibration_strategy','reference261_calibration','control','control_strategy','reference261_control','incumbent','selection','historical_reference_binding'):require(item[key]==e[key],'Reused main evidence:'+key)
            require(item['model']==e['models'] and item['model_counts']==e['main_model_counts'],'Reused full model evidence')
        else:
            require(item['model_recomputed'] is True and item.get('reuses_main_verified_full',False) is False,'Actually recomputed physical prefix');physical_scope_check(item['input_scope'],n,item['model_counts']['counts'])
    require(e['fit_execution_accounting']==execution_accounting(e),'All actual work summed once;122 alias excluded')


def source_evidence(source,plan):
    require(source['raw_source_hashes']==plan['raw_source_hashes'] and source['original169_source_closure']==plan['original169_source_closure'],'Original169/raw source pins')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:require(source[key]==plan['input_sha256'][str(REBUILD/name)],'Registered original cached source')
    require(source['historical261_reference']==plan['historical261_reference']==reference_binding(),'Full historical reference chain')
    require(source['prior_prepared_sha256']==plan['input_sha256'][str(PRIOR/'prepared/COMPLETION.json')],'Immediate predecessor prepared binding')
    require(source['fresh_training_array_reconstruction'] is False and source['cached_training_basis']=='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.','Honest cached provenance')
    for key in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact','original_scores_exact','matched_ridge_calibration_refits_counted','historical_control_before_candidate_fit','full_independent_before_candidate_selection'):require(source[key] is True,'Source/lifecycle proof:'+key)
    require(source['original_other_target_fits']==source['original_local_fits']==0 and source['reference_numeric_files_fully_decoded'] is True,'Honest model/read scope')


def model_callbacks(ind,train,metadata,arrays,n,*,study,original,expected_orders,reference,historical_binding,reference_proof):
    import monthly_monotone_tree266_kernel as kernel
    import monthly_monotone_tree266_reference as ref
    proofs={};strategies={}
    def callback(view,stage):
        require(set(proofs)==(set() if stage=='calibration' else {'calibration','calibration_strategy','reference261_calibration'}),'Ordered gates exactly once')
        before=old.array_snapshot(arrays);tb=old.array_snapshot(train);vb=structured_digest(view)
        fn=ind.verify_calibration_models if stage=='calibration' else ind.verify_control_models
        check=fn(train,metadata,arrays,view,n=n);receipt(check,n,ind.CALIBRATION_FLAGS if stage=='calibration' else ind.CONTROL_FLAGS,stage)
        bundle_preservation(view,n,stage)
        require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and vb==structured_digest(view),'Independent pre-gate mutation')
        layout=reference_layout(arrays,metadata['monthly'],view,n);cap=ref.bind_verified_inputs(view,layout,check,stage=stage)
        reference_consumed_check(reference_proof,reference,historical_binding,layout,view,stage)
        strategy=(kernel.calibration_decisions if stage=='calibration' else kernel.control_decisions)(study,arrays,metadata['monthly'],view,n=n)
        if stage=='calibration':
            exact_single(original,strategy,'Original matched Ridge');verify=ind.verify_incumbent_selection(arrays,old.meta_prefix(metadata['monthly'],n),*strategy,n=n)
        else:verify=ind.verify_control_selection(arrays,old.meta_prefix(metadata['monthly'],n),view,*strategy,n=n)
        skind='incumbent' if stage=='calibration' else 'control_selection';receipt(verify,n,ind.INCUMBENT_FLAGS if stage=='calibration' else ind.CONTROL_SELECTION_FLAGS,skind)
        compare=ref.compare_calibration_reference if stage=='calibration' else ref.compare_control_reference
        historical=compare(reference,view,layout,independent_receipt=check,verification_binding=cap,orders=strategy[0],scores=strategy[1],pre_scores=strategy[2])
        reference_consumed_check(reference_proof,reference,historical_binding,layout,view,stage)
        envelope=reference_envelope(historical,layout,historical_binding);reference_receipt(envelope,n,view['width'],stage,historical_binding)
        require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and vb==structured_digest(view),'Strategy/reference mutated sources or models')
        strategy_proof=dict(passed=True,events=n,pre_scores_exact=True,final_scores_exact=True,ordered_choices_exact=True,producer_episode_calls=1,independent=verify)
        if stage=='calibration':strategy_proof.update(matched_ranks_exact=True,delta_exact_zero=True)
        proofs.update({stage:check,stage+'_strategy':strategy_proof,'reference261_'+stage:envelope});strategies[stage]=strategy
        return check
    return (lambda view:callback(view,'calibration')),(lambda view:callback(view,'control')),proofs,strategies


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
        require(orders[arm]==oo[arm][:n],'Order prefix');exact_array(scores[arm],ss[arm][:n],'Final prefix');exact_array(pres[arm],pp[arm][:n],'Pre-score prefix')


def bundle_parity(full,part,n):
    q=n*full['width']
    for role in ROLES:
        for key in ('raw','available'):exact_array(part[role][key],full[role][key][:q],'Forecast prefix/'+role+'/'+key)
        exact_array(part[role]['rank'],full[role]['rank'][:n],'Rank prefix/'+role)
    ids=part['training_row_ids'];require(isinstance(ids,np.ndarray) and ids.ndim==1 and ids.dtype==full['training_row_ids'].dtype,'Original row ID dtype')
    if not ids.size:
        exact_array(part['label_values'],np.empty(0,dtype=np.float64),'Empty prefix labels')
        return
    lookup={int(row):i for i,row in enumerate(full['training_row_ids'])}
    require(all(int(row) in lookup for row in ids),'Preserved prefix global row IDs')
    take=np.array([lookup[int(row)] for row in ids],dtype=np.int64)
    exact_array(part['label_values'],full['label_values'][take],'Preserved label prefix bits')

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

def matched_strategy_check(value,n):return strategy_check(value,n,True)

def prepare_one(study,train,metadata,arrays,ind,n,expected_orders,expected_original,*,reference,historical_binding,reference_proof):
    import monthly_monotone_tree266_kernel as kernel
    require(historical_binding==reference_binding(),'Complete historical binding required')
    reference_original_check(reference_proof,reference,historical_binding,metadata['monthly'],n,arrays['baseline'].shape[1])
    before=old.array_snapshot(arrays);tb=old.array_snapshot(train);mb=structured_digest(metadata)
    original=kernel.original_decisions(study,arrays,metadata['monthly'],n=n)
    require(original[0]==expected_orders,'Frozen source original orders before models');exact_single(expected_original,original,'Immediate predecessor original')
    exact_single((reference['orders']['incumbent_local67'][:n],reference['scores']['incumbent_local67'][:n],reference['pre_scores']['incumbent_local67'][:n]),original,'Historical261 original')
    early=ind.verify_incumbent_selection(arrays,old.meta_prefix(metadata['monthly'],n),*original,n=n);receipt(early,n,ind.INCUMBENT_FLAGS,'incumbent')
    require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and mb==structured_digest(metadata),'Early original source mutation')
    cal,control,proofs,strategies=model_callbacks(ind,train,metadata,arrays,n,study=study,original=original,expected_orders=expected_orders,reference=reference,historical_binding=historical_binding,reference_proof=reference_proof)
    bundle=kernel.fit_models(train,metadata,arrays,n=n,calibration_validator=cal,control_validator=control)
    require(set(proofs)=={'calibration','calibration_strategy','reference261_calibration','control','control_strategy','reference261_control'},'Both complete historical/strategy gates before candidate')
    require(bundle['calibration_gate']==proofs['calibration'] and bundle['control_gate']==proofs['control'],'Returned exact pre-model receipts')
    digest=structured_digest(bundle);models=ind.verify_models(train,metadata,arrays,bundle,n=n);receipt(models,n,ind.MODEL_FLAGS,'models')
    require(digest==structured_digest(bundle),'Full verifier mutated supplied model');require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and mb==structured_digest(metadata),'Full verifier source mutation before decisions');bundle_preservation(bundle,n)
    aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(metadata['monthly'],n);ig,ip=incumbent_callback(ind,aa,mm,n,expected_orders)
    got=kernel.decisions(study,aa,mm,bundle,n=n,incumbent_validator=ig,expected_incumbent_orders=expected_orders)
    require(set(ip)=={'incumbent'},'Final incumbent gate');oo,ss,pp,audit=got
    selection=ind.verify_selection(aa,mm,bundle,oo,scores=ss,pre_scores=pp,audit=audit,n=n);receipt(selection,n,ind.SELECTION_FLAGS,'selection',width=arrays['baseline'].shape[1])
    exact_single(original,(oo[ARMS[0]],ss[ARMS[0]],pp[ARMS[0]]),'Returned original')
    exact_single(strategies['control'],(oo[ARMS[1]],ss[ARMS[1]],pp[ARMS[1]]),'Returned exact pre-candidate unconstrained control')
    require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and mb==structured_digest(metadata) and digest==structured_digest(bundle),'Immutable source/models/reference')
    evidence=dict(early_incumbent=early,original_reference=dict(passed=True,events=n,source_incumbent_orders_exact=True,prior_incumbent_exact=True,reference261_incumbent_exact=True),**proofs,models=models,incumbent=ip['incumbent'],selection=selection,main_model_counts=bundle['audit'],historical_reference_binding=historical_binding)
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
    import monthly_monotone_tree266_kernel as kernel
    import monthly_monotone_tree266_independent as ind
    study=old.local();m,arrays,px,op=study.load()
    require(len(m['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 arrays')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');metadata=read(REBUILD/'MONTHLY_META.json')
    require(metadata['monthly']==m and metadata['training']==read(study.arch.P/'META.json')['training'],'Original source identities')
    prior=read(PRIOR/'prepared/COMPLETION.json')
    for name in ('SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json'):require(prior['artifacts_sha256'][name]==sha(PRIOR/'prepared'/name),'Prior prepared binding')
    require(read(PRIOR/'prepared/META.json')==m,'Prior identity')
    original_orders=read(LOCAL/'local_linear_ORDERS.json');require(read(PRIOR/'prepared/ORDERS.json')[ARMS[0]]==original_orders,'Original source/predecessor choices')
    original_scores=ev.load_arrays(PRIOR/'prepared/SCORES.npz')[ARMS[0]];original_pres=ev.load_arrays(PRIOR/'prepared/PRE_SCORES.npz')[ARMS[0]]
    historical=plan['historical261_reference'];reference,reference_proof=load_reference(historical);require(reference['meta']==m,'Historical261 monthly identity')
    before=old.array_snapshot(arrays);tb=old.array_snapshot(train);rb=structured_digest(reference)
    bundle,wanted,e=prepare_one(study,train,metadata,arrays,ind,122,original_orders,(original_orders,original_scores,original_pres),reference=reference,historical_binding=historical,reference_proof=reference_proof)
    cached=[];physical=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(m,n);cb,cr=incumbent_callback(ind,aa,mm,n,original_orders[:n]);snapshot=structured_digest(bundle)
        got=kernel.decisions(study,aa,mm,bundle,n=n,incumbent_validator=cb,expected_incumbent_orders=original_orders[:n]);exact_prefix(wanted,got,n);require(set(cr)=={'incumbent'},'Cached original gate')
        oo,ss,pp,au=got;ci=ind.verify_selection(aa,mm,bundle,oo,scores=ss,pre_scores=pp,audit=au,n=n);receipt(ci,n,ind.SELECTION_FLAGS,'selection',width=49)
        require(snapshot==structured_digest(bundle),'Cached model mutation');cached.append(dict(prefix=n,exact=True,incumbent=cr['incumbent'],selection=ci))
        if n in PREFIXES and n!=122:
            print(json.dumps(dict(stage='266_physical_model_prefix',events=n)),flush=True)
            pt,pm,pa,scope=physical_inputs(train,metadata,arrays,n)
            small,gg,pe=prepare_one(study,pt,pm,pa,ind,n,original_orders[:n],(original_orders[:n],original_scores[:n],original_pres[:n]),reference=reference,historical_binding=historical,reference_proof=reference_proof)
            bundle_parity(bundle,small,n);exact_prefix(wanted,gg,n)
            pe.update(prefix=n,exact=True,model_recomputed=True,input_scope=scope,model=pe.pop('models'),model_counts=pe.pop('main_model_counts'));physical.append(pe)
    reuse={k:v for k,v in e.items()};reuse.update(prefix=122,exact=True,model_recomputed=False,reuses_main_verified_full=True,model=reuse.pop('models'),model_counts=reuse.pop('main_model_counts'));physical.append(reuse)
    require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train) and rb==structured_digest(reference),'All original and historical arrays immutable')
    e.update(cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True,historical_reference_unchanged=True)
    e['fit_execution_accounting']=execution_accounting(e);orders,scores,pres,audit=wanted
    values={'MODEL_AUDIT.json':dict(model=bundle['audit'],labels=bundle['label_preservation'],**{role:bundle[role]['audit'] for role in ROLES}),'DECISION_AUDIT.json':audit,'ORDERS.json':orders,'META.json':m,
      'SOURCE_PARITY.json':dict(original169_source_closure=plan['original169_source_closure'],raw_source_hashes=plan['raw_source_hashes'],original_training_hash=sha(REBUILD/'TRAIN_INPUTS.npz'),original_identity_hash=sha(REBUILD/'MONTHLY_META.json'),saved169_parity_hash=sha(REBUILD/'PARITY.json'),prior_prepared_sha256=sha(PRIOR/'prepared/COMPLETION.json'),historical261_reference=historical,cached_training_basis='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.',fresh_training_array_reconstruction=False,original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True,original_scores_exact=True,original_other_target_fits=0,original_local_fits=0,matched_ridge_calibration_refits_counted=True,historical_control_before_candidate_fit=True,full_independent_before_candidate_selection=True,reference_numeric_files_fully_decoded=True),'INDEPENDENT_PREPARE.json':e}
    preparation_evidence(e,values['MODEL_AUDIT.json'],audit);source_evidence(values['SOURCE_PARITY.json'],plan);gate(expected)
    dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{role+'_'+key:bundle[role][key] for role in ROLES for key in ('raw','available','rank')})
    np.savez_compressed(dest/'LABELS.npz',training_row_ids=bundle['training_row_ids'],label_values=bundle['label_values'])
    for name,value in (('SCORES',scores),('PRE_SCORES',pres)):np.savez_compressed(dest/(name+'.npz'),**value)
    for name,value in values.items():dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={name:sha(dest/name) for name in PREP_FILES},new_returns_computed=False,model_fit_accounting='Actual Ridge/control/candidate plus all independent stage canonical/SVD/refits/walks/shape and selector work;123cached fit0;physical122 aliases main'))
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
    import monthly_monotone_tree266_admission as admission
    return admission.required_metrics(metrics,incumbent)


def select_first88(metrics,orders):
    import monthly_monotone_tree266_admission as admission
    return admission.select_first88(metrics,orders)


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
    import monthly_monotone_tree266_independent as ind
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
    import monthly_monotone_tree266_independent as ind
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
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=dict(strict_pit=False,reused_sample=True,global_novelty='UNKNOWN',conditional_raw_prediction_shape_only=True))
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),phase_terminals_sha256=sha(PHASE_TERMINALS),new_trials=3,model_audit_sha256=sha(OUT/'prepared/MODEL_AUDIT.json'),independent_prepare_sha256=sha(OUT/'prepared/INDEPENDENT_PREPARE.json'),protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==262 and sum(x['trials'] for x in state['completed'])==1297,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review fixed conditional monotone-tree replication; first88 dual-reference choice immutable and frozen observers unchanged.')
    old.commit_state(before,state,'monotone_tree266_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

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
            for summary in (r['metrics'],r['slices']['full122']):require(type(summary['cost_per_side']) in (int,float) and summary['cost_per_side']==0.,'ZERO configured cost exact')
            for row in data['TRADES']:require(type(row['cost']) in (int,float) and row['cost']==0.,'ZERO trade cost exact')
            for row in data['HOLDINGS']:require(type(row['fee']) in (int,float) and row['fee']==0.,'ZERO period fee exact')
            for value in (r['metrics']['fees_paid'],r['slices']['full122']['fees_paid']):require(type(value) in (int,float) and value==0.,'ZERO aggregate fee exact')

def digest_string(value,label):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),label+' final SHA256 required')
    return value

def reference_closure_check(pins=None):
    for path,h in REFERENCE_SHA.items():
        require(sha(path)==h,'Frozen261 reference changed:'+path.name)
        if pins is not None:require(pins.get(str(path))==h,'Missing full261 reference pin:'+str(path))
    c=read(REFERENCE/'COMPLETION.json');p=read(REFERENCE/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and c['protected_files']==21444,'Closed261 historical comparator')
    require(p['status']=='COMPLETE' and p['events']==122 and p['new_returns_computed'] is False and set(p['artifacts_sha256'])==PREP_FILES,'Complete261 prepared scope')
    require(c['plan_sha256']==p['plan_sha256']==REFERENCE_SHA[REFERENCE/'PLAN.json'],'261 reference PLAN chain')
    for name,h in p['artifacts_sha256'].items():require(REFERENCE_SHA[REFERENCE/'prepared'/name]==h,'261 complete prepared member binding')
    require(c['model_audit_sha256']==p['artifacts_sha256']['MODEL_AUDIT.json'] and c['independent_prepare_sha256']==p['artifacts_sha256']['INDEPENDENT_PREPARE.json'],'261 completion model proof links')
    return reference_binding()

def reference_binding():
    return dict(schema='monthly_monotone_tree266_historical_binding_v1',files_sha256={str(k):v for k,v in REFERENCE_SHA.items()},completion_sha256=REFERENCE_SHA[REFERENCE/'COMPLETION.json'],prepared_completion_sha256=REFERENCE_SHA[REFERENCE/'prepared/COMPLETION.json'],plan_sha256=REFERENCE_SHA[REFERENCE/'PLAN.json'],prepared_members=sorted(PREP_FILES),npz_decoded=False,hash_scope='All13 complete file bytes; hashing is not numerical prefix consumption')

def load_reference(binding):
    require(binding==reference_closure_check(),'Verified complete reference before decoding')
    # Actual preparation has already passed gate(). Decoding reads full frozen
    # files; subsequent reference comparisons consume only the declared prefix.
    dest=REFERENCE/'prepared'
    reference=dict(forecasts=ev.load_arrays(dest/'FORECASTS.npz'),model_audit=read(dest/'MODEL_AUDIT.json'),labels=ev.load_arrays(dest/'LABELS.npz'),scores=ev.load_arrays(dest/'SCORES.npz'),pre_scores=ev.load_arrays(dest/'PRE_SCORES.npz'),orders=read(dest/'ORDERS.json'),meta=read(dest/'META.json'))
    proof=_ReferenceProof(reference,binding,_REFERENCE_TOKEN);_ISSUED_REFERENCES.append(proof)
    return reference,proof

def strategy_check(value,n,calibration):
    import monthly_monotone_tree266_independent as ind
    integer(value['events'],'Strategy events',n)
    for key in ('passed','pre_scores_exact','final_scores_exact','ordered_choices_exact'):require(value[key] is True,'Pre-candidate strategy:'+key)
    if calibration:
        require(value['matched_ranks_exact'] is True and value['delta_exact_zero'] is True,'Ridge exactly original')
    integer(value['producer_episode_calls'],'Pre-candidate producer selector',1)
    kind='incumbent' if calibration else 'control_selection';receipt(value['independent'],n,ind.INCUMBENT_FLAGS if calibration else ind.CONTROL_SELECTION_FLAGS,kind)

def reference_layout(arrays,meta,view,n):
    from copy import deepcopy
    import monthly_monotone_tree266_reference as ref
    # This extraction is after actual independent model verification of view;
    # status is deliberately absent, and ref derives it from cohort/U itself.
    return dict(n=n,width=view['width'],monthly=old.meta_prefix(meta,n),original_u=(np.asarray(arrays['eligible'])[:n]&np.asarray(arrays['forecast_available'])[:n]).copy(),annual_cohorts=[{k:deepcopy(row[k]) for k in ref.COHORT_KEYS} for row in view['calibration_ridge']['audit']['annual']])

def reference_envelope(comparison,layout,binding):
    return dict(reference_prepared_sha256=binding['prepared_completion_sha256'],source_layout_sha256=structured_digest(layout),consumed_years=[row['year'] for row in layout['annual_cohorts']],annual_cohorts=layout['annual_cohorts'],original_u_indices=[np.flatnonzero(row).tolist() for row in layout['original_u']],comparison=comparison)

def reference_receipt(value,n,width,stage,binding):
    import monthly_monotone_tree266_reference as ref
    require(set(value)=={'reference_prepared_sha256','source_layout_sha256','consumed_years','annual_cohorts','original_u_indices','comparison'},'Full reference envelope')
    require(value['reference_prepared_sha256']==binding['prepared_completion_sha256'],'Frozen reference manifest binding');digest_string(value['source_layout_sha256'],'Verified source layout')
    require(value['consumed_years']==[row['year'] for row in value['annual_cohorts']] and value['consumed_years']==sorted(set(value['consumed_years'])),'Consumed chronological reference years')
    require(len(value['original_u_indices'])==n,'Original prefix U rows')
    for ids in value['original_u_indices']:require(type(ids) is list and ids==sorted(set(ids)) and all(type(i) is int and 0<=i<width for i in ids),'Full original U indices')
    c=value['comparison'];require(c['schema']=='monthly_monotone_tree266_historical_'+stage+'_reference_v1','Reference stage schema');integer(c['checked_events'],'Reference events',n);integer(c['width'],'Reference width',width)
    for key in ('passed','source_layout_independently_verified_by_caller','reference_manifest_verified_by_caller','original_query_domain_used','prefix_fit_state_from_verified_source','full_annual_fit_not_borrowed','exact_bits_and_types'):require(c[key] is True,'Reference exact proof:'+key)
    require(c['receipt_authentication_performed'] is False and c['candidate_models_computed'] is False and c['computes_returns'] is False and c['model_fits']==0 and type(c['numeric_tolerance']) is float and c['numeric_tolerance']==0.,'Reference comparison scope')
    require(c['reference_role']==('control' if stage=='calibration' else 'candidate') and c['current_role']==('calibration_ridge' if stage=='calibration' else 'unconstrained_control'),'Historical economic role mapping')
    counts_check(c['counts'],ref.COUNT_FIELDS,'Reference comparison counts');cc=c['counts']
    require(cc['model_fits']==cc['returns_computed']==0 and cc['verification_binding_checks']==1,'Comparison not training')
    for key,expected in (('forecast_numeric_cells_compared',2*n*width),('availability_cells_compared',n*width),('score_numeric_cells_compared',2*n*width),('ordered_events_compared',n),('annual_metadata_rows_compared',len(value['annual_cohorts']))):integer(cc[key],key,expected)
    if stage=='calibration':
        require(cc['tree_models_compared']==0 and cc['calibration_incumbent_score_cells_compared']==2*n*width and cc['calibration_incumbent_ordered_events_compared']==n,'Ridge/original strategy comparison')
    else:
        require(all(cc[k]==0 for k in ('label_values_compared','label_row_ids_compared','ridge_models_compared','calibration_incumbent_score_cells_compared','calibration_incumbent_ordered_events_compared')),'Tree reference comparison roles')

def exact_array(actual,expected,label):
    require(isinstance(actual,np.ndarray) and isinstance(expected,np.ndarray) and actual.shape==expected.shape and actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes(),label+' exact shape/dtype/bits')

_REFERENCE_TOKEN=object()
_ISSUED_REFERENCES=[]

class _ReferenceProof:
    """Host-issued identity/source binding, not a hostile-Python security boundary."""
    __slots__=('_reference','_baseline','_binding','_sealed')
    def __init__(self,reference,binding,token):
        from copy import deepcopy
        require(token is _REFERENCE_TOKEN,'Reference proof issuer')
        object.__setattr__(self,'_reference',reference)
        # One full decoded private copy at the admitted full-file loading step.
        # Prefix validation below never walks future numeric bodies in this copy.
        object.__setattr__(self,'_baseline',deepcopy(reference))
        object.__setattr__(self,'_binding',deepcopy(binding));object.__setattr__(self,'_sealed',True)
    def __setattr__(self,key,value):raise TypeError('Immutable issued reference proof')
    def __deepcopy__(self,memo):return self


def reference_proof_check(proof,reference,binding):
    require(type(proof) is _ReferenceProof and any(proof is p for p in _ISSUED_REFERENCES),'Host-issued historical source proof required')
    require(proof._reference is reference and proof._binding==binding==reference_binding(),'Same verified historical object and complete binding')


def reference_original_projection(reference,meta,n,width):
    from copy import deepcopy
    import monthly_monotone_tree266_reference as ref
    total,prefix=ref._monthly(reference['meta'],n,width,'Historical immutable prefix')
    _,wanted=ref._monthly(meta,n,width,'Current immutable prefix');ref._equal(prefix,wanted,'Historical immutable identity')
    result=dict(meta=prefix,orders=deepcopy(reference['orders']['incumbent_local67'][:n]))
    for field in ('scores','pre_scores'):
        value=reference[field]['incumbent_local67'];result[field]=ref._array_prefix(value,(total,width),n,'float','Historical immutable original '+field)
    return result


def reference_original_check(proof,reference,binding,meta,n,width):
    reference_proof_check(proof,reference,binding)
    actual=reference_original_projection(reference,meta,n,width)
    expected=reference_original_projection(proof._baseline,meta,n,width)
    require(structured_digest(actual)==structured_digest(expected),'Verified historical original reference mutated')


def reference_consumed_projection(reference,layout,view,stage):
    """Only the fields actually consumed by the fixed reference helper gates."""
    from copy import deepcopy
    import monthly_monotone_tree266_reference as ref
    require(stage in ('calibration','control'),'Explicit reference projection stage')
    n,w,meta,annual=ref._layout(layout);total,_=ref._monthly(reference['meta'],n,w,'Historical prefix header')
    result=dict(original=reference_original_projection(reference,meta,n,w),roles={},labels=None)
    roles=(('calibration_ridge','control','matched_pooled_ridge'),)
    if stage=='control':roles+=(('unconstrained_control','candidate','weak_pooled_tree'),)
    # No ordinary query: neither annual values nor label bodies are accessed.
    if annual:
        ids=view['training_row_ids'];size=len(ids);oldids=reference['labels']['training_row_ids']
        wanted=ref._array_prefix(oldids,oldids.shape,size,'int','Historical consumed label IDs')
        ref._array_equal(ids,wanted,'Historical consumed global IDs')
        values=reference['labels']['label_values']
        result['labels']=dict(training_row_ids=wanted,label_values=ref._array_prefix(values,(len(oldids),),size,'float','Historical consumed labels'))
    for newrole,oldrole,arm in roles:
        record=dict(forecasts={},annual=[],orders=deepcopy(reference['orders'][arm][:n]))
        for key,kind in (('raw','float'),('available','bool'),('rank','float')):
            shape=(total,w) if key=='rank' else (total*w,);stop=n if key=='rank' else n*w
            record['forecasts'][key]=ref._array_prefix(reference['forecasts'][oldrole+'_'+key],shape,stop,kind,'Historical consumed '+key)
        for field in ('scores','pre_scores'):record[field]=ref._array_prefix(reference[field][arm],(total,w),n,'float','Historical consumed '+field)
        if annual:
            rows=ref._annual_index(reference['model_audit'][oldrole]['annual'],[r['year'] for r in annual],'Historical consumed annual')
            ridge=oldrole=='control'
            for scope in annual:
                row=rows[scope['year']]
                keys=('year','cutoff','training_row_ids','status')+(('lower_bound','training_months') if ridge else ())
                value={key:deepcopy(row[key]) for key in keys}
                if scope['status']=='fit':
                    value.update(ref._project_query(row,scope['query_row_ids'],'Historical consumed query'))
                    for key in (('weights','mean','scale','beta','intercept','gram','rhs') if ridge else ('tree',)):value[key]=deepcopy(row[key])
                record['annual'].append(value)
        result['roles'][newrole]=record
    return result


def reference_consumed_check(proof,reference,binding,layout,view,stage):
    reference_proof_check(proof,reference,binding)
    actual=reference_consumed_projection(reference,layout,view,stage)
    expected=reference_consumed_projection(proof._baseline,layout,view,stage)
    require(structured_digest(actual)==structured_digest(expected),'Verified historical consumed reference mutated:'+stage)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    elif a.phase=='prepare':require(bool(a.plan_sha256),'PLAN hash required');prepare_recorded(a.plan_sha256)
    else:require(bool(a.plan_sha256),'PLAN hash required');globals()[a.phase](a.plan_sha256)
