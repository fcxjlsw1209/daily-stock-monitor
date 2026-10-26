"""257: correlation geometry with frozen diagonal diagnostic, weak local-quarter only.

All new actual weights, forecasts, orders and returns require immutable registration. Original labels remain unchanged.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_correlation_distance257'
PRIOR=ROOT/'outputs/monthly_local_distance256'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CONTEXT=ROOT/'work/monthly_correlation_distance257_context.json'
CONTRACT=ROOT/'work/monthly_correlation_distance257_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_correlation_distance257_TESTS.json'
REVIEW=ROOT/'work/monthly_correlation_distance257_CODE_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_local_distance256_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_correlation_distance257_RESULT_REVIEW.json'
ARMS=('incumbent_local67','diag_mass_local67','corr_mass_local67')
ROLES=('control','diagonal','candidate')
DIAGNOSTIC_FILES=('FORECASTS.npz','ORDERS.json','SCORES.npz','PRE_SCORES.npz','META.json','COMPLETION.json')
PREFIXES=(0,1,53,88,122)
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.0,selection_tolerance=0.0,independent_numeric_system='augmented_least_squares_SVD')
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=tuple(name for stem in ('host','kernel','independent') for name in ('monthly_correlation_distance257_'+stem+'.py','test_monthly_correlation_distance257_'+stem+'.py'))
PREP_FILES={'FORECASTS.npz','LABELS.npz','SCORES.npz','PRE_SCORES.npz','MODEL_AUDIT.json','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()


def files():return [ROOT/'work'/p for p in code_names]


def codes():return {str(p):sha(p) for p in files()}


def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('correlation_distance257_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==252 and sum(x['trials'] for x in state['completed'])==1267,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Predecessor changed')


def predecessor_binding(state):
    record=state['completed'][-1]['completion']
    require(Path(record['path'])==PRIOR/'COMPLETION.json' and record['sha256']==sha(PRIOR/'COMPLETION.json'),'Predecessor central closure binding')


def registration_check(state,expected):
    initial_state(state)
    require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration absent')


def contract_check():
    c=read(CONTEXT);d=read(CONTRACT)
    require(c['arms']==list(ARMS) and c['registered_trials']==3 and c['promotable_arms']==[ARMS[2]],'Three fixed arms; only correlation promotable')
    require(d['context_sha256']==sha(CONTEXT),'Implementation contract binding')
    require(c['actual_labels_computed'] is False and c['actual_predictions_computed'] is False and c['actual_returns_computed'] is False,'Prospective specification')
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
    paths=files()+[CONTEXT,CONTRACT,TESTS,REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',PRIOR_REVIEW,old.PROTECTED]
    paths += [PRIOR/'prepared'/name for name in DIAGNOSTIC_FILES]
    paths += [REBUILD/name for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json','PARITY.json')]
    for path in paths:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in review['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_correlation_distance257_independent as ind
    import monthly_correlation_distance257_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_correlation_distance257_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),implementation_contract=contract,implementation_contract_sha256=sha(CONTRACT),runtime=ev.runtime(),input_sha256=pins,raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,actual_labels_before_registration=0,actual_weights_before_registration=0,actual_predictions_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits='Count original annual/local, diagonal local, correlation local and independent actual solves separately; both experimental pooled fits zero; annual correlation builds descriptive only',numerical_verification=NUMERICAL_VERIFICATION)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),last_goal_turn_classification='progress',next_step='Validate original and saved256 diagonal before correlation local distance; only corr can pass first88 ZERO selection; diagonal always diagnostic.')
    old.commit_state(before,state,'correlation_distance257_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


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


def solve_counts(value,label):
    counts_check(value,('pooled_log_solves','local_linear_solves','total_regression_solves'),label)
    require(value['total_regression_solves']==value['pooled_log_solves']+value['local_linear_solves'],label+' sum')


def producer_counts(value,label):
    counts_check(value,('target_fits','log_refits','local_evaluations','local_solves','actual_regression_solves'),label)
    require(value['target_fits']==4*value['log_refits'] and value['local_solves']<=value['local_evaluations'],label+' fit scope')
    require(value['actual_regression_solves']==value['target_fits']+value['log_refits']+value['local_solves'],label+' solve sum')


def model_counts_check(value,n):
    import monthly_correlation_distance257_independent as ind
    counts_check(value,ind.MODEL_COUNT_FIELDS,'Model counts')
    require(value['valid_training_rows']<=value['training_rows'],'Model cohort counts')
    require(value['control_target_fits']==4*value['control_log_refits'] and value['control_local_solves']<=value['control_local_evaluations'],'Original model solve scope')
    for role in ('diag','candidate'):
        require(value[role+'_pooled_fits']==value[role+'_log_refits']==0,'Experimental pooled/log solves prohibited')
        require(value[role+'_local_evaluations']==value['control_local_evaluations'],'Same experimental query cohort')
        require(value[role+'_local_solves']<=value[role+'_kernel_evaluations']<=value[role+'_local_evaluations'],'Experimental local/kernel scope')
        require(value[role+'_changed_weight_evaluations']<=value[role+'_kernel_evaluations'],'Changed kernel scope')
    require(value['diag_equal_scale_evaluations']+value['diag_changed_weight_evaluations']<=value['diag_kernel_evaluations'],'Diagonal equal-scale copies and changed weights disjoint')
    require(value['candidate_identity_correlation_evaluations']<=value['candidate_kernel_evaluations'],'Identity correlation scope')
    require(value['correlation_matrix_builds']==value['control_log_refits'],'One descriptive correlation build per reused annual context')
    require(value['actual_regression_solves']==sum(value[k] for k in ('control_target_fits','control_log_refits','control_local_solves','diag_local_solves','candidate_local_solves')),'Model solve sum excludes matrix builds')
    if n==0:require(all(v==0 for v in value.values()),'Empty prefix no training rows or model fits')


def model_preservation(model,labels,n):
    import monthly_correlation_distance257_independent as ind
    require(model['schema']=='monthly_correlation_distance257_model_audit_v1','Model audit schema')
    c=model['counts'];model_counts_check(c,n)
    for key in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','candidate_pooled_copied','candidate_local_constant_copied','candidate_local_linear_only','diagonal_forecast_reproduced','correlation_floor_coordinates_retained','identity_correlation_reuses_diagonal','residuals_are_mature_in_sample'):
        require(model.get(key) is True,'Model preservation:'+key)
    for key in ('residuals_are_oof','computes_returns'):require(model.get(key) is False,'Model scope:'+key)
    require(type(model.get('mass_relative_tolerance')) is float and model['mass_relative_tolerance']==float(16*np.finfo(float).eps),'Prospective weight mass tolerance')
    expected={'schema','rows','valid_rows','target_keys','target_sha256','common_good_sha256','labels_unchanged','common_good_unchanged','original_row_order_preserved','recentered','raw_target_reconstructed'}
    require(type(labels) is dict and set(labels)==expected and labels['schema']=='monthly_correlation_distance257_labels_preserved_v1','Exact unchanged-label proof schema')
    integer(labels['rows'],'Preserved label rows',c['training_rows']);integer(labels['valid_rows'],'Preserved valid label rows',c['valid_training_rows'])
    require(labels['target_keys']==list(ind.TARGETS) and type(labels['target_sha256']) is dict and set(labels['target_sha256'])==set(ind.TARGETS),'All four unchanged targets')
    for digest in list(labels['target_sha256'].values())+[labels['common_good_sha256']]:
        require(type(digest) is str and len(digest)==64 and all(x in '0123456789abcdef' for x in digest),'Canonical preserved target/good SHA')
    for key in ('labels_unchanged','common_good_unchanged','original_row_order_preserved'):require(labels[key] is True,'Labels preservation:'+key)
    for key in ('recentered','raw_target_reconstructed'):require(labels[key] is False,'No new labels:'+key)


def bundle_preservation(bundle,n):
    model_preservation(bundle['audit'],bundle['label_preservation'],n)
    for role in ('diagonal','candidate'):
        for key in ('source_log','local_constant'):
            a,b=bundle['control']['raw'][key],bundle[role]['raw'][key]
            require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'Copied raw bytes:'+role+'/'+key)
        for key in ('available','local_available'):
            a,b=bundle['control'][key],bundle[role][key]
            require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'Copied model support:'+role+'/'+key)


def receipt(check,n,flags,kind,*,width=None):
    import monthly_correlation_distance257_independent as ind
    require(type(check) is dict,'Independent receipt object')
    kind={'model':'models'}.get(kind,kind)
    standard={'control':ind.CONTROL_FLAGS,'diagnostic_models':ind.DIAGNOSTIC_MODEL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'diagnostic_selection':ind.DIAGNOSTIC_SELECTION_FLAGS,'selection':ind.SELECTION_FLAGS}
    require(kind in standard,'Known receipt kind');integer(n,'Requested events');integer(check.get('checked_events'),'Independent events',n)
    require(check.get('passed') is True and all(check.get(k) is True for k in tuple(flags)+tuple(standard[kind])),'Independent flags:'+kind)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Oracle independence:'+kind)
    def flag(key):require(check.get(key) is True,'Full independent proof:'+kind+'/'+key)
    def tolerance(key,value):
        actual=check.get(key);require(type(actual) is float and np.isfinite(actual) and actual==value,'Exact prospective tolerance:'+kind+'/'+key)
    if kind in ('control','diagnostic_models','models'):
        for key in ('continuous_atol','continuous_rtol','rank_tolerance'):tolerance(key,NUMERICAL_VERIFICATION[key])
        flag('exact_discrete')
    if kind=='control':
        flag('full_control_supplied')
        require(check.get('candidate_labels_computed') is False and check.get('candidate_models_computed') is False,'Control before new experimental models')
        integer(check.get('episode_calls'),'Control episode calls',0)
        pc=check.get('producer_equivalent_fit_counts');sc=check.get('independent_solve_counts')
        producer_counts(pc,'Control producer-equivalent counts');solve_counts(sc,'Control independent solves')
        require(sc['pooled_log_solves']==pc['log_refits'] and sc['local_linear_solves']==pc['local_solves'],'Control independent actual solve scope')
        if n==0:require(not any(pc.values()) and not any(sc.values()),'Empty control no model fits')
    elif kind in ('diagnostic_models','models'):
        if kind=='diagnostic_models':
            flag('full_diagonal_supplied');flag('full_source_inputs_supplied')
            require(check.get('correlation_moments_computed') is False and check.get('correlation_models_computed') is False,'Diagnostic gate before correlation')
            integer(check.get('episode_calls'),'Diagnostic model gate episode calls',0)
            c=check.get('counts');diagnostic_counts_check(c,n)
            roles={'control':'control','diagonal':'candidate'}
            integer(check.get('independent_matrix_builds'),'Diagnostic no correlation matrices',0)
        else:
            for key in ('full_bundle_supplied','full_labels_supplied','full_control_supplied','full_diagonal_supplied','full_candidate_supplied'):flag(key)
            require(check.get('independent_numeric_system')==NUMERICAL_VERIFICATION['independent_numeric_system'],'Independent numerical method')
            c=check.get('counts');model_counts_check(c,n);roles={'control':'control','diagonal':'diag','candidate':'candidate'}
            integer(check.get('independent_matrix_builds'),'Independent descriptive matrix builds',c['correlation_matrix_builds'])
        require(check.get('producer_equivalent_counts')==c,'Equivalent counts binding')
        sc=check.get('independent_solve_counts');require(type(sc) is dict and set(sc)==set(roles)|{'total_regression_solves'},'Independent model solve counts')
        for role,prefix in roles.items():
            solve_counts(sc[role],role+' independent solves')
            require(sc[role]['pooled_log_solves']==c[prefix+'_log_refits'] and sc[role]['local_linear_solves']==c[prefix+'_local_solves'],'Independent solves versus actual fit scope')
        integer(sc['total_regression_solves'],'Independent total solves',sum(sc[r]['total_regression_solves'] for r in roles))
    else:
        if kind=='diagnostic_selection':
            for key in ('full_diagonal_scores_supplied','full_diagonal_pre_scores_supplied'):flag(key)
            require(check.get('correlation_selection_computed') is False,'Diagnostic selection gate before correlation')
            integer(check.get('episode_calls'),'One independent diagonal selection',1)
        else:
            for key in ('full_scores_supplied','full_pre_scores_supplied'):flag(key)
        tolerance('numerical_tolerance',0.)
        integer(check.get('ordered_choices_checked'),'Ordered choices',3*n if kind=='selection' else n)
        if kind=='incumbent':require(check.get('candidate_selection_computed') is False,'Original checked before experimental selection')
        elif kind=='selection':
            for key in ('full_bundle_supplied','full_audit_supplied'):flag(key)
            c=check.get('counts');counts_check(c,ind.COUNT_FIELDS,'Decision counts')
            require(c['events']==n and c['symbols_per_event']>0 and c['initial_partial_events']+c['formal_events']==n,'Decision event scope')
            if width is not None:integer(c['symbols_per_event'],'Exact decision width',width)
            require(c['incumbent_episode_calls']==c['diag_episode_calls']==c['candidate_episode_calls']==1 and c['new_model_fits']==c['predictive_model_fits']==0,'Decision-only single episode scope')


def fit_binding(control,diagnostic,models,model_audit,n):
    c=model_audit['counts'];model_counts_check(c,n);diagnostic_counts_check(diagnostic['counts'],n)
    require(models['counts']==c,'Independent/producer model counts binding')
    require(control['independent_solve_counts']==models['independent_solve_counts']['control']==diagnostic['independent_solve_counts']['control'],'Repeated independent control solve scope')
    require(diagnostic['independent_solve_counts']['diagonal']==models['independent_solve_counts']['diagonal'],'Repeated independent diagnostic solve scope')
    pc=control['producer_equivalent_fit_counts'];dc=diagnostic['counts']
    for key in ('target_fits','log_refits','local_evaluations','local_solves'):require(pc[key]==c['control_'+key]==dc['control_'+key],'Control producer-equivalent fit count binding')
    for key in ('pooled_fits','log_refits','local_evaluations','local_solves','kernel_evaluations','equal_scale_evaluations','changed_weight_evaluations'):
        require(c['diag_'+key]==dc['candidate_'+key],'Diagnostic producer-equivalent fit count binding')
    for key in ('training_rows','valid_training_rows'):require(c[key]==dc[key],'Same diagnostic consumed training cohort')


def execution_accounting(e):
    entries=[]
    invocations=[('main',122,e['control'],e['diagnostic_models'],e['models'],e['main_model_counts'])]
    invocations += [('physical',v['prefix'],v['control'],v['diagnostic_models'],v['model'],v['model_counts']) for v in e['physical_prefixes'] if v['prefix']!=122]
    for scope,n,control,diagnostic,models,audit in invocations:
        fit_binding(control,diagnostic,models,audit,n)
        a=audit['counts']['actual_regression_solves'];b=control['independent_solve_counts']['total_regression_solves'];d=diagnostic['independent_solve_counts']['total_regression_solves'];c=models['independent_solve_counts']['total_regression_solves']
        entries.append(dict(scope=scope,events=n,producer_regression_solves=a,independent_control_gate_solves=b,independent_diagnostic_gate_solves=d,independent_model_check_solves=c,total_regression_solves=a+b+c+d,producer_correlation_matrix_builds=audit['counts']['correlation_matrix_builds'],independent_correlation_matrix_builds=models['independent_matrix_builds']))
    keys=('producer_regression_solves','independent_control_gate_solves','independent_diagnostic_gate_solves','independent_model_check_solves','total_regression_solves','producer_correlation_matrix_builds','independent_correlation_matrix_builds')
    return dict(schema='monthly_correlation_distance257_fit_execution_accounting_v1',model_invocations=len(entries),physical122_reuses_main=True,entries=entries,totals={k:sum(v[k] for v in entries) for k in keys},decision_only_fits=0,correlation_matrix_builds_are_predictive_fits=False)


def preparation_evidence(e,model_audit,decision_audit):
    kinds=('control','diagnostic_models','models','incumbent','diagnostic_selection','selection')
    for key in kinds:receipt(e[key],122,(),key,width=49)
    require(e['all123_order_prefixes'] is True and e['original_arrays_unchanged'] is True and e['original_training_unchanged'] is True,'Preservation proofs')
    require(e['main_model_counts']==model_audit['model'],'Saved main model audit binding')
    model_preservation(model_audit['model'],model_audit['labels'],122)
    fit_binding(e['control'],e['diagnostic_models'],e['models'],model_audit['model'],122)
    require(decision_audit['schema']=='monthly_correlation_distance257_decisions_v1' and len(decision_audit['events'])==122 and decision_audit['counts']==e['selection']['counts'],'Saved decision audit scope')
    require(decision_audit['incumbent_gate']==e['incumbent'] and decision_audit['diagnostic_gate']==e['diagnostic_selection'],'Saved pre-correlation selection receipts')
    for key in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact','diagonal_decisions_reproduced'):require(decision_audit[key] is True,'Decision preservation:'+key)
    require(decision_audit['computes_returns'] is False,'Decisions before outcomes')
    pins=e['diagnostic_cache_sha256'];diagnostic_parity_check(e['diagnostic_parity'],122,pins,True)
    cached=e['cached_prefixes'];physical=e['physical_prefixes'];require(type(cached) is list and len(cached)==123,'All cached prefixes')
    for n,v in enumerate(cached):
        integer(v['prefix'],'Cached prefix',n);require(v['exact'] is True,'Exact cached prefix')
        receipt(v['incumbent'],n,(),'incumbent');receipt(v['diagnostic_selection'],n,(),'diagnostic_selection');diagnostic_parity_check(v['diagnostic_parity'],n,pins,False)
    require(type(physical) is list and len(physical)==len(PREFIXES),'All model prefixes')
    for n,v in zip(PREFIXES,physical):
        integer(v['prefix'],'Physical prefix',n);require(v['exact'] is True,'Exact physical prefix')
        for key in kinds:receipt(v['model' if key=='models' else key],n,(),key,width=49)
        fit_binding(v['control'],v['diagnostic_models'],v['model'],v['model_counts'],n);diagnostic_parity_check(v['diagnostic_parity'],n,pins,True)
        if n==122:
            require(v['model_recomputed'] is False and v.get('reuses_main_verified_full') is True,'Physical122 reuse, no new fit')
            for key in kinds:require(v['model' if key=='models' else key]==e[key],'Physical122 main receipt reuse')
            require(v['model_counts']==e['main_model_counts'] and v['diagnostic_parity']==e['diagnostic_parity'],'Physical122 main audit reuse')
        else:require(v['model_recomputed'] is True and v.get('reuses_main_verified_full',False) is False,'Physically recomputed prefix')
    require(e['fit_execution_accounting']==execution_accounting(e),'Actual solve accounting with122 excluded from repeat count')


def source_evidence(source,plan):
    require(source['raw_source_hashes']==plan['raw_source_hashes'],'Saved raw source pins')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        require(source[key]==plan['input_sha256'][str(REBUILD/name)],'Registered cached source:'+name)
    require(source['fresh_training_array_reconstruction'] is False and source['cached_training_basis']=='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.','Honest cache provenance scope')
    for key in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact','saved256_diagnostic_exact'):require(source[key] is True,'Source preservation:'+key)
    require(set(source['saved256_cache_sha256'])==set(DIAGNOSTIC_FILES),'Full diagnostic source scope')
    for name,h in source['saved256_cache_sha256'].items():require(plan['input_sha256'].get(str(PRIOR/'prepared'/name))==h,'Registered saved256 source binding')


def model_callbacks(ind,train,metadata,arrays,n,saved):
    receipts={};width=arrays['baseline'].shape[1]
    def control(result):
        require(not receipts,'Control gate once')
        check=ind.verify_control(train,metadata,arrays,result,n=n);receipt(check,n,(),'control');receipts['control']=check;return check
    def diagnostic(result):
        require(set(receipts)=={'control'},'Original model gate before diagnostic gate')
        exact_saved_forecast(result,saved,n,width)
        check=ind.verify_diagnostic_models(train,metadata,arrays,result,n=n);receipt(check,n,(),'diagnostic_models');receipts['diagnostic_models']=check;return check
    return control,diagnostic,receipts



def exact_prefix(wanted,got,n):
    oo,ss,pp,_=wanted;orders,scores,pres,_=got
    require(set(orders)==set(scores)==set(pres)==set(ARMS),'Exactly three decision arms')
    for arm in ARMS:
        require(orders[arm]==oo[arm][:n],'Order prefix')
        np.testing.assert_array_equal(scores[arm],ss[arm][:n]);np.testing.assert_array_equal(pres[arm],pp[arm][:n])


def bundle_parity(full,part,n):
    q=n*full['width']
    for arm in ROLES:
        for key,value in part[arm]['raw'].items():np.testing.assert_array_equal(value,full[arm]['raw'][key][:q])
        for key in ('available','local_available'):np.testing.assert_array_equal(part[arm][key],full[arm][key][:q])
    wanted={int(row):value for row,value in zip(full['training_row_ids'],full['label_values'])}
    np.testing.assert_array_equal(part['label_values'],np.array([wanted[int(row)] for row in part['training_row_ids']]))


def diagnostic_counts_check(c,n):
    import monthly_correlation_distance257_independent as ind
    counts_check(c,ind.DIAGNOSTIC_MODEL_COUNT_FIELDS,'Frozen256 diagnostic model counts')
    require(c['valid_training_rows']<=c['training_rows'] and c['control_target_fits']==4*c['control_log_refits'],'Diagnostic original cohort/fits')
    require(c['candidate_pooled_fits']==c['candidate_log_refits']==0,'Diagnostic no pooled/log fits')
    require(c['control_local_solves']<=c['control_local_evaluations']==c['candidate_local_evaluations'],'Diagnostic same query cohort')
    require(c['candidate_local_solves']<=c['candidate_kernel_evaluations']<=c['candidate_local_evaluations'],'Diagnostic local/kernel counts')
    require(c['candidate_equal_scale_evaluations']+c['candidate_changed_weight_evaluations']<=c['candidate_kernel_evaluations'],'Diagnostic equal branch counts')
    require(c['actual_regression_solves']==sum(c[k] for k in ('control_target_fits','control_log_refits','control_local_solves','candidate_local_solves')),'Diagnostic producer equivalent counts')
    if n==0:require(not any(c.values()),'Empty diagnostic no fits')

def diagnostic_parity_check(value,n,pins,model):
    expected={'events','cache_sha256','orders_exact','scores_exact','pre_scores_exact'}|({'forecast_exact'} if model else set())
    require(type(value) is dict and set(value)==expected,'Exact saved256 parity receipt fields');integer(value['events'],'Saved256 parity events',n)
    require(value['cache_sha256']==pins and type(pins) is dict and set(pins)==set(DIAGNOSTIC_FILES),'Saved256 full cache bindings')
    for h in pins.values():require(type(h) is str and len(h)==64 and all(x in '0123456789abcdef' for x in h),'Saved256 canonical digest')
    for key in expected-{'events','cache_sha256'}:require(value[key] is True,'Saved256 exact parity:'+key)

def diagnostic_cache(plan):
    """Called only after registration; every full cache file is explicitly pinned."""
    dest=PRIOR/'prepared';pins={name:sha(dest/name) for name in DIAGNOSTIC_FILES}
    for name,h in pins.items():require(plan['input_sha256'].get(str(dest/name))==h,'Registered diagnostic cache '+name)
    c=read(dest/'COMPLETION.json');require(c['status']=='COMPLETE' and c['events']==122 and c['new_returns_computed'] is False,'Frozen256 prepared completion')
    require(c['plan_sha256']==sha(PRIOR/'PLAN.json') and set(c['artifacts_sha256'])==PREP_FILES,'Frozen256 full prepared manifest')
    for name,h in c['artifacts_sha256'].items():require(sha(dest/name)==h,'Frozen256 prepared artifact changed')
    forecasts=ev.load_arrays(dest/'FORECASTS.npz');scores=ev.load_arrays(dest/'SCORES.npz');pres=ev.load_arrays(dest/'PRE_SCORES.npz');orders=read(dest/'ORDERS.json');meta=read(dest/'META.json')
    return dict(forecasts=forecasts,scores=scores,pres=pres,orders=orders,meta=meta,input_sha256=pins)

def exact_saved_forecast(forecast,saved,n,width):
    for key in ('source_log','local_constant','local_linear'):
        actual=np.asarray(forecast['raw'][key]);expected=np.asarray(saved['forecasts']['candidate_'+key][:n*width])
        require(actual.shape==expected.shape==(n*width,) and actual.dtype==expected.dtype,'Saved diagonal forecast axes');np.testing.assert_array_equal(actual,expected)
    for key in ('available','local_available'):
        actual=np.asarray(forecast[key]);expected=np.asarray(saved['forecasts']['candidate_'+key][:n*width])
        require(actual.shape==expected.shape==(n*width,) and actual.dtype==expected.dtype==np.dtype(bool),'Saved diagonal availability axes');np.testing.assert_array_equal(actual,expected)

def exact_saved_selection(orders,scores,pres,saved,arm,n,width):
    require(orders==saved['orders'][arm][:n],'Frozen256 full ordered record parity')
    for value,key in ((scores,'scores'),(pres,'pres')):
        a=np.asarray(value);b=np.asarray(saved[key][arm][:n]);require(a.shape==b.shape==(n,width) and a.dtype==b.dtype,'Saved256 full score axes');np.testing.assert_array_equal(a,b)

def decision_callbacks(ind,arrays,meta,bundle,n,saved):
    receipts={};width=arrays['baseline'].shape[1]
    def incumbent(orders,scores,pres):
        require(not receipts,'Incumbent selection gate once')
        exact_saved_selection(orders,scores,pres,saved,ARMS[0],n,width)
        check=ind.verify_incumbent(arrays,meta,orders,scores,pres,n=n);receipt(check,n,(),'incumbent');receipts['incumbent']=check;return check
    def diagnostic(orders,scores,pres):
        require(set(receipts)=={'incumbent'},'Incumbent selection gate before diagonal')
        exact_saved_selection(orders,scores,pres,saved,ARMS[1],n,width)
        check=ind.verify_diagnostic_selection(arrays,meta,bundle,orders,scores,pres,n=n);receipt(check,n,(),'diagnostic_selection');receipts['diagnostic_selection']=check;return check
    return incumbent,diagnostic,receipts

def parity_receipt(saved,n,model):
    result=dict(events=n,cache_sha256=saved['input_sha256'],orders_exact=True,scores_exact=True,pre_scores_exact=True)
    if model:result['forecast_exact']=True
    return result

def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_correlation_distance257_kernel as kernel
    import monthly_correlation_distance257_independent as ind
    study=old.local();m,arrays,px,op=study.load()
    require(len(m['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 arrays')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');metadata=read(REBUILD/'MONTHLY_META.json')
    require(metadata['monthly']==m and metadata['training']==read(study.arch.P/'META.json')['training'],'Original training/query identities')
    saved=diagnostic_cache(plan);require(saved['meta']==m,'Frozen256 metadata identity')
    before=old.array_snapshot(arrays);train_before=old.array_snapshot(train)
    cg,dg,cr=model_callbacks(ind,train,metadata,arrays,122,saved)
    bundle=kernel.fit_models(train,metadata,arrays,n=122,control_validator=cg,diagnostic_validator=dg)
    require(set(cr)=={'control','diagnostic_models'},'Verified original and saved256 diagonal before correlation')
    model_check=ind.verify_models(train,metadata,arrays,bundle,n=122);receipt(model_check,122,ind.MODEL_FLAGS,'models')
    bundle_preservation(bundle,122)
    ig,dig,ir=decision_callbacks(ind,arrays,m,bundle,122,saved)
    wanted=kernel.decisions(study,arrays,m,bundle,n=122,incumbent_validator=ig,expected_incumbent_orders=read(LOCAL/'local_linear_ORDERS.json'),diagnostic_validator=dig,expected_diagnostic_orders=saved['orders'][ARMS[1]])
    require(set(ir)=={'incumbent','diagnostic_selection'},'Verified original and diagnostic decisions before correlation')
    orders,scores,pres,audit=wanted
    check=ind.verify_selection(arrays,m,bundle,orders,scores=scores,pre_scores=pres,audit=audit,n=122);receipt(check,122,ind.SELECTION_FLAGS,'selection',width=49)
    cached=[];physical=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(m,n)
        cb,db,cb_receipts=decision_callbacks(ind,aa,mm,bundle,n,saved)
        got=kernel.decisions(study,aa,mm,bundle,n=n,incumbent_validator=cb,expected_incumbent_orders=orders[ARMS[0]][:n],diagnostic_validator=db,expected_diagnostic_orders=saved['orders'][ARMS[1]][:n])
        exact_prefix(wanted,got,n);require(set(cb_receipts)=={'incumbent','diagnostic_selection'},'Prefix original and diagnostic gates')
        cached.append(dict(prefix=n,exact=True,**cb_receipts,diagnostic_parity=parity_receipt(saved,n,False)))
        if n in PREFIXES and n!=122:
            print(json.dumps(dict(stage='257_physical_model_prefix',events=n)),flush=True)
            cg2,dg2,cr2=model_callbacks(ind,train,metadata,arrays,n,saved)
            small=kernel.fit_models(train,metadata,arrays,n=n,control_validator=cg2,diagnostic_validator=dg2)
            require(set(cr2)=={'control','diagnostic_models'},'Prefix original/diagonal model gates')
            mi=ind.verify_models(train,metadata,arrays,small,n=n);receipt(mi,n,ind.MODEL_FLAGS,'models')
            bundle_preservation(small,n)
            bundle_parity(bundle,small,n)
            ig2,dig2,ir2=decision_callbacks(ind,aa,mm,small,n,saved)
            gg=kernel.decisions(study,aa,mm,small,n=n,incumbent_validator=ig2,expected_incumbent_orders=orders[ARMS[0]][:n],diagnostic_validator=dig2,expected_diagnostic_orders=saved['orders'][ARMS[1]][:n]);exact_prefix(wanted,gg,n)
            oo,ss,pp,au=gg
            si=ind.verify_selection(aa,mm,small,oo,scores=ss,pre_scores=pp,audit=au,n=n);receipt(si,n,ind.SELECTION_FLAGS,'selection',width=49)
            physical.append(dict(prefix=n,exact=True,model_recomputed=True,**cr2,model=mi,**ir2,selection=si,model_counts=small['audit'],diagnostic_parity=parity_receipt(saved,n,True)))
    physical.append(dict(prefix=122,exact=True,model_recomputed=False,reuses_main_verified_full=True,**cr,model=model_check,**ir,selection=check,model_counts=bundle['audit'],diagnostic_parity=parity_receipt(saved,122,True)))
    require(before==old.array_snapshot(arrays) and train_before==old.array_snapshot(train),'Original arrays/training changed')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    forecasts={arm+'_'+key:val for arm in ROLES for key,val in bundle[arm]['raw'].items()}
    for arm in ROLES:
        for key in ('available','local_available'):forecasts[arm+'_'+key]=bundle[arm][key]
    np.savez_compressed(dest/'FORECASTS.npz',**forecasts)
    np.savez_compressed(dest/'LABELS.npz',training_row_ids=bundle['training_row_ids'],label_values=bundle['label_values'])
    for name,value in [('SCORES',scores),('PRE_SCORES',pres)]:np.savez_compressed(dest/(name+'.npz'),**value)
    values={'MODEL_AUDIT.json':dict(model=bundle['audit'],labels=bundle['label_preservation'],control=bundle['control']['audit'],diagonal=bundle['diagonal']['audit'],candidate=bundle['candidate']['audit']),'DECISION_AUDIT.json':audit,'ORDERS.json':orders,'META.json':m,'SOURCE_PARITY.json':dict(raw_source_hashes=plan['raw_source_hashes'],original_training_hash=sha(REBUILD/'TRAIN_INPUTS.npz'),original_identity_hash=sha(REBUILD/'MONTHLY_META.json'),saved169_parity_hash=sha(REBUILD/'PARITY.json'),cached_training_basis='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.',fresh_training_array_reconstruction=False,original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True,saved256_cache_sha256=saved['input_sha256'],saved256_diagnostic_exact=True),'INDEPENDENT_PREPARE.json':dict(**cr,models=model_check,**ir,selection=check,diagnostic_cache_sha256=saved['input_sha256'],diagnostic_parity=parity_receipt(saved,122,True),cached_prefixes=cached,physical_prefixes=physical,main_model_counts=bundle['audit'],all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True)}
    values['INDEPENDENT_PREPARE.json']['fit_execution_accounting']=execution_accounting(values['INDEPENDENT_PREPARE.json'])
    preparation_evidence(values['INDEPENDENT_PREPARE.json'],values['MODEL_AUDIT.json'],values['DECISION_AUDIT.json']);source_evidence(values['SOURCE_PARITY.json'],plan)
    require(values['INDEPENDENT_PREPARE.json']['diagnostic_cache_sha256']==values['SOURCE_PARITY.json']['saved256_cache_sha256'],'Independent/source diagnostic cache bindings')
    for name,value in values.items():dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,model_fit_accounting='Main counts plus newly fitted physical prefixes0/1/53/88; prefix122 reuses main. Independent oracle solves are separately audited.'))
    print(json.dumps(dict(status='PREPARED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['events']==122 and c['new_returns_computed'] is False and set(c['artifacts_sha256'])==PREP_FILES,'Prepared scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    preparation_evidence(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),read(OUT/'prepared/MODEL_AUDIT.json'),read(OUT/'prepared/DECISION_AUDIT.json'))
    source_evidence(read(OUT/'prepared/SOURCE_PARITY.json'),read(OUT/'PLAN.json'))
    require(read(OUT/'prepared/INDEPENDENT_PREPARE.json')['diagnostic_cache_sha256']==read(OUT/'prepared/SOURCE_PARITY.json')['saved256_cache_sha256'],'Independent/source diagnostic cache bindings')
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
    invalid={arm:required_metrics(metrics[arm],arm==ARMS[0]) for arm in ARMS}
    reference=metrics[ARMS[0]];candidate=metrics[ARMS[2]]
    require(all(len(orders[a])>=88 for a in ARMS),'First88 ordered holdings required')
    changed=[i for i in range(88) if set(orders[ARMS[2]][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    close='same_close_zero' not in invalid[ARMS[2]] and old.guard(candidate['same_close_zero'],reference['same_close_zero'],.005)
    opening='next_open_zero' not in invalid[ARMS[2]] and old.guard(candidate['next_open_zero'],reference['next_open_zero'],0.)
    support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed)
    admitted=bool(close and opening and support)
    return dict(primary=ARMS[2] if admitted else ARMS[0],admitted=admitted,guards=dict(same_close=bool(close),next_open=bool(opening),support=bool(support)),changed_months=changed,undefined_sharpe=invalid,first88_only=True,later34_used=False,diagnostic_arm_never_promoted=True)

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
    import monthly_correlation_distance257_independent as ind
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
    import monthly_correlation_distance257_independent as ind
    check=ind.audit_outcomes(px,op,orders,exports,results,meta);outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    dump(dest/'INDEPENDENT_OUTCOME.json',check);dump(dest/'RESULTS.json',dict(selection=f['selection'],arms=results,reused_sample=True,strict_pit=False,zero_cost_selection=True,next_open_cost_diagnostic_only=True,observer_promotion=False))
    gate(expected);dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,frozen_sha256=sha(OUT/'selected/FROZEN.json'),paths=9,slices=45,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE',full_cagr={a:results[a]['same_close_zero']['metrics']['cagr'] for a in ARMS},completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def result_review_check(expected):
    review=read(RESULT_REVIEW)
    require(review['passed'] is True and review['unresolved_blockers']==[],'Saved evidence review passed')
    for key,path in [('plan_sha256',OUT/'PLAN.json'),('context_sha256',CONTEXT),('prepared_completion_sha256',OUT/'prepared/COMPLETION.json'),('results_completion_sha256',OUT/'results/COMPLETION.json')]:
        require(review[key]==sha(path),'Saved review binding:'+key)
    require(review['plan_sha256']==expected,'Saved review explicit PLAN')
    pins={str(Path(k) if Path(k).is_absolute() else ROOT/k):v for k,v in review['input_sha256'].items()}
    required=[OUT/'PLAN.json',CONTEXT,CONTRACT,OUT/'prepared/COMPLETION.json',OUT/'selected/FROZEN.json',OUT/'selected/TRAIN_RESULTS.json',OUT/'results/COMPLETION.json']
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
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),new_trials=3,model_audit_sha256=sha(OUT/'prepared/MODEL_AUDIT.json'),independent_prepare_sha256=sha(OUT/'prepared/INDEPENDENT_PREPARE.json'),protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==253 and sum(x['trials'] for x in state['completed'])==1270,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review257 weak local-slot correlation-distance conditional replication; target70 subject to independent goal audit; frozen observers unchanged.')
    old.commit_state(before,state,'correlation_distance257_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    elif a.phase=='prepare':require(bool(a.plan_sha256),'PLAN hash required');prepare_recorded(a.plan_sha256)
    else:require(bool(a.plan_sha256),'PLAN hash required');globals()[a.phase](a.plan_sha256)
