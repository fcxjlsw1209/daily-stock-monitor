"""255: stock/ETF-centered training targets in only the existing weak5% slot.

All new actual labels, forecasts, orders and returns require immutable registration.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_asset_target255'
PRIOR=ROOT/'outputs/monthly_calendar_memory254'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CONTEXT=ROOT/'work/monthly_asset_target255_context.json'
CONTRACT=ROOT/'work/monthly_asset_target255_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_asset_target255_TESTS.json'
REVIEW=ROOT/'work/monthly_asset_target255_CODE_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_calendar_memory254_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_asset_target255_RESULT_REVIEW.json'
ARMS=('incumbent_local67','asset_centered_local67')
PREFIXES=(0,1,53,88,122)
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.0,selection_tolerance=0.0,independent_numeric_system='augmented_least_squares_SVD')
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=tuple(name for stem in ('host','kernel','independent') for name in ('monthly_asset_target255_'+stem+'.py','test_monthly_asset_target255_'+stem+'.py'))
PREP_FILES={'FORECASTS.npz','LABELS.npz','SCORES.npz','PRE_SCORES.npz','MODEL_AUDIT.json','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()


def files():return [ROOT/'work'/p for p in code_names]


def codes():return {str(p):sha(p) for p in files()}


def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('asset_target255_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==250 and sum(x['trials'] for x in state['completed'])==1263,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Predecessor changed')


def predecessor_binding(state):
    record=state['completed'][-1]['completion']
    require(Path(record['path'])==PRIOR/'COMPLETION.json' and record['sha256']==sha(PRIOR/'COMPLETION.json'),'Predecessor central closure binding')


def registration_check(state,expected):
    initial_state(state)
    require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration absent')


def contract_check():
    c=read(CONTEXT);d=read(CONTRACT)
    require(c['arms']==list(ARMS) and c['registered_trials']==2 and c['promotable_arms']==[ARMS[1]],'Two fixed arms')
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
    paths += [REBUILD/name for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json','PARITY.json')]
    for path in paths:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in review['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_asset_target255_independent as ind
    import monthly_asset_target255_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_asset_target255_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),implementation_contract=contract,implementation_contract_sha256=sha(CONTRACT),runtime=ev.runtime(),input_sha256=pins,raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,actual_labels_before_registration=0,actual_predictions_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits='Count actual annual and local solves in preparation; not zero',numerical_verification=NUMERICAL_VERIFICATION)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),last_goal_turn_classification='progress',next_step='Validate unchanged model and same-cohort stock/ETF-centered target; only weak5%slot; freeze first88 ZERO selection before diagnostics.')
    old.commit_state(before,state,'asset_target255_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


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
    import monthly_asset_target255_independent as ind
    counts_check(value,ind.MODEL_COUNT_FIELDS,'Model counts')
    require(value['valid_training_rows']<=value['training_rows'] and value['changed_label_rows']<=value['valid_training_rows'],'Model cohort counts')
    total=0
    for role in ('control','candidate'):
        require(value[role+'_target_fits']==4*value[role+'_log_refits'] and value[role+'_local_solves']<=value[role+'_local_evaluations'],'Model solve scope')
        total+=sum(value[role+'_'+key] for key in ('target_fits','log_refits','local_solves'))
    require(value['actual_regression_solves']==total,'Model solve sum')
    if n==0:require(all(v==0 for v in value.values()),'Empty prefix no training rows or model fits')


def receipt(check,n,flags,kind,*,width=None):
    import monthly_asset_target255_independent as ind
    require(type(check) is dict,'Independent receipt object')
    kind={'model':'models'}.get(kind,kind)
    require(kind in ('control','models','incumbent','selection'),'Known receipt kind')
    integer(n,'Requested events');integer(check.get('checked_events'),'Independent events',n)
    standard_flags={'control':ind.CONTROL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS}[kind]
    require(check.get('passed') is True and all(check.get(k) is True for k in tuple(flags)+tuple(standard_flags)),'Independent flags:'+kind)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Oracle independence:'+kind)
    def flag(key):require(check.get(key) is True,'Full independent proof:'+kind+'/'+key)
    def tolerance(key,value):
        actual=check.get(key)
        require(type(actual) is float and np.isfinite(actual) and actual==value,'Exact prospective tolerance:'+kind+'/'+key)
    if kind in ('control','models'):
        for key in ('continuous_atol','continuous_rtol','rank_tolerance'):tolerance(key,NUMERICAL_VERIFICATION[key])
        flag('exact_discrete');flag('full_control_supplied')
    if kind=='control':
        require(check.get('candidate_labels_computed') is False and check.get('candidate_models_computed') is False,'Control before new candidate')
        integer(check.get('episode_calls'),'Control episode calls',0)
        pc=check.get('producer_equivalent_fit_counts');sc=check.get('independent_solve_counts')
        producer_counts(pc,'Control producer-equivalent counts');solve_counts(sc,'Control independent solves')
        require(sc['pooled_log_solves']==pc['log_refits'] and sc['local_linear_solves']==pc['local_solves'],'Control independent actual solve scope')
        if n==0:require(not any(pc.values()) and not any(sc.values()),'Empty control no model fits')
    elif kind=='models':
        for key in ('full_bundle_supplied','full_labels_supplied','full_candidate_supplied'):flag(key)
        require(check.get('independent_numeric_system')==NUMERICAL_VERIFICATION['independent_numeric_system'],'Independent numerical method')
        c=check.get('counts');model_counts_check(c,n)
        counts_check(check.get('producer_equivalent_counts'),ind.MODEL_COUNT_FIELDS,'Producer-equivalent model counts')
        require(check['producer_equivalent_counts']==c,'Equivalent counts binding')
        sc=check.get('independent_solve_counts')
        require(type(sc) is dict and set(sc)=={'control','candidate','total_regression_solves'},'Independent model solve counts')
        for role in ('control','candidate'):
            solve_counts(sc[role],role+' independent solves')
            require(sc[role]['pooled_log_solves']==c[role+'_log_refits'] and sc[role]['local_linear_solves']==c[role+'_local_solves'],'Independent solves versus actual fit scope')
        integer(sc['total_regression_solves'],'Independent total solves',sum(sc[r]['total_regression_solves'] for r in ('control','candidate')))
    else:
        for key in ('full_scores_supplied','full_pre_scores_supplied'):flag(key)
        tolerance('numerical_tolerance',0.)
        integer(check.get('ordered_choices_checked'),'Ordered choices',n if kind=='incumbent' else 2*n)
        if kind=='incumbent':require(check.get('candidate_selection_computed') is False,'Original checked before candidate selection')
        else:
            for key in ('full_bundle_supplied','full_audit_supplied'):flag(key)
            c=check.get('counts');counts_check(c,ind.COUNT_FIELDS,'Decision counts')
            require(c['events']==n and c['symbols_per_event']>0 and c['initial_partial_events']+c['formal_events']==n,'Decision event scope')
            if width is not None:integer(c['symbols_per_event'],'Exact decision width',width)
            require(c['incumbent_episode_calls']==c['candidate_episode_calls']==1 and c['new_model_fits']==c['predictive_model_fits']==0,'Decision-only single episode scope')


def fit_binding(control,models,model_audit,n):
    c=model_audit['counts'];model_counts_check(c,n)
    require(models['counts']==c,'Independent/producer model counts binding')
    require(control['independent_solve_counts']==models['independent_solve_counts']['control'],'Repeated independent control solve scope')
    pc=control['producer_equivalent_fit_counts']
    for key in ('target_fits','log_refits','local_evaluations','local_solves'):
        require(pc[key]==c['control_'+key],'Control producer-equivalent fit count binding')


def execution_accounting(e):
    entries=[]
    invocations=[('main',122,e['control'],e['models'],e['main_model_counts'])]
    invocations += [('physical',v['prefix'],v['control'],v['model'],v['model_counts']) for v in e['physical_prefixes'] if v['prefix']!=122]
    for scope,n,control,models,audit in invocations:
        fit_binding(control,models,audit,n)
        a=audit['counts']['actual_regression_solves'];b=control['independent_solve_counts']['total_regression_solves'];c=models['independent_solve_counts']['total_regression_solves']
        entries.append(dict(scope=scope,events=n,producer_regression_solves=a,independent_control_gate_solves=b,independent_model_check_solves=c,total_regression_solves=a+b+c))
    sums={key:sum(v[key] for v in entries) for key in ('producer_regression_solves','independent_control_gate_solves','independent_model_check_solves','total_regression_solves')}
    return dict(schema='monthly_asset_target255_fit_execution_accounting_v1',model_invocations=len(entries),physical122_reuses_main=True,entries=entries,totals=sums,decision_only_fits=0)


def preparation_evidence(e,model_audit,decision_audit):
    import monthly_asset_target255_independent as ind
    kinds=[('control',ind.CONTROL_FLAGS),('models',ind.MODEL_FLAGS),('incumbent',ind.INCUMBENT_FLAGS),('selection',ind.SELECTION_FLAGS)]
    for key,flags in kinds:receipt(e[key],122,flags,key,width=49)
    require(e['all123_order_prefixes'] is True and e['original_arrays_unchanged'] is True and e['original_training_unchanged'] is True,'Preservation proofs')
    require(e['main_model_counts']==model_audit['model'],'Saved main model audit binding')
    fit_binding(e['control'],e['models'],model_audit['model'],122)
    require(decision_audit['schema']=='monthly_asset_target255_decisions_v1' and len(decision_audit['events'])==122 and decision_audit['counts']==e['selection']['counts'],'Saved decision audit scope')
    require(decision_audit['incumbent_gate']==e['incumbent'],'Saved pre-candidate incumbent receipt')
    for key in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact'):
        require(decision_audit[key] is True,'Decision preservation:'+key)
    require(decision_audit['computes_returns'] is False,'Decisions before outcomes')
    cached=e['cached_prefixes'];physical=e['physical_prefixes']
    require(type(cached) is list and len(cached)==123,'All cached prefixes')
    for n,v in enumerate(cached):
        integer(v['prefix'],'Cached prefix',n);require(v['exact'] is True,'Exact cached prefix')
        receipt(v['incumbent'],n,ind.INCUMBENT_FLAGS,'incumbent')
    require(type(physical) is list and len(physical)==len(PREFIXES),'All model prefixes')
    for n,v in zip(PREFIXES,physical):
        integer(v['prefix'],'Physical prefix',n);require(v['exact'] is True,'Exact physical prefix')
        for key,flags in kinds:receipt(v['model' if key=='models' else key],n,flags,key,width=49)
        fit_binding(v['control'],v['model'],v['model_counts'],n)
        if n==122:
            require(v['model_recomputed'] is False and v.get('reuses_main_verified_full') is True,'Physical122 reuse, no new fit')
            for key,_ in kinds:require(v['model' if key=='models' else key]==e[key],'Physical122 main receipt reuse')
            require(v['model_counts']==e['main_model_counts'],'Physical122 main counts reuse')
        else:
            require(v['model_recomputed'] is True and v.get('reuses_main_verified_full',False) is False,'Physically recomputed prefix')
    require(e['fit_execution_accounting']==execution_accounting(e),'Actual solve accounting with122 excluded from repeat count')


def source_evidence(source,plan):
    require(source['raw_source_hashes']==plan['raw_source_hashes'],'Saved raw source pins')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        require(source[key]==plan['input_sha256'][str(REBUILD/name)],'Registered cached source:'+name)
    require(source['fresh_training_array_reconstruction'] is False and source['cached_training_basis']=='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.','Honest cache provenance scope')
    for key in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact'):require(source[key] is True,'Source preservation:'+key)


def model_callbacks(ind,train,metadata,arrays,n):
    receipts={}
    def control(result):
        require(not receipts,'Control gate once')
        check=ind.verify_control(train,metadata,arrays,result,n=n);receipt(check,n,ind.CONTROL_FLAGS,'control')
        require(check.get('candidate_labels_computed') is False and check.get('candidate_models_computed') is False and check.get('episode_calls')==0,'Control before new labels/models')
        receipts['control']=check;return check
    return control,receipts


def incumbent_callback(ind,arrays,meta,n):
    receipts={}
    def incumbent(orders,scores,pres):
        require(not receipts,'Incumbent gate once')
        check=ind.verify_incumbent(arrays,meta,orders,scores,pres,n=n);receipt(check,n,ind.INCUMBENT_FLAGS,'incumbent')
        receipts['incumbent']=check;return check
    return incumbent,receipts


def exact_prefix(wanted,got,n):
    oo,ss,pp,_=wanted;orders,scores,pres,_=got
    require(set(orders)==set(scores)==set(pres)==set(ARMS),'Exactly two decision arms')
    for arm in ARMS:
        require(orders[arm]==oo[arm][:n],'Order prefix')
        np.testing.assert_array_equal(scores[arm],ss[arm][:n]);np.testing.assert_array_equal(pres[arm],pp[arm][:n])


def bundle_parity(full,part,n):
    q=n*full['width']
    for arm in ('control','candidate'):
        for key,value in part[arm]['raw'].items():np.testing.assert_array_equal(value,full[arm]['raw'][key][:q])
        for key in ('available','local_available'):np.testing.assert_array_equal(part[arm][key],full[arm][key][:q])
    wanted={int(row):value for row,value in zip(full['training_row_ids'],full['label_values'])}
    np.testing.assert_array_equal(part['label_values'],np.array([wanted[int(row)] for row in part['training_row_ids']]))


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_asset_target255_kernel as kernel
    import monthly_asset_target255_independent as ind
    study=old.local();m,arrays,px,op=study.load()
    require(len(m['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 arrays')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');metadata=read(REBUILD/'MONTHLY_META.json')
    require(metadata['monthly']==m and metadata['training']==read(study.arch.P/'META.json')['training'],'Original training/query identities')
    before=old.array_snapshot(arrays);train_before=old.array_snapshot(train)
    cg,cr=model_callbacks(ind,train,metadata,arrays,122)
    bundle=kernel.fit_models(train,metadata,arrays,n=122,control_validator=cg)
    require(set(cr)=={'control'},'Verified unchanged control before new labels')
    model_check=ind.verify_models(train,metadata,arrays,bundle,n=122);receipt(model_check,122,ind.MODEL_FLAGS,'models')
    ig,ir=incumbent_callback(ind,arrays,m,122)
    wanted=kernel.decisions(study,arrays,m,bundle,n=122,incumbent_validator=ig,expected_incumbent_orders=read(LOCAL/'local_linear_ORDERS.json'))
    require(set(ir)=={'incumbent'},'Verified original decisions before candidate')
    orders,scores,pres,audit=wanted
    check=ind.verify_selection(arrays,m,bundle,orders,scores=scores,pre_scores=pres,audit=audit,n=122);receipt(check,122,ind.SELECTION_FLAGS,'selection',width=49)
    cached=[];physical=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(m,n)
        cb,cb_receipts=incumbent_callback(ind,aa,mm,n)
        got=kernel.decisions(study,aa,mm,bundle,n=n,incumbent_validator=cb,expected_incumbent_orders=orders[ARMS[0]][:n])
        exact_prefix(wanted,got,n);require(set(cb_receipts)=={'incumbent'},'Prefix original gate')
        cached.append(dict(prefix=n,exact=True,incumbent=cb_receipts['incumbent']))
        if n in PREFIXES and n!=122:
            print(json.dumps(dict(stage='255_physical_model_prefix',events=n)),flush=True)
            cg2,cr2=model_callbacks(ind,train,metadata,arrays,n)
            small=kernel.fit_models(train,metadata,arrays,n=n,control_validator=cg2)
            require(set(cr2)=={'control'},'Prefix model control gate')
            mi=ind.verify_models(train,metadata,arrays,small,n=n);receipt(mi,n,ind.MODEL_FLAGS,'models')
            bundle_parity(bundle,small,n)
            ig2,ir2=incumbent_callback(ind,aa,mm,n)
            gg=kernel.decisions(study,aa,mm,small,n=n,incumbent_validator=ig2,expected_incumbent_orders=orders[ARMS[0]][:n]);exact_prefix(wanted,gg,n)
            oo,ss,pp,au=gg
            si=ind.verify_selection(aa,mm,small,oo,scores=ss,pre_scores=pp,audit=au,n=n);receipt(si,n,ind.SELECTION_FLAGS,'selection',width=49)
            physical.append(dict(prefix=n,exact=True,model_recomputed=True,control=cr2['control'],model=mi,incumbent=ir2['incumbent'],selection=si,model_counts=small['audit']))
    physical.append(dict(prefix=122,exact=True,model_recomputed=False,reuses_main_verified_full=True,control=cr['control'],model=model_check,incumbent=ir['incumbent'],selection=check,model_counts=bundle['audit']))
    require(before==old.array_snapshot(arrays) and train_before==old.array_snapshot(train),'Original arrays/training changed')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    forecasts={arm+'_'+key:val for arm in ('control','candidate') for key,val in bundle[arm]['raw'].items()}
    for arm in ('control','candidate'):
        for key in ('available','local_available'):forecasts[arm+'_'+key]=bundle[arm][key]
    np.savez_compressed(dest/'FORECASTS.npz',**forecasts)
    np.savez_compressed(dest/'LABELS.npz',training_row_ids=bundle['training_row_ids'],label_values=bundle['label_values'])
    for name,value in [('SCORES',scores),('PRE_SCORES',pres)]:np.savez_compressed(dest/(name+'.npz'),**value)
    values={'MODEL_AUDIT.json':dict(model=bundle['audit'],labels=bundle['label_audit'],control=bundle['control']['audit'],candidate=bundle['candidate']['audit']),'DECISION_AUDIT.json':audit,'ORDERS.json':orders,'META.json':m,'SOURCE_PARITY.json':dict(raw_source_hashes=plan['raw_source_hashes'],original_training_hash=sha(REBUILD/'TRAIN_INPUTS.npz'),original_identity_hash=sha(REBUILD/'MONTHLY_META.json'),saved169_parity_hash=sha(REBUILD/'PARITY.json'),cached_training_basis='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.',fresh_training_array_reconstruction=False,original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True),'INDEPENDENT_PREPARE.json':dict(control=cr['control'],models=model_check,incumbent=ir['incumbent'],selection=check,cached_prefixes=cached,physical_prefixes=physical,main_model_counts=bundle['audit'],all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True)}
    values['INDEPENDENT_PREPARE.json']['fit_execution_accounting']=execution_accounting(values['INDEPENDENT_PREPARE.json'])
    preparation_evidence(values['INDEPENDENT_PREPARE.json'],values['MODEL_AUDIT.json'],values['DECISION_AUDIT.json']);source_evidence(values['SOURCE_PARITY.json'],plan)
    for name,value in values.items():dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,model_fit_accounting='Main counts plus newly fitted physical prefixes0/1/53/88; prefix122 reuses main. Independent oracle solves are separately audited.'))
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
    require(set(metrics)==set(orders)==set(ARMS),'Two fixed arms')
    invalid={arm:required_metrics(metrics[arm],arm==ARMS[0]) for arm in ARMS}
    reference=metrics[ARMS[0]];candidate=metrics[ARMS[1]]
    require(all(len(orders[a])>=88 for a in ARMS),'First88 ordered holdings required')
    changed=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    close='same_close_zero' not in invalid[ARMS[1]] and old.guard(candidate['same_close_zero'],reference['same_close_zero'],.005)
    opening='next_open_zero' not in invalid[ARMS[1]] and old.guard(candidate['next_open_zero'],reference['next_open_zero'],0.)
    support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed)
    admitted=bool(close and opening and support)
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=dict(same_close=bool(close),next_open=bool(opening),support=bool(support)),changed_months=changed,undefined_sharpe=invalid,first88_only=True,later34_used=False)

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
    require(check.get('passed') is True and check.get('checked_paths')==6 and check.get('checked_slices')==30 and check.get('independent_accounts') is True and check.get('independent_of_production_model_and_engine') is True and check.get('terminal_liquidation') is False,'6paths30slices independent accounts')
    require(set(check['paths'])==set(ARMS),'Outcome arms')
    for paths in check['paths'].values():
        require(set(paths)=={p[0] for p in ev.PATHS},'Outcome paths')
        for p in paths.values():require(p['passed'] is True and p['monthly_executions']==122 and set(p['slices'])=={'full122','first88','early53','recent35','later34'},'Account slices')


def train_check(training,audit):
    import monthly_asset_target255_independent as ind
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
    import monthly_asset_target255_independent as ind
    check=ind.audit_outcomes(px,op,orders,exports,results,meta);outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    dump(dest/'INDEPENDENT_OUTCOME.json',check);dump(dest/'RESULTS.json',dict(selection=f['selection'],arms=results,reused_sample=True,strict_pit=False,zero_cost_selection=True,next_open_cost_diagnostic_only=True,observer_promotion=False))
    gate(expected);dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,frozen_sha256=sha(OUT/'selected/FROZEN.json'),paths=6,slices=30,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}))
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
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['frozen_sha256']==sha(OUT/'selected/FROZEN.json') and c['paths']==6 and c['slices']==30 and set(c['artifacts_sha256'])==RESULT_FILES,'Complete results')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'results'/name)==h,'Changed result')
    check=read(OUT/'results/INDEPENDENT_OUTCOME.json');outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    result=read(OUT/'results/RESULTS.json');require(result['selection']==f['selection'],'Result selection');result_review_check(expected)
    protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive')
    before=CENTRAL.read_bytes();state=json.loads(before);registration_check(state,expected)
    pick=result['selection']['primary'];arm=result['arms'][pick];m=arm['same_close_zero']['metrics'];best=state['best_training_selected_candidate']
    if result['selection']['admitted'] and m['cagr']>best['cagr']:
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=read(CONTEXT)['limitations'])
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),new_trials=2,model_audit_sha256=sha(OUT/'prepared/MODEL_AUDIT.json'),independent_prepare_sha256=sha(OUT/'prepared/INDEPENDENT_PREPARE.json'),protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=2,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=6,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==251 and sum(x['trials'] for x in state['completed'])==1265,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review255 weak-slot same-cohort asset-type target conditional replication; target70 subject to independent goal audit; frozen observers unchanged.')
    old.commit_state(before,state,'asset_target255_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

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
