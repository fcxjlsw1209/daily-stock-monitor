"""264: positive-gain continuous survival versus endpoint age, full noncalm core.

Only the root executes actual lifecycle phases; this host creates no synthetic source fallback.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
from copy import deepcopy
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_gain_survival264'
PRIOR=ROOT/'outputs/monthly_risk_clock263'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
CONTEXT=ROOT/'work/monthly_gain_survival264_context.json'
CONTRACT=ROOT/'work/monthly_gain_survival264_IMPLEMENTATION_CONTRACT.json'
RESULT_REVIEW=ROOT/'work/monthly_gain_survival264_RESULT_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_risk_clock263_RESULT_REVIEW.json'
PHASE_TERMINALS=OUT/'PHASE_TERMINALS.json'
SOURCE_SHA='e136a6bc21ec3a65e59cae3d6a28be6514cca471c5e1023e4a6eb88e8516f119'
PREDECESSOR_SHA='0d602a5083d4bde0fae6b010b5cbd5178813059062ea7b2cef787cc306a64b83'
NUMERICAL_VERIFICATION=dict(original_core_atol=1e-12,original_core_rtol=0.,canonical_numeric_tolerance=0.,rank_tolerance=0.,selection_tolerance=0.)
TESTS=ROOT/'work/monthly_gain_survival264_TESTS_V2.json'
REVIEW=ROOT/'work/monthly_gain_survival264_CODE_REVIEW.json'
ARMS=('incumbent_local67','endpoint_gain_age_control','continuous_gain_survival_candidate')
PREFIXES=(0,1,53,88,122)
WORK_FIELDS=('full_source_slot_inspections','cached_slot_validations','momentum_evaluations','return_evaluations','statistic_evaluations','ratio_evaluations','rank_evaluations','scalar_rank_evaluations','original_episode_calls','original_source_price_observations','original_source_cells','positive_gain_anchors','survival_comparisons','endpoint_comparisons','positive_raw_multiplications','original_source_snapshot_price_observations','original_source_snapshot_hashes')
ZERO_SOURCE_COUNTS=('endpoint_nonzero_deltas','survival_nonzero_deltas','endpoint_changed_holdings','survival_changed_holdings','candidate_control_changed_holdings')
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=('monthly_gain_survival264_host.py','test_monthly_gain_survival264_host.py','monthly_gain_survival264_kernel.py','test_monthly_gain_survival264_kernel.py','monthly_gain_survival264_independent.py','test_monthly_gain_survival264_independent.py','gain_survival264_sources.py')
PREP_FILES={'FEATURES.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
SOURCE_REVIEW=ROOT/'work/gain_survival264_SOURCE_CONTRACT.json'
PRE_REG_REVIEW=ROOT/'work/gain_survival264_PRE_REG_REVIEW_V2.json'
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
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
    return dict(schema='monthly_gain_survival264_original169_closure_v1',chain_sha256={str(k):v for k,v in SOURCE_CHAIN169.items()},artifacts_sha256=dict(artifacts),source_input_count=len(plan['input_sha256']),npz_decoded=False,source_cache_rebuilt=False)


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
    mods=[old.module('gain_survival264_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in mods]);stream=io.StringIO()
    r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    require(r.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=r.testsRun,runtime=ev.runtime(),code_sha256=codes()))

def code_review():
    r=read(REVIEW)
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['code_sha256']==codes(),'Numerical code review')
    require(r['context_sha256']==sha(CONTEXT) and r['implementation_contract_sha256']==sha(CONTRACT) and r['actual_returns_computed'] is False,'Prospective review scope')
    return r

def initial_state(state):
    require(len(state['completed'])==259 and sum(x['trials'] for x in state['completed'])==1288,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Predecessor changed')


def predecessor_binding(state):
    record=state['completed'][-1]['completion']
    require(Path(record['path'])==PRIOR/'COMPLETION.json' and record['sha256']==sha(PRIOR/'COMPLETION.json')==PREDECESSOR_SHA,'Predecessor central closure binding')


def registration_check(state,expected):
    initial_state(state)
    require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration absent')

def register():
    require(not OUT.exists(),'Fresh strategy directory required')
    before=CENTRAL.read_bytes();state=json.loads(before);initial_state(state);require(not state['registered_queue'],'Registered pending experiment')
    t=read(TESTS);require(t['passed'] is True and t['synthetic_only'] is True and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Formal tests')
    c=read(PRIOR/'COMPLETION.json');require(c['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and c['protected_files']==21444 and c['plan_sha256']==sha(PRIOR/'PLAN.json'),'Predecessor closure')
    predecessor_binding(state)
    context=read(CONTEXT);require(context['arms']==list(ARMS) and context['registered_trials']==3 and context['new_predictive_model_fits']==0,'Fixed arms')
    require(read(CONTRACT)['context']['sha256']==sha(CONTEXT),'Implementation context binding')
    prior_review=read(PRIOR_REVIEW);require(prior_review['passed'] is True and prior_review['unresolved_blockers']==[] and c['result_review_sha256']==sha(PRIOR_REVIEW),'Predecessor saved review')
    review=code_review();pins=dict(read(PRIOR/'PLAN.json')['input_sha256']);source_closure=bind_source_closure(pins)
    require(context.get('frozen') is True and read(CONTRACT).get('frozen') is True and context.get('registration_authorized') is True and read(CONTRACT).get('registration_authorized') is True,'Final approved prospective contracts required')
    require(c.get('target_reached') is False and state['best_training_selected_candidate']['cagr']<.7,'Goal already reached; no successor registration')
    require(context['predecessor']['directory']==PRIOR.name and context['predecessor']['completion_sha256']==sha(PRIOR/'COMPLETION.json') and context['predecessor']['result_review_sha256']==sha(PRIOR_REVIEW),'Final predecessor closure binding')
    paper=read(PRE_REG_REVIEW);require(paper['passed'] is True and paper['blocking_findings']==[],'Independent prospective review')
    paper_review_check(context,read(CONTRACT),paper)
    for path in files()+[CONTEXT,CONTRACT,TESTS,REVIEW,PRIOR_REVIEW,PRE_REG_REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',old.PROTECTED]:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in dict(read(CONTRACT)['input_sha256'],**review['input_sha256']).items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    source=read(SOURCE_REVIEW);source_review_check(source)
    for path in (SOURCE_REVIEW,CALENDAR):old.bind(pins,path)
    for k,v in source['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_gain_survival264_independent as ind
    import monthly_gain_survival264_kernel as kernel
    import gain_survival264_sources as sources
    for implementation in (ind,kernel,sources):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_gain_survival264_v1',original169_source_closure=source_closure,created_utc=now(),context=context,context_sha256=sha(CONTEXT),implementation_contract_sha256=sha(CONTRACT),numerical_verification=NUMERICAL_VERIFICATION,runtime=ev.runtime(),input_sha256=pins,
        raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits=0,new_predictive_model_fits=0,descriptive_fit_definition='Original signed raw with matched positive-gain endpoint age and continuous first-breach survival quality on full original U0; no new momentum/risk estimator or predictive fits. Source/statistic/rank/episode work separately accounted.')
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare full original U0 253-price gain survival after original source/incumbent gates; first88 ZERO only.')
    old.commit_state(before,state,'gain_survival264_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');source_review_check(read(SOURCE_REVIEW));require(p['implementation_contract_sha256']==sha(CONTRACT) and p['numerical_verification']==NUMERICAL_VERIFICATION,'Implementation/numerical contract changed');require(p['context_sha256']==sha(CONTEXT) and p['context']==read(CONTEXT) and p['runtime']==ev.runtime(),'Context/runtime changed')
    old.verify(p['input_sha256']);require(p['original169_source_closure']==source_closure_check(p['input_sha256']),'Original169 full source chain');code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);registration_check(s,expected)
    return p

def decisions(study,arrays,meta,features,n,expected_orders,certificate,*,original_certificate,original_anchor,width=None):
    import monthly_gain_survival264_kernel as kernel
    import monthly_gain_survival264_independent as ind
    iv,sv,gates=callbacks(ind,arrays,meta,n,expected_orders,features=features,certificate=certificate,original_certificate=original_certificate,original_anchor=original_anchor,width=width)
    orders,scores,pres,audit=kernel.decisions(study,arrays,meta,features,n,incumbent_validator=iv,expected_incumbent_orders=expected_orders,source_validator=sv)
    require(set(gates)=={'incumbent','source'} and audit['incumbent_gate']==gates['incumbent'] and audit['source_gate']==gates['source'],'Cached early gate receipts')
    check=ind.verify_selection(arrays,meta,features,orders,scores=scores,pre_scores=pres,audit=audit,n=n)
    selection_check(check,n,ind,width);require(check['counts']==audit['counts'],'Cached selection counts')
    return (orders,scores,pres,audit),dict(prefix=n,gates=gates,selection=check,counts=audit['counts'],original_source_certificate_sha256=original_anchor['certificate_sha256'],consumed_original_source_sha256=gates.original_prefix_sha256,exact=True)


def selection_check(check,n,ind,width=None):return receipt(check,n,ind,'selection',width=width)


def full_check(check,n,ind,width=None):return receipt(check,n,ind,'full',width=width)


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_gain_survival264_independent as ind
    study=old.local();meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 axes')
    before=old.array_snapshot(arrays);rawpanel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(canonical_sources(raw)==canonical_sources(plan['raw_source_hashes']),'Original64 source map');old.verify(raw)
    parity=study.verify_panel_parity(rawpanel,px,op);calendar=read(CALENDAR)['sessions'];review=read(SOURCE_REVIEW);source_review_check(review)
    panel={'adj':rawpanel['adj']};expected_orders=read(LOCAL/'local_linear_ORDERS.json')
    print(json.dumps(dict(stage='264_full_original_source_core_gates',events=122)),flush=True)
    result,main=prepare_one(study,arrays,meta,panel,calendar,review,122,expected_orders,width=49)
    features,orders,scores,pres,audit=result;require(type(features) is list and len(features)==122,'Full feature list')
    fingerprint=digest(features);cached=[];physical=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n)
        (oo,ss,pr,au),proof=decisions(study,aa,mm,features[:n],n,expected_orders[:n],main['gates']['source'],original_certificate=main['original_source'],original_anchor=main['original_source_anchor'],width=49)
        require(au['events']==audit['events'][:n],'All123 decision event prefixes')
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'All123 exact order prefixes')
            np.testing.assert_array_equal(ss[arm],scores[arm][:n]);np.testing.assert_array_equal(pr[arm],pres[arm][:n])
        cached.append(proof)
        if n in PREFIXES:
            if n==122:
                physical.append(dict(main,reused_full=True,source_recomputed=False));continue
            print(json.dumps(dict(stage='264_physical_prefix',events=n)),flush=True)
            aa,mm,pp,cc=physical_inputs(arrays,meta,panel,calendar,n)
            (ff,po,ps,pt,pa),item=prepare_one(study,aa,mm,pp,cc,review,n,expected_orders[:n],width=49)
            require(ff==features[:n] and pa['events']==au['events'] and pa['counts']==au['counts'] and po==oo,'Physical feature/domain/order prefix')
            for arm in ARMS:np.testing.assert_array_equal(ps[arm],ss[arm]);np.testing.assert_array_equal(pt[arm],pr[arm])
            physical.append(item)
    require(digest(features)==fingerprint and old.array_snapshot(arrays)==before,'Full features/original27 arrays unchanged')
    evidence=dict(main=main,cached_prefixes=cached,physical_prefixes=physical,full_features_sha256=fingerprint,
       full_source_certificate_sha256=digest(main['gates']['source']),full_original_source_certificate_sha256=main['original_source_anchor']['certificate_sha256'],all123_order_prefixes=True,original_arrays_unchanged=True,
       original_eligible_unchanged=True,new_model_fits=0,predictive_model_fits=0)
    evidence['execution_accounting']=execution_accounting(evidence)
    completion=dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,new_returns_computed=False,
       new_model_fits=0,new_predictive_model_fits=0,descriptive_counts=audit['counts'])
    preparation_check(evidence,completion,ind,audit,features);gate(expected)
    dest=OUT/'prepared';dest.mkdir();dump(dest/'FEATURES.json',features)
    np.savez_compressed(dest/'SCORES.npz',**scores);np.savez_compressed(dest/'PRE_SCORES.npz',**pres)
    source=dict(original169_source_closure=plan['original169_source_closure'],raw_source_hashes=raw,execution_prices=parity,source_review_sha256=sha(SOURCE_REVIEW),
       original_array_fingerprints=old.array_fingerprints(arrays),original_arrays_unchanged=True,original_orders_exact=True,
       source_metadata_not_inherited_actual_coverage=True,new253_source_and_gain_checks_passed=True)
    for name,value in [('DECISION_AUDIT.json',audit),('META.json',meta),('ORDERS.json',orders),('SOURCE_PARITY.json',source),('INDEPENDENT_PREPARE.json',evidence)]:dump(dest/name,value)
    completion['artifacts_sha256']={name:sha(dest/name) for name in PREP_FILES};dump(dest/'COMPLETION.json',completion)
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',counts=audit['counts'],execution_accounting=evidence['execution_accounting'],completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    plan=gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREP_FILES,'Prepared manifest scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    import monthly_gain_survival264_independent as ind
    preparation_check(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),c,ind,read(OUT/'prepared/DECISION_AUDIT.json'),read(OUT/'prepared/FEATURES.json'))
    source=read(OUT/'prepared/SOURCE_PARITY.json')
    require(canonical_sources(source['raw_source_hashes'])==canonical_sources(plan['raw_source_hashes']) and source['source_review_sha256']==sha(SOURCE_REVIEW),'Saved source identity')
    require(all(source.get(k) is True for k in ('original_arrays_unchanged','original_orders_exact','source_metadata_not_inherited_actual_coverage','new253_source_and_gain_checks_passed')),'Saved source preservation')
    require(source['original169_source_closure']==plan['original169_source_closure'],'Saved full original source chain')
    return read(OUT/'prepared/META.json'),read(OUT/'prepared/ORDERS.json')


def preparation_check(e,completion,ind,audit,features):
    require(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged')),'Prepared preservation')
    for v in (e,completion):require(v.get('new_model_fits')==0,'No new models')
    require(completion['new_predictive_model_fits']==0 and completion['new_returns_computed'] is False and completion['events']==122,'Prepared scope')
    require(type(features) is list and len(features)==122 and digest(features)==e['full_features_sha256'],'Full features binding')
    require(e['full_source_certificate_sha256']==digest(e['main']['gates']['source']),'Full source certificate binding')
    require(e['full_original_source_certificate_sha256']==e['main']['original_source_anchor']['certificate_sha256'],'Full original U0 certificate binding')
    counts_check(audit['counts'],122,ind,49);require(len(audit['events'])==122 and audit['counts']==completion['descriptive_counts'],'Full audit scope')
    require(e['main']['prefix']==122 and e['main']['reused_full'] is False and e['main']['source_recomputed'] is True,'Main execution')
    require([x['prefix'] for x in e['cached_prefixes']]==list(range(123)),'All123 cached receipts')
    require([x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'Five physical prefixes')
    for item in [e['main']]+e['physical_prefixes']+e['cached_prefixes']:
        n=item['prefix'];kind='selection' if 'selection' in item else 'full'
        require(item['exact'] is True,'Exact prefix');counts_check(item['counts'],n,ind,49)
        receipt(item['gates']['incumbent'],n,ind,'incumbent',width=49)
        receipt(item['gates']['source'],n,ind,'cached_source' if kind=='selection' else 'source',width=49)
        receipt(item[kind],n,ind,kind,width=49)
        require(item['counts']==item[kind]['counts'],'Receipt/count binding')
        sourcecounts=dict(item['counts'])
        for k in ZERO_SOURCE_COUNTS:sourcecounts[k]=0
        require(sourcecounts==item['gates']['source']['feature_counts'],'Source counts binding')
    # Saved complete-evidence validation; this is not a cached decision call.
    saved_anchors={}
    for item in [e['main']]+e['physical_prefixes']:
        anchor=validated_original_anchor(item['original_source'],item['prefix'],ind,width=49)
        require(digest(item['original_source_anchor'])==digest(anchor),'Saved full original anchor')
        projection=original_prefix_certificate(item['original_source'],anchor,item['prefix'])
        require(item['consumed_original_source_sha256']==digest(projection),'Full consumed original projection')
        saved_anchors[item['prefix']]=anchor
        original_feature_binding(features[:item['prefix']],item['original_source'],item['prefix'],49)
        producer_original_work_check(item['producer_original_source_work'],item['counts'])
        original_snapshot_binding(item['original_source'],item['original_source'],item['producer_original_source_work'])
        for kind in ('source','full'):
            proof=item['gates']['source'] if kind=='source' else item['full']
            for key in ('original_source_price_observations','original_source_cells'):
                require(proof[key]==0,'Original fullpast numerical reconstruction reused')
            original_snapshot_binding(proof,item['original_source'],item['producer_original_source_work'])
            require(proof.get('original_source_proof_sha256')==digest(item['original_source']) and proof.get('original_source_proof_events')==item['prefix'],'Saved independent original proof binding')
    for item in e['cached_prefixes']:
        require(item['original_source_certificate_sha256']==e['full_original_source_certificate_sha256'],'Cached original U0 certificate binding')
        projection=original_prefix_certificate(e['main']['original_source'],saved_anchors[122],item['prefix'])
        require(item['consumed_original_source_sha256']==digest(projection),'Cached consumed original projection binding')
    for item in e['physical_prefixes']:
        reuse=item['prefix']==122
        require(item['reused_full'] is reuse and item['source_recomputed'] is (not reuse),'Physical122 reuse declaration')
        if reuse:require(item==dict(e['main'],reused_full=True,source_recomputed=False),'Exact reused full proof')
    require(e['main']['counts']==audit['counts'] and e['main']['gates']['source']==audit['source_gate'] and e['main']['gates']['incumbent']==audit['incumbent_gate'],'Main audit/gate binding')
    require(e['execution_accounting']==execution_accounting(e),'Actual descriptive execution accounting')


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
        support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed)
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
    import monthly_gain_survival264_independent as ind
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
    import monthly_gain_survival264_independent as ind
    zero_cost_check(exports,results)
    check=ind.audit_outcomes(px,op,orders,exports,results,meta);outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    dump(dest/'INDEPENDENT_OUTCOME.json',check);dump(dest/'RESULTS.json',dict(selection=f['selection'],arms=results,reused_sample=True,strict_pit=False,zero_cost_selection=True,next_open_cost_diagnostic_only=True,observer_promotion=False))
    gate(expected);dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,frozen_sha256=sha(OUT/'selected/FROZEN.json'),paths=9,slices=45,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE',full_cagr={a:results[a]['same_close_zero']['metrics']['cagr'] for a in ARMS},completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def close(expected):
    meta,orders=prepared(expected);f=frozen(expected,orders);c=read(OUT/'results/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['frozen_sha256']==sha(OUT/'selected/FROZEN.json') and c['paths']==9 and c['slices']==45 and set(c['artifacts_sha256'])==RESULT_FILES,'Complete results')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'results'/name)==h,'Changed result')
    check=read(OUT/'results/INDEPENDENT_OUTCOME.json');outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    result=read(OUT/'results/RESULTS.json');require(result['selection']==f['selection'],'Result selection');result_review_check(expected)
    protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive coverage')
    before=CENTRAL.read_bytes();state=json.loads(before);registration_check(state,expected)
    pick=result['selection']['primary'];arm=result['arms'][pick];m=arm['same_close_zero']['metrics'];best=state['best_training_selected_candidate']
    if result['selection']['admitted'] and m['cagr']>best['cagr']:
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=dict(strict_pit=False,reused_sample=True,global_duplicate='UNKNOWN',selection='Full noncalm .52 core on original U0 before E; continuous versus endpoint gain-age contrast isolates first-breach chronology with common gain weights; calm/partial unchanged'))
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),phase_terminals_sha256=sha(PHASE_TERMINALS),new_trials=3,new_model_fits=0,new_predictive_model_fits=0,descriptive_statistics=read(OUT/'prepared/DECISION_AUDIT.json')['counts'],protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==260 and sum(x['trials'] for x in state['completed'])==1291,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review matched continuous gain survival versus endpoint gain-age; frozen observers unchanged; target subject to independent goal audit.')
    old.commit_state(before,state,'gain_survival264_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)


def source_review_check(source):
    import gain_survival264_sources as helper
    require(source.get('schema')==helper.SOURCE_SCHEMA and source.get('passed') is True and source.get('metadata_only') is True,'264 metadata source schema')
    require(all(source.get(k) is False for k in ('source_admitted','strict_pit','prices_loaded','npz_decoded','actual_features_orders_returns_computed','registered_strategy')),'Source metadata only')
    require(source['potential_clock_is_not_state_or_U0_admission'] is True and source['actual253_slot_coverage']==source['original_U0_and_core_parity']=='UNKNOWN_UNTIL_REGISTERED_PREPARE','Source support remains unknown')
    require(source['raw_source_count']==64 and canonical_sources(source['raw_source_hashes'])==canonical_sources(read(LOCAL/'PLAN.json')['raw_source_hashes']),'Original64 source identity')
    require(source['panel_first_session']=='2008-01-02' and source['events']==122 and source['width']==49 and source['formal_monthends']==121,'Original source axes')
    require(source['calendar_sha256']==sha(CALENDAR),'Official calendar binding')
    sessions=read(CALENDAR)['sessions'];integer(source['origin_session_position'],'Official origin',sessions.index('2008-01-02'))
    metadata_path=Path(source['metadata_path']);metadata_path=metadata_path if metadata_path.is_absolute() else ROOT/metadata_path
    require(source['metadata_sha256']==sha(metadata_path),'Original metadata binding');meta=read(metadata_path)
    require(len(source['clock_events'])==122 and len(meta['dates'])==122,'Full source event inventory')
    bounds=read(ROOT/'work/peer239_maturity_policy.json')['securities']
    require(source['listing_bounds']==bounds and len(bounds)==2,'Original listing bounds unchanged')
    require(source['identity_cells']==helper.identities(meta,bounds,122),'Exact metadata issuer identity inventory')
    for q,e in enumerate(source['clock_events']):
        require(e==helper.clock(q,meta['dates'][q],meta['symbols'][q],sessions),'Exact264 potential253 clock')
        require(len(e['symbols'])==49,'Full original inventory')
    require(all(source['source_policy'][k] is False for k in ('additional_provider','additional_sector_mapping','additional_vox')),'No new sector/source augmentation')
    binding=read(CONTEXT)['source_policy']['source_contract_binding']
    require(type(SOURCE_SHA) is str and len(SOURCE_SHA)==64 and binding['path']==str(SOURCE_REVIEW.relative_to(ROOT)) and binding['sha256']==sha(SOURCE_REVIEW)==SOURCE_SHA,'Frozen264 source binding')
    return source


def source_metadata():
    require(not SOURCE_REVIEW.exists(),'Source metadata write-once')
    import gain_survival264_sources as helper
    value=helper.build_metadata_contract()
    require(value['metadata_only'] is True and value['prices_loaded'] is False and value['npz_decoded'] is False,'Metadata construction only')
    dump(SOURCE_REVIEW,value)
    print(json.dumps(dict(status='SOURCE_METADATA_ONLY',path=str(SOURCE_REVIEW),sha256=sha(SOURCE_REVIEW),source_admitted=False)),flush=True)

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


def integer(value,label,expected=None):
    require(type(value) is int and value>=0,'Nonnegative integer '+label)
    if expected is not None:require(value==expected,'Integer scope '+label)
    return value

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def canonical_sources(mapping):
    require(type(mapping) is dict and len(mapping)==64,'Complete64 source mapping')
    answer={}
    for key,value in mapping.items():
        require(type(key) is str and bool(key) and 'CloudStorage' not in key and '..' not in Path(key).parts,'Local source path')
        path=Path(key) if Path(key).is_absolute() else ROOT/key
        path=path.resolve();require(path.is_relative_to(ROOT),'Source outside local root')
        name=path.relative_to(ROOT).as_posix();require(name not in answer,'Source path collision')
        require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Source SHA')
        answer[name]=value
    return answer

def counts_check(counts,n,ind,width=None):
    require(type(counts) is dict and set(counts)==set(ind.COUNT_FIELDS),'Complete264 descriptive counts')
    for key,value in counts.items():integer(value,key)
    integer(counts['events'],'events',n)
    w=counts['symbols_per_event'];require(w>0,'Positive width')
    if width is not None:integer(w,'width',width)
    require(counts['new_model_fits']==counts['predictive_model_fits']==0 and counts['original_episode_calls']==3,'Three original episodes and no model fits')
    partial=counts['initial_partial_events'];require(partial<=min(1,n),'Named partial count')
    formal=n-partial;require(counts['calm_events']+counts['active_events']==formal,'Formal state partition')
    require(counts['available_events']+counts['empty_domain_events']==counts['active_events'],'Active branch partition without risk fallback')
    require(counts['original_core_cells']==w*formal and counts['original_finite_cells']<=w*formal,'Original full-width core domain')
    H=counts['required_cells'];A=counts['positive_gain_anchors']
    require(H<=counts['original_finite_cells'] and counts['available_events']<=H<=w*counts['available_events'],'Complete active U0 source domain')
    require(counts['price_observations']==253*H and counts['gain_return_observations']==252*H and counts['gain_statistics']==H,'Exact253 prices/252 returns/one gain statistic')
    require(counts['endpoint_ranked_cells']==counts['survival_ranked_cells']==H,'Both arms rank the complete same U0')
    Z=counts['no_positive_gain_cells'];N=counts['nonpositive_original_cells']
    require(Z<=H and N<=H and H-Z<=A<=252*(H-Z),'Positive gain anchor scope')
    require(counts['quality_ratio_evaluations']==2*(H-Z) and counts['positive_raw_multiplications']==2*(H-N),'Quality and signed transform operations')
    require(counts['endpoint_comparisons']==A and counts['survival_comparisons']<=252*A,'Chronological comparison scope')
    require(counts['eligible_cells']<=w*n,'Original eligibility count')
    for key in ('endpoint_identity_events','survival_identity_events'):require(counts[key]<=counts['available_events'],'Whole-query raw identity scope')
    for key in ('endpoint_nonzero_deltas','survival_nonzero_deltas'):require(counts[key]<=H,'Delta only on ranked cells')
    for key in ('endpoint_changed_holdings','survival_changed_holdings','candidate_control_changed_holdings'):require(counts[key]<=counts['available_events'],'Changed event count')


def descriptive_work(c,*,full,selecting):
    return dict(full_source_slot_inspections=c['price_observations'] if full else 0,
        cached_slot_validations=0 if full else c['price_observations'],momentum_evaluations=0,
        return_evaluations=c['gain_return_observations'],statistic_evaluations=c['gain_statistics'],
        ratio_evaluations=c['quality_ratio_evaluations'],rank_evaluations=2*c['available_events'] if selecting else 0,
        scalar_rank_evaluations=c['endpoint_ranked_cells']+c['survival_ranked_cells'] if selecting else 0,
        positive_gain_anchors=c['positive_gain_anchors'],survival_comparisons=c['survival_comparisons'],
        endpoint_comparisons=c['endpoint_comparisons'],positive_raw_multiplications=c['positive_raw_multiplications'])


def receipt(check,n,ind,kind,*,width=None):
    flags={'source':ind.FEATURE_FLAGS,'cached_source':ind.FEATURE_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS,'full':ind.PREPARE_FLAGS}[kind]
    require(type(check) is dict and check.get('passed') is True and all(check.get(k) is True for k in flags),'Independent '+kind+' flags')
    integer(check.get('checked_events'),'checked_events',n)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent isolation')
    for k in WORK_FIELDS:integer(check.get(k),k)
    integer(check['original_episode_calls'],'Actual independent episodes',0 if kind in ('source','cached_source') else 1 if kind=='incumbent' else 3)
    require(type(check.get('numerical_tolerance')) in (int,float) and check['numerical_tolerance']==0.,'Exact canonical/selection verification')
    if kind!='incumbent':require(type(check.get('rank_tolerance')) in (int,float) and check['rank_tolerance']==0. and check.get('exact_discrete') is True,'Exact ranks/discrete')
    if kind in ('source','cached_source'):
        require(check.get('full_features_supplied') is True and check.get('source_raw_targets_none') is True and check.get('candidate_selection_computed') is False and check.get('source_rebuilt') is (kind=='source'),'Raw pre-rank source gate scope')
        c=check['feature_counts'];counts_check(c,n,ind,width)
        require(all(c[k]==0 for k in ZERO_SOURCE_COUNTS),'Raw source cannot select or fund')
    elif kind=='incumbent':
        require(check.get('full_scores_supplied') is True and check.get('full_pre_scores_supplied') is True and check.get('candidate_selection_computed') is False,'Incumbent-only scope')
        integer(check.get('ordered_choices_checked'),'Incumbent choices',n)
        require(all(check[k]==0 for k in WORK_FIELDS if k!='original_episode_calls'),'No source/stat/rank work in incumbent gate')
        return check
    else:
        require(all(check.get(k) is True for k in ('full_features_supplied','full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Complete selection evidence')
        integer(check.get('ordered_choices_checked'),'Three ordered choice sets',3*n)
        c=check['counts'];counts_check(c,n,ind,width)
    full=kind in ('source','full');selecting=kind in ('selection','full')
    expected=descriptive_work(c,full=full,selecting=selecting)
    require(all(check[k]==v for k,v in expected.items()),'Actual new source/math/rank work')
    integer(check['original_source_cells'],'No repeated original fullpast reconstruction cells',0)
    integer(check['original_source_price_observations'],'No repeated original fullpast reconstruction observations',0)
    snapshot_work_check(check,c['original_core_cells'],full)
    return check

def snapshot_work_check(check,original_cells,required):
    observations=check['original_source_snapshot_price_observations']
    passes=check['original_source_snapshot_hashes']
    integer(observations,'Source snapshot numeric observations')
    integer(passes,'Source snapshot numeric passes',int(required and original_cells>0))
    if required:
        sha0=check.get('original_source_snapshot_sha256')
        require(type(sha0) is str and len(sha0)==64 and all(x in '0123456789abcdef' for x in sha0),'Consumed source snapshot SHA')
        require(observations>=original_cells if original_cells else observations==0,'Source snapshot consumed matrix observations')
    else:integer(observations,'No cached/incumbent snapshot price reads',0)


def original_snapshot_binding(check,original,work):
    require(check.get('original_source_snapshot_sha256')==original.get('original_source_snapshot_sha256'),'Same physical consumed source snapshot')
    integer(check['original_source_snapshot_price_observations'],'Exact consumed source snapshot observations',work['original_source_price_observations'])
    integer(check['original_source_snapshot_hashes'],'Exact numeric snapshot pass',int(work['original_source_price_observations']>0))


def unranked_prefix(features,n):
    require(type(features) is list and type(n) is int and 0<=n<=len(features),'Consumed feature prefix')
    result=[]
    for event in features[:n]:
        require(type(event) is dict and 'endpoint_target' in event and 'survival_target' in event,'Rank field schema')
        item=deepcopy(event);item['endpoint_target']=None;item['survival_target']=None;result.append(item)
    return result

def paper_review_check(context,contract,paper):
    require(paper.get('passed') is True and paper.get('draft_only') is True and paper.get('blocking_findings')==[],'Independent paper review scope')
    expected={'work/monthly_gain_survival264_context.REVIEW_DRAFT.json','work/monthly_gain_survival264_IMPLEMENTATION_CONTRACT.REVIEW_DRAFT.json'}
    documents=paper['reviewed_documents'];require(expected<=set(documents),'Both original reviewed draft bindings')
    for path,digest0 in documents.items():require(sha(ROOT/path)==digest0,'Reviewed paper document changed:'+path)
    for entry in paper['referenced_interfaces_verified'].values():
        require(sha(ROOT/entry['path'])==entry['sha256'],'Reviewed paper interface changed')
    for obj in (context,contract):
        r=obj['pre_registration_review'];require(r['path']==str(PRE_REG_REVIEW.relative_to(ROOT)) and r['sha256']==sha(PRE_REG_REVIEW),'Prospective review receipt binding')
        require(obj['reviewed_draft_sha256']=={p:documents[p] for p in sorted(expected)},'Unchanged reviewed drafts')

def callbacks(ind,arrays,meta,n,expected_orders,*,original_certificate,original_anchor,original_proof=None,panel=None,calendar=None,review=None,features=None,certificate=None,width=None):
    require(type(expected_orders) is list and len(expected_orders)==n,'Exact saved incumbent prefix required')
    projection=original_prefix_certificate(original_certificate,original_anchor,n)
    records=_GateRecords(digest(projection))
    def incumbent(orders,scores,pres):
        require(orders==expected_orders,'Saved original orders before new statistics')
        records['incumbent']=receipt(ind.verify_incumbent(arrays,meta,orders,scores,pres,n=n),n,ind,'incumbent',width=width)
        return records['incumbent']
    def source(values):
        raw_digest=digest(values)
        require('incumbent' in records,'Incumbent gate must precede source/statistics gate')
        original_feature_binding(values,projection,n,width)
        if panel is not None:
            require(original_proof is not None,'Independent issued original proof capability required')
            proof=ind.verify_features(panel,meta,calendar,arrays,review,values,n=n,original_proof=original_proof);kind='source'
            require(proof.get('original_source_proof_sha256')==original_anchor['certificate_sha256'] and proof.get('original_source_proof_events')==n,'Raw source original capability binding')
            require(all(proof.get(k)==original_certificate.get(k) for k in ('original_source_snapshot_sha256','original_source_snapshot_price_observations','original_source_snapshot_hashes')),'Raw source consumed snapshot receipt binding')
        else:
            require(type(certificate) is dict and type(certificate.get('checked_events')) is int and certificate['checked_events']>=n,'Cached covering source certificate')
            receipt(certificate,certificate['checked_events'],ind,'source',width=width)
            require(type(features) is list and digest(values)==digest(unranked_prefix(features,n)),'Cached exact unranked feature prefix')
            proof=ind.verify_features_cached(arrays,meta,values,n=n);kind='cached_source'
        require(digest(values)==raw_digest,'Source verifier mutated raw feature input')
        records['source']=receipt(proof,n,ind,kind,width=width)
        return records['source']
    return incumbent,source,records

_ORIGINAL_ANCHOR_TOKEN=object()

class _OriginalAnchor(dict):
    """Runtime capability produced only by full source verification; JSON-safe evidence."""
    def __init__(self,record,token=None):
        require(token is _ORIGINAL_ANCHOR_TOKEN,'Validated original source anchor required')
        dict.__init__(self,record)
    def _readonly(self,*args,**kwargs):raise TypeError('Immutable verified original source anchor')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_readonly
    def __deepcopy__(self,memo):return self

class _GateRecords(dict):
    def __init__(self,prefix_sha256):
        super().__init__();self.original_prefix_sha256=prefix_sha256

def validated_original_anchor(certificate,n,ind,*,width=None):
    """Full gate only; hashes the complete verified certificate and every event once."""
    original_source_check(certificate,n,ind,width=width)
    record=dict(schema='monthly_gain_survival264_original_source_anchor_v1',checked_events=n,certificate_sha256=digest(certificate),original_event_sha256=tuple(digest(e) for e in certificate['original_events']))
    return _OriginalAnchor(record,_ORIGINAL_ANCHOR_TOKEN)

def original_prefix_certificate(certificate,anchor,n):
    """Read/hash consumed bodies only; a bare caller-provided SHA is not a capability."""
    require(type(anchor) is _OriginalAnchor,'Full source gate validated anchor required')
    integer(n,'Consumed original prefix');require(n<=anchor['checked_events'],'Original anchor coverage')
    require(type(certificate) is dict and type(certificate.get('original_events')) is list and type(certificate.get('u0_indices')) is list,'Original certificate containers')
    events=certificate['original_events'][:n];ids=certificate['u0_indices'][:n]
    require(len(events)==len(ids)==n,'Consumed original evidence coverage')
    for q,(event,u0) in enumerate(zip(events,ids)):
        require(digest(event)==anchor['original_event_sha256'][q] and u0==event['u0_indices'],'Consumed original event changed')
    return dict(schema='monthly_gain_survival264_original_source_prefix_v1',checked_events=n,original_events=events,u0_indices=ids)

def original_source_check(check,n,ind,*,width=None):
    require(type(check) is dict and check.get('passed') is True and all(check.get(k) is True for k in ind.ORIGINAL_SOURCE_FLAGS),'Original source flags')
    integer(check.get('checked_events'),'Original source events',n);integer(check.get('episode_calls'),'Original source episodes',0)
    for key in WORK_FIELDS:integer(check.get(key),'Original source '+key)
    require(all(check[k]==0 for k in WORK_FIELDS if k not in ('original_source_price_observations','original_source_cells','original_source_snapshot_price_observations','original_source_snapshot_hashes')),'No new264 work at original gate')
    require(check.get('new_target_ranks_computed') is False and check.get('candidate_selection_computed') is False and check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Original source-only scope')
    require(check.get('C_source_atol')==1e-12 and check.get('C_source_rtol')==0. and check.get('rank_tolerance')==0.,'Frozen original C tolerance')
    events=check['original_events'];require(type(events) is list and len(events)==n and type(check['u0_indices']) is list and len(check['u0_indices'])==n,'Complete original U0 certificate')
    cells=finite_cells=0
    fields=('original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63')
    for q,e in enumerate(events):
        require(type(e) is dict and set(e)==set(ind.ORIGINAL_EVENT_KEYS),'Original event schema')
        integer(e['query_index'],'Original query',q)
        require(type(e['date']) is str and type(e['initial_partial']) is bool and type(e['calm']) is bool,'Original event metadata')
        ids=e['u0_indices'];require(type(ids) is list and ids==sorted(set(ids)) and all(type(j) is int and j>=0 for j in ids),'Original U0 indices')
        require(ids==check['u0_indices'][q],'Original U0 duplicate evidence')
        if e['initial_partial']:
            require(q==0 and e['date']=='2016-08-17' and ids==[] and all(e[k] is None for k in fields),'Partial original source scope')
        else:
            w=len(e['original_rank']);require(w>0 and all(type(e[k]) is list and len(e[k])==w for k in fields),'Original full-width evidence')
            if width is not None:integer(w,'Original evidence width',width)
            require(all(j<w for j in ids) and ids==[j for j,x in enumerate(e['original_raw']) if x is not None],'Original finite U0 identity')
            cells+=w;finite_cells+=len(ids)
    integer(check.get('original_checked_cells'),'Original checked cells',cells);integer(check.get('original_source_cells'),'Original fullpast cells',cells)
    integer(check.get('original_finite_cells'),'Original finite cells',finite_cells)
    require(check['original_source_price_observations']>=cells,'Original fullpast observations')
    snapshot_work_check(check,cells,True)
    return check


def original_feature_binding(features,certificate,n,width=None):
    require(type(features) is list and len(features)>=n,'Original feature prefix')
    keys=('original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63')
    for q,(f,e) in enumerate(zip(features[:n],certificate['original_events'][:n])):
        require(f['query_index']==q and all(f[k]==e[k] for k in ('date','initial_partial','calm','u0_indices')),'Early original U0/event identity')
        if e['initial_partial']:require(f['cells']==[],'Partial has no source cells');continue
        w=len(e['original_rank']);require(len(f['cells'])==w and (width is None or w==width),'Original feature width')
        for j,cell in enumerate(f['cells']):
            require(cell['column_index']==j and cell['symbol']==f['symbols'][j] and cell['original_finite'] is (j in e['u0_indices']),'Original source cell identity')
            require(digest({k:cell[k] for k in keys})==digest({k:e[k][j] for k in keys}),'Original source numeric evidence exact')

def physical_inputs(arrays,meta,panel,calendar,n):
    integer(n,'Physical prefix');require(n<=len(meta['dates']),'Physical range')
    aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n)
    pp={k:(v.loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else v.iloc[:0].copy()) for k,v in panel.items()}
    cc=old.feature_prefix_calendar(calendar,meta,n)
    return aa,mm,pp,cc


def prepare_one(study,arrays,meta,panel,calendar,review,n,expected_orders,*,width=None):
    import monthly_gain_survival264_kernel as kernel
    import monthly_gain_survival264_independent as ind
    original=ind.verify_original_source(panel,meta,calendar,arrays,n=n)
    anchor=validated_original_anchor(original,n,ind,width=width)
    capability=ind.validated_original_proof(original,n=n,width=width)
    iv,sv,gates=callbacks(ind,arrays,meta,n,expected_orders,original_certificate=original,original_anchor=anchor,original_proof=capability,panel=panel,calendar=calendar,review=review,width=width)
    result=kernel.prepare(study,arrays,meta,panel,calendar,review,n,incumbent_validator=iv,expected_incumbent_orders=expected_orders,source_validator=sv)
    features,orders,scores,pres,audit=result
    require(set(gates)=={'incumbent','source'} and audit['incumbent_gate']==gates['incumbent'] and audit['source_gate']==gates['source'],'Both early gates preserved')
    require(orders[ARMS[0]]==expected_orders,'Original orders exact')
    original_feature_binding(features,original,n,width)
    check=ind.verify_prepared(panel,meta,calendar,arrays,review,features,orders,scores=scores,pre_scores=pres,audit=audit,n=n,original_proof=capability)
    full_check(check,n,ind,width);require(check['counts']==audit['counts'],'Complete full audit counts')
    for receipt0 in (gates['source'],check):
        for k in ('original_source_price_observations','original_source_cells'):require(receipt0[k]==0,'Original fullpast numerical reconstruction reused')
        require(receipt0.get('original_source_proof_sha256')==digest(original) and receipt0.get('original_source_proof_events')==n,'Independent original proof exact binding')
    work=producer_original_work(meta,calendar,n,audit['counts'])
    for receipt0 in (original,gates['source'],check):original_snapshot_binding(receipt0,original,work)
    return result,dict(prefix=n,original_source=original,original_source_anchor=anchor,consumed_original_source_sha256=gates.original_prefix_sha256,producer_original_source_work=work,gates=gates,full=check,counts=audit['counts'],exact=True,reused_full=False,source_recomputed=True)


def producer_original_work(meta,calendar,n,counts):
    formal=[q for q in range(n) if meta['dates'][q]!='2016-08-17']
    if formal:
        start=calendar.index('2008-01-02');end=calendar.index(meta['dates'][formal[-1]])
        rows=end-start+1;symbols=sorted({s for q in formal for s in meta['symbols'][q]})
    else:rows=0;symbols=[]
    out=dict(fullpast_rows=rows,union_symbols=symbols,original_source_price_observations=rows*len(symbols),original_source_cells=sum(len(meta['symbols'][q]) for q in formal))
    producer_original_work_check(out,counts);return out


def producer_original_work_check(work,counts):
    require(type(work) is dict and set(work)=={'fullpast_rows','union_symbols','original_source_price_observations','original_source_cells'},'Producer original source work schema')
    for k in ('fullpast_rows','original_source_price_observations','original_source_cells'):integer(work[k],k)
    syms=work['union_symbols'];require(type(syms) is list and all(type(x) is str and x for x in syms) and syms==sorted(set(syms)),'Producer original source symbol union')
    require(work['original_source_price_observations']==work['fullpast_rows']*len(syms) and work['original_source_cells']==counts['original_core_cells'],'Producer single matrix fullpast work')
    require((counts['original_core_cells']>0)==bool(work['fullpast_rows'])==bool(syms),'No original numeric for empty/partial-only prefix')

def execution_accounting(e):
    physical=[e['main']]+[x for x in e['physical_prefixes'] if not x['reused_full']]
    cached=e['cached_prefixes'];proofs=[]
    for item in physical:proofs.extend([item['original_source'],item['gates']['incumbent'],item['gates']['source'],item['full']])
    for item in cached:proofs.extend([item['gates']['incumbent'],item['gates']['source'],item['selection']])
    def csum(items,key):return sum(x['counts'][key] for x in items)
    allcalls=physical+cached
    prod=dict.fromkeys(WORK_FIELDS,0)
    for item in physical:
        for k,v in descriptive_work(item['counts'],full=True,selecting=True).items():prod[k]+=v
        prod['original_source_price_observations']+=item['producer_original_source_work']['original_source_price_observations']
        prod['original_source_cells']+=item['counts']['original_core_cells']
    for item in cached:
        for k,v in descriptive_work(item['counts'],full=False,selecting=True).items():prod[k]+=v
    prod['original_episode_calls']=csum(allcalls,'original_episode_calls')
    independent={k:sum(x[k] for x in proofs) for k in WORK_FIELDS};total={k:prod[k]+independent[k] for k in WORK_FIELDS}
    rank_work={owner:dict(full_scalar_rank_evaluations=csum(physical,'endpoint_ranked_cells')+csum(physical,'survival_ranked_cells'),cached_scalar_rank_evaluations=csum(cached,'endpoint_ranked_cells')+csum(cached,'survival_ranked_cells'),scalar_rank_evaluations=values['scalar_rank_evaluations']) for owner,values in (('producer',prod),('independent',independent))}
    metadata_work=dict(full_certificate_validations=len(physical),full_certificate_hashes=len(physical),full_event_hashes=sum(x['prefix'] for x in physical),prefix_projection_validations=len(allcalls),prefix_event_hashes=sum(x['prefix'] for x in allcalls),prefix_projection_hashes=len(allcalls),cached_full_certificate_validations=0,cached_full_certificate_hashes=0,cached_event_hashes=sum(x['prefix'] for x in cached),cached_projection_hashes=len(cached),scope='Host runtime original-source capability creation and consumed-prefix gates only; complete saved-evidence revalidation is separate, not cached decision execution. No price access or new statistics in these metadata checks.')
    return dict(physical_preparations=len(physical),physical_122_reused=True,cached_decisions=len(cached),producer=prod,independent=independent,total=total,rank_work=rank_work,original_certificate_metadata_work=metadata_work,
        rank_evaluations_definition='Two actual rank-domain passes per available nonempty event, including exact raw identity; zero on inactive/empty.',scalar_rank_evaluations_definition='Two canonical scalar ranks per ranked cell; raw source gates compute no ranks.',
        original_source_gate_calls=len(physical),original_core_checked_cells=sum(x['original_source']['original_checked_cells'] for x in physical),
        original_source_work_definition='Producer converts one fullpast matrix per preparation. Independent original proof gate runs once per physical preparation and reconstructs each formal query fullpast; raw/full gates consume issued capability, frame identity and exact event bindings with zero repeated pandas reconstruction. Original/raw/full each hash their consumed source snapshot; hash observations and passes are separate. Cached decisions read no fullpast.',
        source_snapshot_work_definition='Numeric snapshot hashing over consumed fullpast official grid and consumed formal symbol union: once at original gate and once at each raw/full reuse; static empty-prefix header hashes excluded from numeric hash passes. No future rows/columns or cached source reads.',
        source_and_cached_statistic_recomputation_disclosed=True,return_vector_steps_computed=total['return_evaluations'],new_model_fits=0,predictive_model_fits=0)

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
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','source-metadata','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='source-metadata':source_metadata()
    elif a.phase=='test':tests()
    elif a.phase=='register':register()
    elif a.phase=='prepare':require(bool(a.plan_sha256),'Plan hash required');prepare_recorded(a.plan_sha256)
    else:require(bool(a.plan_sha256),'Plan hash required');globals()[a.phase](a.plan_sha256)
