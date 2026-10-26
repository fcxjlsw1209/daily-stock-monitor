"""262: matched total versus sector tracking risk on the same long relative numerator.

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
OUT=ROOT/'outputs/monthly_tracking_risk262'
PRIOR=ROOT/'outputs/monthly_weak_tree261'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
CONTEXT=ROOT/'work/monthly_tracking_risk262_context.json'
CONTRACT=ROOT/'work/monthly_tracking_risk262_IMPLEMENTATION_CONTRACT.json'
RESULT_REVIEW=ROOT/'work/monthly_tracking_risk262_RESULT_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_weak_tree261_RESULT_REVIEW.json'
PHASE_TERMINALS=OUT/'PHASE_TERMINALS.json'
SOURCE_SHA='96b656fdef4f5473f23224d1709673ec2c5aac1edb323217ce82957ce54e0fbd'
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,canonical_numeric_tolerance=0.,rank_tolerance=0.,selection_tolerance=0.)
TESTS=ROOT/'work/monthly_tracking_risk262_TESTS.json'
REVIEW=ROOT/'work/monthly_tracking_risk262_CODE_REVIEW.json'
ARMS=('incumbent_local67','sector_total_risk_control','sector_tracking_risk_candidate')
PREFIXES=(0,1,53,88,122)
WORK_FIELDS=('full_source_slot_inspections','cached_slot_validations','statistic_evaluations','rank_evaluations','original_episode_calls')
ZERO_SOURCE_COUNTS=('total_nonzero_deltas','tracking_nonzero_deltas','total_changed_holdings','tracking_changed_holdings','candidate_control_changed_holdings')
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=('monthly_tracking_risk262_host.py','test_monthly_tracking_risk262_host.py','monthly_tracking_risk262_kernel.py','test_monthly_tracking_risk262_kernel.py','monthly_tracking_risk262_independent.py','test_monthly_tracking_risk262_independent.py','tracking_risk262_sources.py')
PREP_FILES={'FEATURES.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
SOURCE_REVIEW=ROOT/'work/tracking_risk262_source_contract.json'
PRE_REG_REVIEW=ROOT/'work/tracking_risk262_PRE_REG_REVIEW.json'
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
    return dict(schema='monthly_tracking_risk262_original169_closure_v1',chain_sha256={str(k):v for k,v in SOURCE_CHAIN169.items()},artifacts_sha256=dict(artifacts),source_input_count=len(plan['input_sha256']),npz_decoded=False,source_cache_rebuilt=False)


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
    mods=[old.module('tracking_risk262_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==257 and sum(x['trials'] for x in state['completed'])==1282,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Predecessor changed')


def predecessor_binding(state):
    record=state['completed'][-1]['completion']
    require(Path(record['path'])==PRIOR/'COMPLETION.json' and record['sha256']==sha(PRIOR/'COMPLETION.json'),'Predecessor central closure binding')


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
    require(context.get('frozen') is True and read(CONTRACT).get('frozen') is True,'Final prospective contracts required')
    paper=read(PRE_REG_REVIEW);require(paper['passed'] is True and paper['blockers']==[],'Independent prospective review')
    paper_review_check(context,read(CONTRACT),paper)
    for path in files()+[CONTEXT,CONTRACT,TESTS,REVIEW,PRIOR_REVIEW,PRE_REG_REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',old.PROTECTED]:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in dict(read(CONTRACT)['input_sha256'],**review['input_sha256']).items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    source=read(SOURCE_REVIEW);source_review_check(source)
    for path in (SOURCE_REVIEW,CALENDAR):old.bind(pins,path)
    for k,v in source['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_tracking_risk262_independent as ind
    import monthly_tracking_risk262_kernel as kernel
    import tracking_risk262_sources as sources
    for implementation in (ind,kernel,sources):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_tracking_risk262_v1',original169_source_closure=source_closure,created_utc=now(),context=context,context_sha256=sha(CONTEXT),implementation_contract_sha256=sha(CONTRACT),numerical_verification=NUMERICAL_VERIFICATION,runtime=ev.runtime(),input_sha256=pins,
        raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits=0,new_predictive_model_fits=0,descriptive_fit_definition='Canonical R252 and current63 total/tracking sample risk; no predictive labels or fits. All repeated source/statistic/rank and episode work separately accounted.')
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare shared65-slot sector-relative risk sources with original source/incumbent gates; first88 ZERO only.')
    old.commit_state(before,state,'tracking_risk262_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');source_review_check(read(SOURCE_REVIEW));require(p['implementation_contract_sha256']==sha(CONTRACT) and p['numerical_verification']==NUMERICAL_VERIFICATION,'Implementation/numerical contract changed');require(p['context_sha256']==sha(CONTEXT) and p['context']==read(CONTEXT) and p['runtime']==ev.runtime(),'Context/runtime changed')
    old.verify(p['input_sha256']);require(p['original169_source_closure']==source_closure_check(p['input_sha256']),'Original169 full source chain');code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);registration_check(s,expected)
    return p

def decisions(study,arrays,meta,features,n,expected_orders,certificate,*,width=None):
    import monthly_tracking_risk262_kernel as kernel
    import monthly_tracking_risk262_independent as ind
    iv,sv,gates=callbacks(ind,arrays,meta,n,expected_orders,features=features,certificate=certificate,width=width)
    orders,scores,pres,audit=kernel.decisions(study,arrays,meta,features,n,incumbent_validator=iv,expected_incumbent_orders=expected_orders,source_validator=sv)
    require(set(gates)=={'incumbent','source'} and audit['incumbent_gate']==gates['incumbent'] and audit['source_gate']==gates['source'],'Cached early gate receipts')
    check=ind.verify_selection(arrays,meta,features,orders,scores=scores,pre_scores=pres,audit=audit,n=n)
    selection_check(check,n,ind,width);require(check['counts']==audit['counts'],'Cached selection counts')
    return (orders,scores,pres,audit),dict(prefix=n,gates=gates,selection=check,counts=audit['counts'],exact=True)


def selection_check(check,n,ind,width=None):return receipt(check,n,ind,'selection',width=width)


def full_check(check,n,ind,width=None):return receipt(check,n,ind,'full',width=width)


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_tracking_risk262_independent as ind
    study=old.local();meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 axes')
    before=old.array_snapshot(arrays);rawpanel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(canonical_sources(raw)==canonical_sources(plan['raw_source_hashes']),'Original64 source map');old.verify(raw)
    parity=study.verify_panel_parity(rawpanel,px,op);calendar=read(CALENDAR)['sessions'];review=read(SOURCE_REVIEW);source_review_check(review)
    import tracking_risk262_sources as sources
    original_panel={'adj':rawpanel['adj']};panel=sources.attach_vox(original_panel,review,calendar);expected_orders=read(LOCAL/'local_linear_ORDERS.json')
    print(json.dumps(dict(stage='262_full_original_source_tracking_gates',events=122)),flush=True)
    result,main=prepare_one(study,arrays,meta,panel,calendar,review,122,expected_orders,width=49,original_panel=original_panel)
    features,orders,scores,pres,audit=result;require(type(features) is list and len(features)==122,'Full feature list')
    fingerprint=digest(features);cached=[];physical=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n)
        (oo,ss,pr,au),proof=decisions(study,aa,mm,features[:n],n,expected_orders[:n],main['gates']['source'],width=49)
        require(au['events']==audit['events'][:n],'All123 decision event prefixes')
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'All123 exact order prefixes')
            np.testing.assert_array_equal(ss[arm],scores[arm][:n]);np.testing.assert_array_equal(pr[arm],pres[arm][:n])
        cached.append(proof)
        if n in PREFIXES:
            if n==122:
                physical.append(dict(main,reused_full=True,source_recomputed=False));continue
            print(json.dumps(dict(stage='262_physical_prefix',events=n)),flush=True)
            aa,mm,pp,cc=physical_inputs(arrays,meta,panel,calendar,n)
            _,_,original_pp,_=physical_inputs(arrays,meta,original_panel,calendar,n)
            (ff,po,ps,pt,pa),item=prepare_one(study,aa,mm,pp,cc,review,n,expected_orders[:n],width=49,original_panel=original_pp)
            require(ff==features[:n] and pa['events']==au['events'] and pa['counts']==au['counts'] and po==oo,'Physical feature/domain/order prefix')
            for arm in ARMS:np.testing.assert_array_equal(ps[arm],ss[arm]);np.testing.assert_array_equal(pt[arm],pr[arm])
            physical.append(item)
    require(digest(features)==fingerprint and old.array_snapshot(arrays)==before,'Full features/original27 arrays unchanged')
    evidence=dict(main=main,cached_prefixes=cached,physical_prefixes=physical,full_features_sha256=fingerprint,
       full_source_certificate_sha256=digest(main['gates']['source']),all123_order_prefixes=True,original_arrays_unchanged=True,
       original_eligible_unchanged=True,new_model_fits=0,predictive_model_fits=0)
    evidence['execution_accounting']=execution_accounting(evidence)
    completion=dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,new_returns_computed=False,
       new_model_fits=0,new_predictive_model_fits=0,descriptive_counts=audit['counts'])
    preparation_check(evidence,completion,ind,audit,features);gate(expected)
    dest=OUT/'prepared';dest.mkdir();dump(dest/'FEATURES.json',features)
    np.savez_compressed(dest/'SCORES.npz',**scores);np.savez_compressed(dest/'PRE_SCORES.npz',**pres)
    source=dict(original169_source_closure=plan['original169_source_closure'],raw_source_hashes=raw,execution_prices=parity,source_review_sha256=sha(SOURCE_REVIEW),
       original_array_fingerprints=old.array_fingerprints(arrays),original_arrays_unchanged=True,original_orders_exact=True,
       source_metadata_not_inherited_actual_coverage=True,new65_source_and_canonical_checks_passed=True,vox_sha256=sha(ROOT/review['vox_prices_path']))
    for name,value in [('DECISION_AUDIT.json',audit),('META.json',meta),('ORDERS.json',orders),('SOURCE_PARITY.json',source),('INDEPENDENT_PREPARE.json',evidence)]:dump(dest/name,value)
    completion['artifacts_sha256']={name:sha(dest/name) for name in PREP_FILES};dump(dest/'COMPLETION.json',completion)
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',counts=audit['counts'],execution_accounting=evidence['execution_accounting'],completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    plan=gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREP_FILES,'Prepared manifest scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    import monthly_tracking_risk262_independent as ind
    preparation_check(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),c,ind,read(OUT/'prepared/DECISION_AUDIT.json'),read(OUT/'prepared/FEATURES.json'))
    source=read(OUT/'prepared/SOURCE_PARITY.json')
    require(canonical_sources(source['raw_source_hashes'])==canonical_sources(plan['raw_source_hashes']) and source['source_review_sha256']==sha(SOURCE_REVIEW),'Saved source identity')
    require(all(source.get(k) is True for k in ('original_arrays_unchanged','original_orders_exact','source_metadata_not_inherited_actual_coverage','new65_source_and_canonical_checks_passed')),'Saved source preservation')
    require(source['original169_source_closure']==plan['original169_source_closure'] and source['vox_sha256']==read(SOURCE_REVIEW)['vox_prices_sha256'],'Saved full original/VOX source chain')
    return read(OUT/'prepared/META.json'),read(OUT/'prepared/ORDERS.json')


def preparation_check(e,completion,ind,audit,features):
    require(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged')),'Prepared preservation')
    for v in (e,completion):require(v.get('new_model_fits')==0,'No new models')
    require(completion['new_predictive_model_fits']==0 and completion['new_returns_computed'] is False and completion['events']==122,'Prepared scope')
    require(type(features) is list and len(features)==122 and digest(features)==e['full_features_sha256'],'Full features binding')
    require(e['full_source_certificate_sha256']==digest(e['main']['gates']['source']),'Full source certificate binding')
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
        for k in ('total_nonzero_deltas','tracking_nonzero_deltas','total_changed_holdings','tracking_changed_holdings','candidate_control_changed_holdings'):sourcecounts[k]=0
        require(sourcecounts==item['gates']['source']['feature_counts'],'Source counts binding')
    for item in [e['main']]+e['physical_prefixes']:
        original_source_check(item['original_source'],item['prefix'],ind)
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
    import monthly_tracking_risk262_independent as ind
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
    import monthly_tracking_risk262_independent as ind
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
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=dict(strict_pit=False,reused_sample=True,global_duplicate='UNKNOWN',selection='Original50/52 core and E preserved; shared risk standardization/domain/auxmix confounds versus incumbent; denominator isolated only between new arms'))
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),phase_terminals_sha256=sha(PHASE_TERMINALS),new_trials=3,new_model_fits=0,new_predictive_model_fits=0,descriptive_statistics=read(OUT/'prepared/DECISION_AUDIT.json')['counts'],protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==258 and sum(x['trials'] for x in state['completed'])==1285,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review matched total versus tracking risk conditional contrast; frozen observers unchanged; target subject to independent goal audit.')
    old.commit_state(before,state,'tracking_risk262_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)


def source_review_check(source):
    require(source.get('schema')=='tracking_risk262_source_contract_v1','Source contract schema')
    require(all(source.get(k) is True for k in ('passed','metadata_checks_passed','metadata_only','raw_source_hashes_verified')),'Source metadata checks')
    require(all(source.get(k) is False for k in ('source_admitted','strict_pit','prices_loaded','npz_decoded','loader_executed','actual_features_computed','actual_models_fitted','actual_orders_computed','actual_returns_computed','observers_modified')),'Metadata-only source scope')
    require(source['known_source_blockers']==[] and source['actual_source_coverage']==source['actual_common_rank_support']=='UNKNOWN_UNTIL_REGISTERED_PREPARE','Prospective numeric coverage')
    require(source['raw_source_count']==64 and canonical_sources(source['raw_source_hashes'])==canonical_sources(read(LOCAL/'PLAN.json')['raw_source_hashes']),'Original64 source identity')
    require(source['panel_first_session']=='2008-01-02' and source['calendar_sha256']==sha(CALENDAR),'Original source clock')
    m=source['metadata'];require(m['events']==122 and m['formal_events']==121 and m['width']==49 and m['stocks_per_event']==32 and m['stock_cells']==3904 and len(set(m['etfs']))==17,'Original source inventory')
    clocks=source['clock_events'];require(len(clocks)==122 and len(source['mapping_records'])==3904 and len(source['listing_identity_cells'])==5,'Complete metadata inventories')
    for q,e in enumerate(clocks):
        require(set(e)==set(source['clock_fields']) and e['query_index']==q and e['initial_partial'] is (q==0) and e['numeric_evaluation_required'] is (q!=0),'Official event clock schema')
        t=e['query_position'];integer(t,'Query position')
        if q==0:require(e['numerator_positions']==e['risk_positions']==e['required_positions']==[],'Partial consumes no new prices')
        else:
            require(e['numerator_positions']==[t-252,t-42] and e['risk_positions']==list(range(t-63,t+1)) and e['required_positions']==[t-252]+list(range(t-63,t+1)),'Exact2/64/65 clocks')
            require(len(e['numerator_dates'])==2 and len(e['risk_dates'])==64 and len(e['required_dates'])==65,'Source date scope')
    binding=read(CONTEXT)['source_policy']['source_contract_binding']
    require(binding['path']==str(SOURCE_REVIEW.relative_to(ROOT)) and binding['sha256']==sha(SOURCE_REVIEW)==SOURCE_SHA,'Context source binding')
    early=source['early_listing_policy'];policy=early['policy']
    require(early['sha256']==sha(ROOT/early['path']) and policy==read(ROOT/early['path']),'Early policy exact binding')
    b=policy['path_bound_override'];oldb=policy['original_history_bound']
    require(policy['passed'] is True and policy['source_admitted'] is False and policy['strict_pit'] is False and policy['publication_date']=='2026-06-15','New declared evidence scope')
    require(b['symbol']=='SPCX' and b['issuer_key']==oldb['issuer_key'] and b['not_before']=='2026-06-12' and b['available_from']=='2026-06-16' and b['kind']=='issuer_actual_first_trade','Exact early path-bound clock')
    require(oldb in source['listing_bounds'] and oldb['not_before']=='2026-06-08' and oldb['available_from']=='2026-06-09','Original239 lowerbound retained')
    require(source['vox']['full2015_source_version_preserved'] is True and source['vox']['extension2014_used'] is False and source['vox']['price_field']=='adjClose','Original238 VOX source')
    require(source['structural_policy']['no_whole_event_missing_fallback'] is True,'No missing-source fallback')
    return source

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
    require(type(counts) is dict and set(counts)==set(ind.COUNT_FIELDS),'Complete26 descriptive counts')
    for key,value in counts.items():integer(value,key)
    integer(counts['events'],'events',n)
    if width is not None:integer(counts['symbols_per_event'],'width',width)
    else:require(counts['symbols_per_event']>0,'Positive width')
    require(counts['new_model_fits']==counts['predictive_model_fits']==0 and counts['original_episode_calls']==3,'Three original episodes and no model fits')
    m=counts['mature_cells'];require(counts['total_risk_statistics']==counts['tracking_risk_statistics']==m,'Two risks per mature cell')
    require(all(counts[k]==63*m for k in ('stock_return_observations','sector_return_observations','tracking_return_observations')),'63 formation return observations per instrument')
    require(counts['initial_partial_events']<=min(1,n) and counts['supported_cells']<=m and counts['warmup_cells']+m<=counts['stock_cells'],'Domain counts')
    require(counts['unmapped_warmup_cells']<=counts['warmup_cells'] and counts['zero_total_cells']<=m and counts['zero_tracking_cells']<=m,'Structural and zero counts')
    require(counts['own_price_observations']%65==counts['sector_price_observations']%65==0,'Complete65 slot observations')
    for k in ('total_changed_holdings','tracking_changed_holdings','candidate_control_changed_holdings'):require(counts[k]<=n,'Changed event count')

def receipt(check,n,ind,kind,*,width=None):
    flags={'source':ind.FEATURE_FLAGS,'cached_source':ind.FEATURE_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS,'full':ind.PREPARE_FLAGS}[kind]
    require(type(check) is dict and check.get('passed') is True and all(check.get(k) is True for k in flags),'Independent '+kind+' flags')
    integer(check.get('checked_events'),'checked_events',n)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent isolation')
    for k in WORK_FIELDS:integer(check.get(k),k)
    integer(check['original_episode_calls'],'Actual independent episodes',0 if kind in ('source','cached_source') else 1 if kind=='incumbent' else 3)
    require(check.get('numerical_tolerance')==0.,'Exact canonical/selection verification')
    if kind!='incumbent':require(check.get('rank_tolerance')==0. and check.get('exact_discrete') is True,'Exact ranks/discrete')
    if kind in ('source','cached_source'):
        require(check.get('full_features_supplied') is True and check.get('source_raw_targets_none') is True and check.get('candidate_selection_computed') is False and check.get('source_rebuilt') is (kind=='source'),'Raw pre-rank source gate scope')
        c=check['feature_counts'];counts_check(c,n,ind,width)
        require(all(c[k]==0 for k in ZERO_SOURCE_COUNTS),'Raw source cannot select or fund')
    elif kind=='incumbent':
        require(check.get('full_scores_supplied') is True and check.get('full_pre_scores_supplied') is True and check.get('candidate_selection_computed') is False,'Incumbent-only scope')
        integer(check.get('ordered_choices_checked'),'Incumbent choices',n)
        require(all(check[k]==0 for k in WORK_FIELDS if k!='original_episode_calls'),'No new source/stat/rank work in incumbent gate')
        return check
    else:
        require(all(check.get(k) is True for k in ('full_features_supplied','full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Complete selection evidence')
        integer(check.get('ordered_choices_checked'),'Three ordered choice sets',3*n)
        c=check['counts'];counts_check(c,n,ind,width)
    slots=c['own_price_observations']+c['sector_price_observations']
    expected=dict(full_source_slot_inspections=slots if kind in ('source','full') else 0,cached_slot_validations=slots if kind in ('cached_source','selection') else 0,statistic_evaluations=2*c['mature_cells'],rank_evaluations=0 if kind in ('source','cached_source') else 2*(n-c['initial_partial_events']))
    require(all(check[k]==v for k,v in expected.items()),'Actual source/stat/rank work scope')
    return check

def unranked_prefix(features,n):
    require(type(features) is list and type(n) is int and 0<=n<=len(features),'Consumed feature prefix')
    result=[]
    for event in features[:n]:
        require(type(event) is dict and 'total_target' in event and 'tracking_target' in event,'Rank field schema')
        item=deepcopy(event);item['total_target']=None;item['tracking_target']=None;result.append(item)
    return result


def paper_review_check(context,contract,paper):
    expected={str(CONTEXT.relative_to(ROOT)):'work/monthly_tracking_risk262_context.REVIEWED_DRAFT.json',str(CONTRACT.relative_to(ROOT)):'work/monthly_tracking_risk262_IMPLEMENTATION_CONTRACT.REVIEWED_DRAFT.json'}
    mapping=context['reviewed_draft_snapshots'];require(mapping==contract['reviewed_draft_snapshots'] and set(mapping)==set(expected),'Only two reviewed-draft relocations')
    require(paper['passed'] is True and paper['blockers']==[],'Prospective review passed')
    require(set(expected)<=set(paper['input_sha256']),'Both reviewed draft inputs required')
    for name,digest0 in paper['input_sha256'].items():
        if name in expected:
            entry=mapping[name];require(entry['snapshot_path']==expected[name] and entry['sha256']==digest0,'Exact reviewed draft mapping')
            path=ROOT/entry['snapshot_path']
        else:path=Path(name) if Path(name).is_absolute() else ROOT/name
        require(sha(path)==digest0,'Prospective reviewed input changed:'+name)
    for obj in (context,contract):
        r=obj['pre_registration_review'];require(r['path']==str(PRE_REG_REVIEW.relative_to(ROOT)) and r['sha256']==sha(PRE_REG_REVIEW),'Prospective review receipt binding')


def callbacks(ind,arrays,meta,n,expected_orders,*,panel=None,calendar=None,review=None,features=None,certificate=None,width=None):
    records={}
    require(type(expected_orders) is list and len(expected_orders)==n,'Exact saved incumbent prefix required')
    def incumbent(orders,scores,pres):
        require(orders==expected_orders,'Saved original orders before new statistics')
        records['incumbent']=receipt(ind.verify_incumbent(arrays,meta,orders,scores,pres,n=n),n,ind,'incumbent',width=width)
        return records['incumbent']
    def source(values):
        raw_digest=digest(values)
        require('incumbent' in records,'Incumbent gate must precede source/statistics gate')
        if panel is not None:
            proof=ind.verify_features(panel,meta,calendar,arrays,review,values,n=n);kind='source'
        else:
            require(type(certificate) is dict and type(certificate.get('checked_events')) is int and certificate['checked_events']>=n,'Cached covering source certificate')
            receipt(certificate,certificate['checked_events'],ind,'source',width=width)
            require(type(features) is list and digest(values)==digest(unranked_prefix(features,n)),'Cached exact unranked feature prefix')
            proof=ind.verify_features_cached(arrays,meta,values,n=n);kind='cached_source'
        require(digest(values)==raw_digest,'Source verifier mutated raw feature input')
        records['source']=receipt(proof,n,ind,kind,width=width)
        return records['source']
    return incumbent,source,records

def original_source_check(check,n,ind):
    require(type(check) is dict and check.get('passed') is True and all(check.get(k) is True for k in ind.ORIGINAL_SOURCE_FLAGS),'Original source flags')
    integer(check.get('checked_events'),'Original source events',n)
    integer(check.get('episode_calls'),'Original source episodes',0)
    for key in WORK_FIELDS:integer(check.get(key),'Original source '+key,0)
    require(check.get('new_target_ranks_computed') is False and check.get('candidate_selection_computed') is False and check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Original source-only scope')
    require(check.get('C_source_atol')==1e-12 and check.get('C_source_rtol')==0. and check.get('rank_tolerance')==0.,'Frozen original C tolerance')
    return check


def physical_inputs(arrays,meta,panel,calendar,n):
    integer(n,'Physical prefix');require(n<=len(meta['dates']),'Physical range')
    aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n)
    pp={k:(v.loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else v.iloc[:0].copy()) for k,v in panel.items()}
    cc=old.feature_prefix_calendar(calendar,meta,n)
    return aa,mm,pp,cc


def prepare_one(study,arrays,meta,panel,calendar,review,n,expected_orders,*,width=None,original_panel=None):
    import monthly_tracking_risk262_kernel as kernel
    import monthly_tracking_risk262_independent as ind
    original=original_source_check(ind.verify_original_source(panel if original_panel is None else original_panel,meta,calendar,arrays,n=n),n,ind)
    iv,sv,gates=callbacks(ind,arrays,meta,n,expected_orders,panel=panel,calendar=calendar,review=review,width=width)
    result=kernel.prepare(study,arrays,meta,panel,calendar,review,n,incumbent_validator=iv,expected_incumbent_orders=expected_orders,source_validator=sv)
    features,orders,scores,pres,audit=result
    require(set(gates)=={'incumbent','source'} and audit['incumbent_gate']==gates['incumbent'] and audit['source_gate']==gates['source'],'Both early gates preserved')
    require(orders[ARMS[0]]==expected_orders,'Original orders exact')
    check=ind.verify_prepared(panel,meta,calendar,arrays,review,features,orders,scores=scores,pre_scores=pres,audit=audit,n=n)
    full_check(check,n,ind,width);require(check['counts']==audit['counts'],'Complete full audit counts')
    return result,dict(prefix=n,original_source=original,gates=gates,full=check,counts=audit['counts'],exact=True,reused_full=False,source_recomputed=True)

def execution_accounting(e):
    physical=[e['main']]+[x for x in e['physical_prefixes'] if not x['reused_full']]
    cached=e['cached_prefixes'];proofs=[]
    for item in physical:proofs.extend([item['gates']['incumbent'],item['gates']['source'],item['full']])
    for item in cached:proofs.extend([item['gates']['incumbent'],item['gates']['source'],item['selection']])
    def csum(items,key):return sum(x['counts'][key] for x in items)
    prod=dict(full_source_slot_inspections=csum(physical,'own_price_observations')+csum(physical,'sector_price_observations'),cached_slot_validations=csum(cached,'own_price_observations')+csum(cached,'sector_price_observations'),statistic_evaluations=2*csum(physical+cached,'mature_cells'),rank_evaluations=2*sum(x['prefix']-x['counts']['initial_partial_events'] for x in physical+cached),original_episode_calls=csum(physical+cached,'original_episode_calls'))
    independent={k:sum(x[k] for x in proofs) for k in WORK_FIELDS}
    # Contract rank_evaluations counts two vector-domain passes per formal event,
    # including empty domains. Scalar work is two ranks per supported cell.
    rank_work=dict(
        producer=dict(full_scalar_rank_evaluations=2*csum(physical,'supported_cells'),cached_scalar_rank_evaluations=2*csum(cached,'supported_cells')),
        independent=dict(full_scalar_rank_evaluations=2*csum(physical,'supported_cells'),cached_scalar_rank_evaluations=2*csum(cached,'supported_cells')))
    for label,work in rank_work.items():
        work['scalar_rank_evaluations']=work['full_scalar_rank_evaluations']+work['cached_scalar_rank_evaluations']
        (prod if label=='producer' else independent)['scalar_rank_evaluations']=work['scalar_rank_evaluations']
    total={k:prod[k]+independent[k] for k in WORK_FIELDS+('scalar_rank_evaluations',)}
    return dict(physical_preparations=len(physical),physical_122_reused=True,cached_decisions=len(cached),producer=prod,independent=independent,total=total,
        rank_work=rank_work,rank_evaluations_definition='Two vector-domain passes per formal event, including empty domain; not scalar rank calls.',
        scalar_rank_evaluations_definition='Two canonical scalar ranks per supported cell; zero for empty support; source-only gates compute no ranks.',
        original_source_gate_calls=len(physical),original_core_checked_cells=sum(x['original_source']['original_checked_cells'] for x in physical),
        source_and_cached_statistic_recomputation_disclosed=True,return_vector_steps_computed=63*3*(total['statistic_evaluations']//2),
        new_model_fits=0,predictive_model_fits=0)

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
    elif a.phase=='prepare':require(bool(a.plan_sha256),'Plan hash required');prepare_recorded(a.plan_sha256)
    else:require(bool(a.plan_sha256),'Plan hash required');globals()[a.phase](a.plan_sha256)
