"""Fixed long-leader domain with original Local67 scoring; prospective zero-cost bridge."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_leader_bridge253'
PRIOR=ROOT/'outputs/monthly_peer_path252_v2'
REFERENCE=ROOT/'outputs/monthly_basket_core251'
import monthly_basket_core251_host as ref_host
import monthly_basket_core251_independent as ref_ind
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
CONTEXT=ROOT/'work/monthly_leader_bridge253_context.json'
TESTS=ROOT/'work/monthly_leader_bridge253_TESTS.json'
REVIEW=ROOT/'work/monthly_leader_bridge253_CODE_REVIEW.json'
ARMS=('incumbent_local67','long_leader_local67','basket_core251_control')
PREFIXES=(0,1,53,88,122)
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=tuple(name for stem in ('host','kernel','independent') for name in ('monthly_leader_bridge253_'+stem+'.py','test_monthly_leader_bridge253_'+stem+'.py'))
PREP_FILES={'FEATURES.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
SOURCE_REVIEW=ROOT/'work/basket_core251_SOURCE_CONTRACT.json'
PRIOR_REVIEW=ROOT/'work/monthly_peer_path252_RESULT_REVIEW_v2.json'
RESULT_REVIEW=ROOT/'work/monthly_leader_bridge253_RESULT_REVIEW.json'
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()
def files():return [ROOT/'work'/p for p in code_names]
def codes():return {str(p):sha(p) for p in files()}

def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('leader_bridge253_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==248 and sum(x['trials'] for x in state['completed'])==1257,'Predecessor counters')
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
    previous_review=read(PRIOR_REVIEW)
    require(previous_review['passed'] is True and previous_review['unresolved_blockers']==[] and previous_review['plan_sha256']==sha(PRIOR/'PLAN.json') and previous_review['results_completion_sha256']==sha(PRIOR/'results/COMPLETION.json'),'Independently reviewed predecessor')
    context=read(CONTEXT);require(context['arms']==list(ARMS) and context['registered_trials']==3 and context['new_predictive_model_fits']==0 and context['promotable_arms']==[ARMS[1]],'Fixed arms')
    review=code_review();pins=dict(read(PRIOR/'PLAN.json')['input_sha256'])
    for path in files()+[CONTEXT,TESTS,REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',PRIOR_REVIEW,old.PROTECTED]:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in review['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    source=read(SOURCE_REVIEW);source_review_check(source)
    for path in (SOURCE_REVIEW,CALENDAR,Path(ref_host.__file__).resolve(),Path(ref_ind.__file__).resolve(),ref_host.CONTEXT,REFERENCE/'PLAN.json',REFERENCE/'COMPLETION.json',ROOT/'work/monthly_basket_core251_RESULT_REVIEW.json',REFERENCE/'prepared/COMPLETION.json'):
        old.bind(pins,path)
    for name in ref_host.PREP_FILES:old.bind(pins,REFERENCE/'prepared'/name)
    for k,v in source['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_leader_bridge253_independent as ind
    import monthly_leader_bridge253_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_leader_bridge253_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),runtime=ev.runtime(),input_sha256=pins,
        raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits=0,new_predictive_model_fits=0,descriptive_fit_definition='Reuse verified251 exact annual-positive Top6 domain and original Local67 model/pre-scores; new selector eligibility only; no new features or predictive labels.')
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Verify original source/core and both frozen endpoints before restricted original selector; first88 ZERO selection.')
    old.commit_state(before,state,'leader_bridge253_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');source_review_check(read(SOURCE_REVIEW));require(p['context_sha256']==sha(CONTEXT) and p['context']==read(CONTEXT) and p['runtime']==ev.runtime(),'Context/runtime changed')
    old.verify(p['input_sha256']);code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);registration_check(s,expected)
    return p

def decisions(study,arrays,meta,features,n,**kwargs):
    import monthly_leader_bridge253_kernel as kernel
    return kernel.decisions(study,arrays,meta,features,n,**kwargs)


def selection_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.SELECTION_FLAGS),'Independent selection flags')
    require(all(check.get(k) is True for k in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Complete selection input scope')
    require(check['ordered_choices_checked']==3*n and check['checked_events']==n,'Complete event/order scope')
    require(check['numerical_tolerance']==0. and check['producer_modules_imported'] is False and check['computes_returns'] is False,'Exact independent selection scope')


def full_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.PREPARE_FLAGS),'Independent preparation flags')
    require(check.get('full_features_supplied') is True and check.get('original_core_checked') is True and check.get('price_clocks_checked') is True,'Full original core/price source scope')
    require(all(check.get(k) is True for k in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Complete preparation input scope')
    require(check['ordered_choices_checked']==3*n and check['checked_events']==n,'Complete full source/order scope')
    require(check['C_source_atol']==1e-12 and check['C_source_rtol']==0.,'Frozen original weighted core source tolerance')
    require(check['continuous_atol']==2e-10 and check['continuous_rtol']==2e-10 and check['rank_tolerance']==0. and check['exact_discrete'] is True,'Frozen numerical contract')
    require(check['producer_modules_imported'] is False and check['computes_returns'] is False,'Independent preparation scope')


def source_gate_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.SOURCE_GATE_FLAGS),'Independent pre-candidate source flags')
    require(check['checked_events']==n and check['episode_calls']==0 and check['new_target_ranks_computed'] is False and check.get('new_basket_selection_computed') is False,'Source-only scope before target ranks, baskets and all episodes')
    require(check['C_source_atol']==1e-12 and check['C_source_rtol']==0. and check['continuous_atol']==2e-10 and check['continuous_rtol']==2e-10 and check['rank_tolerance']==0.,'Source gate numerical contract')
    require(all(check.get(k) is False for k in ('producer_modules_imported','computes_returns','candidate_selection_computed')),'Independent source-only scope')


def incumbent_gate_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.INCUMBENT_GATE_FLAGS),'Independent pre-candidate incumbent flags')
    require(check['checked_events']==n and check['ordered_choices_checked']==n and check['numerical_tolerance']==0.,'Single incumbent exact scope')
    require(all(check.get(k) is False for k in ('producer_modules_imported','computes_returns','candidate_selection_computed')),'Independent original-only scope')


def pre_gate_check(receipts,n,ind,needs_source):
    require(set(receipts)==({'source','incumbent','endpoints'} if needs_source else {'incumbent','endpoints'}),'Required pre-candidate gates')
    if needs_source:ref_host.source_gate_check(receipts['source'],n,ref_ind)
    ref_host.incumbent_gate_check(receipts['incumbent'],n,ref_ind)
    check=receipts['endpoints']
    require(all(check.get(k) is True for k in ind.ENDPOINT_FLAGS),'Independent endpoint flags')
    require(check.get('passed') is True and check.get('checked_events')==n and check.get('ordered_choices_checked')==2*n,'Two independently verified frozen endpoints')
    require(check.get('numerical_tolerance')==0. and check.get('producer_modules_imported') is False and check.get('computes_returns') is False and check.get('candidate_selection_computed') is False,'Endpoint gate scope')


def gate_callbacks(ind,arrays,meta,panel,calendar,review,features,n,needs_source):
    receipts={}
    def source(source_view):
        require(needs_source and not receipts,'Source gate precedes every episode exactly once')
        check=ref_ind.verify_source_features(panel,meta,calendar,arrays,review,source_view,n=n)
        ref_host.source_gate_check(check,n,ref_ind);receipts['source']=check;return check
    def incumbent(orders,scores,pres):
        require(set(receipts)==({'source'} if needs_source else set()),'Original gate follows source before candidate')
        check=ref_ind.verify_incumbent(arrays,meta,orders,scores,pres,n=n)
        ref_host.incumbent_gate_check(check,n,ref_ind);receipts['incumbent']=check;return check
    def endpoints(orders,scores,pres):
        require(set(receipts)==({'source','incumbent'} if needs_source else {'incumbent'}),'Old endpoint gate before new candidate')
        check=ind.verify_endpoints(arrays,meta,features,orders,scores=scores,pre_scores=pres,n=n)
        receipts['endpoints']=check;pre_gate_check(receipts,n,ind,needs_source);return check
    return source,incumbent,endpoints,receipts


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_leader_bridge253_independent as ind
    import monthly_leader_bridge253_kernel as kernel
    study=old.local();meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original axes')
    before=old.array_snapshot(arrays);rawpanel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Original raw sources');old.verify(raw)
    parity=study.verify_panel_parity(rawpanel,px,op)
    calendar=read(CALENDAR)['sessions'];source_review=read(SOURCE_REVIEW);source_review_check(source_review)
    panel={key:rawpanel[key] for key in ('adj',)}
    features=read(REFERENCE/'prepared/FEATURES.json')
    require(len(features)==122,'Full frozen251 features')
    print(json.dumps(dict(stage='253_source_and_endpoint_gates',events=122)),flush=True)
    sg,ig,eg,pre_gates=gate_callbacks(ind,arrays,meta,panel,calendar,source_review,features,122,True)
    orders,scores,pres,audit=kernel.prepare(study,arrays,meta,panel,calendar,source_review,features,122,source_validator=sg,incumbent_validator=ig,endpoint_validator=eg,expected_endpoints=reference_endpoints(122))
    pre_gate_check(pre_gates,122,ind,True)
    check=ind.verify_prepared(panel,meta,calendar,arrays,source_review,features,orders,scores=scores,pre_scores=pres,audit=audit,n=122)
    full_check(check,122,ind);require(check['counts']==audit['counts'],'Full audit counts')
    require(orders[ARMS[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Exact incumbent orders/scores')
    selection=ind.verify_selection(arrays,meta,features,orders,scores=scores,pre_scores=pres,audit=audit,n=122)
    selection_check(selection,122,ind);require(selection['counts']==audit['counts'],'Selection audit counts')
    checks=[];cached_gates=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n);ff=features[:n]
        _,cg,ce,gr=gate_callbacks(ind,aa,mm,None,None,None,ff,n,False)
        oo,ss,pr,au=decisions(study,aa,mm,ff,n,incumbent_validator=cg,endpoint_validator=ce,expected_endpoints=reference_endpoints(n))
        pre_gate_check(gr,n,ind,False);cached_gates.append(dict(prefix=n,gates=gr))
        require(au['events']==audit['events'][:n],'All123 decisionaudit prefixes')
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'All123 order prefixes')
            np.testing.assert_array_equal(ss[arm],scores[arm][:n]);np.testing.assert_array_equal(pr[arm],pres[arm][:n])
        if n in PREFIXES:
            print(json.dumps(dict(stage='253_physical_prefix',events=n)),flush=True)
            pp={key:frame.loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else frame.iloc[:0].copy() for key,frame in panel.items()}
            cc=old.feature_prefix_calendar(calendar,meta,n)
            sg,ig,eg,pg=gate_callbacks(ind,aa,mm,pp,cc,source_review,ff,n,True)
            po,ps,pt,pa=kernel.prepare(study,aa,mm,pp,cc,source_review,ff,n,source_validator=sg,incumbent_validator=ig,endpoint_validator=eg,expected_endpoints=reference_endpoints(n))
            pre_gate_check(pg,n,ind,True)
            require(pa==au and po==oo,'Physical source/domain/order prefix')
            for arm in ARMS:
                np.testing.assert_array_equal(ps[arm],ss[arm]);np.testing.assert_array_equal(pt[arm],pr[arm])
            ci=ind.verify_prepared(pp,mm,cc,aa,source_review,ff,po,scores=ps,pre_scores=pt,audit=pa,n=n)
            full_check(ci,n,ind);require(ci['counts']==pa['counts'],'Physical audit counts')
            cs=ind.verify_selection(aa,mm,ff,po,scores=ps,pre_scores=pt,audit=pa,n=n)
            selection_check(cs,n,ind);require(cs['counts']==pa['counts'],'Physical selection counts')
            checks.append(dict(prefix=n,full=ci,selection=cs,counts=pa['counts'],pre_candidate_gates=pg,exact=True))
    require(old.array_snapshot(arrays)==before,'Original27 arrays mutated');gate(expected)
    dest=OUT/'prepared';dest.mkdir();dump(dest/'FEATURES.json',features)
    np.savez_compressed(dest/'SCORES.npz',**scores);np.savez_compressed(dest/'PRE_SCORES.npz',**pres)
    for name,value in [('DECISION_AUDIT.json',audit),('META.json',meta),('ORDERS.json',orders),('SOURCE_PARITY.json',dict(raw_source_hashes=raw,execution_prices=parity,source_review_sha256=sha(SOURCE_REVIEW),original_array_fingerprints=old.array_fingerprints(arrays),original_arrays_unchanged=True,original_orders_exact=True,reference251_prepared_sha256=sha(REFERENCE/'prepared/COMPLETION.json'),original_full49_pre_preserved=True,control251_exact=True)),('INDEPENDENT_PREPARE.json',dict(full=check,selection=selection,pre_candidate_gates=pre_gates,all123_cached_control_gates=cached_gates,physical_prefixes=checks,all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_input_unchanged=True,experimental_domain_adds_positive_M252=True,original_full49_pre_preserved=True,control251_exact=True,new_model_fits=0))]:dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,new_model_fits=0,new_predictive_model_fits=0,descriptive_counts=audit['counts']))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',counts=audit['counts'],completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREP_FILES,'Prepared scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    import monthly_leader_bridge253_independent as ind
    preparation_check(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),c,ind,read(OUT/'prepared/DECISION_AUDIT.json'))
    require(len(read(OUT/'prepared/FEATURES.json'))==122,'Full source feature scope')
    return read(OUT/'prepared/META.json'),read(OUT/'prepared/ORDERS.json')


def preparation_check(e,completion,ind,audit):
    require(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_input_unchanged')),'Prepared preservation proof')
    require(e.get('experimental_domain_adds_positive_M252') is True and e.get('original_full49_pre_preserved') is True and e.get('control251_exact') is True,'Explicit domain and preserved architecture')
    require(e['new_model_fits']==0 and completion['new_model_fits']==0 and completion['new_predictive_model_fits']==0 and completion['new_returns_computed'] is False and completion['events']==122,'Prepared scope')
    full_check(e['full'],122,ind);selection_check(e['selection'],122,ind)
    pre_gate_check(e['pre_candidate_gates'],122,ind,True)
    require([x['prefix'] for x in e['all123_cached_control_gates']]==list(range(123)),'All123 cached incumbent gates')
    for item in e['all123_cached_control_gates']:pre_gate_check(item['gates'],item['prefix'],ind,False)
    require(e['full']['counts']==e['selection']['counts']==audit['counts']==completion['descriptive_counts'],'Full counts binding')
    require(len(audit['events'])==122 and audit['counts']['new_model_fits']==audit['counts']['predictive_model_fits']==0,'Audit scope')
    require([x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'All5 independent physical prefixes')
    for item,n in zip(e['physical_prefixes'],PREFIXES):
        require(item['exact'] is True,'Exact physical prefix')
        pre_gate_check(item['pre_candidate_gates'],n,ind,True)
        full_check(item['full'],n,ind);selection_check(item['selection'],n,ind)
        require(item['full']['counts']==item['selection']['counts']==item['counts'] and item['counts']['events']==n,'Prefix count binding')


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
    require(all(len(orders[a])>=88 for a in ARMS),'First88 ordered holdings required')
    undefined={a:required_metrics(metrics[a],a==ARMS[0]) for a in ARMS}
    candidate=ARMS[1]
    changed=[i for i in range(88) if set(orders[candidate][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    guards=dict(close='same_close_zero' not in undefined[candidate] and old.guard(metrics[candidate]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),open='next_open_zero' not in undefined[candidate] and old.guard(metrics[candidate]['next_open_zero'],metrics[ARMS[0]]['next_open_zero'],0.),support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed))
    admitted=all(guards.values())
    return dict(primary=candidate if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months=changed,undefined_sharpe=undefined,diagnostic_control_not_selectable=ARMS[2],first88_only=True,later34_used=False)


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
    require(check.get('passed') is True and check.get('checked_paths')==9 and check.get('checked_slices')==45 and check.get('independent_accounts') is True and check.get('independent_of_production_model_and_engine') is True and check.get('terminal_liquidation') is False,'9paths45slices independent accounts')
    require(set(check['paths'])==set(ARMS),'Outcome arms')
    for paths in check['paths'].values():
        require(set(paths)=={p[0] for p in ev.PATHS},'Outcome paths')
        for p in paths.values():require(p['passed'] is True and p['monthly_executions']==122 and set(p['slices'])=={'full122','first88','early53','recent35','later34'},'Account slices')

def train_check(training,audit):
    import monthly_leader_bridge253_independent as ind
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
    import monthly_leader_bridge253_independent as ind
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
    required=[OUT/'PLAN.json',CONTEXT,OUT/'prepared/COMPLETION.json',OUT/'selected/FROZEN.json',OUT/'selected/TRAIN_RESULTS.json',OUT/'results/COMPLETION.json']
    required += [OUT/'prepared'/name for name in PREP_FILES]
    required += [OUT/'results'/name for name in RESULT_FILES]
    for path in required:require(pins.get(str(path))==sha(path),'Saved review coverage:'+str(path))
    old.verify(pins)
    return review


def close(expected):
    meta,orders=prepared(expected);f=frozen(expected,orders);c=read(OUT/'results/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['frozen_sha256']==sha(OUT/'selected/FROZEN.json') and c['paths']==9 and c['slices']==45 and set(c['artifacts_sha256'])==RESULT_FILES,'Complete results')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'results'/name)==h,'Changed result')
    check=read(OUT/'results/INDEPENDENT_OUTCOME.json');outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    result=read(OUT/'results/RESULTS.json');require(result['selection']==f['selection'],'Result selection')
    result_review_check(expected)
    protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive coverage')
    before=CENTRAL.read_bytes();state=json.loads(before);registration_check(state,expected)
    pick=result['selection']['primary'];arm=result['arms'][pick];m=arm['same_close_zero']['metrics'];best=state['best_training_selected_candidate']
    if result['selection']['admitted'] and m['cagr']>best['cagr']:
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=read(CONTEXT)['limitations'])
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),new_trials=3,new_model_fits=0,new_predictive_model_fits=0,descriptive_statistics=read(OUT/'prepared/DECISION_AUDIT.json')['counts'],protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==249 and sum(x['trials'] for x in state['completed'])==1260,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review fixed leader-domain bridge and training admission; control251 diagnostic only; frozen observers unchanged; target70 remains subject to independent goal audit.')
    old.commit_state(before,state,'leader_bridge253_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)


def source_review_check(source):
    # The original source contract remains frozen and checks its original context.
    ref_host.source_review_check(source)
    c=read(REFERENCE/'COMPLETION.json')
    require(c['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and c['plan_sha256']==sha(REFERENCE/'PLAN.json'),'Verified251 reference')
    r=read(ROOT/'work/monthly_basket_core251_RESULT_REVIEW.json')
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['plan_sha256']==sha(REFERENCE/'PLAN.json') and r['results_completion_sha256']==sha(REFERENCE/'results/COMPLETION.json'),'Reference independently reviewed')
    p=read(REFERENCE/'prepared/COMPLETION.json')
    require(p['status']=='COMPLETE' and p['plan_sha256']==sha(REFERENCE/'PLAN.json') and set(p['artifacts_sha256'])==ref_host.PREP_FILES,'Reference prepared scope')
    for name,h in p['artifacts_sha256'].items():require(sha(REFERENCE/'prepared'/name)==h,'Reference prepared binding')


def reference_endpoints(n):
    names=('incumbent_local67','long_momentum_basket_core')
    orders=read(REFERENCE/'prepared/ORDERS.json')
    scores=dict(np.load(REFERENCE/'prepared/SCORES.npz',allow_pickle=False))
    pres=dict(np.load(REFERENCE/'prepared/PRE_SCORES.npz',allow_pickle=False))
    return dict(orders={a:orders[a][:n] for a in names},scores={a:scores[a][:n].copy() for a in names},pre_scores={a:pres[a][:n].copy() for a in names})


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
    elif a.phase=='prepare':require(bool(a.plan_sha256),'Plan hash required');prepare_recorded(a.plan_sha256)
    else:require(bool(a.plan_sha256),'Plan hash required');globals()[a.phase](a.plan_sha256)
