"""258: direct downside versus symmetric formation moments on a frozen leader domain.

Only the root executes actual lifecycle phases; this host creates no synthetic source fallback.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_downside_basket258'
PRIOR=ROOT/'outputs/monthly_correlation_distance257_v2'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
CONTEXT=ROOT/'work/monthly_downside_basket258_context.json'
CONTRACT=ROOT/'work/monthly_downside_basket258_IMPLEMENTATION_CONTRACT.json'
RESULT_REVIEW=ROOT/'work/monthly_downside_basket258_RESULT_REVIEW.json'
PRIOR_REVIEW=ROOT/'work/monthly_correlation_distance257_v2_RESULT_REVIEW.json'
PHASE_TERMINALS=ROOT/'work/monthly_downside_basket258_PHASE_TERMINALS.json'
LEGACY_DRAFT=ROOT/'work/monthly_portfolio_path247_DRAFT_01.json'
LEGACY_DRAFT_SHA='c6b43b3af5c2df6d3ae514273dcaa8b18c88618bb9825d9e5b18f3a8e40766ae'
SOURCE_SHA='f3c800cbbb0b2a4677874815b74158d805879cc177ee50f4406df2acf66122a9'
NUMERICAL_VERIFICATION=dict(continuous_atol=2e-10,continuous_rtol=2e-10,direct_nav_atol=2e-12,direct_nav_rtol=2e-12,canonical_objective_tolerance=0.,rank_tolerance=0.,selection_tolerance=0.)
TESTS=ROOT/'work/monthly_downside_basket258_TESTS.json'
REVIEW=ROOT/'work/monthly_downside_basket258_CODE_REVIEW.json'
ARMS=('incumbent_local67','basket_downside_moment','basket_symmetric_moment')
PREFIXES=(0,1,53,88,122)
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=('monthly_downside_basket258_host.py','test_monthly_downside_basket258_host.py','monthly_downside_basket258_kernel.py','test_monthly_downside_basket258_kernel.py','monthly_downside_basket258_independent.py','test_monthly_downside_basket258_independent.py')
PREP_FILES={'FEATURES.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
SOURCE_REVIEW=ROOT/'work/after246_portfolio_source_preflight.json'
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()
def files():return [ROOT/'work'/p for p in code_names]
def codes():return {str(p):sha(p) for p in files()}

def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('downside_basket258_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==253 and sum(x['trials'] for x in state['completed'])==1270,'Predecessor counters')
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
    require(read(CONTRACT)['context_sha256']==sha(CONTEXT),'Implementation context binding')
    prior_review=read(PRIOR_REVIEW);require(prior_review['passed'] is True and prior_review['unresolved_blockers']==[] and c['result_review_sha256']==sha(PRIOR_REVIEW),'Predecessor saved review')
    review=code_review();pins=dict(read(PRIOR/'PLAN.json')['input_sha256'])
    for path in files()+[CONTEXT,CONTRACT,TESTS,REVIEW,PRIOR_REVIEW,LEGACY_DRAFT,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',old.PROTECTED]:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in review['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    source=read(SOURCE_REVIEW);source_review_check(source)
    for path in (SOURCE_REVIEW,CALENDAR):old.bind(pins,path)
    for k,v in source['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_downside_basket258_independent as ind
    import monthly_downside_basket258_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_downside_basket258_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),implementation_contract_sha256=sha(CONTRACT),numerical_verification=NUMERICAL_VERIFICATION,runtime=ev.runtime(),input_sha256=pins,
        raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits=0,new_predictive_model_fits=0,descriptive_fit_definition='Fixed-unit formation paths and zero-target downside/symmetric second moments; no predictive labels or fits.')
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare common253 fixed-unit downside/symmetric moments with original incumbent and independent gates; first88 ZERO only.')
    old.commit_state(before,state,'downside_basket258_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');source_review_check(read(SOURCE_REVIEW));require(p['implementation_contract_sha256']==sha(CONTRACT) and p['numerical_verification']==NUMERICAL_VERIFICATION,'Implementation/numerical contract changed');require(p['context_sha256']==sha(CONTEXT) and p['context']==read(CONTEXT) and p['runtime']==ev.runtime(),'Context/runtime changed')
    old.verify(p['input_sha256']);code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);registration_check(s,expected)
    return p

def decisions(study,arrays,meta,features,n,expected_orders,certificate,*,width=None):
    import monthly_downside_basket258_kernel as kernel
    import monthly_downside_basket258_independent as ind
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
    import monthly_downside_basket258_independent as ind
    study=old.local();meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original122x49/27 axes')
    before=old.array_snapshot(arrays);rawpanel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(canonical_sources(raw)==canonical_sources(plan['raw_source_hashes']),'Original64 source map');old.verify(raw)
    parity=study.verify_panel_parity(rawpanel,px,op);calendar=read(CALENDAR)['sessions'];review=read(SOURCE_REVIEW);source_review_check(review)
    panel={'adj':rawpanel['adj']};expected_orders=read(LOCAL/'local_linear_ORDERS.json')
    print(json.dumps(dict(stage='258_full_original_source_moment_gates',events=122)),flush=True)
    result,main=prepare_one(study,arrays,meta,panel,calendar,review,122,expected_orders,width=49)
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
            print(json.dumps(dict(stage='258_physical_prefix',events=n)),flush=True)
            pp={'adj':panel['adj'].loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else panel['adj'].iloc[:0].copy()}
            cc=old.feature_prefix_calendar(calendar,meta,n)
            (ff,po,ps,pt,pa),item=prepare_one(study,aa,mm,pp,cc,review,n,expected_orders[:n],width=49)
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
    source=dict(raw_source_hashes=raw,execution_prices=parity,source_review_sha256=sha(SOURCE_REVIEW),
       original_array_fingerprints=old.array_fingerprints(arrays),original_arrays_unchanged=True,original_orders_exact=True,
       source_metadata_not_inherited_actual_coverage=True,source_legacy_variance_rule_not_used=True)
    for name,value in [('DECISION_AUDIT.json',audit),('META.json',meta),('ORDERS.json',orders),('SOURCE_PARITY.json',source),('INDEPENDENT_PREPARE.json',evidence)]:dump(dest/name,value)
    completion['artifacts_sha256']={name:sha(dest/name) for name in PREP_FILES};dump(dest/'COMPLETION.json',completion)
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',counts=audit['counts'],execution_accounting=evidence['execution_accounting'],completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    plan=gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREP_FILES,'Prepared manifest scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    import monthly_downside_basket258_independent as ind
    preparation_check(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),c,ind,read(OUT/'prepared/DECISION_AUDIT.json'),read(OUT/'prepared/FEATURES.json'))
    source=read(OUT/'prepared/SOURCE_PARITY.json')
    require(canonical_sources(source['raw_source_hashes'])==canonical_sources(plan['raw_source_hashes']) and source['source_review_sha256']==sha(SOURCE_REVIEW),'Saved source identity')
    require(all(source.get(k) is True for k in ('original_arrays_unchanged','original_orders_exact','source_metadata_not_inherited_actual_coverage','source_legacy_variance_rule_not_used')),'Saved source preservation')
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
        for k in ('downside_changed_holdings','symmetric_changed_holdings','candidate_control_changed_holdings'):sourcecounts[k]=0
        require(sourcecounts==item['gates']['source']['feature_counts'],'Source counts binding')
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
    candidate=ARMS[1];guards={};changes={}
    for reference in (ARMS[0],ARMS[2]):
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
    import monthly_downside_basket258_independent as ind
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
    import monthly_downside_basket258_independent as ind
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
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=dict(strict_pit=False,reused_sample=True,global_duplicate='UNKNOWN',selection=read(CONTEXT)['core_and_domain']['dominance']))
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),phase_terminals_sha256=sha(PHASE_TERMINALS),new_trials=3,new_model_fits=0,new_predictive_model_fits=0,descriptive_statistics=read(OUT/'prepared/DECISION_AUDIT.json')['counts'],protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==254 and sum(x['trials'] for x in state['completed'])==1273,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review conditional downside versus symmetric basket preference; frozen observers unchanged; target subject to independent goal audit.')
    old.commit_state(before,state,'downside_basket258_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)


def source_review_check(source):
    flags=('passed','metadata_only','calendar_clock_verified','metadata_axes_verified','raw_source_hashes_verified','raw_source_map_matches_original','listing_identity_metadata_verified','source_window_recomputed_253')
    require(all(source.get(k) is True for k in flags),'Source metadata review')
    require(all(source.get(k) is False for k in ('strict_pit','registered_strategy','actual_shortlist_computed','prices_loaded','npz_decoded')),'Metadata-only source scope')
    require(source['raw_source_count']==64 and canonical_sources(source['raw_source_hashes'])==canonical_sources(read(LOCAL/'PLAN.json')['raw_source_hashes']),'Original64 source identity')
    require(source['calendar']['sha256']==sha(CALENDAR) and source['calendar']['prices']==253 and source['calendar']['return_intervals']==252 and len(source['calendar']['windows'])==122,'Official calendar windows')
    require(source['metadata']['events']==122 and source['metadata']['width']==49 and source['metadata']['formal_monthends']==121,'Source axes')
    require(ROOT/source['draft_path']==LEGACY_DRAFT and source['draft_sha256']==sha(LEGACY_DRAFT)==LEGACY_DRAFT_SHA,'Pinned legacy247 metadata draft')
    require(sha(SOURCE_REVIEW)==SOURCE_SHA and read(CONTEXT)['input_sha256'][str(SOURCE_REVIEW.relative_to(ROOT))]==SOURCE_SHA,'Context binds legacy source metadata')
    require(source['source_scope']['VOX_used'] is False and source['source_scope']['sector_mapping_used'] is False and source['source_scope']['extra_sources']==[],'Price-only source scope')
    # The old metadata establishes clocks/identities; its old variance policy is not258 support.


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
    require(type(counts) is dict and set(counts)==set(ind.COUNT_FIELDS),'Complete22 descriptive counts')
    for key,value in counts.items():integer(value,key)
    integer(counts['events'],'events',n)
    if width is not None:integer(counts['symbols_per_event'],'width',width)
    else:require(counts['symbols_per_event']>0,'Positive width')
    require(counts['new_model_fits']==counts['predictive_model_fits']==0,'No predictive model fits')
    require(counts['original_episode_calls']==1,'Single original episode including n0')
    require(counts['formation_return_observations']==252*counts['trio_statistics'],'Descriptive return count')

def receipt(check,n,ind,kind,*,width=None):
    flags={'source':ind.FEATURE_FLAGS,'cached_source':ind.FEATURE_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,
           'selection':ind.SELECTION_FLAGS,'full':ind.PREPARE_FLAGS}[kind]
    require(type(check) is dict and check.get('passed') is True and all(check.get(k) is True for k in flags),'Independent '+kind+' flags')
    integer(check.get('checked_events'),'checked_events',n)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent isolation')
    for k in ('original_episode_calls','descriptive_path_evaluations','cached_path_validations'):integer(check.get(k),k)
    integer(check['original_episode_calls'],'Independent bridge count',2 if kind=='full' else 1)
    if kind in ('source','cached_source','full'):
        require(check.get('continuous_atol')==2e-10 and check.get('continuous_rtol')==2e-10 and check.get('nav_atol')==2e-12 and check.get('nav_rtol')==2e-12 and check.get('rank_tolerance')==0. and check.get('exact_discrete') is True,'Frozen continuous versus exact contract')
    else:require(check.get('numerical_tolerance')==0.,'Exact selection/ incumbent contract')
    if kind in ('source','cached_source'):
        require(check.get('full_features_supplied') is True and check.get('candidate_selection_computed') is False and check.get('source_rebuilt') is (kind=='source'),'Source gate scope')
        counts_check(check['feature_counts'],n,ind,width)
        c=check['feature_counts'];require(all(c[k]==0 for k in ('downside_changed_holdings','symmetric_changed_holdings','candidate_control_changed_holdings')),'Source does not select')
        require(check['descriptive_path_evaluations']==(c['trio_statistics'] if kind=='source' else 0) and check['cached_path_validations']==(c['trio_statistics'] if kind=='cached_source' else 0),'Source execution counts')
    elif kind=='incumbent':
        require(check.get('full_scores_supplied') is True and check.get('full_pre_scores_supplied') is True and check.get('candidate_selection_computed') is False,'Incumbent gate scope')
        integer(check.get('ordered_choices_checked'),'Incumbent choices',n)
        require(check['descriptive_path_evaluations']==check['cached_path_validations']==0,'Incumbent gate no new paths')
    else:
        require(all(check.get(k) is True for k in ('full_features_supplied','full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Complete decision inputs')
        integer(check.get('ordered_choices_checked'),'All three ordered choices',3*n);counts_check(check['counts'],n,ind,width)
        c=check['counts'];require(check['cached_path_validations']==c['trio_statistics'] and check['descriptive_path_evaluations']==(c['trio_statistics'] if kind=='full' else 0),'Decision execution counts')
        if kind=='full':require(check.get('descriptive_path_returns_checked') is True,'Full descriptive paths')
    return check

def callbacks(ind,arrays,meta,n,expected_orders,*,panel=None,calendar=None,review=None,features=None,certificate=None,width=None):
    records={}
    require(type(expected_orders) is list and len(expected_orders)==n,'Exact saved incumbent prefix required')
    def incumbent(orders,scores,pres):
        require(orders==expected_orders,'Saved original orders before new statistics')
        records['incumbent']=receipt(ind.verify_incumbent(arrays,meta,orders,scores,pres,n=n),n,ind,'incumbent',width=width)
        return records['incumbent']
    def source(values):
        require('incumbent' in records,'Incumbent gate must precede source/statistics gate')
        if panel is not None:
            proof=ind.verify_features(panel,meta,calendar,arrays,review,values,n=n);kind='source'
        else:
            require(type(certificate) is dict and type(certificate.get('checked_events')) is int and certificate['checked_events']>=n,'Cached covering source certificate')
            receipt(certificate,certificate['checked_events'],ind,'source',width=width)
            require(type(features) is list and values==features[:n],'Cached exact feature prefix')
            proof=ind.verify_features_cached(arrays,meta,values,n=n);kind='cached_source'
        records['source']=receipt(proof,n,ind,kind,width=width)
        return records['source']
    return incumbent,source,records

def prepare_one(study,arrays,meta,panel,calendar,review,n,expected_orders,*,width=None):
    import monthly_downside_basket258_kernel as kernel
    import monthly_downside_basket258_independent as ind
    iv,sv,gates=callbacks(ind,arrays,meta,n,expected_orders,panel=panel,calendar=calendar,review=review,width=width)
    result=kernel.prepare(study,arrays,meta,panel,calendar,review,n,incumbent_validator=iv,expected_incumbent_orders=expected_orders,source_validator=sv)
    features,orders,scores,pres,audit=result
    require(set(gates)=={'incumbent','source'} and audit['incumbent_gate']==gates['incumbent'] and audit['source_gate']==gates['source'],'Both early gates preserved')
    require(orders[ARMS[0]]==expected_orders,'Original orders exact')
    check=ind.verify_prepared(panel,meta,calendar,arrays,review,features,orders,scores=scores,pre_scores=pres,audit=audit,n=n)
    full_check(check,n,ind,width);require(check['counts']==audit['counts'],'Complete full audit counts')
    return result,dict(prefix=n,gates=gates,full=check,counts=audit['counts'],exact=True,reused_full=False,source_recomputed=True)

def execution_accounting(e):
    physical=[e['main']]+[x for x in e['physical_prefixes'] if not x['reused_full']]
    cached=e['cached_prefixes'];proofs=[]
    for item in physical:proofs.extend([item['gates']['incumbent'],item['gates']['source'],item['full']])
    for item in cached:proofs.extend([item['gates']['incumbent'],item['gates']['source'],item['selection']])
    producer_source=sum(x['counts']['trio_statistics'] for x in physical)
    producer_cached=sum(x['counts']['trio_statistics'] for x in cached)
    independent_source=sum(x['descriptive_path_evaluations'] for x in proofs)
    independent_cached=sum(x['cached_path_validations'] for x in proofs)
    return dict(producer_total_path_moment_calculations=producer_source+producer_cached,
        independent_total_path_moment_calculations=independent_source+independent_cached,
        total_path_moment_calculations=producer_source+producer_cached+independent_source+independent_cached,
        total_formation_return_steps_computed=252*(producer_source+producer_cached+independent_source+independent_cached),
        cached_validation_recomputes_path_and_moments=True,
        producer_original_episode_calls=sum(x['counts']['original_episode_calls'] for x in physical+cached),
        physical_preparations=len(physical),physical_122_reused=True,cached_decisions=len(cached),
        producer_source_price_observations=sum(x['counts']['source_price_observations'] for x in physical),
        producer_descriptive_path_evaluations=sum(x['counts']['trio_statistics'] for x in physical),
        producer_cached_path_validations=sum(x['counts']['trio_statistics'] for x in cached),
        independent_original_episode_calls=sum(x['original_episode_calls'] for x in proofs),
        independent_descriptive_path_evaluations=sum(x['descriptive_path_evaluations'] for x in proofs),
        independent_cached_path_validations=sum(x['cached_path_validations'] for x in proofs),
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
