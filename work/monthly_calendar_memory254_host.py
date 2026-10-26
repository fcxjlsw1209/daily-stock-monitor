"""Month-end historical episode memory on original Local67; prospective zero-cost study."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_calendar_memory254'
PRIOR=ROOT/'outputs/monthly_leader_bridge253'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
CONTEXT=ROOT/'work/monthly_calendar_memory254_context.json'
TESTS=ROOT/'work/monthly_calendar_memory254_TESTS.json'
REVIEW=ROOT/'work/monthly_calendar_memory254_CODE_REVIEW.json'
ARMS=('incumbent_local67','calendar_memory_local67','fixed21_memory_control')
PREFIXES=(0,1,53,88,122)
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=tuple(name for stem in ('host','kernel','independent') for name in ('monthly_calendar_memory254_'+stem+'.py','test_monthly_calendar_memory254_'+stem+'.py'))
PREP_FILES={'FEATURES.npz','SOURCE_AUDIT.json','EFFECTS.npz','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PRIOR_REVIEW=ROOT/'work/monthly_leader_bridge253_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_calendar_memory254_RESULT_REVIEW.json'
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()
def files():return [ROOT/'work'/p for p in code_names]
def codes():return {str(p):sha(p) for p in files()}

def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('calendar_memory254_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==249 and sum(x['trials'] for x in state['completed'])==1260,'Predecessor counters')
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
    old.bind(pins,CALENDAR)
    import monthly_calendar_memory254_independent as ind
    import monthly_calendar_memory254_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_calendar_memory254_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),runtime=ev.runtime(),input_sha256=pins,
        raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits=0,new_predictive_model_fits=0,descriptive_fit_definition='New month-end own-history anchors, paired fixed21/calendar utility labels and descriptive shrinkage estimates. Original Local67 yearly predictive models and full49 pre-score unchanged.')
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Verify paired calendar-memory source, exact original incumbent, mature-only estimates and once-only boundary selector; first88 ZERO selection.')
    old.commit_state(before,state,'calendar_memory254_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');require(p['context_sha256']==sha(CONTEXT) and p['context']==read(CONTEXT) and p['runtime']==ev.runtime(),'Context/runtime changed')
    old.verify(p['input_sha256']);code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);registration_check(s,expected)
    return p

def digest_record(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def data_digest(data):return digest_record(old.array_fingerprints(data))


def receipt_check(check,n,flags,kind):
    require(all(check.get(k) is True for k in flags),'Independent '+kind+' flags')
    require(check.get('passed') is True and check.get('checked_events')==n,'Independent '+kind+' event scope')
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent '+kind+' isolation')
    if kind=='source':
        require(check.get('episode_calls')==0 and check.get('memory_effects_computed') is False and check.get('candidate_selection_computed') is False,'Source before prediction and selection')
        require(check.get('full_source_audit_supplied') is True and check.get('continuous_atol')==2e-10 and check.get('continuous_rtol')==2e-10 and check.get('exact_discrete') is True,'Full source numeric contract')
    else:
        require(check.get('full_scores_supplied') is True and check.get('full_pre_scores_supplied') is True and check.get('numerical_tolerance')==0.,'Complete exact score contract')
        require(check.get('ordered_choices_checked')==(3*n if kind=='selection' else n),'All ordered choices')
        if kind=='incumbent':require(check.get('candidate_selection_computed') is False,'Incumbent before candidate')
        if kind=='selection':require(check.get('full_effects_supplied') is True and check.get('full_audit_supplied') is True,'Complete effect and audit contract')


def gate_callbacks(ind,arrays,meta,panel,calendar,data,source_audit,n,*,cached_source=None):
    receipts={}
    def source(view):
        require(not receipts,'Source gate first and once')
        if cached_source is None:
            check=ind.verify_source(panel,calendar,meta,view,source_audit,n=n)
            receipt_check(check,n,ind.SOURCE_FLAGS,'source')
            check=dict(check,source_data_sha256=data_digest(data))
        else:
            require(cached_source.get('passed') is True and cached_source.get('source_data_sha256')==data_digest(data),'Previously independently verified unchanged full source')
            check=dict(passed=True,checked_events=n,cached_source_reference=True,
                full_source_checked_events=cached_source['checked_events'],no_new_source_rebuild_claim=True,
                full_source_receipt_sha256=digest_record(cached_source),full_source_data_sha256=cached_source['source_data_sha256'])
        receipts['source']=check;return check
    def incumbent(orders,scores,pres):
        require(set(receipts)=={'source'},'Incumbent after source, before candidate')
        check=ind.verify_incumbent(arrays,meta,orders,scores,pres,n=n)
        receipt_check(check,n,ind.INCUMBENT_FLAGS,'incumbent');receipts['incumbent']=check;return check
    return source,incumbent,receipts


def pre_gate_check(receipts,n,ind,*,cached=False,full_source=None):
    require(set(receipts)=={'source','incumbent'},'Source and original incumbent both required')
    if cached:
        c=receipts['source'];require(c.get('passed') is True and c.get('cached_source_reference') is True and c.get('no_new_source_rebuild_claim') is True and c.get('checked_events')==n and c.get('full_source_checked_events')==122,'Cached source receipt')
        require(full_source is not None and c.get('full_source_receipt_sha256')==digest_record(full_source) and c.get('full_source_data_sha256')==full_source.get('source_data_sha256'),'Exact full-source binding')
    else:receipt_check(receipts['source'],n,ind.SOURCE_FLAGS,'source')
    receipt_check(receipts['incumbent'],n,ind.INCUMBENT_FLAGS,'incumbent')


def exact_prefix(wanted,got,n):
    orders,scores,pres,effects,audit=wanted
    oo,ss,pp,ee,aa=got
    require(aa['events']==audit['events'][:n],'Mature decision audit prefix')
    for arm in ARMS:
        require(oo[arm]==orders[arm][:n],'Ordered prefix')
        for x,y in ((ss,scores),(pp,pres),(ee,effects)):
            np.testing.assert_array_equal(x[arm],y[arm][:n])


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_calendar_memory254_independent as ind
    import monthly_calendar_memory254_kernel as kernel
    study=old.local();meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original axes')
    before=old.array_snapshot(arrays);rawpanel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Original raw sources');old.verify(raw)
    parity=study.verify_panel_parity(rawpanel,px,op)
    calendar=read(CALENDAR)['sessions']
    panel={key:rawpanel[key] for key in ('adj','open','close','high','low','volume')}
    print(json.dumps(dict(stage='254_build_paired_memory',events=122)),flush=True)
    data,source_audit=kernel.build_features(panel,calendar,meta,n=122)
    data_before=data_digest(data);source_before=digest_record(source_audit)
    sg,ig,gates=gate_callbacks(ind,arrays,meta,panel,calendar,data,source_audit,122)
    wanted=kernel.decisions(study,arrays,meta,data,n=122,source_validator=sg,incumbent_validator=ig,expected_incumbent_orders=read(LOCAL/'local_linear_ORDERS.json'))
    orders,scores,pres,effects,audit=wanted;pre_gate_check(gates,122,ind)
    check=ind.verify_selection(arrays,meta,data,orders,scores=scores,pre_scores=pres,effects=effects,audit=audit,n=122)
    receipt_check(check,122,ind.SELECTION_FLAGS,'selection');require(check['counts']==audit['counts'],'Independent counts')
    cached=[];physical=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n)
        sg,ig,cg=gate_callbacks(ind,aa,mm,None,None,data,None,n,cached_source=gates['source'])
        got=kernel.decisions(study,aa,mm,data,n=n,source_validator=sg,incumbent_validator=ig,expected_incumbent_orders=orders[ARMS[0]][:n])
        pre_gate_check(cg,n,ind,cached=True,full_source=gates['source']);exact_prefix(wanted,got,n)
        cached.append(dict(prefix=n,gates=cg,exact=True))
        if n in PREFIXES:
            print(json.dumps(dict(stage='254_physical_prefix',events=n)),flush=True)
            pp={key:frame.loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else frame.iloc[:0].copy() for key,frame in panel.items()}
            dd,sa=kernel.build_features(pp,calendar,mm,n=n)
            sg,ig,pg=gate_callbacks(ind,aa,mm,pp,calendar,dd,sa,n)
            actual=kernel.decisions(study,aa,mm,dd,n=n,source_validator=sg,incumbent_validator=ig,expected_incumbent_orders=orders[ARMS[0]][:n])
            pre_gate_check(pg,n,ind);exact_prefix(wanted,actual,n)
            oo,ss,pr,ee,au=actual
            ci=ind.verify_selection(aa,mm,dd,oo,scores=ss,pre_scores=pr,effects=ee,audit=au,n=n)
            receipt_check(ci,n,ind.SELECTION_FLAGS,'selection');require(ci['counts']==au['counts'],'Physical counts')
            physical.append(dict(prefix=n,gates=pg,selection=ci,counts=au['counts'],exact=True))
    require(old.array_snapshot(arrays)==before,'Original27 arrays mutated')
    require(data_digest(data)==data_before and digest_record(source_audit)==source_before,'Verified source data/audit mutated');gate(expected)
    dest=OUT/'prepared';dest.mkdir()
    for name,val in [('FEATURES',data),('SCORES',scores),('PRE_SCORES',pres),('EFFECTS',effects)]:np.savez_compressed(dest/(name+'.npz'),**val)
    for name,val in [('SOURCE_AUDIT.json',source_audit),('DECISION_AUDIT.json',audit),('META.json',meta),('ORDERS.json',orders),('SOURCE_PARITY.json',dict(raw_source_hashes=raw,execution_prices=parity,original_array_fingerprints=old.array_fingerprints(arrays),original_arrays_unchanged=True,original_orders_exact=True,original_full49_pre_preserved=True,source_data_sha256=data_before,source_audit_sha256=source_before)),('INDEPENDENT_PREPARE.json',dict(selection=check,pre_candidate_gates=gates,all123_cached_control_gates=cached,physical_prefixes=physical,all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_input_unchanged=True,original_full49_pre_preserved=True,new_predictive_model_fits=0))]:dump(dest/name,val)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,new_model_fits=0,new_predictive_model_fits=0,descriptive_counts=audit['counts']))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',counts=audit['counts'],completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def preparation_check(e,completion,ind,audit):
    require(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_input_unchanged','original_full49_pre_preserved')),'Preservation proof')
    require(e['new_predictive_model_fits']==completion['new_predictive_model_fits']==0 and completion['new_returns_computed'] is False and completion['events']==122,'Preparation scope')
    pre_gate_check(e['pre_candidate_gates'],122,ind)
    receipt_check(e['selection'],122,ind.SELECTION_FLAGS,'selection')
    require(e['selection']['counts']==audit['counts']==completion['descriptive_counts'],'Counts binding')
    require(len(audit['events'])==122,'Event inventory')
    require([x['prefix'] for x in e['all123_cached_control_gates']]==list(range(123)),'123cached prefixes')
    for item in e['all123_cached_control_gates']:
        require(item['exact'] is True,'Exact cached prefix');pre_gate_check(item['gates'],item['prefix'],ind,cached=True,full_source=e['pre_candidate_gates']['source'])
    require([x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'5physical prefixes')
    for item in e['physical_prefixes']:
        n=item['prefix'];require(item['exact'] is True,'Exact physical prefix')
        pre_gate_check(item['gates'],n,ind);receipt_check(item['selection'],n,ind.SELECTION_FLAGS,'selection')
        require(item['selection']['counts']==item['counts'],'Physical count binding')


def prepared(expected):
    gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREP_FILES,'Prepared scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    import monthly_calendar_memory254_independent as ind
    preparation_check(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),c,ind,read(OUT/'prepared/DECISION_AUDIT.json'))
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
    require(all(len(orders[a])>=88 for a in ARMS),'First88 ordered holdings required')
    undefined={a:required_metrics(metrics[a],a==ARMS[0]) for a in ARMS}
    candidate=ARMS[1]
    changed=[i for i in range(88) if set(orders[candidate][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    guards=dict(close='same_close_zero' not in undefined[candidate] and old.guard(metrics[candidate]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),open='next_open_zero' not in undefined[candidate] and old.guard(metrics[candidate]['next_open_zero'],metrics[ARMS[0]]['next_open_zero'],0.),support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed))
    c=metrics[candidate];d=metrics[ARMS[2]]
    guards['calendar_vs_fixed21']=('same_close_zero' not in undefined[ARMS[2]] and 'same_close_zero' not in undefined[candidate]
        and all(c[p]['training']['cagr']>=d[p]['training']['cagr']-1e-12 for p in ('same_close_zero','next_open_zero'))
        and c['same_close_zero']['training']['sharpe']>=d['same_close_zero']['training']['sharpe'])
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
    import monthly_calendar_memory254_independent as ind
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
    import monthly_calendar_memory254_independent as ind
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
    require(len(state['completed'])==250 and sum(x['trials'] for x in state['completed'])==1263,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review paired calendar-memory replication and first88 admission; fixed21 diagnostic not promotable; observers unchanged; target70 remains subject to independent goal audit.')
    old.commit_state(before,state,'calendar_memory254_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)


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
