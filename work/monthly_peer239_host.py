"""Fixed mature same-sector peer momentum, prospective three-arm lifecycle."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_peer239'
PRIOR=ROOT/'outputs/monthly_ownsector_gate238'
LOCAL=old.LOCAL
SOURCE=ROOT/'outputs/monthly_peer239_maturity_audit'
CENTRAL=old.CENTRAL
CONTEXT=ROOT/'work/monthly_peer239_context.json'
TESTS=ROOT/'work/monthly_peer239_TESTS.json'
REVIEW=ROOT/'work/monthly_peer239_CODE_REVIEW.json'
ARMS=('incumbent_local67','same_sector_peer_median_rank','matched_absolute_momentum_rank')
KEYS=('mature','supported','momentum','peer_median','peer_target','absolute_target')
COUNTS=('events','stock_cells','mature_cells','supported_cells','endpoint_observations','new_model_fits')
PREFIXES=(0,1,53,88,122)
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=('monthly_peer239_host.py','test_monthly_peer239_host.py','monthly_peer239_kernel.py','test_monthly_peer239_kernel.py','monthly_peer239_independent.py','test_monthly_peer239_independent.py')
PREP_FILES={'FEATURES.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()
def files():return [ROOT/'work'/p for p in code_names]
def codes():return {str(p):sha(p) for p in files()}

def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('peer239_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in mods]);stream=io.StringIO()
    r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    require(r.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=r.testsRun,runtime=ev.runtime(),code_sha256=codes()))

def source_binding():
    c=read(SOURCE/'COMPLETION.json');r=read(SOURCE/'REPORT.json');p=read(SOURCE/'SOURCE_PLAN.json')
    require(c['status']=='COMPLETED_MATURE_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED' and c['admitted'] is True
        and c['plan_sha256']==sha(SOURCE/'SOURCE_PLAN.json') and c['report_sha256']==sha(SOURCE/'REPORT.json')
        and c['independent_review_sha256']==sha(SOURCE/'INDEPENDENT_REVIEW.json'),'Mature source closure')
    require(r['admitted'] is True and not r['gaps'] and r['counts']==c['counts'] and r['counts']['events']==122
        and r['counts']['stock_cells']==3904 and r['original_inventory_preserved'] is True,'Full inventory source admission')
    review=read(SOURCE/'INDEPENDENT_REVIEW.json');require(review['passed'] is True and all(x['exact'] for x in review['physical_prefixes']) and [x['prefix'] for x in review['physical_prefixes']]==list(PREFIXES),'Source prefix proof')
    pins=dict(p['input_sha256'])
    for path in (SOURCE/'SOURCE_PLAN.json',SOURCE/'COMPLETION.json',SOURCE/'REPORT.json',SOURCE/'INDEPENDENT_REVIEW.json'):old.bind(pins,path)
    return dict(input_sha256=pins,source_report_sha256=sha(SOURCE/'REPORT.json'),counts=r['counts'],strict_pit=False)

def code_review():
    r=read(REVIEW)
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['code_sha256']==codes(),'Numerical code review')
    require(r['context_sha256']==sha(CONTEXT) and r['actual_returns_computed'] is False,'Prospective review scope')
    return r

def initial_state(state):
    require(len(state['completed'])==234 and sum(x['trials'] for x in state['completed'])==1215,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name and state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Predecessor/incumbent changed')

def registration_check(state,expected):
    initial_state(state)
    require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration absent')

def register():
    require(not OUT.exists(),'Fresh strategy directory required')
    before=CENTRAL.read_bytes();state=json.loads(before);initial_state(state);require(not state['registered_queue'],'Registered pending experiment')
    t=read(TESTS);require(t['passed'] is True and t['synthetic_only'] is True and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Formal tests')
    c=read(PRIOR/'COMPLETION.json');require(c['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and c['protected_files']==21444 and c['plan_sha256']==sha(PRIOR/'PLAN.json'),'Predecessor closure')
    context=read(CONTEXT);require(context['arms']==list(ARMS) and context['registered_trials']==3 and context['new_model_fits']==0,'Fixed arms')
    review=code_review();source=source_binding();pins=dict(read(PRIOR/'PLAN.json')['input_sha256'])
    for k,v in source['input_sha256'].items():old.bind(pins,Path(k),v)
    for path in files()+[CONTEXT,TESTS,REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',old.PROTECTED]:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in review['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_peer239_independent as ind
    if hasattr(ind,'dependency_files'):
        for path in ind.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_peer239_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),runtime=ev.runtime(),input_sha256=pins,
        source_binding=source,raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare mature-sector peer and matched absolute long momentum targets; select on first88 zero-cost paths.')
    old.commit_state(before,state,'peer239_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');require(p['context_sha256']==sha(CONTEXT) and p['context']==read(CONTEXT) and p['runtime']==ev.runtime(),'Context/runtime changed')
    old.verify(p['input_sha256']);require(p['source_binding']==source_binding(),'Source binding changed');code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);registration_check(s,expected)
    return p

def validate_values(values,arrays,n):
    require(set(values)==set(KEYS),'Six fixed values')
    shape=np.asarray(arrays['baseline']).shape;require(len(shape)==2 and type(n) is int and 0<=n<=shape[0],'Value axes')
    for k,v in values.items():
        require(v.shape==shape and v.dtype==np.dtype(bool if k in KEYS[:2] else np.float64),'Value dtype/shape '+k)
        if k not in KEYS[:2]:require(not np.isinf(v[:n]).any(),'Finite derived arithmetic')
    require(not (values['supported'][:n]&~values['mature'][:n]).any(),'Support requires maturity')
    for k,mask in (('momentum','mature'),('peer_median','supported'),('peer_target','supported'),('absolute_target','supported')):
        require(np.array_equal(np.isfinite(values[k][:n]),values[mask][:n]),'Exact availability '+k)
    for k in KEYS[-2:]:require((np.abs(values[k][:n][values['supported'][:n]])<=.5).all(),'Bounded percentile targets')

def decisions(study,arrays,meta,values,n):
    validate_values(values,arrays,n);before=old.array_snapshot(arrays)
    pre=study.modified_pre_score(study.configs()[4],arrays,n);policy=read(LOCAL/'INCUMBENT_POLICY.json')
    orders,scores,pres={},{},{}
    targets=(np.full(arrays['baseline'].shape,np.nan),values['peer_target'],values['absolute_target'])
    for arm,target in zip(ARMS,targets):
        calls=[]
        def boundary(changed,good,names,effects,original,pol):
            require(not calls and np.array_equal(good,arrays['eligible'][:n]),'Exactly one original-E episode')
            calls.append(True);pres[arm]=changed.copy()
            return study.ep.select_boundary(changed,good,names,effects,original,pol)
        chosen,score=study.adapter.score_and_choose('incumbent66',0. if arm==ARMS[0] else .05,target,arrays,meta,n,
            pre_score=pre.copy(),effects=arrays['effects'],episode_policy=policy,boundary_selector=boundary)
        require(len(calls)==1,'Single episode required')
        scores[arm]=score.copy();orders[arm]=study.rp.make_orders(chosen,score,meta)
    require(old.array_snapshot(arrays)==before,'Original cache mutated')
    return orders,scores,pres

def source_prefix(source,n):
    from copy import deepcopy
    r=deepcopy(source);r['events']=r['events'][:n];r['records']=[x for x in r['records'] if x['query_index']<n];rr=r['records'];r['gaps']=[x for x in rr if x['gap_reasons']]
    r['counts']=dict(events=n,stock_cells=len(rr),mature_cells=sum(x['history_status']=='observed_full_window' for x in rr),warmup_cells=sum(x['history_status']=='official_pre_inception' for x in rr),partial_cells=sum(x['initial_partial'] for x in rr),supported_cells=sum(x['structural_peer_support'] for x in rr),deferred_sector_cells=sum('missing_sector' in x['deferred_gap_reasons'] for x in rr),deferred_endpoints=sum(sum(y.startswith('endpoint:') for y in x['deferred_gap_reasons']) for x in rr),gap_cells=len(r['gaps']))
    return r

def feature_check(check,values,ind):
    require(all(check.get(k) is True for k in ind.FEATURE_FLAGS),'Independent feature flag missing')
    require(check['numerical_tolerance']==0. and check['producer_modules_imported'] is False and check['computes_returns'] is False,'Feature exact independent scope')
    c=check['counts'];require(set(c)==set(COUNTS),'Feature count keys')
    require(c['events']==len(values['mature']) and c['mature_cells']==int(values['mature'].sum()) and c['supported_cells']==int(values['supported'].sum()) and c['endpoint_observations']==2*c['mature_cells'] and c['new_model_fits']==0,'Feature coverage')

def selection_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.SELECTION_FLAGS) and check['ordered_choices_checked']==3*n and check['full_scores_supplied'] is True and check['full_pre_scores_supplied'] is True,'Independent selection coverage')
    require(check['numerical_tolerance']==0. and check['checked_events']==n and check['producer_modules_imported'] is False and check['computes_returns'] is False,'Exact score checks')

def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_peer239_kernel as kernel
    import monthly_peer239_independent as ind
    study=old.local();meta,arrays,px,op=study.load();require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original axes')
    before=old.array_snapshot(arrays);rawpanel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Original raw sources');old.verify(raw)
    parity=study.verify_panel_parity(rawpanel,px,op);panel={'adj':rawpanel['adj']};calendar=read(old.CALENDAR)['sessions'];source=read(SOURCE/'REPORT.json')
    values,audit=kernel.build_features(panel,meta,calendar,arrays,source);validate_values(values,arrays,122)
    checked=ind.verify_features(panel,meta,calendar,arrays,source,values,audit);feature_check(checked,values,ind)
    require(checked['counts']==audit['counts'],'Independent counts mismatch')
    orders,scores,pres=decisions(study,arrays,meta,values,122);require(orders[ARMS[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Incumbent exact order parity')
    selected=ind.verify_selection(arrays,meta,values,orders,scores=scores,pre_scores=pres,n=122);selection_check(selected,122,ind)
    checks=[]
    for n in PREFIXES:
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n);cc=old.feature_prefix_calendar(calendar,meta,n);rr=source_prefix(source,n)
        pp={'adj':panel['adj'].loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else panel['adj'].iloc[:0].copy()}
        vv,au=kernel.build_features(pp,mm,cc,aa,rr);ci=ind.verify_features(pp,mm,cc,aa,rr,vv,au);feature_check(ci,vv,ind)
        for key in KEYS:np.testing.assert_array_equal(vv[key],values[key][:n])
        require(au['events']==audit['events'][:n],'Feature evidence prefix mismatch')
        oo,ss,pr=decisions(study,aa,mm,vv,n);cs=ind.verify_selection(aa,mm,vv,oo,scores=ss,pre_scores=pr,n=n);selection_check(cs,n,ind)
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'Order feature prefix mismatch');np.testing.assert_array_equal(ss[arm],scores[arm][:n]);np.testing.assert_array_equal(pr[arm],pres[arm][:n])
        checks.append(dict(prefix=n,feature=ci,selection=cs,source_rows=len(pp['adj']),source_last_session=str(pp['adj'].index[-1].date()) if n else None,calendar_rows=len(cc),exact=True))
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n);vv={k:v[:n].copy() for k,v in values.items()};oo,ss,pr=decisions(study,aa,mm,vv,n)
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'All123 order prefix');np.testing.assert_array_equal(ss[arm],scores[arm][:n]);np.testing.assert_array_equal(pr[arm],pres[arm][:n])
    require(old.array_snapshot(arrays)==before,'Original27 arrays changed');gate(expected)
    dest=OUT/'prepared';dest.mkdir();np.savez_compressed(dest/'FEATURES.npz',**values)
    for name,value in [('FEATURE_AUDIT.json',audit),('META.json',meta),('ORDERS.json',orders),('SOURCE_PARITY.json',dict(raw_source_hashes=raw,execution_prices=parity,original_array_fingerprints=old.array_fingerprints(arrays),original_arrays_unchanged=True,original_orders_exact=True,source_report_sha256=sha(SOURCE/'REPORT.json'))),('INDEPENDENT_PREPARE.json',dict(feature=checked,selection=selected,physical_prefixes=checks,all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,source_inventory_preserved=True,new_model_fits=0))]:dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,counts=audit['counts'],artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,new_model_fits=0))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',counts=audit['counts'],completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def prepared(expected):
    gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREP_FILES,'Prepared scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    import monthly_peer239_independent as ind
    with np.load(OUT/'prepared/FEATURES.npz',allow_pickle=False) as data:values={k:data[k].copy() for k in data.files}
    meta=read(OUT/'prepared/META.json');validate_values(values,{'baseline':np.empty((len(meta['dates']),49))},122)
    preparation_check(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),values,read(OUT/'prepared/FEATURE_AUDIT.json'),c,read(SOURCE/'REPORT.json'),ind)
    return meta,read(OUT/'prepared/ORDERS.json')

def preparation_check(e,values,audit,completion,source,ind):
    require(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged','source_inventory_preserved')),'Prepared preservation proof')
    require(e['new_model_fits']==0 and completion['new_model_fits']==0 and completion['new_returns_computed'] is False,'Prepared scope')
    def counts(n):
        v={k:x[:n] for k,x in values.items()};sc=source_prefix(source,n)['counts']
        return dict(events=n,stock_cells=sc['stock_cells'],mature_cells=int(v['mature'].sum()),supported_cells=int(v['supported'].sum()),endpoint_observations=2*int(v['mature'].sum()),new_model_fits=0)
    expected=counts(122);require(expected['mature_cells']==source['counts']['mature_cells'] and expected['supported_cells']==source['counts']['supported_cells'],'Prepared source coverage')
    require(completion['counts']==audit['counts']==e['feature']['counts']==expected,'Prepared audit coverage')
    feature_check(e['feature'],values,ind);selection_check(e['selection'],122,ind)
    require([x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'All five physical prefixes required')
    for item,n in zip(e['physical_prefixes'],PREFIXES):
        require(item['exact'] is True and item['feature']['counts']==counts(n),'Exact prefix coverage')
        feature_check(item['feature'],{k:v[:n] for k,v in values.items()},ind);selection_check(item['selection'],n,ind)

def select_first88(metrics,orders):
    require(set(metrics)==set(orders)==set(ARMS),'Three fixed arms')
    changed=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    versus=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[2]][i]['symbols'])]
    support=lambda x:len(x)>=3 and any(i<53 for i in x) and any(i>=53 for i in x)
    guards=dict(close_vs_incumbent=old.guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),
        open_vs_incumbent=old.guard(metrics[ARMS[1]]['next_open_zero'],metrics[ARMS[0]]['next_open_zero'],0.),
        close_vs_absolute_control=old.guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_absolute_control=old.guard(metrics[ARMS[1]]['next_open_zero'],metrics[ARMS[2]]['next_open_zero'],0.),changed_support=support(changed),changed_support_vs_control=support(versus))
    return dict(primary=ARMS[1] if all(guards.values()) else ARMS[0],admitted=all(guards.values()),guards=guards,changed_months=changed,changed_months_vs_control=versus,first88_only=True,later34_used=False)

def select(expected):
    meta,orders=prepared(expected);require(not (OUT/'selected').exists(),'Selection write-once');px,op=ev.prices();engine=ev.engine();metrics={}
    for arm in ARMS:
        metrics[arm]={}
        for path,mode,cost in ev.PATHS[:2]:
            _,eq,_,_=engine.replay(px,op,orders[arm][:88],cost=cost,execution=mode,start=pd.Timestamp(meta['dates'][0]),end=pd.Timestamp(meta['ends'][87]))
            require(str(eq.index[-1].date())==meta['ends'][87],'Training cutoff');metrics[arm][path]=old.train_metrics(engine,eq,meta)
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
    import monthly_peer239_independent as ind
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
    import monthly_peer239_independent as ind
    check=ind.audit_outcomes(px,op,orders,exports,results,meta);outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    dump(dest/'INDEPENDENT_OUTCOME.json',check);dump(dest/'RESULTS.json',dict(selection=f['selection'],arms=results,reused_sample=True,strict_pit=False,zero_cost_selection=True,next_open_cost_diagnostic_only=True,observer_promotion=False))
    gate(expected);dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,frozen_sha256=sha(OUT/'selected/FROZEN.json'),paths=9,slices=45,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE',full_cagr={a:results[a]['same_close_zero']['metrics']['cagr'] for a in ARMS},completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def close(expected):
    meta,orders=prepared(expected);f=frozen(expected,orders);c=read(OUT/'results/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['frozen_sha256']==sha(OUT/'selected/FROZEN.json') and c['paths']==9 and c['slices']==45 and set(c['artifacts_sha256'])==RESULT_FILES,'Complete results')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'results'/name)==h,'Changed result')
    check=read(OUT/'results/INDEPENDENT_OUTCOME.json');outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    result=read(OUT/'results/RESULTS.json');require(result['selection']==f['selection'],'Result selection')
    protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive coverage')
    before=CENTRAL.read_bytes();state=json.loads(before);registration_check(state,expected)
    pick=result['selection']['primary'];arm=result['arms'][pick];m=arm['same_close_zero']['metrics'];best=state['best_training_selected_candidate']
    if result['selection']['admitted'] and m['cagr']>best['cagr']:
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=read(CONTEXT)['limitations'])
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),new_trials=3,new_model_fits=0,protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==235 and sum(x['trials'] for x in state['completed'])==1218,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review peer-centering and matched absolute momentum, retain frozen observers; 70% goal subject to verified outcome.')
    old.commit_state(before,state,'peer239_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    else:require(bool(a.plan_sha256),'Plan hash required');globals()[a.phase](a.plan_sha256)
