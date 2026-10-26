"""Exact-rational core-conditioned auxiliary ordering; preregistered lifecycle."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_core_conditioned243'
PRIOR=ROOT/'outputs/monthly_block_sharpe242'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
CONTEXT=ROOT/'work/monthly_core_conditioned243_context.json'
TESTS=ROOT/'work/monthly_core_conditioned243_TESTS.json'
REVIEW=ROOT/'work/monthly_core_conditioned243_CODE_REVIEW.json'
ARMS=('incumbent_local67','core_conditioned_aux','aux_rank_control')
PREFIXES=(0,1,53,88,122)
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=('monthly_core_conditioned243_host.py','test_monthly_core_conditioned243_host.py','monthly_core_conditioned243_kernel.py','test_monthly_core_conditioned243_kernel.py','monthly_core_conditioned243_independent.py','test_monthly_core_conditioned243_independent.py')
PREP_FILES={'SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()
def files():return [ROOT/'work'/p for p in code_names]
def codes():return {str(p):sha(p) for p in files()}

def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('core_conditioned243_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==238 and sum(x['trials'] for x in state['completed'])==1227,'Predecessor counters')
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
    review=code_review();pins=dict(read(PRIOR/'PLAN.json')['input_sha256'])
    for path in files()+[CONTEXT,TESTS,REVIEW,ROOT/'work/after242_orthogonal_coverage.json',ROOT/'work/after242_score_architecture_review.json',ROOT/'work/after242_capture_coverage.json',PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',old.PROTECTED]:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in review['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_core_conditioned243_independent as ind
    import monthly_core_conditioned243_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_core_conditioned243_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),runtime=ev.runtime(),input_sha256=pins,
        raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare core-conditioned aggregate auxiliary ranking and matched raw-auxiliary rank control; first88 zero-cost selection.')
    old.commit_state(before,state,'core_conditioned243_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');require(p['context_sha256']==sha(CONTEXT) and p['context']==read(CONTEXT) and p['runtime']==ev.runtime(),'Context/runtime changed')
    old.verify(p['input_sha256']);code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);registration_check(s,expected)
    return p

def decisions(study,arrays,meta,n):
    import monthly_core_conditioned243_kernel as kernel
    return kernel.decisions(study,arrays,meta,n)

def selection_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.SELECTION_FLAGS),'Independent selection flags')
    require(check['ordered_choices_checked']==3*n and check['checked_events']==n,'Complete event/order scope')
    require(all(check.get(k) is True for k in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Full independently checked arrays and audit')
    require(check['numerical_tolerance']==0. and check['producer_modules_imported'] is False and check['computes_returns'] is False,'Independent exact score scope')

def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_core_conditioned243_independent as ind
    study=old.local();meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original axes')
    before=old.array_snapshot(arrays);rawpanel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Original raw sources');old.verify(raw)
    parity=study.verify_panel_parity(rawpanel,px,op)
    orders,scores,pres,audit=decisions(study,arrays,meta,122)
    require(orders[ARMS[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Exact incumbent order/score parity')
    selected=ind.verify_selection(arrays,meta,orders,scores=scores,pre_scores=pres,audit=audit,n=122);selection_check(selected,122,ind)
    checks=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n)
        oo,ss,pr,au=decisions(study,aa,mm,n)
        require(au['events']==audit['events'][:n],'Exact audit event prefix')
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'All123 order prefixes')
            np.testing.assert_array_equal(ss[arm],scores[arm][:n]);np.testing.assert_array_equal(pr[arm],pres[arm][:n])
        if n in PREFIXES:
            ci=ind.verify_selection(aa,mm,oo,scores=ss,pre_scores=pr,audit=au,n=n);selection_check(ci,n,ind)
            checks.append(dict(prefix=n,selection=ci,exact=True,counts=au['counts']))
    require(old.array_snapshot(arrays)==before,'Original27 arrays mutated');gate(expected)
    dest=OUT/'prepared';dest.mkdir();np.savez_compressed(dest/'SCORES.npz',**scores);np.savez_compressed(dest/'PRE_SCORES.npz',**pres)
    exports=[('DECISION_AUDIT.json',audit),('META.json',meta),('ORDERS.json',orders),
        ('SOURCE_PARITY.json',dict(raw_source_hashes=raw,execution_prices=parity,original_array_fingerprints=old.array_fingerprints(arrays),original_arrays_unchanged=True,original_orders_exact=True)),
        ('INDEPENDENT_PREPARE.json',dict(selection=selected,physical_prefixes=checks,all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,new_model_fits=0))]
    for name,value in exports:dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,counts=audit['counts'],artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,new_model_fits=0))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',counts=audit['counts'],completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def prepared(expected):
    gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREP_FILES,'Prepared scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    import monthly_core_conditioned243_independent as ind
    meta=read(OUT/'prepared/META.json');audit=read(OUT/'prepared/DECISION_AUDIT.json')
    preparation_check(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),audit,c,ind)
    return meta,read(OUT/'prepared/ORDERS.json')

def preparation_check(e,audit,completion,ind):
    require(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged')),'Prepared preservation proof')
    require(e['new_model_fits']==0 and completion['new_model_fits']==0 and completion['new_returns_computed'] is False,'Prepared scope')
    require(completion['counts']==audit['counts']==e['selection']['counts'] and audit['counts']['events']==122 and audit['counts']['new_model_fits']==0 and len(audit['events'])==122,'Prepared counts')
    selection_check(e['selection'],122,ind)
    require([x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'All5 independent physicalprefixes')
    for item,n in zip(e['physical_prefixes'],PREFIXES):
        require(item['exact'] is True and item['counts']==item['selection']['counts'] and item['counts']['events']==n and item['counts']['new_model_fits']==0,'Exact prefix counts')
        selection_check(item['selection'],n,ind)

def select_first88(metrics,orders):
    require(set(metrics)==set(orders)==set(ARMS),'Three fixed arms')
    changed=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    versus=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[2]][i]['symbols'])]
    support=lambda x:len(x)>=3 and any(i<53 for i in x) and any(i>=53 for i in x)
    guards=dict(close_vs_incumbent=old.guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),
        open_vs_incumbent=old.guard(metrics[ARMS[1]]['next_open_zero'],metrics[ARMS[0]]['next_open_zero'],0.),
        close_vs_aux_rank_control=old.guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_aux_rank_control=old.guard(metrics[ARMS[1]]['next_open_zero'],metrics[ARMS[2]]['next_open_zero'],0.),changed_support=support(changed),changed_support_vs_control=support(versus))
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
    import monthly_core_conditioned243_independent as ind
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
    import monthly_core_conditioned243_independent as ind
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
    require(len(state['completed'])==239 and sum(x['trials'] for x in state['completed'])==1230,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review core-conditioned auxiliary ordering versus original and rank-only control; retain frozen observers; 70% goal subject to verified outcome.')
    old.commit_state(before,state,'core_conditioned243_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    else:require(bool(a.plan_sha256),'Plan hash required');globals()[a.phase](a.plan_sha256)
