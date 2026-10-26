"""Median versus mean of fixed shifted annual momentum windows; preregistered lifecycle."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_endpoint_median245'
PRIOR=ROOT/'outputs/monthly_up_participation244'
LOCAL=old.LOCAL
CENTRAL=old.CENTRAL
CONTEXT=ROOT/'work/monthly_endpoint_median245_context.json'
TESTS=ROOT/'work/monthly_endpoint_median245_TESTS.json'
REVIEW=ROOT/'work/monthly_endpoint_median245_CODE_REVIEW.json'
ARMS=('incumbent_local67','endpoint_median','endpoint_mean_control')
PREFIXES=(0,1,53,88,122)
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=('monthly_endpoint_median245_host.py','test_monthly_endpoint_median245_host.py','monthly_endpoint_median245_kernel.py','test_monthly_endpoint_median245_kernel.py','monthly_endpoint_median245_independent.py','test_monthly_endpoint_median245_independent.py')
PREP_FILES={'FEATURES.npz','FEATURE_AUDIT.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
SOURCE_REVIEW=ROOT/'work/endpoint_median245_SOURCE_REVIEW.json'
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()
def files():return [ROOT/'work'/p for p in code_names]
def codes():return {str(p):sha(p) for p in files()}

def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('endpoint_median245_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==240 and sum(x['trials'] for x in state['completed'])==1233,'Predecessor counters')
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
    for path in files()+[CONTEXT,TESTS,REVIEW,ROOT/'work/after244_endpoint_coverage.json',ROOT/'work/after244_endpoint_definition_review.json',PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',old.PROTECTED]:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in review['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    source=read(SOURCE_REVIEW);source_review_check(source)
    for path in (SOURCE_REVIEW,CALENDAR):old.bind(pins,path)
    for k,v in source['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_endpoint_median245_independent as ind
    import monthly_endpoint_median245_kernel as kernel
    for implementation in (ind,kernel):
        for path in implementation.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_endpoint_median245_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),runtime=ev.runtime(),input_sha256=pins,
        raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare exact median and mean of21 shifted annual-return windows; first88 zero-cost selection.')
    old.commit_state(before,state,'endpoint_median245_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    p=read(OUT/'PLAN.json');source_review_check(read(SOURCE_REVIEW));require(p['context_sha256']==sha(CONTEXT) and p['context']==read(CONTEXT) and p['runtime']==ev.runtime(),'Context/runtime changed')
    old.verify(p['input_sha256']);code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);registration_check(s,expected)
    return p

def decisions(study,arrays,meta,values,n):
    import monthly_endpoint_median245_kernel as kernel
    return kernel.decisions(study,arrays,meta,values,n)

def selection_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.SELECTION_FLAGS),'Independent selection flags')
    require(all(check.get(k) is True for k in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Complete independent score inputs')
    require(check['ordered_choices_checked']==3*n and check['checked_events']==n,'Complete event/order scope')
    require(check['numerical_tolerance']==0. and check['producer_modules_imported'] is False and check['computes_returns'] is False,'Independent exact score scope')

def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_endpoint_median245_independent as ind
    import monthly_endpoint_median245_kernel as kernel
    import monthly_risk_consensus219_host as source
    study=old.local();meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original axes')
    before=old.array_snapshot(arrays);rawpanel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Original raw sources');old.verify(raw)
    parity=study.verify_panel_parity(rawpanel,px,op);panel=source.price_inputs(rawpanel,meta)
    calendar=read(CALENDAR)['sessions'];values,audit=kernel.build_features(panel,meta,calendar)
    feature=ind.verify_features(panel,meta,calendar,values,audit)
    feature_check(feature,122,ind);require(feature['counts']==audit['counts'],'Feature counts parity');bounds=listing_bound_check(audit,read(SOURCE_REVIEW))
    orders,scores,pres,decisionaudit=decisions(study,arrays,meta,values,122)
    require(orders[ARMS[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Exact incumbent order/score parity')
    selected=ind.verify_selection(arrays,meta,values,orders,scores=scores,pre_scores=pres,audit=decisionaudit,n=122);selection_check(selected,122,ind);require(selected['counts']==decisionaudit['counts'],'Decision counts parity')
    checks=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n)
        vv={k:v[:n].copy() for k,v in values.items()}
        oo,ss,pr,au=decisions(study,aa,mm,vv,n)
        require(au['events']==decisionaudit['events'][:n],'All123 decisionaudit prefixes')
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'All123 order prefixes')
            np.testing.assert_array_equal(ss[arm],scores[arm][:n]);np.testing.assert_array_equal(pr[arm],pres[arm][:n])
        if n in PREFIXES:
            pp=source.feature_prefix_panel(panel,meta,n)
            small,sa=kernel.build_features(pp,mm,calendar)
            fc=ind.verify_features(pp,mm,calendar,small,sa);feature_check(fc,n,ind);require(fc['counts']==sa['counts'],'Prefix feature counts parity')
            require(sa['event_history']==audit['event_history'][:n],'Physical source audit prefix')
            for k in values:
                if n:np.testing.assert_array_equal(small[k],values[k][:n])
                else:require(small[k].size==0,'Empty price prefix')
            ci=ind.verify_selection(aa,mm,vv,oo,scores=ss,pre_scores=pr,audit=au,n=n);selection_check(ci,n,ind);require(ci['counts']==au['counts'],'Prefix decision counts parity')
            checks.append(dict(prefix=n,selection=ci,feature=fc,feature_counts=sa['counts'],decision_counts=au['counts'],exact=True))
    require(old.array_snapshot(arrays)==before,'Original27 arrays mutated');gate(expected)
    dest=OUT/'prepared';dest.mkdir();np.savez_compressed(dest/'FEATURES.npz',**values);np.savez_compressed(dest/'SCORES.npz',**scores);np.savez_compressed(dest/'PRE_SCORES.npz',**pres)
    exports=[('FEATURE_AUDIT.json',audit),('DECISION_AUDIT.json',decisionaudit),('META.json',meta),('ORDERS.json',orders),
        ('SOURCE_PARITY.json',dict(raw_source_hashes=raw,execution_prices=parity,listing_bounds=bounds,source_review_sha256=sha(SOURCE_REVIEW),original_array_fingerprints=old.array_fingerprints(arrays),original_arrays_unchanged=True,original_orders_exact=True)),
        ('INDEPENDENT_PREPARE.json',dict(feature=feature,selection=selected,physical_prefixes=checks,all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,new_model_fits=0))]
    for name,value in exports:dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,new_model_fits=0))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',available_cells=int(values['available'].sum()),completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def prepared(expected):
    gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREP_FILES,'Prepared scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared file')
    import monthly_endpoint_median245_independent as ind
    preparation_check(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),c,ind,read(OUT/'prepared/FEATURE_AUDIT.json'),read(OUT/'prepared/DECISION_AUDIT.json'))
    listing_bound_check(read(OUT/'prepared/FEATURE_AUDIT.json'),read(SOURCE_REVIEW))
    return read(OUT/'prepared/META.json'),read(OUT/'prepared/ORDERS.json')

def preparation_check(e,completion,ind,feature_audit,decision_audit):
    require(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged')),'Prepared preservation proof')
    require(e['new_model_fits']==0 and completion['new_model_fits']==0 and completion['new_returns_computed'] is False and completion['events']==122,'Prepared scope')
    selection_check(e['selection'],122,ind);feature_check(e['feature'],122,ind)
    require(e['feature']['counts']==feature_audit['counts'] and e['selection']['counts']==decision_audit['counts'],'Full audit counts binding')
    require(len(feature_audit['event_history'])==len(decision_audit['events'])==122,'Full audit event rows')
    require([x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'All5 independent physicalprefixes')
    for item,n in zip(e['physical_prefixes'],PREFIXES):
        require(item['exact'] is True,'Exact prefix')
        selection_check(item['selection'],n,ind);feature_check(item['feature'],n,ind)
        require(item['feature']['counts']==item['feature_counts'] and item['selection']['counts']==item['decision_counts'],'Prefix audit count binding')
        require(item['feature_counts']['events']==item['decision_counts']['events']==n,'Prefix audit event counts')

def select_first88(metrics,orders):
    require(set(metrics)==set(orders)==set(ARMS),'Three fixed arms')
    changed=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    versus=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[2]][i]['symbols'])]
    support=lambda x:len(x)>=3 and any(i<53 for i in x) and any(i>=53 for i in x)
    guards=dict(close_vs_incumbent=old.guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),
        open_vs_incumbent=old.guard(metrics[ARMS[1]]['next_open_zero'],metrics[ARMS[0]]['next_open_zero'],0.),
        close_vs_all_control=old.guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_all_control=old.guard(metrics[ARMS[1]]['next_open_zero'],metrics[ARMS[2]]['next_open_zero'],0.),changed_support=support(changed),changed_support_vs_control=support(versus))
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
    import monthly_endpoint_median245_independent as ind
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
    import monthly_endpoint_median245_independent as ind
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
    require(len(state['completed'])==241 and sum(x['trials'] for x in state['completed'])==1236,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review robust endpoint momentum aggregation versus same-window mean; retain frozen observers; goal subject to verified outcome.')
    old.commit_state(before,state,'endpoint_median245_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)


def source_review_check(source):
    require(source['passed'] is True and source['strict_pit'] is False and source['listing_identity_metadata_verified'] is True,'Source metadata scope')
    require(source['context_sha256']==sha(CONTEXT) and source['coverage_sha256']==sha(ROOT/'work/after244_endpoint_coverage.json'),'Source context binding')
    require(source['raw_source_hashes']==read(LOCAL/'PLAN.json')['raw_source_hashes'] and len(source['raw_source_hashes'])==64,'Exact original64 source lineage')
    for key in ('metadata_only','source_admission_metadata_only','raw_source_hashes_verified','raw_source_map_matches_original','metadata_axes_verified','calendar_clock_verified','source_window_recomputed_273'):
        require(source.get(key) is True,'Metadata evidence '+key)
    for key in ('actual_support_computed','actual_new_features_computed','actual_new_orders_computed','actual_new_returns_computed','actual_model_fits_computed','npz_decoded','prices_loaded','loader_executed','observers_modified'):
        require(source.get(key) is False,'Prospective source boundary '+key)


def feature_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.FEATURE_FLAGS),'Independent full-feature flags')
    require(check['events']==n and check['producer_modules_imported'] is False and check['computes_returns'] is False and check['numerical_tolerance']==0.,'Independent feature scope')

def listing_bound_check(audit,review):
    require(review['passed'] is True and review['strict_pit'] is False and review['listing_identity_metadata_verified'] is True,'Bound source review')
    identities={(x['query_index'],x['column_index']):x for x in review['listing_identity_cells']}
    require(len(identities)==len(review['listing_identity_cells']),'Unique listed identity cells')
    checked=0;missing=0;seen=set()
    for event in audit['event_history']:
        for cell in event['cells']:
            for bound in review['listing_bounds']:
                if cell['symbol']!=bound['symbol']:continue
                key=(event['query_index'],cell['column_index']);proof=identities.get(key)
                require(proof is not None and key not in seen,'Unexpected or duplicated bound ticker occurrence')
                require(proof['symbol']==cell['symbol'] and proof['date']==event['date'] and proof['issuer_key']==bound['issuer_key'],'Complete bound issuer identity')
                require(proof['new_price_start']==event['price_start'] and proof['new_price_end']==event['price_end'],'Full273 source bound clock')
                seen.add(key)
                if bound['available_from']<=event['date']:
                    checked+=1
                    if event['price_start'] is not None and event['price_start']<bound['not_before']:
                        require(cell['source_available'] is False,'Valid full pre-listing window would indicate identity contamination')
                        missing+=1
    n=len(audit['event_history'])
    require(seen=={k for k in identities if k[0]<n},'Missing reviewed bound issuer occurrence')
    return dict(passed=True,clock='t-314..t-42',known_bound_cells_checked=checked,prebound_unavailable_cells=missing,identity_cells_checked=len(seen),original_eligibility_modified=False,strict_pit=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    else:require(bool(a.plan_sha256),'Plan hash required');globals()[a.phase](a.plan_sha256)
