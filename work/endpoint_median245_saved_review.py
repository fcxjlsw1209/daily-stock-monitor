"""Saved245 artifact/hash audit only; no price loading, replay or new feature math."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, importlib.util, json, math
ROOT=Path('/Users/shiweili/code/monthly-stock-research')
BASE=ROOT/'outputs/monthly_endpoint_median245'
OUTPUT=ROOT/'work/monthly_endpoint_median245_RESULT_REVIEW.json'
ARMS=('incumbent_local67','endpoint_median','endpoint_mean_control')
PATHS=(('same_close_zero','close',0.),('next_open_zero','open',0.),('next_open_cost','open',.0005))
SLICES=('full122','first88','early53','recent35','later34')
PINS={}
def need(v,msg):
    if not v:raise ValueError(msg)
def digest(p):
    d=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):d.update(x)
    return d.hexdigest()
def bind(p,expected=None):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    need(p.resolve().is_relative_to(ROOT),'Local path only');h=digest(p)
    need(expected is None or h==expected,'SHA mismatch '+str(p));PINS[str(p.relative_to(ROOT))]=h;return h
def read(p,expected=None):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    bind(p,expected);return json.loads(p.read_text())
def literal(path,name):
    bind(path)
    for n in ast.parse((ROOT/path).read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
    raise ValueError('Missing static schema '+name)
def flags(r,keys):need(all(r.get(k) is True for k in keys),'Independent flags')
def manifest(folder,completion,names):
    need(set(completion['artifacts_sha256'])==set(names),'Exact artifact set')
    for name,h in completion['artifacts_sha256'].items():bind(folder/name,h)
def counts(r,c,n):
    need(r['counts']==c and c['events']==n,'Counts/event binding')
    need(r['producer_modules_imported'] is False and r['computes_returns'] is False and r['numerical_tolerance']==0.,'Independent scope')
def guard(a,b,gain):
    return a['training']['cagr']>=b['training']['cagr']+gain-1e-12 and a['training']['sharpe']>=b['training']['sharpe']-.05 and a['training']['max_drawdown']>=b['training']['max_drawdown']-.03 and all(a[k]['cagr']>=b[k]['cagr']-.025 for k in ('early','recent'))
def changes(orders,a,b,n):return [i for i in range(n) if set(orders[a][i]['symbols'])!=set(orders[b][i]['symbols'])]
def select(training,orders):
    need(set(training)==set(ARMS),'Training arms')
    need(all(set(x)=={'same_close_zero','next_open_zero'} for x in training.values()),'Zero-cost-only training')
    a,b,c=(training[x] for x in ARMS);x=changes(orders,ARMS[1],ARMS[0],88);y=changes(orders,ARMS[1],ARMS[2],88)
    support=lambda z:len(z)>=3 and any(i<53 for i in z) and any(i>=53 for i in z)
    g=dict(close_vs_incumbent=guard(a=b['same_close_zero'],b=a['same_close_zero'],gain=.005),open_vs_incumbent=guard(b['next_open_zero'],a['next_open_zero'],0.),close_vs_all_control=guard(b['same_close_zero'],c['same_close_zero'],0.),open_vs_all_control=guard(b['next_open_zero'],c['next_open_zero'],0.),changed_support=support(x),changed_support_vs_control=support(y))
    return dict(primary=ARMS[1] if all(g.values()) else ARMS[0],admitted=all(g.values()),guards=g,changed_months=x,changed_months_vs_control=y,first88_only=True,later34_used=False)

def main(args):
    need(not OUTPUT.exists(),'Write-once result review')
    plan=read(BASE/'PLAN.json',args.plan_sha256)
    context=read('work/monthly_endpoint_median245_context.json',plan['context_sha256']);need(context==plan['context'],'Context exact')
    test=read('work/monthly_endpoint_median245_TESTS.json','e844c9f139e04eb82b8bb0e8344f04fe3713ea6957424d1f691b5e4e6cb36b20')
    code=read('work/monthly_endpoint_median245_CODE_REVIEW.json','95164066e6c40085370ed27cd83b3f7ed46264b6bb8f44f1bd50690cd7baab90')
    need(test['passed'] is True and test['synthetic_only'] is True and test['tests']==78 and code['passed'] is True and code['unresolved_blockers']==[],'Formal preregistration evidence')
    need(test['code_sha256']==code['code_sha256'] and len(test['code_sha256'])==6,'Six final code pins')
    for p,h in test['code_sha256'].items():bind(p,h);need(plan['input_sha256'][p]==h,'PLAN code binding')
    for p in ('work/monthly_endpoint_median245_TESTS.json','work/monthly_endpoint_median245_CODE_REVIEW.json','work/monthly_endpoint_median245_context.json'):
        need(plan['input_sha256'][str(ROOT/p)]==PINS[p],'PLAN metadata binding')
    source=read('work/endpoint_median245_SOURCE_REVIEW.json','972c0975c91e6e95141941c54aa13ad82882be048975589876bee09b85880ab9')
    need(source['raw_source_hashes']==plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64,'Raw64 source map')
    need(source['context_sha256']==plan['context_sha256'] and source['strict_pit'] is False,'Source context/PIT')
    for p in ('work/endpoint_median245_SOURCE_REVIEW.json','work/endpoint_median245_source_metadata.py','work/after244_endpoint_coverage.json','work/after244_endpoint_definition_review.json'):
        bind(p,plan['input_sha256'][str(ROOT/p)])
    pc=read(BASE/'prepared/COMPLETION.json','d23e25acb9b20c650e17228b4ae9a4ccc1b688478c1024dd7e6490a5d48e8786')
    need(pc['status']=='COMPLETE' and pc['plan_sha256']==args.plan_sha256 and pc['new_returns_computed'] is False and pc['new_model_fits']==0,'Prepared closure')
    manifest(BASE/'prepared',pc,{'FEATURES.npz','FEATURE_AUDIT.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'})
    meta=read(BASE/'prepared/META.json');orders=read(BASE/'prepared/ORDERS.json');fa=read(BASE/'prepared/FEATURE_AUDIT.json');da=read(BASE/'prepared/DECISION_AUDIT.json');ip=read(BASE/'prepared/INDEPENDENT_PREPARE.json');sp=read(BASE/'prepared/SOURCE_PARITY.json')
    need(len(meta['dates'])==len(meta['symbols'])==122 and all(len(s)==49 for s in meta['symbols']),'Original122x49 axis')
    need(orders[ARMS[0]]==read('outputs/monthly_local_linear70/local_linear_ORDERS.json'),'Exact original incumbent orders')
    need(sp['raw_source_hashes']==plan['raw_source_hashes'] and sp['source_review_sha256']==PINS['work/endpoint_median245_SOURCE_REVIEW.json'],'Source parity binding')
    need(sp['original_arrays_unchanged'] is True and sp['original_orders_exact'] is True and len(sp['original_array_fingerprints'])==27,'Original27/order preservation')
    need(all(sp['execution_prices'][k] is True for k in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact')),'Execution source parity')
    fields=literal('work/monthly_endpoint_median245_independent.py','FEATURE_FLAGS');selections=literal('work/monthly_endpoint_median245_independent.py','SELECTION_FLAGS')
    need(len(fields)==22 and len(selections)==22,'Expected independent schemas')
    flags(ip['feature'],fields);flags(ip['selection'],selections);counts(ip['feature'],fa['counts'],122);counts(ip['selection'],da['counts'],122)
    need(ip['selection']['ordered_choices_checked']==366 and all(ip['selection'][k] is True for k in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Full independent decisions')
    need(all(ip[k] is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged')) and ip['new_model_fits']==0,'Preservation flags')
    need([p['prefix'] for p in ip['physical_prefixes']]==[0,1,53,88,122],'Five physical prefixes')
    for p in ip['physical_prefixes']:
        n=p['prefix'];need(p['exact'] is True,'Exact prefix');flags(p['feature'],fields);flags(p['selection'],selections);counts(p['feature'],p['feature_counts'],n);counts(p['selection'],p['decision_counts'],n)
        need(p['selection']['ordered_choices_checked']==3*n and all(p['selection'][k] is True for k in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Full prefix scores')
    need(fa['counts']['checked_cells']==5978 and fa['counts']['available_cells']==fa['counts']['source_available_cells']==5973 and fa['counts']['unavailable_cells']==5,'Declared source counts')
    need(fa['counts']['security_price_observations']==273*5973 and fa['counts']['endpoint_pairs']==21*5973 and fa['counts']['median_estimates']==fa['counts']['mean_estimates']==5973 and fa['counts']['ranked_cells']==2*5973,'273/21/rank metadata counts')
    need(fa['counts']['new_model_fits']==fa['counts']['predictive_model_fits']==0,'Zero feature fits')
    need(plan['runtime']==test['runtime'],'Pinned runtime agreement')
    gaps=[dict(query_index=e['query_index'],column_index=c['column_index'],date=e['date'],symbol=c['symbol'],reason=c['source_reason']) for e in fa['event_history'] for c in e['cells'] if not c['source_available']]
    expected={(x['query_index'],x['column_index'],x['date'],x['symbol']) for x in source['listing_identity_cells']}
    need({(x['query_index'],x['column_index'],x['date'],x['symbol']) for x in gaps}==expected and len(gaps)==5,'Only five reviewed source gaps')
    need(sp['listing_bounds']==dict(passed=True,clock='t-314..t-42',known_bound_cells_checked=5,prebound_unavailable_cells=5,identity_cells_checked=5,original_eligibility_modified=False,strict_pit=False),'Listing check receipt')
    calendar=read(source['calendar']['path'],source['calendar']['sha256'])['sessions']
    need(len(source['calendar']['windows'])==len(source['calendar']['pair_clocks'])==122,'All source metadata windows')
    for e,w,pairs in zip(fa['event_history'],source['calendar']['windows'],source['calendar']['pair_clocks']):
        i=e['query_index'];t=e['session_position']
        need(e['date']==meta['dates'][i]==w['date']==pairs['date'] and e['symbols']==meta['symbols'][i],'Feature identity axis')
        need(e['source_dates']==calendar[t-314:t-41] and len(e['source_dates'])==273,'Full273 official dates')
        need(e['price_start']==w['price_start']==e['source_dates'][0] and e['price_end']==w['price_end']==e['source_dates'][-1],'273 source metadata clock')
        need(e['pair_dates']==[dict(start=p['start_date'],end=p['end_date']) for p in pairs['pairs']] and len(e['pair_dates'])==21,'All21 endpoint date proofs')
        need(e['source_available_indices']==e['available_indices'],'Common own-source availability')
        for c in e['cells']:
            need(c['source_available']==c['available'],'Actual shared support')
            for key in ('raw_median','raw_mean'):
                value=c[key]
                if c['available']:need(type(value) is list and len(value)==2 and all(type(v) is str for v in value) and int(value[1])>0,'Canonical rational audit structure')
                else:need(value is None and c['candidate'] is None and c['control'] is None,'Missing common targets')
    for arm in ARMS[1:]:need(orders[arm][0]==orders[ARMS[0]][0],'Partial unchanged')
    need(da['counts']['active_events']==121 and da['counts']['original_episode_calls']==3 and da['counts']['new_model_fits']==da['counts']['predictive_model_fits']==0,'Decision scope')
    frozen=read(BASE/'selected/FROZEN.json');training=read(BASE/'selected/TRAIN_RESULTS.json')
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==PINS[str((BASE/'prepared/COMPLETION.json').relative_to(ROOT))] and frozen['orders_sha256']==PINS[str((BASE/'prepared/ORDERS.json').relative_to(ROOT))] and frozen['training_sha256']==PINS[str((BASE/'selected/TRAIN_RESULTS.json').relative_to(ROOT))],'Frozen selection dependencies')
    need(frozen['selection']==select(training,orders),'Exact saved first88-only guards')
    rc=read(BASE/'results/COMPLETION.json',args.results_completion_sha256)
    need(rc['status']=='COMPLETE' and rc['plan_sha256']==args.plan_sha256 and rc['frozen_sha256']==PINS[str((BASE/'selected/FROZEN.json').relative_to(ROOT))] and rc['paths']==9 and rc['slices']==45,'Results closure')
    names={a+'_'+p+'_'+kind+'.json' for a in ARMS for p,_,_ in PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}
    manifest(BASE/'results',rc,names)
    results=read(BASE/'results/RESULTS.json');outcome=read(BASE/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['reused_sample'] is True and results['strict_pit'] is False and results['zero_cost_selection'] is True and results['next_open_cost_diagnostic_only'] is True and results['observer_promotion'] is False,'Result policy labels')
    need(outcome['passed'] is True and outcome['checked_paths']==9 and outcome['checked_slices']==45 and outcome['independent_accounts'] is True and outcome['independent_of_production_model_and_engine'] is True and outcome['terminal_liquidation'] is False,'Independent9/45 receipt')
    accounting=ROOT/'work/ptr169_results_independent.py';bind(accounting,'6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59')
    need(outcome['accounting_source_sha256']==digest(accounting),'Pinned independent metric checker')
    spec=importlib.util.spec_from_file_location('saved245_metrics_only',accounting);acc=importlib.util.module_from_spec(spec);spec.loader.exec_module(acc)
    metric_fields=0;train_fields=0;zero_streams=0;summary={};training_residuals=[]
    prior=read('outputs/monthly_up_participation244/results/RESULTS.json')
    for arm in ARMS:
        need(set(results['arms'][arm])==set(outcome['paths'][arm])=={p[0] for p in PATHS},'Three paths perarm');summary[arm]={}
        for path,execution,cost in PATHS:
            rec=results['arms'][arm][path];check=outcome['paths'][arm][path]
            need(check['passed'] is True and check['monthly_executions']==122 and set(check['slices'])==set(rec['slices'])==set(SLICES),'Five complete independent slices')
            need(rec['metrics']['execution']==execution and rec['metrics']['cost_per_side']==cost and rec['metrics']['start']=='2016-08-17' and rec['metrics']['end']=='2026-09-24','Timing/cost/fullperiod labels')
            for split in SLICES:acc.check_metrics(check['slices'][split],rec['slices'][split],arm+'/'+path+'/'+split);metric_fields+=len(check['slices'][split])
            need(rec['slices']['full122']==rec['metrics'],'Fullmetrics identity')
            if cost==0.:
                zero_streams+=1;need(rec['metrics']['fees_paid']==0.,'Zero cost metrics')
                holds=read(BASE/'results'/(arm+'_'+path+'_HOLDINGS.json'));trades=read(BASE/'results'/(arm+'_'+path+'_TRADES.json'))
                need(all(x['fee']==0. for x in holds) and all(x['cost']==0. for x in trades),'All zero-cost execution fees exact0')
            for short,full in (('training','first88'),('early','early53'),('recent','recent35')):
                if path=='next_open_cost':continue
                acc.check_metrics(check['slices'][full],training[arm][path][short],arm+'/'+path+'/'+short);train_fields+=len(check['slices'][full])
                for k,v in check['slices'][full].items():
                    if isinstance(v,(float,int)) and not isinstance(v,bool) and training[arm][path][short][k]!=v:training_residuals.append(dict(arm=arm,path=path,split=short,field=k,delta=training[arm][path][short][k]-v))
            if arm==ARMS[0]:need(rec['metrics']==prior['arms'][arm][path]['metrics'] and rec['slices']==prior['arms'][arm][path]['slices'],'Exact incumbent metrics/slices against244')
            summary[arm][path]={s:{k:rec['slices'][s][k] for k in ('start','end','cagr','sharpe','max_drawdown')} for s in SLICES}
    need(metric_fields==405 and train_fields==162 and zero_streams==6,'Exact saved metric coverage')
    closure=read(BASE/'COMPLETION.json',args.completion_sha256)
    need(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==args.plan_sha256 and closure['result_completion_sha256']==args.results_completion_sha256 and closure['protected_files']==21444 and closure['observers_modified'] is False and closure['new_trials']==3 and closure['new_model_fits']==0,'Formal close')
    centralpath=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json';central=json.loads(centralpath.read_text())
    need(len(central['completed'])==241 and sum(x['trials'] for x in central['completed'])==1236 and not central['registered_queue'] and central['current']==BASE.name,'Postclose counters/queue')
    need(central['best_training_selected_candidate']['directory']=='monthly_local_linear70' and closure['target_reached'] is False,'Retained incumbent target notmet')
    changes_summary={a:dict(full122=changes(orders,a,ARMS[0],122),first88=changes(orders,a,ARMS[0],88)) for a in ARMS[1:]}
    report=dict(schema='monthly_endpoint_median245_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],input_sha256=dict(PINS),implementation=dict(path=str(Path(__file__)),sha256=digest(Path(__file__))),
        reviewer_role='Author of source metadata/static code review; independent consumer of saved artifacts and existing separate scalar accounting evidence. Did not replay245, fit models or decode actualNPZ.',
        plan_sha256=args.plan_sha256,prepared_completion_sha256=digest(BASE/'prepared/COMPLETION.json'),results_completion_sha256=args.results_completion_sha256,completion_sha256=args.completion_sha256,
        formal_tests=78,prepared_artifacts=9,result_artifacts=38,source_counts=fa['counts'],decision_counts=da['counts'],source_gaps=gaps,source_listing_receipt=sp['listing_bounds'],feature_flags_checked=22,selection_flags_checked=22,all123_order_prefixes=True,physical_prefixes=[0,1,53,88,122],
        selection=frozen['selection'],changes_vs_incumbent=changes_summary,metrics=summary,
        verification=dict(independent_paths=9,independent_slices=45,saved_metric_fields_compared=metric_fields,training_groups=18,training_metric_fields_compared=train_fields,total_saved_metric_fields_compared=metric_fields+train_fields,training_machine_residuals=training_residuals,incumbent_all_metrics_and_slices_exact244=True,zero_cost_streams_exact_zero_fees=6,original27_array_fingerprint_count=27,original_orders_exact=True,original_E_unchanged=True,source64_preserved=True,protected21444_verified_by_root_closer_not_rescanned_by_reviewer=True),
        cost_answer=dict(reported_67_67_source='incumbent_local67/same_close_zero/full122/cagr',same_close_zero_cagr=results['arms'][ARMS[0]]['same_close_zero']['metrics']['cagr'],same_close_zero_fees=0.,next_open_zero_cagr=results['arms'][ARMS[0]]['next_open_zero']['metrics']['cagr'],next_open_5bp_per_side_cagr=results['arms'][ARMS[0]]['next_open_cost']['metrics']['cagr'],zero_cost_selection=True,cost_path_diagnostic_only=True),
        post_close_central=dict(sha256=digest(centralpath),snapshot_only_not_registration_pin=True,completed_entries=241,trials=1236,queue_empty=True,current=central['current'],best=central['best_training_selected_candidate'],target_reached=False),
        limitations=context['limitations'],strict_pit=False,reused_sample=True,actual_source_prices_loaded=False,npz_decoded=False,features_recomputed=False,models_refitted=False,orders_recomputed=False,returns_replayed=False,saved_metrics_only=True,observers_modified=False,registered_files_modified=False)
    with OUTPUT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(OUTPUT),sha256=digest(OUTPUT),passed=True,metric_fields=metric_fields,training_fields=train_fields,training_residuals=len(training_residuals),input_pins=len(PINS),selection=frozen['selection'],full_zero_cagr={a:summary[a]['same_close_zero']['full122']['cagr'] for a in ARMS})))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-sha256',required=True);p.add_argument('--results-completion-sha256',required=True);p.add_argument('--completion-sha256',required=True);main(p.parse_args())
