"""Daily-versus-endpoint same-sector reference aggregation; prospective zero-cost lifecycle."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_peer_path252'
PRIOR=ROOT/'outputs/monthly_basket_core251'
LOCAL=old.LOCAL
SOURCE=ROOT/'outputs/monthly_peer239_maturity_audit'
CENTRAL=old.CENTRAL
CONTEXT=ROOT/'work/monthly_peer_path252_CONTEXT.json'
TESTS=ROOT/'work/monthly_peer_path252_TESTS.json'
REVIEW=ROOT/'work/monthly_peer_path252_CODE_REVIEW.json'
ARMS=('incumbent_local67','daily_peer_path_relative','endpoint_peer_relative_control')
KEYS=('mature','supported','path_target','endpoint_target')
COUNTS=('events','stock_cells','mature_cells','supported_cells','price_observations','new_model_fits')
PREFIXES=(0,1,53,88,122)
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
code_names=('monthly_peer_path252_host.py','test_monthly_peer_path252_host.py','monthly_peer_path252_kernel.py','test_monthly_peer_path252_kernel.py','monthly_peer_path252_independent.py','test_monthly_peer_path252_independent.py')
PREP_FILES={'FEATURES.npz','FEATURE_AUDIT.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz','META.json','DAILY_SOURCE.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
SOURCE_CONTRACT=ROOT/'work/monthly_peer_path252_SOURCE_CONTRACT.json'
PRIOR_REVIEW=ROOT/'work/monthly_basket_core251_RESULT_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_peer_path252_RESULT_REVIEW.json'
RESULT_FILES={a+'_'+p[0]+'_'+kind+'.json' for a in ARMS for p in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}

def now():return datetime.now(timezone.utc).isoformat()
def files():return [ROOT/'work'/p for p in code_names]
def codes():return {str(p):sha(p) for p in files()}

def tests():
    require(not TESTS.exists(),'Tests write-once')
    mods=[old.module('peer_path252_test_'+str(i),files()[j]) for i,j in enumerate((1,3,5))]
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
    require(len(state['completed'])==247 and sum(x['trials'] for x in state['completed'])==1254,'Predecessor counters')
    require(state['completed'][-1]['directory']==PRIOR.name and state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Predecessor/incumbent changed')

def registration_check(state,expected):
    initial_state(state)
    require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration absent')

def register():
    require(not OUT.exists(),'Fresh strategy directory required')
    before=CENTRAL.read_bytes();state=json.loads(before);initial_state(state);require(not state['registered_queue'],'Registered pending experiment')
    t=read(TESTS);require(t['passed'] is True and t['synthetic_only'] is True and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Formal tests')
    c=read(PRIOR/'COMPLETION.json');require(c['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and c['protected_files']==21444 and c['plan_sha256']==sha(PRIOR/'PLAN.json'),'Predecessor closure')
    prior_binding=state['completed'][-1]['completion'];require(Path(prior_binding['path'])==PRIOR/'COMPLETION.json' and prior_binding['sha256']==sha(PRIOR/'COMPLETION.json'),'Central predecessor closure')
    previous_review=read(PRIOR_REVIEW);require(previous_review['passed'] is True and previous_review['unresolved_blockers']==[] and previous_review['plan_sha256']==sha(PRIOR/'PLAN.json') and previous_review['results_completion_sha256']==sha(PRIOR/'results/COMPLETION.json'),'Predecessor independent review')
    contract_check()
    context=read(CONTEXT);require(context['arms']==list(ARMS) and context['registered_trials']==3 and context['new_model_fits']==0,'Fixed arms')
    review=code_review();source=source_binding();pins=dict(read(PRIOR/'PLAN.json')['input_sha256'])
    for k,v in source['input_sha256'].items():old.bind(pins,Path(k),v)
    for path in files()+[CONTEXT,TESTS,REVIEW,SOURCE_CONTRACT,PRIOR_REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',old.PROTECTED]:old.bind(pins,path)
    for k,v in context['input_sha256'].items():old.bind(pins,ROOT/k,v)
    for k,v in review['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    for k,v in read(SOURCE_CONTRACT)['input_sha256'].items():old.bind(pins,Path(k) if Path(k).is_absolute() else ROOT/k,v)
    import monthly_peer_path252_independent as ind
    import monthly_peer_path252_kernel as kernel
    for module in (ind,kernel):
        if hasattr(module,'dependency_files'):
            for path in module.dependency_files():old.bind(pins,Path(path))
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected scope')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    plan=dict(schema='monthly_peer_path252_v1',created_utc=now(),context=context,context_sha256=sha(CONTEXT),runtime=ev.runtime(),input_sha256=pins,
        source_binding=source,raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=21444,
        actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_model_fits=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Validate original core, full211 stock sources and incumbent before new same-sector path/endpoint targets; select first88 ZERO.')
    old.commit_state(before,state,'peer_path252_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN hash')
    contract_check();p=read(OUT/'PLAN.json');require(p['context_sha256']==sha(CONTEXT) and p['context']==read(CONTEXT) and p['runtime']==ev.runtime(),'Context/runtime changed')
    old.verify(p['input_sha256']);require(p['source_binding']==source_binding(),'Source binding changed');code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);registration_check(s,expected)
    return p

def contract_check():
    c=read(SOURCE_CONTRACT)
    require(c['source_only'] is True and c['actual_features_computed'] is False and c['actual_returns_computed'] is False and c['full_daily_source_coverage']=='UNKNOWN','Prospective source contract scope')
    require(isinstance(c['input_sha256'],dict) and c['input_sha256'],'Source contract evidence bindings')
    return c


def validate_values(values,arrays,n):
    require(set(values)==set(KEYS),'Four fixed feature values')
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and type(n) is int and 0<=n<=shape[0],'Value axes')
    for k,v in values.items():
        require(isinstance(v,np.ndarray) and v.shape==shape and v.dtype==np.dtype(bool if k in KEYS[:2] else np.float64),'Value dtype/shape '+k)
    require(not (values['supported'][:n]&~values['mature'][:n]).any(),'Support requires maturity')
    for k in KEYS[2:]:
        require(not np.isinf(values[k][:n]).any(),'No infinite target')
        require(np.array_equal(np.isfinite(values[k][:n]),values['supported'][:n]),'Exact target availability '+k)
        require((np.abs(values[k][:n][values['supported'][:n]])<=.5).all(),'Bounded targets')


def incumbent_decision(study,arrays,meta,n):
    pre=study.modified_pre_score(study.configs()[4],arrays,n)
    calls=[]
    def boundary(changed,good,names,effects,original,pol):
        require(not calls and np.array_equal(good,arrays['eligible'][:n]),'Exactly one original-E incumbent episode')
        calls.append(changed.copy())
        return study.ep.select_boundary(changed,good,names,effects,original,pol)
    chosen,score=study.adapter.score_and_choose('incumbent66',0.,np.full(arrays['baseline'].shape,np.nan),arrays,meta,n,
        pre_score=pre.copy(),effects=arrays['effects'],episode_policy=read(LOCAL/'INCUMBENT_POLICY.json'),boundary_selector=boundary)
    require(len(calls)==1,'Single incumbent episode required')
    return study.rp.make_orders(chosen,score,meta),score.copy(),calls[0]


def decisions(study,arrays,meta,values,n,*,incumbent_bundle=None,incumbent_validator=None,expected_incumbent_orders=None):
    validate_values(values,arrays,n);before=old.array_snapshot(arrays)
    pre=study.modified_pre_score(study.configs()[4],arrays,n);policy=read(LOCAL/'INCUMBENT_POLICY.json')
    bundle=incumbent_bundle if incumbent_bundle is not None else incumbent_decision(study,arrays,meta,n)
    if expected_incumbent_orders is not None:require(bundle[0]==expected_incumbent_orders,'Exact frozen incumbent order parity before candidate scoring')
    if incumbent_validator is not None:incumbent_validator(*bundle)
    orders={ARMS[0]:bundle[0]};scores={ARMS[0]:bundle[1].copy()};pres={ARMS[0]:bundle[2].copy()}
    for arm,key in zip(ARMS[1:],KEYS[2:]):
        calls=[]
        def boundary(changed,good,names,effects,original,pol):
            require(not calls and np.array_equal(good,arrays['eligible'][:n]),'Exactly one original-E candidate episode')
            calls.append(True);pres[arm]=changed.copy()
            return study.ep.select_boundary(changed,good,names,effects,original,pol)
        chosen,score=study.adapter.score_and_choose('incumbent66',.05,values[key],arrays,meta,n,
            pre_score=pre.copy(),effects=arrays['effects'],episode_policy=policy,boundary_selector=boundary)
        require(len(calls)==1,'Single candidate episode required')
        scores[arm]=score.copy();orders[arm]=study.rp.make_orders(chosen,score,meta)
    require(old.array_snapshot(arrays)==before,'Original cache mutated')
    return orders,scores,pres


def source_prefix(source,n):
    from copy import deepcopy
    require(type(n) is int and n>=0 and n<=len(source['events']),'Source prefix range')
    r={k:deepcopy(v) for k,v in source.items() if k not in ('events','records','gaps','counts')}
    r['events']=deepcopy(source['events'][:n]);r['records']=[]
    for row in source['records']:
        require(isinstance(row,dict) and type(row.get('query_index')) is int and row['query_index']>=0,'Canonical source record query')
        if row['query_index']>=n:continue
        r['records'].append(deepcopy(row))
    rr=r['records'];r['gaps']=[x for x in rr if x['gap_reasons']]
    r['counts']=dict(events=n,stock_cells=len(rr),mature_cells=sum(x['history_status']=='observed_full_window' for x in rr),warmup_cells=sum(x['history_status']=='official_pre_inception' for x in rr),partial_cells=sum(x['initial_partial'] for x in rr),supported_cells=sum(x['structural_peer_support'] for x in rr),deferred_sector_cells=sum('missing_sector' in x['deferred_gap_reasons'] for x in rr),deferred_endpoints=sum(sum(y.startswith('endpoint:') for y in x['deferred_gap_reasons']) for x in rr),gap_cells=len(r['gaps']))
    return r


def independent_scope(check):
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent non-return scope')


def source_gate_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.ORIGINAL_SOURCE_FLAGS),'Original source flags')
    require(check['checked_events']==n and check['episode_calls']==0 and check['new_target_ranks_computed'] is False and check['candidate_selection_computed'] is False,'Original source before scoring')
    require(check['C_source_atol']==1e-12 and check['C_source_rtol']==0. and check['continuous_atol']==2e-10 and check['continuous_rtol']==2e-10 and check['rank_tolerance']==0.,'Frozen original source tolerances')
    independent_scope(check)


def incumbent_gate_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.INCUMBENT_GATE_FLAGS),'Original incumbent flags')
    require(check['checked_events']==n and check['ordered_choices_checked']==n and check['numerical_tolerance']==0.,'Exact original incumbent')
    require(check['candidate_selection_computed'] is False,'Incumbent-only gate')
    independent_scope(check)


def daily_source_check(check,source,n,ind):
    require(all(check.get(k) is True for k in ind.SOURCE_FLAGS),'Daily source flags')
    require(check['counts']==source['counts'] and check['counts']['events']==n and check.get('checked_events')==n and check['numerical_tolerance']==0.,'Exact daily source coverage')
    require(check.get('full_source_report_supplied') is True and check.get('episode_calls')==0 and check.get('new_target_ranks_computed') is False and check.get('candidate_selection_computed') is False,'Daily source-only pre-ranking gate')
    independent_scope(check)


def feature_check(check,values,audit,ind):
    require(all(check.get(k) is True for k in ind.FEATURE_FLAGS),'Independent feature flags')
    require(check['numerical_tolerance']==0. and check.get('full_features_supplied') is True and check.get('rank_tolerance')==0. and check.get('exact_discrete') is True,'Exact complete feature arithmetic')
    independent_scope(check)
    c=check['counts'];require(set(c)==set(COUNTS) and c==audit['counts'],'Feature count schema')
    require(c['events']==len(values['mature']) and check.get('checked_events')==c['events'] and check.get('events')==c['events'] and c['mature_cells']==int(values['mature'].sum()) and c['supported_cells']==int(values['supported'].sum()) and c['new_model_fits']==0,'Feature coverage')


def selection_check(check,n,ind):
    require(all(check.get(k) is True for k in ind.SELECTION_FLAGS) and check['ordered_choices_checked']==3*n and check['full_scores_supplied'] is True and check['full_pre_scores_supplied'] is True,'Independent full selection')
    require(check['numerical_tolerance']==0. and check['checked_events']==n,'Exact score scope')
    independent_scope(check)


def prepare_one(study,arrays,meta,panel,calendar,source239,n,kernel,ind,*,expected_incumbent_orders=None):
    # No candidate feature/rank/order exists until each independent pre-gate passes.
    original=ind.verify_original_source(panel,meta,calendar,arrays,n=n);source_gate_check(original,n,ind)
    source=kernel.build_source(panel,meta,calendar,arrays,source239)
    sourced=ind.verify_source(panel,meta,calendar,arrays,source239,source);daily_source_check(sourced,source,n,ind)
    incumbent=incumbent_decision(study,arrays,meta,n)
    require(isinstance(expected_incumbent_orders,list) and incumbent[0]==expected_incumbent_orders,'Exact frozen incumbent orders before any new features')
    original_orders=ind.verify_incumbent(arrays,meta,*incumbent,n=n);incumbent_gate_check(original_orders,n,ind)
    values,audit=kernel.build_features(panel,meta,calendar,arrays,source);validate_values(values,arrays,n)
    feature=ind.verify_features(panel,meta,calendar,arrays,source,values,audit);feature_check(feature,values,audit,ind)
    orders,scores,pres=decisions(study,arrays,meta,values,n,incumbent_bundle=incumbent,expected_incumbent_orders=expected_incumbent_orders)
    selected=ind.verify_selection(arrays,meta,values,orders,scores=scores,pre_scores=pres,n=n);selection_check(selected,n,ind)
    evidence=dict(original_source=original,daily_source=sourced,incumbent=original_orders,feature=feature,selection=selected)
    return values,audit,source,orders,scores,pres,evidence


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation write-once')
    import monthly_peer_path252_kernel as kernel
    import monthly_peer_path252_independent as ind
    study=old.local();meta,arrays,px,op=study.load();require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original axes')
    before=old.array_snapshot(arrays);rawpanel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Original raw sources');old.verify(raw)
    parity=study.verify_panel_parity(rawpanel,px,op);panel={'adj':rawpanel['adj']};calendar=read(old.CALENDAR)['sessions'];source239=read(SOURCE/'REPORT.json')
    print(json.dumps(dict(stage='252_full_source_and_features',events=122)),flush=True)
    expected_incumbent=read(LOCAL/'local_linear_ORDERS.json')
    values,audit,source,orders,scores,pres,evidence=prepare_one(study,arrays,meta,panel,calendar,source239,122,kernel,ind,expected_incumbent_orders=expected_incumbent)
    require(orders[ARMS[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Exact original order parity')
    checks=[]
    for n in PREFIXES:
        print(json.dumps(dict(stage='252_physical_prefix',prefix=n)),flush=True)
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n);cc=old.feature_prefix_calendar(calendar,meta,n);rr=source_prefix(source239,n)
        pp={'adj':panel['adj'].loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else panel['adj'].iloc[:0].copy()}
        vv,au,so,oo,ss,pr,ci=prepare_one(study,aa,mm,pp,cc,rr,n,kernel,ind,expected_incumbent_orders=expected_incumbent[:n])
        for key in KEYS:np.testing.assert_array_equal(vv[key],values[key][:n])
        require(au['events']==audit['events'][:n],'Exact feature audit prefixes')
        require(so['events']==source['events'][:n] and so['records']==[r for r in source['records'] if r['query_index']<n],'Exact daily source prefixes')
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'Exact physical order prefix');np.testing.assert_array_equal(ss[arm],scores[arm][:n]);np.testing.assert_array_equal(pr[arm],pres[arm][:n])
        checks.append(dict(prefix=n,**ci,counts=so['counts'],source_rows=len(pp['adj']),source_last_session=str(pp['adj'].index[-1].date()) if n else None,calendar_rows=len(cc),exact=True))
    cached=[]
    for n in range(123):
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n);vv={k:v[:n].copy() for k,v in values.items()}
        receipt=[]
        def check_inc(*bundle):
            c=ind.verify_incumbent(aa,mm,*bundle,n=n);incumbent_gate_check(c,n,ind);receipt.append(c)
        oo,ss,pr=decisions(study,aa,mm,vv,n,incumbent_validator=check_inc,expected_incumbent_orders=expected_incumbent[:n])
        require(len(receipt)==1,'Cached prefix incumbent gate')
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'All123 cached order prefix');np.testing.assert_array_equal(ss[arm],scores[arm][:n]);np.testing.assert_array_equal(pr[arm],pres[arm][:n])
        cached.append(dict(prefix=n,incumbent=receipt[0],exact=True))
    require(old.array_snapshot(arrays)==before,'Original27 arrays unchanged');gate(expected)
    dest=OUT/'prepared';dest.mkdir();np.savez_compressed(dest/'FEATURES.npz',**values);np.savez_compressed(dest/'SCORES.npz',**scores);np.savez_compressed(dest/'PRE_SCORES.npz',**pres)
    proof=dict(**evidence,physical_prefixes=checks,cached_prefixes=cached,all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,source_inventory_preserved=True,new_model_fits=0)
    for name,value in [('FEATURE_AUDIT.json',audit),('DAILY_SOURCE.json',source),('META.json',meta),('ORDERS.json',orders),('SOURCE_PARITY.json',dict(raw_source_hashes=raw,execution_prices=parity,original_array_fingerprints=old.array_fingerprints(arrays),original_arrays_unchanged=True,original_orders_exact=True,source_report_sha256=sha(SOURCE/'REPORT.json'))),('INDEPENDENT_PREPARE.json',proof)]:dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,counts=audit['counts'],artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,new_model_fits=0))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',counts=audit['counts'],completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def preparation_check(e,values,audit,completion,source,ind):
    require(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged','source_inventory_preserved')),'Preservation evidence')
    require(e['new_model_fits']==completion['new_model_fits']==0 and completion['new_returns_computed'] is False,'Prepare no-return/no-model scope')
    require(completion['counts']==audit['counts']==source['counts']==e['feature']['counts'],'Prepared coverage')
    n=len(values['mature']);source_gate_check(e['original_source'],n,ind);daily_source_check(e['daily_source'],source,n,ind);incumbent_gate_check(e['incumbent'],n,ind);feature_check(e['feature'],values,audit,ind);selection_check(e['selection'],n,ind)
    require([x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'Five physical prefixes required')
    for item,prefix in zip(e['physical_prefixes'],PREFIXES):
        require(item['exact'] is True,'Exact physical prefix')
        source_gate_check(item['original_source'],prefix,ind);incumbent_gate_check(item['incumbent'],prefix,ind);selection_check(item['selection'],prefix,ind)
        rows=[r for r in source['records'] if r['query_index']<prefix]
        counts=dict(events=prefix,stock_cells=len(rows),mature_cells=int(values['mature'][:prefix].sum()),supported_cells=int(values['supported'][:prefix].sum()),price_observations=211*sum(not r['initial_partial'] for r in rows),new_model_fits=0)
        require(item['counts']==counts,'Prefix source count')
        daily_source_check(item['daily_source'],{'counts':counts},prefix,ind)
        feature_check(item['feature'],{k:v[:prefix] for k,v in values.items()},{'counts':counts},ind)
    require([x['prefix'] for x in e['cached_prefixes']]==list(range(123)),'All123 cached prefix receipts')
    for item in e['cached_prefixes']:
        require(item['exact'] is True,'Exact cached prefix');incumbent_gate_check(item['incumbent'],item['prefix'],ind)


def prepared(expected):
    gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and set(c['artifacts_sha256'])==PREP_FILES,'Complete preparation scope')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed prepared artifact')
    import monthly_peer_path252_independent as ind
    with np.load(OUT/'prepared/FEATURES.npz',allow_pickle=False) as data:values={k:data[k].copy() for k in data.files}
    meta=read(OUT/'prepared/META.json');validate_values(values,{'baseline':np.empty((len(meta['dates']),49))},122)
    preparation_check(read(OUT/'prepared/INDEPENDENT_PREPARE.json'),values,read(OUT/'prepared/FEATURE_AUDIT.json'),c,read(OUT/'prepared/DAILY_SOURCE.json'),ind)
    return meta,read(OUT/'prepared/ORDERS.json')


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
    changed=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    versus=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[2]][i]['symbols'])]
    support=lambda x:len(x)>=3 and any(i<53 for i in x) and any(i>=53 for i in x)
    guards=dict(close_vs_incumbent='same_close_zero' not in undefined[ARMS[1]] and old.guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),
        open_vs_incumbent='next_open_zero' not in undefined[ARMS[1]] and old.guard(metrics[ARMS[1]]['next_open_zero'],metrics[ARMS[0]]['next_open_zero'],0.),
        close_vs_endpoint_control=all('same_close_zero' not in undefined[a] for a in ARMS[1:]) and old.guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_endpoint_control=all('next_open_zero' not in undefined[a] for a in ARMS[1:]) and old.guard(metrics[ARMS[1]]['next_open_zero'],metrics[ARMS[2]]['next_open_zero'],0.),changed_support=support(changed),changed_support_vs_control=support(versus))
    return dict(primary=ARMS[1] if all(guards.values()) else ARMS[0],admitted=all(guards.values()),guards=guards,changed_months=changed,changed_months_vs_control=versus,undefined_sharpe=undefined,first88_only=True,later34_used=False)

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
    import monthly_peer_path252_independent as ind
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
    import monthly_peer_path252_independent as ind
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
    reviewed=read(RESULT_REVIEW);require(reviewed['passed'] is True and reviewed['unresolved_blockers']==[] and reviewed['plan_sha256']==expected and reviewed['results_completion_sha256']==sha(OUT/'results/COMPLETION.json'),'Independent saved result review')
    protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive coverage')
    before=CENTRAL.read_bytes();state=json.loads(before);registration_check(state,expected)
    pick=result['selection']['primary'];arm=result['arms'][pick];m=arm['same_close_zero']['metrics'];best=state['best_training_selected_candidate']
    if result['selection']['admitted'] and m['cagr']>best['cagr']:
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=read(CONTEXT)['limitations'])
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),new_trials=3,new_model_fits=0,protected_files=21444,result_review_sha256=sha(RESULT_REVIEW),best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(result['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(state['completed'])==248 and sum(x['trials'] for x in state['completed'])==1257,'Counter advance')
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=result['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review peer reference aggregation and endpoint control; no automatic observer promotion; target remains subject to verified zero-cost outcome.')
    old.commit_state(before,state,'peer_path252_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    else:require(bool(a.plan_sha256),'Plan hash required');globals()[a.phase](a.plan_sha256)
