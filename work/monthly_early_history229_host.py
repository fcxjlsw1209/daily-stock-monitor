"""Source-only historical as-of forecasts preceding the 122-month trading axis."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import unittest
import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev
import monthly_joint_error226_host as previous
import monthly_ineligible_source227_host as prior_source

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_early_history229'
PRIOR=ROOT/'outputs/monthly_expanded_error228'
SOURCE227=ROOT/'outputs/monthly_ineligible_source227'
CENTRAL=previous.CENTRAL
LOCAL=previous.LOCAL
REBUILD=previous.REBUILD
CALENDAR=previous.CALENDAR
PROTECTED=previous.PROTECTED
CONTEXT=ROOT/'work/monthly_early_history229_context.json'
REVIEW=ROOT/'work/early_history229_SOURCE_REVIEW.json'
CODE_REVIEW=ROOT/'work/early_history229_CODE_REVIEW.json'
TESTS=ROOT/'work/monthly_early_history229_TESTS.json'
CHECKPOINT=ROOT/'work/after228_research_checkpoint.json'
PRIOR_SHA='8d65e817a0c34e79950b957f3263d651b792fbaf6c128681a98f1be1ed59df7f'
PREFIXES=(0,12,24,48,67)
TARGETS=('simple_excess','log_excess','vol_scaled','loss_utility')
sha,read,require,dump=ev.sha,ev.read,ev.require,ev.dump


def now():return datetime.now(timezone.utc).isoformat()


def contract():
    return dict(schema='early_history229_source_v1',source_only=True,strategy_trials=0,
        new_portfolio_returns=False,new_ranks=False,new_labels=False,
        hypothesis='The current122-query source begins2016; add earlier as-of errors using existing exact historical query identities/features, without altering current choices or tuning the18-month threshold.',
        query_axis='First67 original TRAIN natural monthends2011-01-31 through2016-07-29,49 exact identities per month; separate from original122 current trading queries.',
        features='Unchanged cached train_plain10, original historical49 rank reference, neutral-fill semantics. No new feature construction, pool expansion, or reranking.',
        query_good='finite cached plain10 AND(original train_eligible OR295past official-session finitepositive asset closes). Never use future train_good/common_good/target validity to admit a query.',
        forecast_model='Frozen local70 annual priorDecember31,5calendar years,minimum18 mature months,ownminimum18; Ridgealpha10,12monthdecay, original targets/common_good only. One union-query replay; coldstart retained as unavailable.',
        historical_axis='New forecasts remain separate. Any strategy using them requires another prospective PLAN and a separate-history-axis kernel.',
        labels='Copy source227 labels/common_good/added/vol63 at exact original early training rows, including their original clipped-centering reference; labels are only future risk-error outcomes and never additional source training.',
        initial_training='Original training archive starts2011;2011/2012 cannot satisfy18 mature months. Do not claim a complete five-year archive in earliest years.',
        preservation='Original27 currentstrategy arrays, original training inputs, source227 forecasts/labels and all protected archives remain byte-identical.',
        prefixes=list(PREFIXES),physical_prefix_policy='Before each prefix forecast call, discard training rows whose label ends exceed priorDec31 of latest query year; also truncate prices at latest query date.',
        fit_accounting='Count new fourtarget sourcefits, pooledlogrefits and localcorrection solves; report independent and prefix repetitions separately. This is source preparation, not zero-fit.',
        strict_PIT=False,limitations=['Pre2016 pool_origins are inherited_pre2016_historical_proxy; no strict historical-pool certification.',
        'Retrospective adjustedprices and neutral-filled feature ranks remain inherited research assumptions.',
        'Historical as-of replays are not archived real-time predictions; expanded support need not improve returns.'],
        observer_writes=False,original_strategy_changed=False,parameter_search=False)


def code_files():
    return [ROOT/'work'/n for n in ('monthly_early_history229_host.py','test_monthly_early_history229_host.py',
        'monthly_early_history229_model.py','test_monthly_early_history229_model.py',
        'monthly_early_history229_independent.py','test_monthly_early_history229_independent.py')]


def state_check(s):
    require(len(s['completed'])==225 and sum(x['trials'] for x in s['completed'])==1192,'Strategy count changed')
    require(s['completed'][-1]['directory']==PRIOR.name,'Round228 must remain latest strategy')
    require(s['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(s.get('target_reached',False) is False,'Goal already marked reached')


def commit_state(before,after,tag):
    # Existing compare-and-replace implementation; its filename namespace is historical,
    # while each backup remains uniquely identified by exact pre-write content hash.
    prior_source.commit_state(before,after,'early229_'+tag)


def early_meta(tm,limit=None):
    dates,ends,names=tm['dates'],tm['ends'],tm['symbols']
    require(len(dates)==len(ends)==len(names),'Training metadata axes')
    require(dates==sorted(set(dates)),'Sorted unique signal dates')
    eligible=[i for i,d in enumerate(dates) if d<'2016-08-17']
    require(eligible==list(range(len(eligible))),'Early queries must be a prefix')
    n=len(eligible) if limit is None else limit
    require(type(n) is int and 0<=n<=len(eligible),'Invalid prefix count')
    return {k:(v[:n] if isinstance(v,list) and len(v)==len(dates) else v) for k,v in tm.items()}


def model_inputs(train,tm,limit=None):
    hm=early_meta(tm,limit)
    widths=np.asarray([len(s) for s in tm['symbols']],dtype=int)
    offsets=np.r_[0,np.cumsum(widths)]
    np.testing.assert_array_equal(train['offsets'],offsets)
    dates=np.asarray(train['dates']);ends=np.asarray(train['ends'])
    symbols=np.asarray([s for names in tm['symbols'] for s in names],dtype=str)
    np.testing.assert_array_equal(dates.astype('datetime64[D]'),np.repeat(tm['dates'],widths).astype('datetime64[D]'))
    np.testing.assert_array_equal(ends.astype('datetime64[D]'),np.repeat(tm['ends'],widths).astype('datetime64[D]'))
    require(len(dates)==len(symbols),'Training row axis')
    q=int(offsets[len(hm['dates'])]);keep=np.ones(len(dates),bool)
    if limit is not None:
        if q:
            year=int(str(hm['dates'][-1])[:4]);cut=np.datetime64('%04d-12-31'%(year-1),'D')
            keep=ends.astype('datetime64[D]')<=cut
        else:keep[:]=False
    good=np.asarray(train['train_eligible'])
    require(good.shape==dates.shape and good.dtype.kind=='b','Original train_eligible must be boolean')
    x=np.asarray(train['train_plain'])
    require(x.shape==(len(dates),10),'Original train_plain dimensions')
    return (dates[keep].copy(),ends[keep].copy(),symbols[keep].copy(),x[keep].copy(),
        {k:np.asarray(train['target_'+k])[keep].copy() for k in TARGETS},np.asarray(train['common_good'])[keep].copy(),
        dates[:q].copy(),symbols[:q].copy(),x[:q].copy(),good[:q].copy()),hm


def slice_labels(labels,train,hm):
    q=sum(map(len,hm['symbols']))
    for k in ('dates','ends','offsets'):require(snapshot({k:labels[k]})==snapshot({k:train[k]}),'Exact source227 label axis differs: '+k)
    expected={'target_'+k for k in TARGETS}|{'dates','ends','offsets','common_good','added','vol63'}
    require(set(labels)==expected,'Source227 label fields changed')
    for k,v in labels.items():require(v.ndim==1 and len(v)==(len(train['offsets']) if k=='offsets' else len(train['dates'])),'Source227 label dimensions: '+k)
    result={k:(v[:len(hm['dates'])+1].copy() if k=='offsets' else v[:q].copy()) for k,v in labels.items()}
    verify_label_copy(result,labels,hm)
    return result


def verify_label_copy(result,labels,hm):
    q=sum(map(len,hm['symbols']));require(set(result)==set(labels),'Copied label field set')
    for k,v in labels.items():
        expected=v[:len(hm['dates'])+1] if k=='offsets' else v[:q]
        require(snapshot({k:result[k]})==snapshot({k:expected}),'Copied label bytes differ: '+k)
        require(not np.shares_memory(result[k],v),'Copied labels must be detached: '+k)


def require_check(check,forecast):
    import monthly_early_history229_independent as independent
    for key in independent.FORECAST_FLAGS:require(check.get(key) is True,'Independent evidence missing: '+key)
    require(check.get('numerical_tolerance')==1e-12 and check.get('producer_modules_imported') is False and check.get('computes_returns') is False and check.get('shared_numerical_primitives') is False,'Independent scope/tolerance differs')
    require(check.get('counts')==forecast['audit']['counts'] and set(check['counts'])==set(independent.COUNT_FIELDS),'Independent fit accounting differs')


def test():
    require(not TESTS.exists(),'Test receipt exists')
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(previous.module('early229test'+str(i),p)) for i,p in enumerate(code_files()[1::2])])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True);require(result.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,tests=result.testsRun,synthetic_only=True,runtime=ev.runtime(),code_sha256={str(p):sha(p) for p in code_files()}))


def register():
    require(not OUT.exists(),'New source directory required')
    before=CENTRAL.read_bytes();s=json.loads(before);state_check(s);require(not s['registered_queue'],'Active experiment')
    t=read(TESTS);require(t['passed'] and t['synthetic_only'] and t['runtime']==ev.runtime(),'Tests/runtime mismatch')
    require(t['code_sha256']=={str(p):sha(p) for p in code_files()},'Numerical code changed after tests')
    r=read(REVIEW);require(r['passed'] is True and r['metadata_only'] is True and r['context_sha256']==sha(CONTEXT),'Metadata source review missing')
    c=read(CODE_REVIEW);require(c['passed'] is True and c['unresolved_blockers']==[],'Code review incomplete')
    require(c['code_sha256']=={str(p):sha(p) for p in code_files()} and c['context_sha256']==sha(CONTEXT),'Code review stale')
    require(read(CONTEXT)==contract(),'Contract differs')
    pins=dict(read(PRIOR/'PLAN.json')['input_sha256'])
    for path,digest in r['input_sha256'].items():previous.bind(pins,Path(path),digest)
    import monthly_early_history229_model as model
    import monthly_early_history229_independent as independent
    for p in code_files()+[REVIEW,CODE_REVIEW,TESTS,CHECKPOINT,CONTEXT,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json',PRIOR/'RESEARCH_REVIEW.json',Path(prior_source.__file__)]+list(model.dependency_files())+list(independent.dependency_files()):previous.bind(pins,p)
    require(sha(PRIOR/'COMPLETION.json')==PRIOR_SHA,'Prior close differs')
    previous.verify(pins);protected=read(PROTECTED);require(len(protected)==21444,'Protection count');previous.verify(protected)
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    dump(OUT/'PLAN.json',dict(created_utc=now(),runtime=ev.runtime(),contract=contract(),input_sha256=pins,
        raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],protected_files=len(protected),prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_queries_forecast_before_registration=0,actual_new_fits_before_registration=0,actual_portfolio_returns_before_registration=0))
    digest=sha(OUT/'PLAN.json');s.update(current=OUT.name,current_status='source_registered_before_early_forecasts',registered_queue=[dict(directory=OUT.name,plan_sha256=digest)],updated_utc=now(),last_goal_turn_classification='progress')
    commit_state(before,s,'register');print(json.dumps(dict(status='REGISTERED',plan_sha256=digest)),flush=True)


def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit PLAN SHA differs');p=read(OUT/'PLAN.json')
    require(p['runtime']==ev.runtime() and p['contract']==contract() and read(CONTEXT)==contract(),'Runtime/contract mismatch')
    previous.verify(p['input_sha256']);require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Prior state differs')
    s=read(CENTRAL);state_check(s);require(s['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Registration mismatch')
    return p


def snapshot(arrays):return {k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()}


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Prepared source write-once')
    import monthly_early_history229_model as model
    import monthly_early_history229_independent as independent
    study=previous.local();m,a,px,op=study.load();require(a['baseline'].shape==(122,49) and len(a)==27,'Current arrays schema')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');meta=read(REBUILD/'MONTHLY_META.json')
    require(meta['monthly']==m and meta['training']==read(study.arch.P/'META.json')['training'],'Original metadata mismatch')
    original=ev.load_arrays(study.target.P/'TRAIN_INPUTS.npz');require(snapshot(train)==snapshot(original),'Original training bytes changed')
    labels=ev.load_arrays(SOURCE227/'prepared/LABELS.npz');cached=ev.load_arrays(SOURCE227/'prepared/FORECASTS.npz')
    before=[snapshot(x) for x in (train,a,labels,cached)]
    args,hm=model_inputs(train,meta['training']);require(len(hm['dates'])==67 and len(args[6])==3283,'Expected early67x49 axis')
    panel,info=study.arch.old.panel_load(m['etfs']);require(study.raw_source_hashes(info)==plan['raw_source_hashes'],'Raw source differs')
    parity=study.verify_panel_parity(panel,px,op);calendar=read(CALENDAR)['sessions']
    support=prior_source.signal_support(panel['adj'],args[6],args[7],calendar)
    print('Source229: earlier as-of forecasts',flush=True)
    forecast=model.forecasts(*args,expansion_allowed=support)
    print('Source229: independent forecast verification',flush=True)
    check=independent.check_forecasts(*args,expansion_allowed=support,result=forecast)
    require_check(check,forecast)
    risk_labels=slice_labels(labels,train,hm);prefixes=[]
    for n in PREFIXES:
        smallargs,sm=model_inputs(train,meta['training'],limit=n);q=len(smallargs[6])
        small_support=prior_source.signal_support(panel['adj'].loc[:sm['dates'][-1]],smallargs[6],smallargs[7],calendar) if n else np.zeros(0,bool)
        np.testing.assert_array_equal(small_support,support[:q])
        small=model.forecasts(*smallargs,expansion_allowed=small_support)
        for k in ('available','local_available'):np.testing.assert_array_equal(small[k],forecast[k][:q])
        for k in forecast['raw']:np.testing.assert_array_equal(small['raw'][k],forecast['raw'][k][:q])
        prefixes.append(dict(prefix=n,query_rows=q,training_rows=len(smallargs[0]),latest_label_end=str(np.asarray(smallargs[1]).max()) if len(smallargs[1]) else None,audit=small['audit']))
        print('Source229: prefix '+str(n)+' verified',flush=True)
    require(before==[snapshot(x) for x in (train,a,labels,cached)],'Original input arrays mutated')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{'raw_'+k:v for k,v in forecast['raw'].items()},available=forecast['available'],local_available=forecast['local_available'],query_eligible=args[9],expansion_allowed=support)
    np.savez_compressed(dest/'LABELS.npz',**risk_labels)
    dump(dest/'META.json',dict(history=hm,current_query_axis_unchanged=True,training_reference=str(REBUILD/'MONTHLY_META.json')))
    dump(dest/'SOURCE_AUDIT.json',dict(forecasts=forecast['audit'],forecast_prefixes=prefixes,original_arrays_unchanged=True,
        copied_label_rows=len(args[6]),copied_label_exact=True,source227_label_sha256=sha(SOURCE227/'prepared/LABELS.npz'),
        raw_price_parity=parity,original_trading_queries=122,new_history_months=67,no_new_orders=True,no_portfolio_returns=True,strict_PIT=False))
    dump(dest/'INDEPENDENT.json',check)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},source_only=True,strategy_trials=0,portfolio_returns=False))
    print(json.dumps(dict(status='SOURCE_PREPARED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def close(expected):
    gate(expected);require(not (OUT/'COMPLETION.json').exists(),'Source closure write-once')
    p=read(OUT/'prepared/COMPLETION.json');require(p['status']=='COMPLETE' and p['plan_sha256']==expected and p['source_only'] is True,'Incomplete source')
    require(set(p['artifacts_sha256'])=={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'},'Prepared artifact set')
    for n,h in p['artifacts_sha256'].items():require(sha(OUT/'prepared'/n)==h,'Changed source artifact')
    saved_audit=read(OUT/'prepared/SOURCE_AUDIT.json')
    require_check(read(OUT/'prepared/INDEPENDENT.json'),dict(audit=saved_audit['forecasts']))
    protected=read(PROTECTED);require(len(protected)==21444,'Protection count');previous.verify(protected)
    before=CENTRAL.read_bytes();s=json.loads(before);state_check(s);audit=read(OUT/'prepared/SOURCE_AUDIT.json')
    dump(OUT/'COMPLETION.json',dict(status='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,
        prepared_sha256=sha(OUT/'prepared/COMPLETION.json'),protected_files=len(protected),source_only=True,observers_modified=False,
        strategy_trials=0,new_portfolio_returns=False,original_strategy_unchanged=True,forecasts=audit['forecasts']))
    s.setdefault('source_experiments',[]).append(dict(directory=OUT.name,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json')),strategy_trials=0))
    s.update(current=OUT.name,current_status='source_completed_independently_verified',registered_queue=[],updated_utc=now(),last_goal_turn_classification='progress',next_step='Preregister separate-history-axis joint/diagonal error risk test; unchanged current122tradingqueries and18monththreshold.')
    commit_state(before,s,'close');print(json.dumps(dict(status='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(OUT/'COMPLETION.json'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':test()
    elif a.phase=='register':register()
    else:require(bool(a.plan_sha256),'Explicit PLAN SHA required');globals()[a.phase](a.plan_sha256)
