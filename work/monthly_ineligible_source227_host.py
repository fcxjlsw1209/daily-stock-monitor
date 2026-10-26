"""Prospective source-only extension of historically ineligible original query rows.
No strategy selection, new orders, risk statistics or portfolio return computation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev
import monthly_joint_error226_host as previous
import monthly_identity_transfer206_host as source206

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ineligible_source227'
CENTRAL=previous.CENTRAL
PRIOR=ROOT/'outputs/monthly_joint_error226'
LOCAL=previous.LOCAL
REBUILD=previous.REBUILD
SOURCE=previous.SOURCE
CALENDAR=previous.CALENDAR
PROTECTED=previous.PROTECTED
REVIEW=ROOT/'work/ineligible_source227_SOURCE_REVIEW.json'
TESTS=ROOT/'work/monthly_ineligible_source227_TESTS.json'
CHECKPOINT=ROOT/'work/after226_research_checkpoint.json'
CONTEXT=ROOT/'work/monthly_ineligible_source227_context.json'
PRIOR_SHA='80e1f20981a27348b4d2623d9103d3f3425d90d0ce083c4a76fb88937e68b414'
PREFIXES=(0,1,53,88,122)
LABEL_CUTOFFS=('2019-12-31','2023-12-31','2025-12-31')
sha,read,require,dump=ev.sha,ev.read,ev.require,ev.dump


def now(): return datetime.now(timezone.utc).isoformat()

def contract():
    return dict(schema='original_domain_ineligible_source227_v1', source_only=True,
        strategy_trials=0, parameter_search=False, portfolio_returns=False,
        original_monthly_queries=122, original_query_width=49, original_training_rows=8820,
        new_query_domain='Same exact historical query rows; finite cached plain10 and295 positivefinite official-session past asset closes, formerly not event_good; independent of future labels.',
        features='Exact cached plain10; unchanged historical49 rank reference and neutral-fill semantics. No new feature calculation or reranking.',
        forecasts='Replay original event_good with original annual training/cohort/targets; exact old cached3 raw forecasts and masks parity; separately predict allowed formerly ineligible rows; copy old cached rows exactly after verification.',
        training='Original common_good and four cached targets only; original priorDecember31/5calendar-year cutoff,18monthminimum, Ridgealpha10/12monthdecay; local18month rule unchanged. Extra labels never enter model fitting.',
        labels='Only originalTRAIN row identities; add !train_eligible finite plain10 with295 positivefinite past official asset closes,64past volatility closes with sampleSTD>0,and complete positivefinite asset+SPY official future path through exact next-monthend. Fixed clips1/1/5/1 then subtract ORIGINAL common_good clipped mean. Never enlarge mean cohort or revive eligible rejected rows.',
        old_label_parity_atol=1e-12, old_label_parity_rtol=0., old_export_labels_exact_copy=True,
        old_forecast_parity='Exact array equality including NaNs and masks before stitch.',
        label_old_stricter_path_exceptions='Record old labels failing stricter new path; retain existing original values/cohort rather than revise history.',
        model_prefixes=list(PREFIXES), label_physical_cutoffs=list(LABEL_CUTOFFS),
        source_model_fit_counts='Count original-domain replays, expanded-domain fits and independent/prefix repeats separately. Source extension is not zero-fit.',
        initial_and_terminal_query='Keep exact partial first and terminal query dates for forecasts. Labels use only existing training naturalmonthend identities, never truncated query end.',
        limitations=['Expanded forecasts are historical replays, not archived real-time forecasts.',
          'Formerly ineligible feature states may be outside useful prediction domain.',
          'Finite cached ranks may contain neutral-filled missing raw factors; price warmup does not prove every OHLCV feature complete.',
          'Retrospective adjusted prices, historical proxy Top32 and assumed ETF17 remain; strict point-in-time certification false.',
          'No inference of return improvement from extra observations; a separate prospective strategy experiment is required.'],
        observer_writes=False, original_trading_pool_changed=False, original_strategy_changed=False)


def code_files():
    return [ROOT/'work'/n for n in ('monthly_ineligible_source227_host.py','test_monthly_ineligible_source227_host.py',
        'monthly_ineligible_source227_model.py','test_monthly_ineligible_source227_model.py',
        'monthly_ineligible_source227_independent.py','test_monthly_ineligible_source227_independent.py')]


def state_check(state):
    require(len(state['completed'])==224 and sum(r['trials'] for r in state['completed'])==1189,'Completed strategy count changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round226 must remain latest strategy')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(not state.get('target_reached',False),'Goal already marked reached')


def commit_state(before, after, tag):
    require(CENTRAL.read_bytes()==before,'Concurrent central update')
    digest=hashlib.sha256(before).hexdigest()
    backup=CENTRAL.parent/('PRE_SOURCE227_'+tag+'_'+digest+'.json')
    if not backup.exists():
        with backup.open('xb') as f: f.write(before)
    require(sha(backup)==digest,'Central backup differs')
    with tempfile.NamedTemporaryFile(dir=CENTRAL.parent,delete=False) as f:
        name=Path(f.name);f.write((json.dumps(after,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    try:
        require(CENTRAL.read_bytes()==before,'Concurrent central change before replace')
        os.replace(name,CENTRAL)
    finally: name.unlink(missing_ok=True)


def signal_support(adj,event_dates,event_symbols,calendar):
    raw_dates=np.asarray(event_dates)
    require(raw_dates.ndim==1 and raw_dates.dtype.kind not in 'biufc','Explicit dates required')
    date_index=pd.DatetimeIndex(event_dates)
    require(date_index.tz is None and not date_index.hasnans and date_index.equals(date_index.normalize()),'Signal dates must be normalized')
    dates=date_index.to_numpy(dtype='datetime64[D]');symbols=np.asarray(event_symbols)
    require(dates.ndim==1 and symbols.shape==dates.shape,'Signal warmup row axes')
    require(not np.isnat(dates).any(),'Invalid signal dates')
    require(isinstance(adj,pd.DataFrame) and isinstance(adj.index,pd.DatetimeIndex) and adj.index.is_unique and adj.index.is_monotonic_increasing and adj.columns.is_unique,'Price frame axes')
    require(adj.index.tz is None and not adj.index.hasnans and adj.index.equals(adj.index.normalize()),'Price dates must be normalized')
    sessions=pd.DatetimeIndex(calendar)
    require(sessions.tz is None and not sessions.hasnans and sessions.equals(sessions.normalize()) and sessions.is_unique and sessions.is_monotonic_increasing,'Calendar sessions')
    output=np.zeros(len(dates),bool)
    for i,(day,symbol) in enumerate(zip(dates,symbols)):
        require(isinstance(symbol,(str,np.str_)) and symbol and symbol==symbol.strip(),'Exact symbol required')
        pos=sessions.get_indexer([pd.Timestamp(day)])[0]
        if pos<294 or symbol not in adj.columns: continue
        past=sessions[pos-294:pos+1]
        if not past.isin(adj.index).all(): continue
        values=adj.loc[past,symbol].to_numpy(float)
        output[i]=bool(np.all(np.isfinite(values)&(values>0.)))
    return output


def slice_training(train,tm,cutoff):
    cut=np.datetime64(cutoff,'D'); require(not np.isnat(cut),'Invalid label cutoff')
    keep=np.asarray(train['ends'],dtype='datetime64[D]')<=cut
    widths=np.array([len(row) for row in tm['symbols']],dtype=int)
    offsets=np.r_[0,np.cumsum(widths)]
    np.testing.assert_array_equal(train['offsets'],offsets)
    require(len(keep)==int(offsets[-1]),'Training identities differ')
    selected=np.asarray(tm['ends'],dtype='datetime64[D]')<=cut
    np.testing.assert_array_equal(keep,np.repeat(selected,widths))
    result={k:np.asarray(v)[keep].copy() if np.asarray(v).ndim and len(v)==len(keep) else np.asarray(v).copy() for k,v in train.items() if k!='offsets'}
    result['offsets']=np.r_[0,np.cumsum(widths[selected])]
    meta=dict(tm)
    for k,v in tm.items():
        if isinstance(v,list) and len(v)==len(widths): meta[k]=[x for x,use in zip(v,selected) if use]
    return result,meta,keep


def test():
    require(not TESTS.exists(),'Test receipt already exists')
    modules=[previous.module('source227test'+str(i),p) for i,p in enumerate(code_files()[1::2])]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True);require(result.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,tests=result.testsRun,synthetic_only=True,runtime=ev.runtime(),code_sha256={str(p):sha(p) for p in code_files()}))


def register():
    require(not OUT.exists(),'Source experiment is write-once')
    before=CENTRAL.read_bytes();state=json.loads(before);state_check(state)
    require(not state['registered_queue'],'Another experiment active')
    tests=read(TESTS);require(tests['passed'] and tests['synthetic_only'] and tests['runtime']==ev.runtime(),'Tests/runtime mismatch')
    require(tests['code_sha256']=={str(p):sha(p) for p in code_files()},'Code changed since tests')
    review=read(REVIEW);require(review['passed'] is True and review['metadata_only'] is True,'Source admission missing')
    require(read(CONTEXT)==contract() and review['context_sha256']==sha(CONTEXT),'Source review contract differs')
    pins=dict(read(PRIOR/'PLAN.json')['input_sha256'])
    for name,digest in review['input_sha256'].items():previous.bind(pins,Path(name),digest)
    import monthly_ineligible_source227_model as model
    import monthly_ineligible_source227_independent as independent
    for p in code_files()+[REVIEW,TESTS,CHECKPOINT,CONTEXT,CALENDAR,PROTECTED,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json']+list(model.dependency_files())+list(independent.dependency_files()):previous.bind(pins,p)
    require(sha(PRIOR/'COMPLETION.json')==PRIOR_SHA,'Predecessor closure changed')
    previous.verify(pins);protected=read(PROTECTED);require(len(protected)==21444,'Full protection count');previous.verify(protected)
    OUT.mkdir()
    with (OUT/'STATE_BEFORE.json').open('xb') as f: f.write(before)
    dump(OUT/'PLAN.json',dict(created_utc=now(),runtime=ev.runtime(),contract=contract(),input_sha256=pins,
        raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),
        protected_files=len(protected),actual_new_queries_or_labels_before_registration=0,actual_new_fits_before_registration=0,
        actual_new_portfolio_returns_before_registration=0))
    digest=sha(OUT/'PLAN.json');state.update(current=OUT.name,current_status='source_registered_before_extended_forecasts_labels',
        registered_queue=[dict(directory=OUT.name,plan_sha256=digest)],updated_utc=now(),last_goal_turn_classification='progress',
        next_step='Build and independently validate original-pool ineligible forecast/label extension; no strategy change.')
    commit_state(before,state,'register');print(json.dumps(dict(status='REGISTERED',plan_sha256=digest)),flush=True)


def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit source PLAN SHA mismatch')
    plan=read(OUT/'PLAN.json');require(plan['runtime']==ev.runtime() and plan['contract']==contract() and read(CONTEXT)==contract(),'Frozen runtime/contract changed')
    previous.verify(plan['input_sha256']);require(sha(OUT/'STATE_BEFORE.json')==plan['prior_central_sha256'],'Prior state changed')
    state=read(CENTRAL);state_check(state)
    require(state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Source registration mismatch')
    return plan


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Prepared source is write-once')
    import monthly_ineligible_source227_model as model
    import monthly_ineligible_source227_independent as independent
    study=previous.local();m,a,px,op=study.load()
    require(a['baseline'].shape==(122,49) and len(a)==27,'Original query schema changed')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');meta=read(REBUILD/'MONTHLY_META.json')
    require(meta['monthly']==m and meta['training']==read(study.arch.P/'META.json')['training'],'Exact source metadata mismatch')
    original=ev.load_arrays(study.target.P/'TRAIN_INPUTS.npz')
    require(set(train)==set(original),'Original cache keys differ')
    for key in train:
        require(train[key].dtype==original[key].dtype and train[key].shape==original[key].shape,'Original cache axes/type differ: '+key)
        np.testing.assert_array_equal(train[key],original[key],err_msg=key)
    original_bytes={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in train.items()}
    arrays_bytes={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in a.items()}
    source=ev.load_arrays(SOURCE/'prepared/FORECASTS.npz')
    panel,info=study.arch.old.panel_load(m['etfs'])
    require(study.raw_source_hashes(info)==plan['raw_source_hashes'],'Raw panel source differs')
    parity=study.verify_panel_parity(panel,px,op)
    calendar=read(CALENDAR)['sessions']
    args=source206.model_inputs(train,meta,a)
    support=signal_support(panel['adj'],args[6],args[7],calendar)
    print('Source227: original replay plus additional query forecasts',flush=True)
    forecasts=model.forecasts(*args,cached_source=source,expansion_allowed=support)
    print('Source227: fixed original-center labels',flush=True)
    labels=model.build_labels(panel['adj'],train,meta['training'],calendar)
    print('Source227: independent source validation',flush=True)
    check_forecasts=independent.check_forecasts(*args,cached_source=source,expansion_allowed=support,result=forecasts)
    check_labels=independent.check_labels(panel['adj'],train,meta['training'],calendar,labels)
    require(check_forecasts['passed'] is True and check_labels['passed'] is True,'Independent check failed')
    prefix=[]
    for n in PREFIXES:
        smallargs=source206.model_inputs(train,meta,a,limit=n);q=n*49
        stored={k:v[:q].copy() for k,v in source.items()}
        small=model.forecasts(*smallargs,cached_source=stored,expansion_allowed=support[:q].copy())
        for k in ('available','local_available'): np.testing.assert_array_equal(small[k],forecasts[k][:q])
        for k in forecasts['raw']:np.testing.assert_array_equal(small['raw'][k],forecasts['raw'][k][:q])
        if n:
            smallpx=panel['adj'].loc[:m['dates'][n-1]]
            np.testing.assert_array_equal(signal_support(smallpx,smallargs[6],smallargs[7],calendar),support[:q])
        prefix.append(dict(prefix=n,query_rows=q,training_rows=len(smallargs[0]),latest_label_end=str(np.asarray(smallargs[1]).max()) if len(smallargs[1]) else None,audit=small['audit']))
        print('Source227: forecast prefix '+str(n)+' verified',flush=True)
    label_prefix=[]
    for cutoff in LABEL_CUTOFFS:
        tr,tm,mask=slice_training(train,meta['training'],cutoff)
        small=model.build_labels(panel['adj'].loc[:cutoff],tr,tm,calendar)
        for k in ('good','added','vol63'):np.testing.assert_array_equal(small[k],labels[k][mask])
        for k in labels['targets']:np.testing.assert_array_equal(small['targets'][k],labels['targets'][k][mask])
        label_prefix.append(dict(cutoff=cutoff,rows=int(mask.sum()),audit=small['audit']))
    require(original_bytes=={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in train.items()},'Training source mutated')
    require(arrays_bytes=={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in a.items()},'Original query strategy arrays mutated')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{'raw_'+k:v for k,v in forecasts['raw'].items()},available=forecasts['available'],local_available=forecasts['local_available'],expansion_allowed=support)
    np.savez_compressed(dest/'LABELS.npz',**{'target_'+k:v for k,v in labels['targets'].items()},common_good=labels['good'],added=labels['added'],vol63=labels['vol63'],dates=train['dates'],ends=train['ends'],offsets=train['offsets'])
    dump(dest/'META.json',meta)
    audit=dict(forecasts=forecasts['audit'],labels=labels['audit'],forecast_prefixes=prefix,label_prefixes=label_prefix,
        original_array_bytes_unchanged=True,raw_price_parity=parity,no_new_orders=True,no_new_portfolio_returns=True,strict_PIT=False)
    dump(dest/'SOURCE_AUDIT.json',audit)
    dump(dest/'INDEPENDENT.json',dict(forecasts=check_forecasts,labels=check_labels))
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},source_only=True,portfolio_returns=False,strategy_trials=0))
    print(json.dumps(dict(status='SOURCE_PREPARED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def close(expected):
    gate(expected);require(not (OUT/'COMPLETION.json').exists(),'Source closure is write-once')
    prepared=read(OUT/'prepared/COMPLETION.json')
    require(prepared['plan_sha256']==expected and prepared['status']=='COMPLETE' and prepared['source_only'] is True,'Incomplete prepared source')
    require(set(prepared['artifacts_sha256'])=={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'},'Source artifacts differ')
    for name,digest in prepared['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==digest,'Source artifact changed')
    checks=read(OUT/'prepared/INDEPENDENT.json');require(all(v['passed'] is True for v in checks.values()),'Independent source check failed')
    protected=read(PROTECTED);require(len(protected)==21444,'Full protection count');previous.verify(protected)
    before=CENTRAL.read_bytes();state=json.loads(before);state_check(state)
    audit=read(OUT/'prepared/SOURCE_AUDIT.json')
    complete=dict(status='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,
        prepared_sha256=sha(OUT/'prepared/COMPLETION.json'),protected_files=len(protected),source_only=True,observers_modified=False,
        original_strategy_unchanged=True,new_portfolio_returns=False,strategy_trials=0,forecasts=audit['forecasts'],labels=audit['labels'])
    dump(OUT/'COMPLETION.json',complete)
    state.setdefault('source_experiments',[]).append(dict(directory=OUT.name,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json')),strategy_trials=0))
    state.update(current=OUT.name,current_status='source_completed_independently_verified',registered_queue=[],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Preregister matched joint-versus-diagonal prediction-error risk using extended same-pool history; original trading rule and18-month threshold unchanged.')
    commit_state(before,state,'close')
    print(json.dumps(dict(status=complete['status'],completion_sha256=sha(OUT/'COMPLETION.json'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':test()
    elif a.phase=='register':register()
    else:
        require(bool(a.plan_sha256),'Explicit source PLAN SHA required')
        globals()[a.phase](a.plan_sha256)
