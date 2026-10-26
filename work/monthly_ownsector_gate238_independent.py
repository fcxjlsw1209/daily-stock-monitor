"""Independent source gate, unchanged Local67 pre-score, and two-arm accounts.

New arithmetic is reconstructed without importing the production gate/host.
Only pinned independent original-score and accounting primitives are reused.
No file output, model training, or real-data entry point is provided.
"""
from datetime import date
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HELPER = ROOT / 'work/monthly_endpoint155_prepare_review.py'
SOURCE_SHA = '0e594355beecc84a705517c8cd2774214e4300b5248e0c772a0306e73c38174f'
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'ownsector_relative_gate')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
VALUE_KEYS = ('original_eligible','candidate_eligible','required','momentum','benchmark_momentum','threshold')
COUNT_FIELDS = ('events','symbols_per_event','checked_cells','original_eligible_cells',
    'weak_events','active_events','required_cells','stock_required_cells','etf_required_cells',
    'source_price_observations','benchmark_price_observations',
    'spy_price_observations','rejected_cells','candidate_eligible_cells','new_model_fits','predictive_model_fits')
FEATURE_FLAGS = ('passed','independent_scalar_rebuild','original_eligibility_exact',
    'weak_state_descriptive_only','initial_partial_unchanged','official_endpoint_offsets_checked',
    'monthend_queries_checked','positive_nonboolean_endpoints_checked','required_source_only_checked',
    'missing_required_source_fails','original_positive_momentum_parity_checked',
    'strict_relative_comparison_checked','max_zero_benchmark_threshold_checked',
    'subset_only_checked','nonpositive_benchmark_redundancy_checked','no_eligibility_rerank',
    'no_model_or_score_changes','audit_checked','descriptive_counts_checked',
    'no_predictive_fits_checked','no_future_or_outcome_inputs',
    'admitted_report_identity_checked','official_report_endpoint_binding_checked',
    'etf_spy_dispatch_checked','dedicated_vox_source_checked','no_additional_spy_gate')
SELECTION_FLAGS = ('passed','independent_original_pre_checked','original_pre_unchanged',
    'original_model_rank_scope_preserved','original_support_checked','original_budget_checked',
    'original_effects_unchanged','final_eligibility_only_checked','subset_only_checked',
    'one_episode_per_arm','initial_partial_unchanged','exact_scores_and_choices_checked')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pinned(path, digest, name):
    require(sha(path) == digest, 'Frozen independent dependency changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def frozen_source_oracle():
    return load_pinned(SOURCE_HELPER, SOURCE_SHA, 'ownsector_gate238_original_score')


def accounting():
    return load_pinned(ACCOUNTING, ACCOUNTING_SHA, 'ownsector_gate238_accounts')


def dependency_files():
    return tuple(sorted(set(frozen_source_oracle().source_files(ROOT)) |
                        {Path(__file__).resolve(), SOURCE_HELPER, ACCOUNTING,
                         ROOT/'work/monthly_allstate_gate237_independent.py',
                         ROOT/'work/test_monthly_upside_retention181_independent.py',
                         ROOT/'work/monthly_upside_retention181_independent.py'}))


def exact_array(actual, expected, label):
    a,e=np.asarray(actual),np.asarray(expected)
    require(a.shape==e.shape and a.dtype==e.dtype and a.tobytes()==e.tobytes(), label+': exact array bytes')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+': keys')
        for key in expected: record_check(actual[key],expected[key],label+'/'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+': list')
        for i,(a,b) in enumerate(zip(actual,expected)):record_check(a,b,label+'/'+str(i))
    elif isinstance(expected,float):
        require(type(actual) is float and np.float64(actual).tobytes()==np.float64(expected).tobytes(),label+': float bytes')
    else:
        require(type(actual) is type(expected) and actual==expected,label+': value/type')


def iso(raw):
    require(isinstance(raw,str),'Canonical ISO date required')
    d=date.fromisoformat(raw)
    require(d.isoformat()==raw,'Canonical ISO date required')
    return pd.Timestamp(d)


def axes(arrays, meta, limit=None):
    require(isinstance(arrays,dict) and all(k in arrays for k in ('eligible','state_weak_narrow')),
            'Original eligibility and weak-state arrays required')
    e,w=np.asarray(arrays['eligible']),np.asarray(arrays['state_weak_narrow'])
    require(e.ndim==2 and e.shape[1]>0,'Eligibility requires nonempty candidate width')
    full,width=e.shape
    require(w.shape==(full,),'Weak-state shape must be [months]')
    require(limit is None or type(limit) is int and 0<=limit<=full,'Invalid physical prefix')
    n=full if limit is None else limit
    require(all(isinstance(v,(bool,np.bool_)) for v in e[:n].flat) and
            all(isinstance(v,(bool,np.bool_)) for v in w[:n].flat),'Exact consumed boolean source values required')
    require(isinstance(meta,dict) and isinstance(meta.get('dates'),list) and isinstance(meta.get('symbols'),list)
            and len(meta['dates'])==len(meta['symbols'])==full,'Metadata axes differ')
    etfs=meta.get('etfs')
    require(isinstance(etfs,(list,tuple)) and len(set(etfs))==len(etfs) and
            all(isinstance(s,str) and s and s.strip()==s for s in etfs),'Explicit unique ETF identities')
    names=[];days=[]
    for raw,ss in zip(meta['dates'][:n],meta['symbols'][:n]):
        day=iso(raw)
        require(not days or day>days[-1],'Query dates must increase')
        require(isinstance(ss,list) and len(ss)==width and len(set(ss))==width and
                all(isinstance(s,str) and s and s.strip()==s for s in ss),'Unique query identities required')
        days.append(day);names.append(ss.copy())
    return e[:n].astype(bool,copy=True),w[:n].astype(bool,copy=True),days,names,n,width


def source_axes(panel, calendar, days):
    require(isinstance(panel,dict) and set(panel)=={'adj','vox'},'Original adj plus dedicated VOX required')
    frame=panel['adj']
    for source in panel.values():
        require(isinstance(source,pd.DataFrame) and isinstance(source.index,pd.DatetimeIndex),'Daily adjusted-price frame required')
        idx=source.index
        require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None and
                idx.equals(idx.normalize()),'Invalid price date axis')
        require(source.columns.is_unique and all(isinstance(s,str) and s and s.strip()==s for s in source.columns),
                'Invalid price identity axis')
    require(list(panel['vox'].columns)==['VOX'],'Dedicated VOX identity axis')
    require(isinstance(calendar,(list,tuple)) and (len(calendar)>0 or not days),'Nonempty official calendar required for queries')
    sessions=[iso(d) for d in calendar]
    require(all(d.weekday()<5 for d in sessions) and all(a<b for a,b in zip(sessions,sessions[1:])),
            'Invalid official calendar order')
    lookup={d:i for i,d in enumerate(sessions)};positions=[]
    for i,day in enumerate(days):
        require(day in lookup,'Query absent from official calendar')
        t=lookup[day];partial=day==pd.Timestamp('2016-08-17')
        if partial:require(i==0,'Initial partial may only be first query')
        else:require(t+1<len(sessions) and sessions[t+1].to_period('M')!=day.to_period('M'),
                     'Official month-end requires next-session witness')
        positions.append(t)
    return frame,sessions,positions


def price(frame, day, symbol):
    require(symbol in frame.columns and day in frame.index,'Required source endpoint missing: '+symbol+'/'+str(day.date()))
    raw=frame.at[day,symbol]
    require(isinstance(raw,numbers.Real) and not isinstance(raw,(bool,np.bool_)),'Nonboolean real endpoint required')
    try:value=float(raw)
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Invalid required endpoint') from exc
    require(math.isfinite(value) and value>0.,'Positive finite endpoint required')
    return value


def momentum(start,end):
    for value in (start,end):
        require(isinstance(value,numbers.Real) and not isinstance(value,(bool,np.bool_)), 'Real endpoint required')
        try:converted=float(value)
        except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Invalid endpoint') from exc
        require(math.isfinite(converted) and converted>0.,'Positive finite endpoint required')
    value=float(end)/float(start)-1.
    if not math.isfinite(value):raise FloatingPointError('Nonfinite endpoint momentum')
    return value


BUSINESS_RULES = {
    'AZN': ('XLV','2023-03-11',None,['azn_20230310_issuer']),
    'BUD': ('XLP','2016-03-15',None,['bud_20160314_issuer_sec','bud_2015_20f_index']),
    'CHL': ('VOX','2016-04-28',None,['chl_2015_20f','chl_20160427_issuer_release']),
    'GOOGL': ('XLK','2016-02-12','2018-09-24',['alphabet_google_2015_index','alphabet_google_2015_body']),
    'PDD': ('XLY','2020-04-25',None,['pdd_2019_20f','pdd_2019_20f_index']),
    'TSLA': ('XLY','2020-02-14',None,['tesla_2019_10k_index','tesla_2019_10k_business']),
}
ORIGINAL_SECTOR_BENCHMARKS = {'XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY'}


def _subset(record, expected, label):
    require(isinstance(record,dict) and all(k in record for k in expected),label+': required keys')
    record_check({k:record[k] for k in expected},expected,label)


def _assignment(row, is_etf, raw_day, symbol):
    old=row.get('original_benchmark');origin=row.get('original_mapping_source')
    require(type(row.get('is_etf')) is bool and row['is_etf']==is_etf,'Report ETF identity differs')
    if is_etf:
        _subset(row,dict(original_benchmark='SPY',benchmark='SPY',mapping_repair='unchanged',
            original_mapping_source='original_META.etfs_SPY',mapping_source='original_META.etfs_SPY',
            mapping_evidence=None),'ETF SPY provenance')
        return 'SPY'
    require(origin in ('query_date_sector_panel','frozen_FOREIGN_override'),'Original stock mapping provenance')
    if old=='XLC':
        wanted=dict(benchmark='VOX',mapping_repair='full_period_communication_proxy',
                    mapping_source='full_period_communication_proxy',mapping_evidence=None)
    elif old is not None:
        require(isinstance(old,str) and old in ORIGINAL_SECTOR_BENCHMARKS,'Original sector benchmark identity')
        wanted=dict(benchmark=old,mapping_repair='unchanged',mapping_source=origin,mapping_evidence=None)
    else:
        require(symbol in BUSINESS_RULES,'Unadmitted missing stock mapping')
        benchmark,start,end,evidence=BUSINESS_RULES[symbol]
        require(raw_day>=start and (end is None or raw_day<end),'Dated mapping outside fixed publication window')
        wanted=dict(benchmark=benchmark,mapping_repair='dated_business_proxy',
                    mapping_source='dated_business_proxy',mapping_evidence=evidence)
    _subset(row,wanted,'Frozen repaired assignment')
    return wanted['benchmark']


def report_bindings(report, old, days, names, positions, sessions, etfs):
    n,width=old.shape
    require(isinstance(report,dict),'Admitted source report required')
    _subset(report,dict(schema='ownsector238_repaired_source_report_v1',status='ADMITTED',
        admitted=True,gaps=[],source_only=True,momentum_computed=False,eligibility_changed=False,
        rankings_computed=False,orders_computed=False,returns_computed=False,model_fits=0,strict_pit=False),
        'Source admission scope')
    _subset(report.get('counts'),dict(symbols_per_event=width),'Source reference width')
    events=report.get('events');rows=report.get('records')
    require(isinstance(events,list) and len(events)>=n and isinstance(rows,list),'Source report axes')
    consumed=[]
    for row in rows:
        require(isinstance(row,dict) and type(row.get('query_index')) is int and row['query_index']>=0,
                'Source record query index')
        if row['query_index']<n:consumed.append(row)
    expected_order=[];bound={}
    for i,(day,ss,t) in enumerate(zip(days,names,positions)):
        raw=str(day.date());partial=raw=='2016-08-17';ids=[j for j in range(width) if old[i,j]]
        required=[] if partial else ids
        pos=[];dates=[];market=[]
        if not partial:
            require(t>=252,'Insufficient official endpoint history')
            pos=[t-252,t-42];dates=[str(sessions[k].date()) for k in pos]
            market=[dict(role=role,symbol='SPY',date=dd,status='valid',reason=None)
                    for role,dd in zip(('market_start','market_end'),dates)]
        _subset(events[i],dict(query_index=i,date=raw,initial_partial=partial,active=not partial,
            original_eligible_indices=ids,required_indices=required,source_positions=pos,
            source_dates=dates,required_count=len(required),admitted_count=len(required),gap_count=0),
            'Official source event binding')
        actual_market=events[i].get('market_endpoints')
        require(isinstance(actual_market,list) and len(actual_market)==len(market),'Source market metadata')
        for actual,expected in zip(actual_market,market):_subset(actual,expected,'Source market endpoint metadata')
        for j in required:expected_order.append((i,j))
    require([(r['query_index'],r.get('column_index')) for r in consumed]==expected_order,
            'Exact ordered source required-cell domain')
    for row,(i,j) in zip(consumed,expected_order):
        day=days[i];raw=str(day.date());symbol=names[i][j];is_etf=symbol in etfs
        _subset(row,dict(query_index=i,column_index=j,date=raw,symbol=symbol,is_etf=is_etf,
            mapping_status='available',admitted=True,gap_reasons=[]),'Required source cell identity')
        benchmark=_assignment(row,is_etf,raw,symbol)
        t=positions[i];start,end=(str(sessions[k].date()) for k in (t-252,t-42))
        expected=[dict(role=role,symbol=sym,date=dd,status='valid',reason=None)
            for role,sym,dd in (('security_start',symbol,start),('security_end',symbol,end),
                               ('benchmark_start',benchmark,start),('benchmark_end',benchmark,end))]
        points=row.get('endpoints');require(isinstance(points,list) and len(points)==4,'Four admitted endpoints required')
        for actual,wanted in zip(points,expected):_subset(actual,wanted,'Official endpoint identity/clock')
        bound[(i,j)]=row
    return bound


def independent_features(panel,meta,calendar,arrays,source_report,limit=None):
    old,weak,days,names,n,width=axes(arrays,meta,limit)
    frame,sessions,positions=source_axes(panel,calendar,days)
    bound=report_bindings(source_report,old,days,names,positions,sessions,set(meta['etfs']))
    values=dict(original_eligible=old.copy(),candidate_eligible=old.copy(),required=np.zeros((n,width),bool),
                momentum=np.full((n,width),np.nan),benchmark_momentum=np.full((n,width),np.nan),threshold=np.full((n,width),np.nan))
    counts={key:0 for key in COUNT_FIELDS};counts.update(events=n,symbols_per_event=width,checked_cells=n*width)
    events=[];cells=[]
    for i,(day,ss,t) in enumerate(zip(days,names,positions)):
        partial=day==pd.Timestamp('2016-08-17');active=bool(not partial)
        ids=[j for j in range(width) if old[i,j]];required=ids.copy() if active else []
        start=end=None;source_positions=[];removed=[];stock_ids=[];etf_ids=[]
        if active:
            require(t>=252,'Insufficient official endpoint history')
            s,u=sessions[t-252],sessions[t-42]
            start,end=str(s.date()),str(u.date());source_positions=[t-252,t-42]
            for j in ids:
                row=bound[(i,j)];benchmark=row['benchmark'];etf=ss[j] in meta['etfs']
                (etf_ids if etf else stock_ids).append(j)
                ms=momentum(price(frame,s,ss[j]),price(frame,u,ss[j]))
                require(ms>0.,'Original E disagrees with reconstructed positive momentum')
                source='vox' if benchmark=='VOX' else 'adj'
                bm=momentum(price(panel[source],s,benchmark),price(panel[source],u,benchmark))
                threshold=max(0.,bm)
                values['required'][i,j]=True;values['momentum'][i,j]=ms
                values['benchmark_momentum'][i,j]=bm;values['threshold'][i,j]=threshold
                values['candidate_eligible'][i,j]=ms>threshold
                if not values['candidate_eligible'][i,j]:removed.append(j)
                if bm<=0.:require(values['candidate_eligible'][i,j],'Nonpositive benchmark redundancy')
                cells.append(dict(query_index=i,column_index=j,date=str(day.date()),symbol=ss[j],is_etf=etf,
                    benchmark=benchmark,benchmark_source=source,original_benchmark=row['original_benchmark'],
                    mapping_repair=row['mapping_repair'],mapping_source=row['mapping_source'],
                    mapping_evidence=None if row['mapping_evidence'] is None else list(row['mapping_evidence']),
                    source_start=start,source_end=end,momentum=ms,benchmark_momentum=bm,
                    threshold=threshold,accepted=bool(values['candidate_eligible'][i,j])))
        chosen=[j for j in range(width) if values['candidate_eligible'][i,j]]
        events.append(dict(query_index=i,date=str(day.date()),symbols=ss,session_position=t,
            weak_narrow=bool(weak[i]),initial_partial=bool(partial),active=active,
            branch='initial_partial' if partial else 'ownsector_relative_gate',
            source_start=start,source_end=end,source_positions=source_positions,
            original_eligible_indices=ids,required_indices=required,stock_required_indices=stock_ids,
            etf_required_indices=etf_ids,rejected_indices=removed,candidate_eligible_indices=chosen))
        counts['original_eligible_cells']+=len(ids);counts['weak_events']+=int(weak[i])
        counts['active_events']+=int(active);counts['required_cells']+=len(required)
        counts['stock_required_cells']+=len(stock_ids);counts['etf_required_cells']+=len(etf_ids)
        counts['source_price_observations']+=2*len(required);counts['benchmark_price_observations']+=2*len(required)
        counts['spy_price_observations']+=2*len(etf_ids)
        counts['rejected_cells']+=len(removed);counts['candidate_eligible_cells']+=len(chosen)
    require(not np.any(values['candidate_eligible']&~old),'Gate must only remove original eligible names')
    audit=dict(schema='monthly_ownsector_gate238_features_v1',**counts,start_offset=252,end_offset=42,
        return_count=210,rankings_recomputed=False,source_missing_filled=False,portfolio_returns_calculated=False,
        event_history=events,cell_history=cells)
    return values,audit


def values_check(values,expected):
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Exactly six gate evidence arrays required')
    for key in VALUE_KEYS:exact_array(values[key],expected[key],'Independent '+key)


def verify_features(panel,meta,calendar,arrays,source_report,values,audit,limit=None):
    expected,ea=independent_features(panel,meta,calendar,arrays,source_report,limit)
    values_check(values,expected);record_check(audit,ea)
    counts={key:ea[key] for key in COUNT_FIELDS}
    return dict(**{key:True for key in FEATURE_FLAGS},**counts,counts=counts,numerical_tolerance=0.,
                producer_modules_imported=False,computes_returns=False)


def selection_evidence(original,meta,values,n):
    old,weak,days,_,_,width=axes(original,meta,n)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Gate evidence keys')
    v={k:np.asarray(values[k])[:n] for k in VALUE_KEYS}
    for key in VALUE_KEYS:
        shape=(n,width)
        dtype=np.dtype(bool) if key in ('original_eligible','candidate_eligible','required') else np.dtype(np.float64)
        require(v[key].shape==shape and v[key].dtype==dtype,'Gate evidence dtype/shape: '+key)
    exact_array(v['original_eligible'],old,'Original E selection identity')
    want=old.copy();required=np.zeros_like(old)
    for i,day in enumerate(days):
        active=bool(day!=pd.Timestamp('2016-08-17'))
        if active:
            for j in range(width):
                if old[i,j]:
                    benchmark=float(v['benchmark_momentum'][i,j]);require(math.isfinite(benchmark),'Required benchmark momentum evidence')
                    threshold=max(0.,benchmark)
                    exact_array(v['threshold'][i:i+1,j:j+1],np.array([[threshold]]),'Canonical threshold')
                    ms=float(v['momentum'][i,j]);require(math.isfinite(ms) and ms>0.,'Original E-positive source parity')
                    required[i,j]=True;want[i,j]=ms>threshold
    exact_array(v['required'],required,'Required endpoint domain')
    for key in ('momentum','benchmark_momentum','threshold'):
        require(np.isnan(v[key][~required]).all(),'Unused numeric evidence must remain NaN: '+key)
    exact_array(v['candidate_eligible'],want,'Canonical gate subset')
    return v,old,width


def selection_oracle(original,meta,values,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']),'Selection prefix')
    v,old,width=selection_evidence(original,meta,values,n)
    if n==0:
        return ({arm:np.empty((0,width)) for arm in ARMS},{arm:[] for arm in ARMS},
                {arm:np.empty((0,width)) for arm in ARMS})
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    exact_array(x['eligible'],old,'Frozen original model eligibility')
    require(x['local'].shape==(n,width),'Original pre shape')
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    scores={};choices={};pres={}
    for arm in ARMS:
        pre=x['local'].copy();pres[arm]=pre.copy();score=pre.copy();rows=[]
        gate=old if arm==ARMS[0] else v['candidate_eligible']
        for i,names in enumerate(x['names']):
            order=sorted([j for j in range(width) if gate[i,j]],key=lambda j:(-pre[i,j],names[j]))
            if len(order)<3:rows.append(order if len(order)==2 else []);continue
            boundary=[j for j in order[2:5] if float(pre[i,order[2]])-float(pre[i,j])<=.02]
            for j in boundary:
                target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
                score[i,j]+=budget[i]*(target-x['support'][i,j])
            winner=min(boundary,key=lambda j:(-score[i,j],names[j]))
            rows.append(sorted(order[:2]+[winner],key=lambda j:(-score[i,j],names[j])))
        scores[arm]=score;choices[arm]=rows
    return scores,choices,pres


def verify_selection(original,meta,values,orders,*,scores=None,pre_scores=None,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres=selection_oracle(original,meta,values,n)
    require(set(orders)==set(ARMS),'Two order arms required')
    wanted={arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(expected[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders,wanted,'Independent ordered selection')
    for supplied,reference,label in ((scores,expected,'post-episode score'),(pre_scores,pres,'original pre-score')):
        if supplied is not None:
            require(isinstance(supplied,dict) and set(supplied)==set(ARMS),'Two '+label+' arms')
            for arm in ARMS:exact_array(supplied[arm],reference[arm],label+'/'+arm)
    return dict(**{key:True for key in SELECTION_FLAGS},ordered_choices_checked=2*n,checked_events=n,
        full_scores_supplied=scores is not None,full_pre_scores_supplied=pre_scores is not None,
        numerical_tolerance=0.,original_source_parity_tolerance=1e-12,
        producer_modules_imported=False,computes_returns=False,
        holdings_counts={arm:{str(k):sum(len(row)==k for row in choices[arm]) for k in (0,2,3)} for arm in ARMS})


def audit_outcomes(px,opening,orders,exports,results,meta):
    """Two arms, three execution paths and five saved slices; no engine reuse."""
    helper=accounting()
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arm names differ')
    require(len(meta['dates'])==len(meta['ends'])==122 and meta['ends'][:-1]==meta['dates'][1:],'Outcome interval axis differs')
    require(meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24' and
        meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30',
        'Outcome fixed slices differ')
    reports={}
    for arm in ARMS:
        require(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and
                [o['end'] for o in orders[arm]]==meta['ends'],'Order axis differs')
        require(set(exports[arm])==set(results[arm])=={p[0] for p in PATHS},'Execution paths differ')
        reports[arm]={}
        for name,execution,cost in PATHS:
            data=exports[arm][name]
            require(set(data)=={'DAILY','TRADES','HOLDINGS','DAILY_POSITIONS'},'Incomplete execution exports')
            audited=helper.audit_path(px,opening,orders[arm],*[data[k] for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')],execution,cost)
            slices=helper.statistics_by_split(audited['days'],audited['values'],meta);wanted=results[arm][name]
            require(set(wanted['slices'])==set(slices),'Fixed five slices differ')
            helper.check_metrics(slices['full122'],wanted['metrics'],arm+'/'+name)
            for split,value in slices.items():helper.check_metrics(value,wanted['slices'][split],arm+'/'+name+'/'+split)
            for key in ('fees_paid','dollar_turnover','trade_rows'):helper.near(audited[key],wanted['metrics'][key],key)
            annual=[]
            for year in sorted({d[:4] for d in audited['days']}):
                ids=[i for i,day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids)<2:continue
                prior=audited['values'][ids[0]-1] if ids[0] else 100000.
                annual.append(dict(year=int(year),return_=audited['values'][ids[-1]]/prior-1.,partial=int(year) in (2016,2026)))
            # Preserve the frozen metric comparison tolerance; feature/selection checks stay exact.
            require(len(wanted['metrics']['annual_returns'])==len(annual),'Annual metric count')
            for got,expected in zip(wanted['metrics']['annual_returns'],annual):
                require(set(got)==set(expected) and got['year']==expected['year'] and got['partial']==expected['partial'],'Annual metric identity')
                helper.near(got['return_'],expected['return_'],'Annual return')
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False,'Account declaration differs')
            for key in ('days','values','daily_cash_nav'):audited.pop(key)
            reports[arm][name]=dict(audited,passed=True,slices=slices)
    return dict(passed=True,checked_paths=6,checked_slices=30,independent_accounts=True,
        independent_of_production_model_and_engine=True,terminal_liquidation=False,
        accounting_source_sha256=ACCOUNTING_SHA,paths=reports)
