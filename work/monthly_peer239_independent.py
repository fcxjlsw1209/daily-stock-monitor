"""Independent mature-peer arithmetic and funded Local67 selection oracle.

No producer/host imports and no real-data entry point. The original pre-score
and account primitives are reused only through the pinned independent helper.
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
FROZEN = ROOT/'work/monthly_ownsector_gate238_independent.py'
FROZEN_SHA = '2ea0629ee99b614bf6464dbff20dab46c703a7811f3467fa41c6bc5059c55be9'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67','same_sector_peer_median_rank','matched_absolute_momentum_rank')
PATHS = (('same_close_zero','close',0.),('next_open_zero','open',0.),('next_open_cost','open',.0005))
VALUE_KEYS = ('mature','supported','momentum','peer_median','peer_target','absolute_target')
COUNT_FIELDS = ('events','stock_cells','mature_cells','supported_cells','endpoint_observations','new_model_fits')
FEATURE_FLAGS = ('passed','independent_scalar_rebuild','mature_source_admission_checked',
    'full_stock_inventory_preserved','official_endpoint_clock_checked','visible_history_bounds_checked',
    'warmup_without_price_reads','whole_issuer_exclusion_checked','complete_peer_set_checked',
    'minimum_two_distinct_peers_checked','all_stocks_before_eligibility','etfs_and_partial_unavailable',
    'positive_nonboolean_endpoints_checked','missing_mature_source_fails','canonical_momentum_checked',
    'conventional_peer_median_checked','shared_supported_rank_domain_checked','average_percentile_ties_checked',
    'negative_momentum_preserved','audit_checked','descriptive_counts_checked','no_future_or_outcome_inputs')
SELECTION_FLAGS = ('passed','independent_original_pre_checked','original_model_rank_scope_preserved',
    'original_eligibility_unchanged','original_support_checked','funded_five_percent_formula_checked',
    'finite_zero_target_is_valid','calm_nonweak_extension_budget_checked','original_budget_checked',
    'core_weights_unchanged','original_effects_unchanged','one_episode_per_arm',
    'initial_partial_unchanged','exact_scores_and_choices_checked')
BOUNDS = {
    'SPCX': dict(symbol='SPCX',issuer_key='https://companiesmarketcap.com/spacex/marketcap/',
        not_before='2026-06-08',available_from='2026-06-09',kind='exchange_new_listing_lower_bound',
        evidence_id='spcx_nasdaq_new_issue'),
    'SKHY': dict(symbol='SKHY',issuer_key='https://companiesmarketcap.com/sk-hynix/marketcap/',
        not_before='2026-07-10',available_from='2026-07-11',kind='issuer_actual_adr_start',
        evidence_id='skhy_adr_listing')}
REPAIRS = {
    'AZN': ('XLV','2023-03-11',None,['azn_20230310_issuer']),
    'BUD': ('XLP','2016-03-15',None,['bud_20160314_issuer_sec','bud_2015_20f_index']),
    'CHL': ('VOX','2016-04-28',None,['chl_2015_20f','chl_20160427_issuer_release']),
    'GOOGL': ('XLK','2016-02-12','2018-09-24',['alphabet_google_2015_index','alphabet_google_2015_body']),
    'PDD': ('XLY','2020-04-25',None,['pdd_2019_20f','pdd_2019_20f_index']),
    'TSLA': ('XLY','2020-02-14',None,['tesla_2019_10k_index','tesla_2019_10k_business'])}
SECTORS = {'XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','XLC'}


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_helper():
    require(sha(FROZEN)==FROZEN_SHA,'Frozen original-score/account helper changed')
    spec=importlib.util.spec_from_file_location('peer239_frozen_independent',FROZEN)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def frozen_source_oracle():
    return frozen_helper().frozen_source_oracle()


def accounting():
    module=frozen_helper().accounting()
    require(sha(module.__file__)==ACCOUNTING_SHA,'Frozen accounting identity changed')
    return module


def dependency_files():
    return tuple(sorted(set(frozen_helper().dependency_files()) |
        {Path(__file__).resolve(),FROZEN,ROOT/'work/test_monthly_ownsector_gate238_independent.py'}))


def exact_array(actual,expected,label):
    a,e=np.asarray(actual),np.asarray(expected)
    require(a.shape==e.shape and a.dtype==e.dtype and a.tobytes()==e.tobytes(),label+': exact bytes')


def record_check(actual,expected,label='audit'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+': keys')
        for k in expected:record_check(actual[k],expected[k],label+'/'+k)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+': list')
        for i,(a,b) in enumerate(zip(actual,expected)):record_check(a,b,label+'/'+str(i))
    elif isinstance(expected,float):
        require(type(actual) is float and np.float64(actual).tobytes()==np.float64(expected).tobytes(),label+': float')
    else:require(type(actual) is type(expected) and actual==expected,label+': type/value')


def subset(actual,expected,label):
    require(isinstance(actual,dict) and all(k in actual for k in expected),label+': required fields')
    record_check({k:actual[k] for k in expected},expected,label)


def iso(raw):
    require(type(raw) is str,'Canonical ISO date required')
    d=date.fromisoformat(raw);require(d.isoformat()==raw,'Canonical ISO date required')
    return pd.Timestamp(d)


def axes(meta,arrays,limit=None):
    require(isinstance(arrays,dict) and 'eligible' in arrays,'Original array axes required')
    e=np.asarray(arrays['eligible']);require(e.ndim==2,'Original eligibility matrix')
    full,width=e.shape
    require(width>0 or full==0,'Nonempty candidate width')
    require(limit is None or type(limit) is int and 0<=limit<=full,'Invalid prefix')
    n=full if limit is None else limit
    require(all(isinstance(v,(bool,np.bool_)) for v in e[:n].flat),'Consumed eligibility must be boolean')
    require(isinstance(meta,dict) and isinstance(meta.get('dates'),list) and isinstance(meta.get('symbols'),list)
            and len(meta['dates'])==len(meta['symbols'])==full,'Full metadata axes')
    etfs=meta.get('etfs');require(isinstance(etfs,(list,tuple)) and
        all(type(s) is str and s and s.strip()==s for s in etfs) and len(set(etfs))==len(etfs),'ETF identities')
    days=[];names=[]
    for raw,ss in zip(meta['dates'][:n],meta['symbols'][:n]):
        d=iso(raw);require(not days or d>days[-1],'Ordered query days')
        require(isinstance(ss,list) and len(ss)==width and len(set(ss))==width and
                all(type(s) is str and s and s.strip()==s for s in ss),'Unique original monthly identities')
        days.append(d);names.append(ss.copy())
    return n,width,days,names,set(etfs)


def source_axes(panel,calendar,days):
    require(isinstance(panel,dict) and set(panel)=={'adj'},'Only original adjusted-close frame')
    p=panel['adj'];require(isinstance(p,pd.DataFrame) and isinstance(p.index,pd.DatetimeIndex),'Price frame')
    idx=p.index
    require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None
        and idx.equals(idx.normalize()),'Price date axis')
    require(p.columns.is_unique and all(type(s) is str and s and s.strip()==s for s in p.columns),'Price identity axis')
    require(isinstance(calendar,(list,tuple)) and (calendar or not days),'Official calendar required')
    sessions=[iso(x) for x in calendar]
    require(all(d.weekday()<5 for d in sessions) and all(a<b for a,b in zip(sessions,sessions[1:])),
            'Official calendar order')
    lookup={d:i for i,d in enumerate(sessions)};positions=[]
    for i,d in enumerate(days):
        require(d in lookup,'Query absent from calendar');t=lookup[d]
        if str(d.date())=='2016-08-17':require(i==0,'Named partial only first')
        else:require(t>=252 and t+1<len(sessions) and sessions[t+1].to_period('M')!=d.to_period('M'),
                     'Official month-end/endpoint history witness')
        positions.append(t)
    return p,sessions,positions


def finite(value,label):
    if not math.isfinite(value):raise FloatingPointError('Nonfinite '+label)
    return float(value)


def endpoint(frame,day,symbol):
    require(symbol in frame.columns and day in frame.index,'Missing mature endpoint')
    raw=frame.at[day,symbol]
    require(isinstance(raw,numbers.Real) and not isinstance(raw,(bool,np.bool_)),'Real nonboolean endpoint')
    try:v=float(raw)
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Invalid endpoint conversion') from exc
    require(math.isfinite(v) and v>0.,'Positive finite endpoint')
    return v


def median(values):
    require(len(values)>0 and all(math.isfinite(float(x)) for x in values),'Nonempty finite median inputs')
    ordered=sorted(float(x) for x in values);m=len(ordered)//2
    return ordered[m] if len(ordered)%2 else finite((ordered[m-1]+ordered[m])/2.,'even median')


def average_percentile(values):
    require(all(math.isfinite(float(x)) for x in values),'Finite rank domain')
    n=len(values);out=[]
    for value in values:
        less=sum(float(other)<float(value) for other in values)
        equal=sum(float(other)==float(value) for other in values)
        twice=2*less+equal+1
        out.append((float(twice)/2.)/float(n)-.5)
    return out


def mapping_check(row,rawday):
    original=row.get('original_sector');sector=row.get('sector')
    require(row.get('mapping_origin') in ('frozen_FOREIGN_override','query_date_sector_panel'),'Sector origin')
    if original is not None:
        require(type(original) is str and original in SECTORS,'Original sector identity')
        wanted=('VOX','full_period_communication_proxy',None) if original=='XLC' else (original,'unchanged',None)
    else:
        rule=REPAIRS.get(row['symbol'])
        if rule is not None and rawday>=rule[1] and (rule[2] is None or rawday<rule[2]):
            wanted=(rule[0],'dated_business_proxy',rule[3])
        else:wanted=(None,'unresolved',None)
    subset(row,dict(sector=wanted[0],mapping_repair=wanted[1],mapping_evidence=wanted[2]),'Sector repair provenance')
    return sector


def report_rows(report,days,names,etfs,sessions,positions):
    n=len(days)
    subset(report,dict(schema='peer239_mature_source_report_v1',status='ADMITTED',admitted=True,gaps=[],
        source_only=True,original_inventory_preserved=True,strict_pit=False,momentum_computed=False,
        rankings_computed=False,orders_computed=False,returns_computed=False,model_fits=0,
        original_failed_audit_preserved=True),'Mature source admission')
    require(isinstance(report.get('events'),list) and len(report['events'])>=n,'Source event prefix')
    require(isinstance(report.get('records'),list) and isinstance(report.get('counts'),dict),'Source records/count metadata')
    rows=[]
    for row in report['records']:
        require(isinstance(row,dict) and type(row.get('query_index')) is int and row['query_index']>=0,'Source record query index')
        if row['query_index']<n:
            require(type(row.get('column_index')) is int and row['column_index']>=0,'Source column index')
            rows.append(row)
    expected_ids=[(i,j) for i,ss in enumerate(names) for j,s in enumerate(ss) if s not in etfs]
    require([(r['query_index'],r.get('column_index')) for r in rows]==expected_ids,'Complete ordered direct-stock inventory')
    by_event=[]
    for i,(day,ss,t) in enumerate(zip(days,names,positions)):
        rawday=str(day.date());partial=rawday=='2016-08-17'
        stocks=[j for j,s in enumerate(ss) if s not in etfs]
        pp=[] if partial else [t-252,t-42];dd=[str(sessions[k].date()) for k in pp]
        group=[r for r in rows if r['query_index']==i]
        keys=[r.get('issuer_key') for r in group]
        require(all(type(k) is str and k and k.strip()==k for k in keys) and len(set(keys))==len(keys),
                'Missing/duplicate whole issuer is not admitted')
        for r in group:
            j=r['column_index'];subset(r,dict(query_index=i,column_index=j,date=rawday,symbol=ss[j],
                initial_partial=partial,source_positions=pp,source_dates=dd,gap_reasons=[]),'Source row identity/clock')
            sector=mapping_check(r,rawday)
            bound=BOUNDS.get(ss[j]);visible=bound if bound is not None and bound['issuer_key']==r['issuer_key'] and bound['available_from']<=rawday else None
            expected_bound=None if partial else visible
            status='initial_partial' if partial else ('official_pre_inception' if visible and dd[0]<visible['not_before'] else 'observed_full_window')
            subset(r,dict(history_status=status,history_bound=expected_bound),'Visible source history bound')
            points=r.get('endpoints');require(isinstance(points,list) and len(points)==(0 if partial else 2),'Endpoint record count')
            original_gaps=[];deferred=[]
            if sector is None:original_gaps.append('missing_sector')
            for point,role,d in zip(points,('security_start','security_end'),dd):
                subset(point,dict(role=role,symbol=ss[j],date=d),'Source endpoint binding')
                before=status=='official_pre_inception' and d<visible['not_before']
                if before:
                    require(point.get('status')=='missing' and point.get('reason') in
                        ('missing_value','missing_column','missing_row'),'Pre-bound price contamination/unclassified absence')
                    gap='endpoint:'+role+':'+point['reason'];original_gaps.append(gap);deferred.append(gap)
                else:subset(point,dict(status='valid',reason=None),'Mature/post-bound valid endpoint declaration')
            if status=='official_pre_inception' and sector is None:deferred.append('missing_sector')
            require(sector is not None or status=='official_pre_inception','Unknown sector outside explicit warmup')
            subset(r,dict(original_gap_reasons=sorted(original_gaps),deferred_gap_reasons=sorted(deferred)),
                   'Source gap/deferred evidence')
        mature=[r for r in group if r['history_status']=='observed_full_window']
        for r in group:
            peers=[]
            if r['history_status']=='observed_full_window':
                peers=[p for p in mature if p['sector']==r['sector'] and p['issuer_key']!=r['issuer_key']]
                peers.sort(key=lambda p:(p['issuer_key'],p['symbol'],p['column_index']))
            ids=[p['column_index'] for p in peers];count=len({p['issuer_key'] for p in peers})
            subset(r,dict(peer_indices=ids,peer_issuer_count=count,structural_peer_support=count>=2),
                   'Complete mature peer set')
        subset(report['events'][i],dict(query_index=i,date=rawday,initial_partial=partial,stock_indices=stocks,
            source_positions=pp,source_dates=dd,mature_indices=[r['column_index'] for r in mature],
            supported_indices=[r['column_index'] for r in group if r['structural_peer_support']]),'Source event')
        by_event.append(group)
    return by_event


def independent_features(panel,meta,calendar,arrays,source_report,limit=None):
    n,width,days,names,etfs=axes(meta,arrays,limit)
    frame,sessions,positions=source_axes(panel,calendar,days)
    groups=report_rows(source_report,days,names,etfs,sessions,positions)
    values={k:(np.zeros((n,width),bool) if k in ('mature','supported') else np.full((n,width),np.nan)) for k in VALUE_KEYS}
    events=[];stock_cells=0
    for i,(day,ss,t,group) in enumerate(zip(days,names,positions,groups)):
        rawday=str(day.date());partial=rawday=='2016-08-17'
        pp=[] if partial else [t-252,t-42];dd=[str(sessions[k].date()) for k in pp]
        stock_cells+=len(group);differences={}
        for r in group:
            j=r['column_index']
            if r['history_status']=='observed_full_window':
                start=endpoint(frame,sessions[pp[0]],ss[j]);end=endpoint(frame,sessions[pp[1]],ss[j])
                values['momentum'][i,j]=finite(end/start-1.,'endpoint momentum')
                values['mature'][i,j]=True
        supported=[r['column_index'] for r in group if r['structural_peer_support']]
        for r in group:
            j=r['column_index']
            if j in supported:
                med=median([values['momentum'][i,p] for p in r['peer_indices']])
                values['peer_median'][i,j]=med;values['supported'][i,j]=True
                differences[j]=finite(float(values['momentum'][i,j])-med,'peer difference')
        q=average_percentile([differences[j] for j in supported])
        c=average_percentile([values['momentum'][i,j] for j in supported])
        for j,a,b in zip(supported,q,c):values['peer_target'][i,j]=a;values['absolute_target'][i,j]=b
        cells=[]
        for r in group:
            j=r['column_index']
            def val(k):return float(values[k][i,j]) if math.isfinite(float(values[k][i,j])) else None
            cells.append(dict(column_index=j,symbol=ss[j],issuer_key=r['issuer_key'],sector=r['sector'],
                history_status=r['history_status'],peer_indices=r['peer_indices'].copy(),peer_issuer_count=r['peer_issuer_count'],
                momentum=val('momentum'),peer_median=val('peer_median'),peer_difference=differences.get(j),
                peer_target=val('peer_target'),absolute_target=val('absolute_target')))
        events.append(dict(query_index=i,date=rawday,source_dates=dd,source_positions=pp,
            stock_indices=[r['column_index'] for r in group],mature_indices=np.flatnonzero(values['mature'][i]).tolist(),
            supported_indices=supported,cells=cells))
    a=int(values['mature'].sum());b=int(values['supported'].sum())
    counts=dict(events=n,stock_cells=stock_cells,mature_cells=a,supported_cells=b,endpoint_observations=2*a,new_model_fits=0)
    return values,dict(schema='monthly_peer239_features_v1',counts=counts,events=events)


def verify_features(panel,meta,calendar,arrays,source_report,values,audit,limit=None):
    expected,ea=independent_features(panel,meta,calendar,arrays,source_report,limit)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Exactly six feature arrays')
    for k in VALUE_KEYS:exact_array(values[k],expected[k],'Independent '+k)
    record_check(audit,ea)
    counts=ea['counts']
    return dict(**{k:True for k in FEATURE_FLAGS},**counts,counts=counts,numerical_tolerance=0.,
                producer_modules_imported=False,computes_returns=False)


def selection_evidence(original,meta,values,n):
    _,width,days,names,etfs=axes(meta,original,n)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Selection feature keys')
    v={k:np.asarray(values[k])[:n] for k in VALUE_KEYS}
    for k in VALUE_KEYS:
        dtype=np.dtype(bool if k in ('mature','supported') else np.float64)
        require(v[k].shape==(n,width) and v[k].dtype==dtype,'Feature shape/dtype '+k)
    require(not (v['supported'] & ~v['mature']).any(),'Supported subset of mature')
    for k in VALUE_KEYS[2:]:
        mask=v['mature'] if k=='momentum' else v['supported']
        require(np.isfinite(v[k][mask]).all() and np.isnan(v[k][~mask]).all(),'Feature availability '+k)
    for i,(d,ss) in enumerate(zip(days,names)):
        forbidden=[j for j,s in enumerate(ss) if s in etfs or str(d.date())=='2016-08-17']
        require(not v['mature'][i,forbidden].any(),'ETF/partial no new features')
        ids=np.flatnonzero(v['supported'][i]).tolist()
        residual=[finite(float(v['momentum'][i,j])-float(v['peer_median'][i,j]),'peer residual') for j in ids]
        for k,raw in (('peer_target',residual),('absolute_target',[float(v['momentum'][i,j]) for j in ids])):
            expected=np.array(average_percentile(raw),np.float64)
            exact_array(v[k][i,ids],expected,'Canonical supported rank '+k)
    return v,width


def _selection(original,meta,values,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']),'Selection prefix')
    v,width=selection_evidence(original,meta,values,n)
    if n==0:
        empty={arm:np.empty((0,width)) for arm in ARMS}
        return empty,{arm:[] for arm in ARMS},{arm:x.copy() for arm,x in empty.items()},dict(
            minimum_remaining_support=None,maximum_budget_sum_error=0.,calm_nonweak_extension_count=0,
            maximum_absolute_delta=0.)
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    require(np.array_equal(x['cw'],np.broadcast_to(np.where(x['calm'],.50,.52)[:,None],(n,width))),
            'Original calm-dependent core coefficient')
    extension=np.asarray(original['extension_available'])[:n]
    forecasts=np.asarray(original['forecast_available'])[:n]
    ranklog=np.asarray(original['rank_log_excess'])[:n]
    require(extension.dtype==forecasts.dtype==np.dtype(bool) and extension.shape==forecasts.shape==ranklog.shape==(n,width),
            'Original extension-mask axes')
    active=(x['calm'] & ~x['weak'])[:,None] & extension & forecasts & x['eligible'] & np.isfinite(ranklog)
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    scores={};choices={};pres={};minimum=1.;sum_error=0.;max_delta=0.
    for arm,key in zip(ARMS,(None,'peer_target','absolute_target')):
        pre=x['local'].copy();delta=np.zeros((n,width));hand=np.zeros((n,width))
        if key is not None:
            good=np.isfinite(v[key]);hand[good]=.05
            delta[good]=.05*(v[key][good]-x['support'][good])
            require(np.isfinite(delta).all(),'Finite funded score delta')
            nonzero=delta!=0.;pre[nonzero]+=delta[nonzero]
        max_delta=max(max_delta,float(np.max(np.abs(delta))))
        require(np.isfinite(pre).all(),'Finite modified original pre')
        pres[arm]=pre.copy();score=pre.copy();rows=[];boundary_mask=np.zeros((n,width),bool)
        for i,names in enumerate(x['names']):
            order=sorted([j for j in range(width) if x['eligible'][i,j]],key=lambda j:(-pre[i,j],names[j]))
            if len(order)<3:rows.append(order if len(order)==2 else []);continue
            boundary=[j for j in order[2:5] if float(pre[i,order[2]])-float(pre[i,j])<=.02]
            for j in boundary:
                boundary_mask[i,j]=True
                target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
                score[i,j]+=budget[i]*(target-x['support'][i,j])
            winner=min(boundary,key=lambda j:(-score[i,j],names[j]))
            rows.append(sorted(order[:2]+[winner],key=lambda j:(-score[i,j],names[j])))
        require(np.isfinite(score).all(),'Finite final scores')
        for i in range(n):
            for j in range(width):
                weights=[float(x['cw'][i,j]),.025*float(x['calm'][i]),.05*float(x['weak'][i]),
                    .01*float(active[i,j]),float(budget[i])*float(boundary_mask[i,j]),float(hand[i,j])]
                remaining=1.-weights[0]
                for w in weights[1:]:remaining-=w
                error=abs(math.fsum(weights+[remaining])-1.)
                require(remaining>=.33-1e-12 and error<=1e-12,'Complete Local67 budget including calm nonweak extension')
                minimum=min(minimum,remaining);sum_error=max(sum_error,error)
        scores[arm]=score;choices[arm]=rows
    return scores,choices,pres,dict(minimum_remaining_support=minimum,maximum_budget_sum_error=sum_error,
        calm_nonweak_extension_count=int(active.sum()),maximum_absolute_delta=max_delta)


def selection_oracle(original,meta,values,n=None):
    scores,choices,pres,_=_selection(original,meta,values,n)
    return scores,choices,pres


def verify_selection(original,meta,values,orders,*,scores=None,pre_scores=None,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,budget=_selection(original,meta,values,n)
    require(isinstance(orders,dict) and set(orders)==set(ARMS),'Three order arms')
    wanted={arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(expected[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders,wanted,'Independent ordered selection')
    for supplied,reference,label in ((scores,expected,'post-episode score'),(pre_scores,pres,'modified pre-score')):
        if supplied is not None:
            require(isinstance(supplied,dict) and set(supplied)==set(ARMS),'Three '+label+' arms')
            for arm in ARMS:exact_array(supplied[arm],reference[arm],label+'/'+arm)
    return dict(**{k:True for k in SELECTION_FLAGS},**budget,checked_events=n,ordered_choices_checked=3*n,
        full_scores_supplied=scores is not None,full_pre_scores_supplied=pre_scores is not None,
        numerical_tolerance=0.,original_source_parity_tolerance=1e-12,
        producer_modules_imported=False,computes_returns=False,
        holdings_counts={arm:{str(k):sum(len(row)==k for row in choices[arm]) for k in (0,2,3)} for arm in ARMS})


def audit_outcomes(px,opening,orders,exports,results,meta):
    """Nine saved paths and 45 slices via independent daily ledger accounting."""
    helper=accounting()
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arms')
    require(len(meta['dates'])==len(meta['ends'])==122 and meta['ends'][:-1]==meta['dates'][1:],'Outcome interval axis')
    require(meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24' and
        meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30',
        'Fixed five outcome slices')
    reports={}
    for arm in ARMS:
        require(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and
                [o['end'] for o in orders[arm]]==meta['ends'],'Order axis')
        require(set(exports[arm])==set(results[arm])=={p[0] for p in PATHS},'Three execution paths')
        reports[arm]={}
        for name,execution,cost in PATHS:
            data=exports[arm][name]
            require(set(data)=={'DAILY','TRADES','HOLDINGS','DAILY_POSITIONS'},'Four execution exports')
            audited=helper.audit_path(px,opening,orders[arm],*[data[k] for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')],execution,cost)
            slices=helper.statistics_by_split(audited['days'],audited['values'],meta);wanted=results[arm][name]
            require(set(wanted['slices'])==set(slices),'Fixed slices')
            helper.check_metrics(slices['full122'],wanted['metrics'],arm+'/'+name)
            for split,value in slices.items():helper.check_metrics(value,wanted['slices'][split],arm+'/'+name+'/'+split)
            for key in ('fees_paid','dollar_turnover','trade_rows'):helper.near(audited[key],wanted['metrics'][key],key)
            annual=[]
            for year in sorted({d[:4] for d in audited['days']}):
                ids=[i for i,day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids)<2:continue
                prior=audited['values'][ids[0]-1] if ids[0] else 100000.
                annual.append(dict(year=int(year),return_=audited['values'][ids[-1]]/prior-1.,partial=int(year) in (2016,2026)))
            require(len(wanted['metrics']['annual_returns'])==len(annual),'Annual count')
            for got,expected in zip(wanted['metrics']['annual_returns'],annual):
                require(set(got)==set(expected) and got['year']==expected['year'] and got['partial']==expected['partial'],'Annual identity')
                helper.near(got['return_'],expected['return_'],'Annual return')
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False,'Account declaration')
            for key in ('days','values','daily_cash_nav'):audited.pop(key)
            reports[arm][name]=dict(audited,passed=True,slices=slices)
    return dict(passed=True,checked_paths=9,checked_slices=45,independent_accounts=True,
        independent_of_production_model_and_engine=True,terminal_liquidation=False,
        accounting_source_sha256=ACCOUNTING_SHA,paths=reports)
