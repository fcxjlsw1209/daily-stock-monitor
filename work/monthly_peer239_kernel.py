"""Pure skipped long-momentum peer-median feature and common-domain control.

Consumes an admitted source report. No source loading, original-E filtering,
model, selection, order, portfolio or return-path calculation.
"""
from pathlib import Path
from numbers import Real
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import monthly_relative_gate235_kernel as axes

VALUE_KEYS=('mature','supported','momentum','peer_median','peer_target','absolute_target')
COUNT_FIELDS=('events','stock_cells','mature_cells','supported_cells','endpoint_observations','new_model_fits')
EVENT_KEYS=('query_index','date','source_dates','source_positions','stock_indices','mature_indices','supported_indices','cells')
CELL_KEYS=('column_index','symbol','issuer_key','sector','history_status','peer_indices','peer_issuer_count','momentum','peer_median','peer_difference','peer_target','absolute_target')
SOURCE_EVENT_KEYS={'query_index','date','initial_partial','stock_indices','source_positions','source_dates','mature_indices','supported_indices'}
SOURCE_RECORD_KEYS={'query_index','column_index','date','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence','peer_indices','peer_issuer_count','structural_peer_support','initial_partial','source_positions','source_dates','endpoints','gap_reasons','original_gap_reasons','history_status','history_bound','deferred_gap_reasons'}
BOUNDS={
    'SPCX':dict(symbol='SPCX',issuer_key='https://companiesmarketcap.com/spacex/marketcap/',not_before='2026-06-08',available_from='2026-06-09',kind='exchange_new_listing_lower_bound',evidence_id='spcx_nasdaq_new_issue'),
    'SKHY':dict(symbol='SKHY',issuer_key='https://companiesmarketcap.com/sk-hynix/marketcap/',not_before='2026-07-10',available_from='2026-07-11',kind='issuer_actual_adr_start',evidence_id='skhy_adr_listing')}
SECTORS={'XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','VOX'}
MISSING_REASONS={'missing_value','missing_row','missing_column'}


def need(ok,message):
    if not ok:raise ValueError(message)


def ident(value):
    return isinstance(value,str) and bool(value) and value.strip()==value


def _input(panel,meta,calendar,arrays,limit):
    frame=axes._frame(panel);sessions=axes._calendar(calendar)
    need(isinstance(arrays,dict) and 'eligible' in arrays,'Original eligibility axes required')
    e=np.asarray(arrays['eligible']);need(e.ndim==2,'Two eligibility axes required')
    total,width=e.shape
    need(width>0 or total==0,'Nonempty security axis for consumed events')
    need(limit is None or type(limit) is int and 0<=limit<=total,'Invalid event prefix')
    n=total if limit is None else limit
    # Values are checked only for structural validity; E never defines the domain.
    axes._booleans(e,n,'eligible')
    need(isinstance(meta,dict) and all(isinstance(meta.get(k),list) and len(meta[k])==total for k in ('dates','symbols')),'Metadata event axes differ')
    etfs=meta.get('etfs');need(isinstance(etfs,(list,tuple)) and all(ident(s) for s in etfs) and len(set(etfs))==len(etfs),'Explicit ETF identities required')
    lookup={day:i for i,day in enumerate(sessions)};queries=[]
    for i in range(n):
        day=axes._day(meta['dates'][i],'Query');names=meta['symbols'][i]
        need(day in lookup and (not queries or day>queries[-1][0]),'Increasing official queries required')
        need(isinstance(names,list) and len(names)==width and all(ident(s) for s in names) and len(set(names))==width,'Query symbols differ')
        partial=str(day.date())=='2016-08-17';need(not partial or i==0,'Named partial must be first')
        t=lookup[day]
        if not partial:need(t>=252 and t+1<len(sessions) and sessions[t+1].to_period('M')!=day.to_period('M'),'Official endpoint history/month-end witness missing')
        queries.append((day,names,t,partial))
    return frame,sessions,queries,set(etfs),width


def _source_top(report,n):
    need(isinstance(report,dict) and report.get('schema')=='peer239_mature_source_report_v1' and report.get('status')=='ADMITTED' and report.get('admitted') is True,'Admitted mature source report required')
    need(report.get('gaps')==[],'No current source gaps permitted')
    for k in ('source_only','original_inventory_preserved','original_failed_audit_preserved'):need(report.get(k) is True,'Missing source invariant '+k)
    for k in ('momentum_computed','rankings_computed','orders_computed','returns_computed','strict_pit'):need(report.get(k) is False,'Source scope differs '+k)
    need(type(report.get('model_fits')) is int and report['model_fits']==0,'No source models')
    need(isinstance(report.get('events'),list) and len(report['events'])>=n and isinstance(report.get('records'),list) and isinstance(report.get('counts'),dict),'Source axes/count metadata required')
    by_query={i:[] for i in range(n)}
    for r in report['records']:
        need(isinstance(r,dict) and type(r.get('query_index')) is int and r['query_index']>=0,'Canonical record query index')
        if r['query_index']>=n:continue
        by_query[r['query_index']].append(r)
    return by_query


def _indices(value,label):
    need(isinstance(value,list) and all(type(x) is int and x>=0 for x in value) and len(set(value))==len(value),'Canonical index list '+label)


def _gap_list(value,label):
    need(isinstance(value,list) and all(ident(s) for s in value) and value==sorted(set(value)),'Canonical '+label)


def _record(r,i,j,day,names,partial,positions,dates):
    need(set(r)==SOURCE_RECORD_KEYS and type(r['column_index']) is int and r['column_index']==j and r['query_index']==i,'Exact record schema/position')
    need(r['date']==day and r['symbol']==names[j] and ident(r['issuer_key']),'Record security/issuer identity')
    need(type(r['initial_partial']) is bool and r['initial_partial']==partial and r['source_dates']==dates and r['source_positions']==positions,'Record official clock')
    need(r['gap_reasons']==[],'Required record has source gaps')
    _indices(r['peer_indices'],'peer_indices');_indices(r['source_positions'],'source_positions')
    _gap_list(r['original_gap_reasons'],'original gap reasons');_gap_list(r['deferred_gap_reasons'],'deferred gap reasons')
    need(r['original_sector'] is None or r['original_sector'] in SECTORS,'Original sector identity')
    need(r['sector'] is None or r['sector'] in SECTORS,'Sector identity')
    need(r['mapping_origin'] in ('frozen_FOREIGN_override','query_date_sector_panel'),'Mapping origin')
    repair=r['mapping_repair'];need(repair in ('unchanged','full_period_communication_proxy','dated_business_proxy','unresolved'),'Mapping repair kind')
    if repair=='unchanged':need(r['sector']==r['original_sector'] and r['sector'] is not None and r['mapping_evidence'] is None,'Unchanged mapping differs')
    elif repair=='full_period_communication_proxy':need(r['original_sector']=='XLC' and r['sector']=='VOX' and r['mapping_evidence'] is None,'Communication mapping differs')
    elif repair=='unresolved':need(r['original_sector'] is None and r['sector'] is None and r['mapping_evidence'] is None,'Unresolved mapping differs')
    else:need(r['original_sector'] is None and r['sector'] is not None and isinstance(r['mapping_evidence'],list) and r['mapping_evidence'] and all(ident(v) for v in r['mapping_evidence']),'Dated mapping evidence required')
    b=r['history_bound'];wanted=BOUNDS.get(r['symbol'])
    visible=wanted if wanted and wanted['issuer_key']==r['issuer_key'] and wanted['available_from']<=day else None
    need(b==(None if partial else visible),'Exact visible admitted history bound required')
    status='initial_partial' if partial else 'official_pre_inception' if b is not None and dates[0]<b['not_before'] else 'observed_full_window'
    need(r['history_status']==status,'History classification differs')
    points=r['endpoints'];need(isinstance(points,list) and len(points)==(0 if partial else 2),'Exact endpoint declarations')
    for p,role,date in zip(points,('security_start','security_end'),dates):
        need(set(p)=={'role','symbol','date','status','reason'} and p['role']==role and p['symbol']==names[j] and p['date']==date,'Endpoint role/identity/date')
        need(p['status'] in ('valid','missing','invalid'),'Endpoint status')
        need((p['status']=='valid' and p['reason'] is None) or (p['status']!='valid' and ident(p['reason'])),'Endpoint reason')
    if status=='official_pre_inception':
        deferred=[]
        for p in points:
            if p['date']<b['not_before']:
                need(p['status']=='missing' and p['reason'] in MISSING_REASONS,'Pre-bound endpoint must be declared missing, never filled')
                deferred.append('endpoint:'+p['role']+':'+p['reason'])
            else:need(p['status']=='valid','Post-bound endpoint gap cannot be deferred')
        if r['sector'] is None:deferred.append('missing_sector')
        need(r['deferred_gap_reasons']==sorted(deferred) and r['original_gap_reasons']==sorted(deferred),'Only documented warmup source gaps may be deferred')
    else:
        need(r['sector'] is not None and r['deferred_gap_reasons']==[] and r['original_gap_reasons']==[],'Ordinary identity/sector gaps cannot disappear')
        need(all(p['status']=='valid' for p in points),'Mature endpoints must be declared valid')
    return status


def _momentum(frame,symbol,start,end):
    need(symbol in frame.columns and start in frame.index and end in frame.index,'Missing mature endpoint source')
    first=axes._price(frame.at[start,symbol],symbol,start);last=axes._price(frame.at[end,symbol],symbol,end)
    result=float(last)/float(first)-1.0
    if not math.isfinite(result):raise FloatingPointError('Nonfinite derived momentum')
    return result


def _median(values):
    ordered=sorted(values);n=len(ordered)
    result=float(ordered[n//2]) if n%2 else (float(ordered[n//2-1])+float(ordered[n//2]))/2.0
    if not math.isfinite(result):raise FloatingPointError('Nonfinite derived median')
    return result


def _rank(value,domain):
    less=sum(x<value for x in domain);equal=sum(x==value for x in domain)
    twice=2*less+equal+1
    result=(float(twice)/2.0)/float(len(domain))-.5
    if not math.isfinite(result):raise FloatingPointError('Nonfinite derived rank')
    return result


def build_features(panel,meta,calendar,arrays,source_report,limit=None):
    frame,sessions,queries,etfs,width=_input(panel,meta,calendar,arrays,limit);n=len(queries)
    source=_source_top(source_report,n)
    values={k:np.zeros((n,width),dtype=bool) if k in ('mature','supported') else np.full((n,width),np.nan,dtype=np.float64) for k in VALUE_KEYS}
    events=[];stock_count=0;mature_count=0;supported_count=0
    for i,(day,names,t,partial) in enumerate(queries):
        d=str(day.date());stock=[j for j,s in enumerate(names) if s not in etfs];positions=[] if partial else [t-252,t-42]
        dates=[str(sessions[k].date()) for k in positions];ev=source_report['events'][i]
        need(isinstance(ev,dict) and set(ev)==SOURCE_EVENT_KEYS and type(ev['query_index']) is int and ev['query_index']==i and ev['date']==d and type(ev['initial_partial']) is bool and ev['initial_partial']==partial,'Source event identity/schema')
        for key in ('stock_indices','mature_indices','supported_indices','source_positions'):_indices(ev[key],key)
        need(ev['stock_indices']==stock and ev['source_positions']==positions and ev['source_dates']==dates,'Source full inventory/official clocks')
        rr=source[i];need([r.get('column_index') for r in rr]==stock,'Exact ordered direct-stock records; no missing or extra cell')
        status={j:_record(r,i,j,d,names,partial,positions,dates) for j,r in zip(stock,rr)}
        issuers=[r['issuer_key'] for r in rr];need(len(issuers)==len(set(issuers)),'Duplicate whole issuer in inventory')
        mapping={r['column_index']:r for r in rr};mature=[j for j in stock if status[j]=='observed_full_window'];supported=[]
        for j in stock:
            r=mapping[j]
            peers=[k for k in mature if j in mature and mapping[k]['sector']==r['sector'] and mapping[k]['issuer_key']!=r['issuer_key']]
            peers.sort(key=lambda k:(mapping[k]['issuer_key'],names[k],k))
            count=len({mapping[k]['issuer_key'] for k in peers});good=j in mature and count>=2
            need(r['peer_indices']==peers and type(r['peer_issuer_count']) is int and r['peer_issuer_count']==count and type(r['structural_peer_support']) is bool and r['structural_peer_support']==good,'Complete mature cohort/support differs')
            if good:supported.append(j)
        need(ev['mature_indices']==mature and ev['supported_indices']==supported,'Source mature/supported indices differ')
        for j in mature:
            values['mature'][i,j]=True;values['momentum'][i,j]=_momentum(frame,names[j],sessions[t-252],sessions[t-42])
        differences={}
        for j in supported:
            values['supported'][i,j]=True;median=_median([float(values['momentum'][i,k]) for k in mapping[j]['peer_indices']])
            values['peer_median'][i,j]=median;diff=float(values['momentum'][i,j])-float(median)
            if not math.isfinite(diff):raise FloatingPointError('Nonfinite derived peer difference')
            differences[j]=diff
        relative=[differences[j] for j in supported];absolute=[float(values['momentum'][i,j]) for j in supported]
        for j in supported:
            values['peer_target'][i,j]=_rank(differences[j],relative);values['absolute_target'][i,j]=_rank(float(values['momentum'][i,j]),absolute)
        cells=[]
        for j in stock:
            r=mapping[j];cell=dict(column_index=j,symbol=names[j],issuer_key=r['issuer_key'],sector=r['sector'],history_status=status[j],peer_indices=list(r['peer_indices']),peer_issuer_count=r['peer_issuer_count'],momentum=float(values['momentum'][i,j]) if j in mature else None,peer_median=float(values['peer_median'][i,j]) if j in supported else None,peer_difference=differences.get(j),peer_target=float(values['peer_target'][i,j]) if j in supported else None,absolute_target=float(values['absolute_target'][i,j]) if j in supported else None)
            need(tuple(cell)==CELL_KEYS,'Cell audit schema');cells.append(cell)
        event=dict(query_index=i,date=d,source_dates=dates,source_positions=positions,stock_indices=stock,mature_indices=mature,supported_indices=supported,cells=cells)
        need(tuple(event)==EVENT_KEYS,'Event audit schema');events.append(event)
        stock_count+=len(stock);mature_count+=len(mature);supported_count+=len(supported)
    counts=dict(events=n,stock_cells=stock_count,mature_cells=mature_count,supported_cells=supported_count,endpoint_observations=2*mature_count,new_model_fits=0)
    need(tuple(counts)==COUNT_FIELDS,'Count audit schema')
    return values,dict(schema='monthly_peer239_features_v1',counts=counts,events=events)


def dependency_files():return [Path(axes.__file__).resolve()]
