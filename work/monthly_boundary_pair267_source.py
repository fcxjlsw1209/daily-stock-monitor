"""267 caller-supplied source adapter: no file IO, NPZ decoder or model fits.

The host authenticates the complete169/Local67/261 files, official calendar,
original study callable and64raw-source panel before calling this module.
Numerical comparisons here are bitwise, not the old reconstruction tolerance.
Full-file hashing/decoding is host work; counters describe only consumed values.
The adapter selects original TopK and supplies prices, but computes no returns.
"""
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
import numbers
import numpy as np
import pandas as pd

INITIAL_PARTIAL='2016-08-17'
CALENDAR_PATH='outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA256='8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
TRAIN_KEYS=('train_plain','event_plain','train_eligible','eligible','offsets')
PREPARED_KEYS=('schema','n','width','months','events','audit','binding')
WORK_FIELDS=('original_pre_score_values_read','original_eligibility_values_read',
 'training_feature_values_read','query_feature_values_read','feature_pairs_compared',
 'eligibility_pairs_compared','selected_signal_prices_read','selected_end_prices_read',
 'selected_price_slots_missing','selected_price_slots_invalid','original_effect_values_read',
 'projected_month_records_checked','projected_event_records_checked')
_ISSUER=object()

def need(ok,message):
    if not ok:raise ValueError(message)

def _day(value,label):
    need(type(value) is str,label+' ISO string')
    try:out=date.fromisoformat(value)
    except (TypeError,ValueError) as exc:raise ValueError(label+' valid date') from exc
    need(out.isoformat()==value,label+' canonical ISO date');return out

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _encoded(value):
    if isinstance(value,np.ndarray):
        need(value.dtype.kind in 'bifu','Numeric array evidence')
        return dict(dtype=value.dtype.str,shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if type(value) is dict:return {k:_encoded(v) for k,v in value.items()}
    if type(value) in (list,tuple):return [_encoded(v) for v in value]
    if isinstance(value,(float,np.floating)):return {'float_hex':float(value).hex()}
    if value is None or type(value) in (str,int,bool):return value
    raise ValueError('Unsupported source evidence scalar')

def _record_hash(value):return _digest(_encoded(value))

def _matrix(value,shape,dtype,label):
    need(isinstance(value,np.ndarray) and value.shape==shape and value.dtype==np.dtype(dtype),label+' shape/dtype')
    return value

def _bits(a,b,label):
    need(isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype,label+' shape/dtype')
    need(a.tobytes()==b.tobytes(),label+' exact bits')

def _prefix_meta(meta,n):
    need(type(meta) is dict and all(type(meta.get(k)) is list for k in ('dates','ends','symbols')),'Monthly metadata')
    total=len(meta['dates']);need(type(n) is int and 0<=n<=total,'Native prefix integer')
    need(len(meta['ends'])==len(meta['symbols'])==total,'Monthly metadata axes')
    out={k:deepcopy(meta[k][:n]) for k in ('dates','ends','symbols')}
    if 'etfs' in meta:out['etfs']=deepcopy(meta['etfs'])
    last=None
    for d,e,ss in zip(out['dates'],out['ends'],out['symbols']):
        need(_day(e,'End')>_day(d,'Signal') and (last is None or d>last),'Ordered signal/end metadata');last=d
        need(type(ss) is list and ss and all(type(s) is str and s and s.strip()==s for s in ss)
             and ss==sorted(set(ss)),'Sorted unique original symbols')
    return out

def build_join(combined_meta,sessions,*,expected_overlap=None):
    """Metadata only; the host supplies SHA-authenticated official sessions.

    All113 metadata rows may be checked at once. Their numerical bodies are not
    present. Formal signal/end monthends need a next-session witness.
    """
    need(type(combined_meta) is dict and {'monthly','training'}<=set(combined_meta),'Combined metadata')
    m,t=combined_meta['monthly'],combined_meta['training']
    mm=_prefix_meta(m,len(m['dates']));tm=_prefix_meta(t,len(t['dates']))
    need(type(sessions) in (list,tuple) and len(sessions)>1,'Official session metadata')
    ds=list(sessions);need(all(_day(d,'Session').weekday()<5 for d in ds) and all(a<b for a,b in zip(ds,ds[1:])),'Ordered unique official weekdays')
    pos={d:i for i,d in enumerate(ds)}
    width=len(mm['symbols'][0]);need(all(len(ss)==width for ss in mm['symbols']),'Fixed monthly width')
    offsets=[0]
    for ss in tm['symbols']:offsets.append(offsets[-1]+len(ss))
    bysignal={d:i for i,d in enumerate(tm['dates'])};rows=[]
    for qi,(signal,end,ss) in enumerate(zip(mm['dates'],mm['ends'],mm['symbols'])):
        need(signal in pos,'Original query is official session')
        if signal==INITIAL_PARTIAL:continue
        p=pos[signal];need(p+1<len(ds) and ds[p+1][:7]!=signal[:7],'Original formal monthend witness')
        if signal not in bysignal:continue
        ti=bysignal[signal];need(tm['ends'][ti]==end,'Shared signal has conflicting maturity')
        need(tm['symbols'][ti]==ss,'Shared original identities/order')
        s,e=_day(signal,'Signal'),_day(end,'End')
        need((e.year,e.month)==(s.year+(s.month==12),s.month%12+1),'Next natural-month label')
        need(end in pos and pos[end]+1<len(ds) and ds[pos[end]+1][:7]!=end[:7],'Official endpoint monthend witness')
        rows.append(dict(source_index=len(rows),query_index=qi,training_index=ti,signal=signal,end=end,
                         symbols=list(ss),training_offset=offsets[ti],event_offset=qi*width))
    if expected_overlap is not None:need(type(expected_overlap) is int and len(rows)==expected_overlap,'Expected joined months')
    out=dict(schema='monthly_boundary_pair267_join_v1',monthly=mm,training=tm,rows=rows,
             width=width,events=len(mm['dates']),training_rows=offsets[-1],offsets=offsets,
             calendar_sha256=_digest(ds),metadata_sha256=_digest(combined_meta),
             metadata_only=True,actual_pair_validity='UNKNOWN',source_files_authenticated=False)
    out['join_sha256']=_digest(out);return out

def original_prefix(study,arrays,monthly,reference,n=None):
    """Compute fixed Local67 P on physical-small copies; exact reference gate.

    reference is host-authenticated {pre_scores,eligible,meta}. This function
    does not authenticate caller file names or mint a manifest certificate.
    """
    total=len(monthly['dates']);n=total if n is None else n;mm=_prefix_meta(monthly,n)
    need(type(reference) is dict and {'pre_scores','eligible','meta'}<=set(reference),'Original reference payload')
    need(_prefix_meta(reference['meta'],n)==mm,'Original reference identity prefix')
    shape=np.shape(arrays['baseline']);need(len(shape)==2 and shape[0]>=n,'Original matrix axes')
    width=shape[1];part={}
    for k,v in arrays.items():
        need(isinstance(v,np.ndarray) and v.ndim>=1 and len(v)>=n,'Original array prefix '+k)
        part[k]=v[:n].copy()
    before={k:_record_hash(v) for k,v in part.items()}
    spec=study.configs()[4]
    need(spec==dict(id=4,base='incumbent_scope67',mode='local_linear'),'Fixed Local67 config4')
    pre=study.modified_pre_score(spec,part,n)
    _matrix(pre,(n,width),'float64','Full Local67 P')
    refp=reference['pre_scores'];refe=reference['eligible']
    need(isinstance(refp,np.ndarray) and refp.ndim==2 and refp.shape[0]>=n and refp.shape[1]==width,'Reference P shape')
    need(isinstance(refe,np.ndarray) and refe.ndim==2 and refe.shape[0]>=n and refe.shape[1]==width,'Reference E shape')
    _bits(pre,refp[:n],'Full Local67 P/reference')
    _matrix(part['eligible'],(n,width),'bool','Original E');_bits(part['eligible'],refe[:n],'Original E/reference')
    need(before=={k:_record_hash(v) for k,v in part.items()},'Original source arrays mutated')
    return pre.copy(),part['eligible'].copy(),dict(passed=True,events=n,width=width,full_local67_p_exact=True,
        upstream_pre_score_substitution=False,original_eligibility_exact=True,source_arrays_unchanged=True,
        source_numeric_cells_compared=2*n*width,original_selector_calls=0,model_fits=0,
        source_file_authentication='Host immutable manifest gate required',prefix_only_math=True)

def _topk(p,e,ss):
    ids=[]
    for j in range(len(ss)):
        if e[j]:need(math.isfinite(float(p[j])),'Finite original eligible P');ids.append(j)
    ids.sort(key=lambda j:(-float(p[j]),ss[j]));return ids[:5],len(ids)

def _mature(headers,years):
    return [i for i,(s,e) in enumerate(headers) if any('%04d-12-31'%(y-6)<s and e<='%04d-12-31'%(y-1) for y in years)]

def _price(adj,day,symbol,work,side):
    if day not in adj.index or symbol not in adj.columns:
        work['selected_price_slots_missing']+=1;return None
    work['selected_'+side+'_prices_read']+=1
    value=adj.at[day,symbol]
    if value is None or value is pd.NA:
        work['selected_price_slots_missing']+=1;return None
    # Match real price types before recognizing NaN; NaT/complex/bool are not missing.
    need(isinstance(value,numbers.Real) and not isinstance(value,(bool,np.bool_)),'Endpoint real nonbool type')
    x=float(value)
    if math.isnan(x):work['selected_price_slots_missing']+=1;return x
    if not math.isfinite(x) or x<=0.:work['selected_price_slots_invalid']+=1
    return x

class _Binding:
    __slots__=('issuer','n','width','headers','query_dates','month_hashes','event_hashes','anchor_sha256')
    def __init__(self,n,width,headers,dates,mh,eh,anchor):
        for k,v in zip(self.__slots__,(_ISSUER,n,width,tuple(headers),tuple(dates),tuple(mh),tuple(eh),anchor)):
            object.__setattr__(self,k,v)
    def __setattr__(self,*args):raise AttributeError('Immutable issued source binding')
    def __deepcopy__(self,memo):raise TypeError('Source binding cannot be copied or reconstructed from JSON')

def _audit(n,width,years,ids,work,mode,anchor=None):
    return dict(schema='monthly_boundary_pair267_source_audit_v1',events=n,width=width,consumed_years=years,
        consumed_month_indices=ids,counts=work,mode=mode,labels_computed=False,models_fitted=0,orders_computed=False,
        portfolio_returns_computed=False,source_files_authenticated=False,npz_decoded=False,
        whole_file_hashing_and_decoding='Host external scope; this module only consumes caller-supplied values',
        canonical_label_price_source="panel['adj']",train_query_features_bitwise=True,
        original_eligibility_bitwise=True,month_gap_filter=False,source_anchor_sha256=anchor)

def prepare_inputs(join,train,arrays,pre_scores,panel,n=None):
    """Fresh physical source consumption; no labels, fitting or future-body scans.

    Mature month feature matrices are compared in full (including NaN payloads
    and signed zero), then only selected TopK endpoints are requested. Missing
    prices remain missing for the kernel's common-month rejection, never filled.
    """
    need(_digest({k:v for k,v in join.items() if k!='join_sha256'})==join['join_sha256'],'Metadata join checksum')
    total,width=join['events'],join['width'];n=total if n is None else n
    need(type(n) is int and 0<=n<=total,'Source native prefix integer')
    dates=join['monthly']['dates'][:n];years=sorted({int(d[:4]) for d in dates if d!=INITIAL_PARTIAL})
    headers=[(r['signal'],r['end']) for r in join['rows']];ids=_mature(headers,years)
    # No ordinary queries: no training keys or panel axes/values are accessed.
    work=dict.fromkeys(WORK_FIELDS,0);months=[dict(signal=s,end=e) for s,e in headers]
    events=[];query_features={}
    _matrix(pre_scores,(total,width),'float64','Original P') if len(pre_scores)==total else _matrix(pre_scores,(n,width),'float64','Physical original P')
    E=arrays['eligible'];effects=arrays['effects']
    need(isinstance(E,np.ndarray) and E.ndim==2 and E.shape[0]>=n and E.shape[1]==width and E.dtype==np.dtype(bool),'Original E prefix')
    need(isinstance(effects,np.ndarray) and effects.ndim==2 and effects.shape[0]>=n and effects.shape[1]==width and effects.dtype==np.dtype('float64'),'Original effects prefix')
    for i in range(n):
        p,e=pre_scores[i].copy(),E[i].copy();work['original_pre_score_values_read']+=width;work['original_eligibility_values_read']+=width
        effects_row=effects[i].copy();work['original_effect_values_read']+=width
        events.append(dict(signal=dates[i],symbols=list(join['monthly']['symbols'][i]),pre_scores=p,eligible=e,
                           features=[None]*width,effects=effects_row))
    if years:
        ep=_matrix(train['event_plain'],(total*width,10),'float64','Query plain10')
        ce=_matrix(train['eligible'],(total,width),'bool','Cached query E')
        for i,d in enumerate(dates):
            if d==INITIAL_PARTIAL:continue
            _bits(E[i],ce[i],'Original/cached query E');work['eligibility_pairs_compared']+=width
            query_features[i]=ep[i*width:(i+1)*width].copy();work['query_feature_values_read']+=10*width
            events[i]['features']=query_features[i].copy()
    if ids:
        tp=_matrix(train['train_plain'],(join['training_rows'],10),'float64','Training plain10')
        te=_matrix(train['train_eligible'],(join['training_rows'],),'bool','Training eligibility')
        off=train['offsets'];need(isinstance(off,np.ndarray),'Offset array metadata')
        off=_matrix(off,(len(join['offsets']),),off.dtype,'Offset metadata')
        need(off.dtype.kind in 'iu','Integer offset metadata')
        need(type(panel) is dict and set(panel)=={'adj'} and isinstance(panel['adj'],pd.DataFrame),'Only original panel adj')
        adj=panel['adj'];need(isinstance(adj.index,pd.DatetimeIndex) and adj.index.is_unique and adj.index.is_monotonic_increasing
            and adj.index.tz is None and adj.index.equals(adj.index.normalize()) and adj.columns.is_unique,'Unique ordered adjusted-price axes')
        for source_index in ids:
            row=join['rows'][source_index];qi,ti,start=row['query_index'],row['training_index'],row['training_offset']
            need(qi<n,'Mature historical query inside consumed prefix')
            need(int(off[ti])==start and int(off[ti+1])==start+width,'Original global row offsets')
            x=tp[start:start+width].copy();g=te[start:start+width].copy()
            work['training_feature_values_read']+=10*width
            _bits(x,query_features[qi],'Training/query plain10');_bits(g,E[qi],'Training/query eligibility')
            work['feature_pairs_compared']+=10*width;work['eligibility_pairs_compared']+=width
            p,e=events[qi]['pre_scores'],events[qi]['eligible'];ss=row['symbols'];chosen,count=_topk(p,e,ss)
            signal_prices=[None]*width;end_prices=[None]*width
            if count>=4:
                for j in chosen:
                    signal_prices[j]=_price(adj,pd.Timestamp(row['signal']),ss[j],work,'signal')
                    end_prices[j]=_price(adj,pd.Timestamp(row['end']),ss[j],work,'end')
            months[source_index].update(symbols=list(ss),pre_scores=p.copy(),eligible=e.copy(),features=x,
                signal_prices=signal_prices,end_prices=end_prices,
                source_query_index=qi,source_training_index=ti,source_training_row_ids=list(range(start,start+width)),
                selected_indices=chosen,eligible_count=count)
    mh=[_record_hash(row) for row in months];eh=[_record_hash(row) for row in events]
    anchor=_digest(dict(n=n,width=width,join_sha256=join['join_sha256'],month_hashes=mh,event_hashes=eh))
    binding=_Binding(n,width,headers,dates,mh,eh,anchor)
    audit=_audit(n,width,years,ids,work,'fresh_physical_source',anchor)
    return dict(schema='monthly_boundary_pair267_prepared_source_v1',n=n,width=width,months=months,events=events,audit=audit,binding=binding)

def project_inputs(prepared,n):
    """Cached prefix: inspect/hash only selected event/month bodies, never future.

    This verifies a previously issued numerical binding, not source files again.
    Full serialization of the prepared input is deliberately not performed.
    """
    b=prepared['binding'];need(type(b) is _Binding and b.issuer is _ISSUER,'Issued source binding required')
    need(prepared['schema']=='monthly_boundary_pair267_prepared_source_v1' and prepared['width']==b.width,'Source projection schema')
    need(type(n) is int and 0<=n<=b.n,'Cached native prefix integer')
    years=sorted({int(d[:4]) for d in b.query_dates[:n] if d!=INITIAL_PARTIAL});ids=_mature(b.headers,years)
    need(len(prepared['months'])==len(b.headers) and len(prepared['events'])>=n,'Source record layout')
    months=[dict(signal=s,end=e) for s,e in b.headers];events=[];work=dict.fromkeys(WORK_FIELDS,0)
    for i in ids:
        row=prepared['months'][i];need(_record_hash(row)==b.month_hashes[i],'Consumed month binding changed')
        months[i]=deepcopy(row);work['projected_month_records_checked']+=1
    for i in range(n):
        row=prepared['events'][i];need(_record_hash(row)==b.event_hashes[i],'Consumed event binding changed')
        events.append(deepcopy(row));work['projected_event_records_checked']+=1
    audit=_audit(n,b.width,years,ids,work,'cached_projection',b.anchor_sha256)
    audit['consumed_projection_sha256']=_record_hash(dict(months=months,events=events))
    return dict(schema='monthly_boundary_pair267_prepared_source_v1',n=n,width=b.width,months=months,events=events,audit=audit,binding=b)
