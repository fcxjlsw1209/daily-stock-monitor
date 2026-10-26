"""Pure source/core gates and exact block-activity breadth integration.

No loaders or replay. Frozen original core validation has priority over new
whole-event source fallback. Native stored zero volume is unresolved and
forces both candidate arms to copy the original event; it is never imputed. All candidate selection follows source and original
incumbent callbacks. Stored volume is a declared proxy, not certified flow.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PINS={
 'monthly_calendar_clock248_v2_kernel.py':'b580b2a2389201f48d81cd78ddecb641699fdbd919c79202746665789b5a6bcb',
 'monthly_volume_breadth249_math.py':'c632e74a76fbc1975fff3e56d96b4ff1fccd314828e3099bd9495d7424a73ee9',
 'monthly_volume_breadth249_adapter.py':'4794e3d2a6b6a3196a0bb5602b781cf324e4cf34a5cb42ba9a3c19e40f7ee2c0'}
def _load(name):
    path=ROOT/'work'/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PINS[name]:raise ValueError('Changed frozen dependency '+name)
    spec=importlib.util.spec_from_file_location('volume249_v2_'+name[:-3],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
old=_load('monthly_calendar_clock248_v2_kernel.py')
exact=_load('monthly_volume_breadth249_math.py')
funding=_load('monthly_volume_breadth249_adapter.py')
require=old.require
ARMS=('incumbent_local67','formation_volume_confirmation_breadth','formation_volume_confirmation_mean_control')
INITIAL_PARTIAL='2016-08-17'
SOURCE_SCHEMA='monthly_volume_breadth249_v2_metadata_source_review_v1'
EVENT_KEYS=('query_index','date','initial_partial','calm','core_weight','symbols','branch','panel_first_session',
 'query_position','next_official_session','trading_price_start','trading_price_end','risk_price_start','risk_price_end',
 'price_start','price_end','volume_start','volume_end','u0_indices','core_source_max_abs_error','missing_indices','zero_indices',
 'transition_ids','continuity_unknown_ids','cells','breadth_rank','mean_rank')
CELL_KEYS=('column_index','symbol','original_finite','original_raw','original_rank','frozen_core','rebuilt_core',
 'core_abs_error','sigma63','source_available','price_gaps','volume_gaps','volume_zeros','measurement')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','calm_formal_events','noncalm_formal_events',
 'original_checked_cells','original_finite_cells','source_checked_cells','source_available_cells','source_missing_cells','source_zero_volume_cells','source_unavailable_cells',
 'active_events','zero_volume_events','missing_source_events','known_transition_events','source_price_observations','source_volume_observations','source_zero_volume_observations',
 'block_covariances','ranked_cells','candidate_nonzero_correction_cells','control_nonzero_correction_cells',
 'candidate_changed_holdings_events','control_changed_holdings_events','original_episode_calls','new_model_fits','predictive_model_fits')
AUDIT_EVENT_KEYS=('query_index','date','branch','extension_active_indices','original_remaining_support','arms')
BRANCHES=('initial_partial','zero_volume','missing_source','known_transition','available')


def dependency_files():
    return tuple(sorted(set(old.dependency_files())|set(funding.dependency_files())|{ROOT/'work'/p for p in PINS}|{ROOT/'work/test_monthly_calendar_clock248_v2_kernel.py'}))


def _frame(frame,label):
    require(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex),label+' frame')
    ix=frame.index
    require(ix.is_unique and ix.is_monotonic_increasing and not ix.hasnans and ix.tz is None and ix.equals(ix.normalize()),label+' date axis')
    require(frame.columns.is_unique and all(type(s) is str and s and s.strip()==s for s in frame.columns),label+' columns')


def _clock(i,meta,b,sessions,lookup,first):
    day=meta['dates'][i];require(day in lookup,'Query official session');t=lookup[day]
    require(t>=356 and t+1<len(sessions),'Full source clock and next witness')
    partial=day==INITIAL_PARTIAL
    require(partial or sessions[t+1][:7]!=day[:7],'Formal official monthend')
    return dict(query_index=i,date=day,initial_partial=partial,calm=bool(b['calm'][i]),core_weight=float(b['cw'][i,0]),symbols=list(meta['symbols'][i]),
      branch='initial_partial' if partial else 'available',panel_first_session=first,query_position=t,next_official_session=sessions[t+1],
      trading_price_start=sessions[t-(252 if b['calm'][i] else 294)],trading_price_end=sessions[t-42],risk_price_start=sessions[t-63],risk_price_end=day,
      price_start=sessions[t-294],price_end=sessions[t-42],volume_start=sessions[t-356],volume_end=sessions[t-42],u0_indices=[],
      core_source_max_abs_error=None,missing_indices=[],zero_indices=[],transition_ids=[],continuity_unknown_ids=[],cells=[],breadth_rank=None,mean_rank=None)


def _indexed(records,n,label):
    require(isinstance(records,list),label+' list');out={}
    for x in records:
        require(type(x) is dict and type(x.get('query_index')) is int and x['query_index']>=0,label+' query index')
        i=x['query_index']
        if i>=n:continue
        require(i not in out,label+' duplicate');out[i]=x
    require(set(out)==set(range(n)),label+' complete consumed events');return out


def _review(review,events,sessions):
    require(type(review) is dict and review.get('schema')==SOURCE_SCHEMA,'Source review schema')
    n=len(events);clocks=_indexed(review.get('clock_events'),n,'Source clock')
    for e in events:
        t=e['query_position'];want={k:e[k] for k in ('query_index','date','initial_partial','symbols','query_position','next_official_session','price_start','price_end','volume_start','volume_end')}
        want.update(numeric_evaluation_required=not e['initial_partial'],price_count=253,return_start=sessions[t-293],return_end=sessions[t-42],return_count=252,
          volume_count=315,volume_extra_leading_sessions=62,first_return_volume=sessions[t-293],first_baseline_start=sessions[t-356],first_baseline_end=sessions[t-294],
          last_return_volume=sessions[t-42],last_baseline_start=sessions[t-105],last_baseline_end=sessions[t-43],block_price_endpoints=[sessions[t-294+21*k] for k in range(13)])
        require(all(type(clocks[e['query_index']].get(k)) is type(v) and clocks[e['query_index']].get(k)==v for k,v in want.items()),'Source clock binding')
    bounds={}
    require(isinstance(review.get('listing_bounds'),list),'Listing bounds')
    for x in review['listing_bounds']:
        require(type(x) is dict and set(x)=={'symbol','issuer_key','not_before','available_from','evidence_id','kind'},'Listing bound fields')
        require(all(type(v) is str and v and v.strip()==v for v in x.values()),'Listing bound strings')
        old._iso(x['not_before'],'Bound date');old._iso(x['available_from'],'Bound available')
        require(x['symbol'] not in bounds,'Duplicate listing');bounds[x['symbol']]=x
    require(isinstance(review.get('identity_cells'),list),'Identity cells');seen={}
    for x in review['identity_cells']:
        require(type(x) is dict and type(x.get('query_index')) is int and x['query_index']>=0,'Identity query')
        i=x['query_index']
        if i>=n:continue
        j=x.get('column_index');require(type(j) is int and 0<=j<49 and (i,j) not in seen,'Identity column/duplicate');seen[i,j]=x
    required={(i,j) for i,e in enumerate(events) for j,s in enumerate(e['symbols']) if s in bounds}
    require(set(seen)==required,'Complete listing identity scope')
    for (i,j),x in seen.items():
        e=events[i];bound=bounds[e['symbols'][j]];t=e['query_position']
        want=dict(date=e['date'],symbol=e['symbols'][j],issuer_key=bound['issuer_key'],not_before=bound['not_before'],available_from=bound['available_from'],bound_visible=bound['available_from']<=e['date'],
          price_start=e['price_start'],price_end=e['price_end'],volume_start=e['volume_start'],volume_end=e['volume_end'],
          price_window_dates_before_bound=sum(d<bound['not_before'] for d in sessions[t-294:t-41]),volume_window_dates_before_bound=sum(d<bound['not_before'] for d in sessions[t-356:t-41]))
        require(all(type(x.get(k)) is type(v) and x.get(k)==v for k,v in want.items()),'Listing identity/clock binding')
    records=review.get('transition_records');rules=review.get('undated_loader_rules');ids=review.get('automatic_known_unresolved_transition_fallback_ids')
    require(isinstance(records,list) and isinstance(rules,list) and isinstance(ids,list),'Transition ledger lists')
    require(all(type(i) is str and i for i in ids) and len(set(ids))==len(ids),'Unique automatic transition IDs')
    byid={}
    for x in records+rules:
        require(type(x) is dict and type(x.get('id')) is str and x['id'] and x['id'] not in byid,'Transition record ID')
        require(isinstance(x.get('symbols'),list) and all(type(s) is str and s for s in x['symbols']),'Transition symbols');byid[x['id']]=x
    require(set(ids)<=set(byid),'Unknown automatic transition ID')
    for rid in ids:
        x=byid[rid]
        require(x.get('known_unresolved_fallback_eligible') is True and x.get('unresolved_unit_transition') is True,'Automatic transition explicit status')
        old._iso(x.get('effective_date'),'Transition effective')
        require(isinstance(x.get('affected_fields'),list) and bool(set(x['affected_fields'])&{'adj','volume'}),'Transition consumed fields')
        visible=x.get('available_from')
        if visible is not None:old._iso(visible,'Transition available')
        require(visible is not None or x.get('retrospective_integrity_basis') is True,'Transition evidence clock')
    return bounds,byid,ids,rules


def _original(adj,events,b):
    formal=[e for e in events if not e['initial_partial']]
    if not formal:return
    end=formal[-1]['date'];past=adj.loc[:end];origin=formal[0]['panel_first_session']
    require(len(past)>0 and str(past.index[0].date())==origin,'Full original rolling history origin')
    names=sorted(set(s for e in formal for s in e['symbols']))
    require(set(names)<=set(past.columns),'Original source missing symbol')
    px=past[names].copy()
    for s in names:
        values=[]
        for x in px[s]:
            if x is None or x is pd.NA or (isinstance(x,Real) and not isinstance(x,(bool,np.bool_)) and pd.isna(x)):values.append(np.nan)
            else:values.append(old._number(x,'Original historical price',False))
        px[s]=np.asarray(values,float)
    with np.errstate(all='ignore'):
        sigma=px.pct_change(fill_method=None).rolling(63).std()
        raw210=px.shift(42)/px.shift(252)-1.
        raw252=(px.shift(42)/px.shift(294)-1.)/sigma.replace(0,np.nan)
    for e in formal:
        i=e['query_index'];names=e['symbols'];day=e['date'];raw=(raw210 if e['calm'] else raw252).loc[day,names].replace([np.inf,-np.inf],np.nan).to_numpy();ranks=old._rank(raw)
        rebuilt=e['core_weight']*ranks;errors=np.abs(rebuilt-b['core'][i]);require((errors<=1e-12).all(),'Original weighted-core source mismatch')
        e['core_source_max_abs_error']=float(errors.max());e['u0_indices']=np.flatnonzero(np.isfinite(raw)).tolist()
        for j,s in enumerate(names):
            e['cells'].append(dict(column_index=j,symbol=s,original_finite=bool(np.isfinite(raw[j])),original_raw=old._shown(raw[j]),original_rank=float(ranks[j]),
              frozen_core=float(b['core'][i,j]),rebuilt_core=float(rebuilt[j]),core_abs_error=float(errors[j]),sigma63=old._shown(sigma.at[pd.Timestamp(day),s]),
              source_available=False,price_gaps=[],volume_gaps=[],volume_zeros=[],measurement=None))


def _window(frame,symbol,days,bound,query,label):
    values=[];gaps=[];zeros=[];errors=[]
    for q,day in enumerate(days):
        reason=None
        if symbol not in frame.columns:reason='missing_symbol';x=None
        elif pd.Timestamp(day) not in frame.index:reason='missing_row';x=None
        else:x=frame.at[pd.Timestamp(day),symbol]
        if reason is None and (x is None or x is pd.NA or (isinstance(x,Real) and not isinstance(x,(bool,np.bool_)) and pd.isna(x))):reason='missing_value'
        if reason is not None:values.append(None);gaps.append(dict(offset=q,date=day,reason=reason));continue
        try:
            v=old._number(x,'Required '+label)
            # Native zero only: a nonzero Real underflowing to binary64 zero
            # remains a hard invalid, never a stored-zero observation.
            if label=='volume' and x==0:
                zeros.append(dict(offset=q,date=day,value=v,reason='unresolved_zero_volume'));values.append(v);continue
            require(v>0,'Positive required '+label)
            if bound and bound['available_from']<=query:require(day>=bound['not_before'],'Visible listing contradiction '+symbol+' '+day+' '+label)
            values.append(v)
        except (ValueError,ArithmeticError) as exc:values.append(None);errors.append(str(exc))
    return values,gaps,zeros,errors


def _features(panel,meta,calendar,review,b,n):
    require(type(panel) is dict and set(panel)=={'adj','volume'},'Adj/volume source panel')
    for k in panel:_frame(panel[k],k)
    first=old._iso(review.get('panel_first_session'),'Panel first session');sessions=old._calendar(calendar);lookup={d:i for i,d in enumerate(sessions)}
    events=[_clock(i,meta,b,sessions,lookup,first) for i in range(n)]
    bounds,records,ids,rules=_review(review,events,sessions)
    adj=panel['adj']
    for e in events:
        if e['initial_partial']:continue
        day=pd.Timestamp(e['date']);require(day in adj.index,'Original source missing query');p=adj.index.get_loc(day);t=e['query_position']
        require(p>=294 and adj.index[p-294:p+1].strftime('%Y-%m-%d').tolist()==sessions[t-294:t+1],'Original consumed clock mismatch')
    _original(adj,events,b)
    for e in events:
        if e['initial_partial']:continue
        t=e['query_position'];errors=[];windows=[]
        for cell in e['cells']:
            s=cell['symbol'];bound=bounds.get(s)
            p,pg,pz,pe=_window(adj,s,sessions[t-294:t-41],bound,e['date'],'price')
            v,vg,vz,ve=_window(panel['volume'],s,sessions[t-356:t-41],bound,e['date'],'volume')
            cell['price_gaps']=pg;cell['volume_gaps']=vg;cell['volume_zeros']=vz;errors+=pe+ve;windows.append((p,v))
            require(not pz,'Price zero must be a hard invalid')
        require(not errors,'; '.join(errors))  # Missing never masks another member's hard error.
        for cell,(p,v) in zip(e['cells'],windows):
            if cell['price_gaps'] or cell['volume_gaps']:e['missing_indices'].append(cell['column_index'])
            if cell['volume_zeros']:
                e['zero_indices'].append(cell['column_index']);continue
            m=exact.measure(p,v);cell['measurement']=m;cell['source_available']=m['available']
        for rid in ids:
            x=records[rid]
            if not set(x['symbols'])&set(e['symbols']):continue
            if not (x.get('retrospective_integrity_basis') is True or (x.get('available_from') is not None and x['available_from']<=e['date'])):continue
            effective=x['effective_date']
            if any(field in x['affected_fields'] and e[start]<effective<=e[end] for field,start,end in (('adj','price_start','price_end'),('volume','volume_start','volume_end'))):e['transition_ids'].append(rid)
        e['continuity_unknown_ids']=[x['id'] for x in rules if set(x['symbols'])&set(e['symbols'])]
        e['branch']='zero_volume' if e['zero_indices'] else ('missing_source' if e['missing_indices'] else ('known_transition' if e['transition_ids'] else 'available'))
        # Target ranks intentionally remain None until the incumbent gate.
    return events


def _measurement(m):
    require(type(m) is dict and tuple(m)==exact.MEASURE_KEYS and m['schema']==exact.SCHEMA,'Measurement schema')
    require(type(m['available']) is bool,'Measurement bool')
    for k,v in (('price_count',253),('volume_count',315),('return_count',252),('block_count',12)):require(type(m[k]) is int and m[k]==v,'Measurement count')
    for k,lim in (('missing_price_indices',253),('missing_volume_indices',315)):
        z=m[k];require(isinstance(z,list) and all(type(q) is int and 0<=q<lim for q in z) and z==sorted(set(z)),'Missing slots')
    if not m['available']:
        require(m['reason']=='missing_source' and (m['missing_price_indices'] or m['missing_volume_indices']),'Missing measurement')
        require(m['high_flags']==[] and m['blocks']==[] and all(m[k] is None for k in ('raw_breadth','raw_mean','positive_block_count','zero_block_count','negative_block_count')),'No missing estimates');return
    require(m['reason']=='available' and not m['missing_price_indices'] and not m['missing_volume_indices'],'Available measurement')
    h=m['high_flags'];require(isinstance(h,list) and len(h)==252 and all(type(x) is int and x in (0,1) for x in h),'High volume flags')
    require(isinstance(m['blocks'],list) and len(m['blocks'])==12,'Twelve blocks');cov=[]
    for k,x in enumerate(m['blocks']):
        require(type(x) is dict and tuple(x)==exact.BLOCK_KEYS,'Block schema')
        for key,v in (('block_index',k),('first_return_index',21*k),('last_return_index',21*k+20),('high_count',sum(h[21*k:21*k+21]))):require(type(x[key]) is int and x[key]==v,'Block count')
        r=exact._rational(x['return_sum']);hr=exact._rational(x['high_return_sum']);c=exact._rational(x['covariance'])
        require(c==(21*hr-x['high_count']*r)/441,'Exact cached covariance algebra');sign=int(c>0)-int(c<0)
        require(type(x['sign']) is int and x['sign']==sign,'Covariance sign');cov.append(c)
    pos=sum(x>0 for x in cov);zero=sum(x==0 for x in cov)
    for k,v in (('positive_block_count',pos),('zero_block_count',zero),('negative_block_count',12-pos-zero)):require(type(m[k]) is int and m[k]==v,'Sign count')
    require(exact._rational(m['raw_breadth'])==Fraction(2*pos+zero-12,24) and exact._rational(m['raw_mean'])==sum(cov,Fraction(0))/12,'Exact cached aggregates')


def _cached(features,meta,b,n):
    require(isinstance(features,list) and len(features)>=n,'Feature prefix')
    for i,e in enumerate(features[:n]):
        require(type(e) is dict and tuple(e)==EVENT_KEYS,'Feature schema')
        require(type(e['query_index']) is int and e['query_index']==i and e['date']==meta['dates'][i] and e['symbols']==meta['symbols'][i],'Feature identity')
        require(type(e['calm']) is bool and e['calm']==bool(b['calm'][i]) and type(e['core_weight']) is float and e['core_weight']==float(b['cw'][i,0]),'Feature state')
        partial=e['date']==INITIAL_PARTIAL;require(e['initial_partial'] is partial and e['branch'] in BRANCHES,'Partial/branch')
        for k in ('panel_first_session','next_official_session','trading_price_start','trading_price_end','risk_price_start','risk_price_end','price_start','price_end','volume_start','volume_end'):old._iso(e[k],k)
        require(type(e['query_position']) is int and e['query_position']>=356 and e['next_official_session']>e['date'] and e['risk_price_end']==e['date'],'Feature clocks')
        require(e['volume_start']<e['price_start']<e['price_end']==e['volume_end']==e['trading_price_end']<e['date'],'Formation clocks')
        if partial:
            require(e['branch']=='initial_partial' and all(e[k]==[] for k in ('u0_indices','missing_indices','zero_indices','transition_ids','continuity_unknown_ids','cells')) and e['core_source_max_abs_error'] is None,'Partial exact bypass');continue
        require(e['next_official_session'][:7]!=e['date'][:7],'Formal witness')
        require(isinstance(e['cells'],list) and len(e['cells'])==49,'Full49 cells');raw=[];missing=[];zero=[];errs=[]
        for j,c in enumerate(e['cells']):
            require(type(c) is dict and tuple(c)==CELL_KEYS and type(c['column_index']) is int and c['column_index']==j and c['symbol']==e['symbols'][j],'Cell identity/schema')
            require(type(c['original_finite']) is bool and type(c['source_available']) is bool,'Cell booleans')
            r=np.nan if c['original_raw'] is None else old._number(c['original_raw'],'Original raw');raw.append(r)
            require(c['original_finite']==bool(np.isfinite(r)),'Original finite domain')
            require(type(c['frozen_core']) is float and c['frozen_core']==float(b['core'][i,j]),'Frozen core anchor')
            if c['sigma63'] is not None:old._number(c['sigma63'],'sigma63')
            z=c['volume_zeros'];require(isinstance(z,list),'Volume zero list')
            for x in z:
                require(type(x) is dict and tuple(x)==('offset','date','value','reason') and type(x['offset']) is int and 0<=x['offset']<315,'Volume zero schema')
                require(type(x['value']) is float and x['value']==0 and x['reason']=='unresolved_zero_volume','Native unresolved volume zero')
                old._iso(x['date'],'Zero date');require(e['volume_start']<=x['date']<=e['volume_end'],'Zero clock')
            require([x['offset'] for x in z]==sorted(set(x['offset'] for x in z)),'Unique sorted zero slots')
            if z:
                zero.append(j);require(c['measurement'] is None and c['source_available'] is False,'Zero volume has no measurement')
            else:
                _measurement(c['measurement']);require(c['source_available'] is c['measurement']['available'],'Source support')
            for key,mkey,lim,start,end in (('price_gaps','missing_price_indices',253,'price_start','price_end'),('volume_gaps','missing_volume_indices',315,'volume_start','volume_end')):
                gaps=c[key];require(isinstance(gaps,list),'Gaps list')
                for g in gaps:
                    require(type(g) is dict and tuple(g)==('offset','date','reason') and type(g['offset']) is int and 0<=g['offset']<lim and g['reason'] in ('missing_symbol','missing_row','missing_value'),'Gap schema')
                    old._iso(g['date'],'Gap date');require(e[start]<=g['date']<=e[end],'Gap clock')
                offsets=[g['offset'] for g in gaps];require(offsets==sorted(set(offsets)),'Unique sorted missing slots')
                if not z:require(offsets==c['measurement'][mkey],'Exact missing slots')
                elif key=='volume_gaps':require(not set(offsets)&{x['offset'] for x in z},'Zero is not ordinary missing')
            if c['price_gaps'] or c['volume_gaps']:missing.append(j)
            require(c['source_available'] is (not z and not c['price_gaps'] and not c['volume_gaps']),'Exact source availability union')
        ranks=old._rank(raw)
        for j,c in enumerate(e['cells']):
            rebuilt=e['core_weight']*float(ranks[j]);err=abs(rebuilt-c['frozen_core']);errs.append(err)
            require(type(c['original_rank']) is float and c['original_rank']==float(ranks[j]) and c['rebuilt_core']==rebuilt and c['core_abs_error']==err and err<=1e-12,'Original rank/core parity')
        require(e['u0_indices']==np.flatnonzero(np.isfinite(raw)).tolist() and e['missing_indices']==missing and e['zero_indices']==zero and e['core_source_max_abs_error']==max(errs),'Exact domains/counts')
        for k in ('transition_ids','continuity_unknown_ids'):require(isinstance(e[k],list) and all(type(x) is str and x for x in e[k]) and len(set(e[k]))==len(e[k]),'Transition IDs')
        branch='zero_volume' if zero else ('missing_source' if missing else ('known_transition' if e['transition_ids'] else 'available'));require(e['branch']==branch,'Whole event fallback')
    return features[:n]


def _strict_equal(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return tuple(actual)==tuple(expected) and all(_strict_equal(actual[k],expected[k]) for k in expected)
    if isinstance(expected,list):return len(actual)==len(expected) and all(_strict_equal(a,b) for a,b in zip(actual,expected))
    if isinstance(expected,float):return actual.hex()==expected.hex()
    return actual==expected


def _finalize_ranks(features,source_stage):
    # Called only after original-incumbent ordered choices and independent gate.
    for e in features:
        for key,rawkey in (('breadth_rank','raw_breadth'),('mean_rank','raw_mean')):
            expected=exact.rank49([c['measurement'][rawkey] for c in e['cells']]) if e['branch']=='available' else None
            if source_stage:
                require(e[key] is None,'Source stage must not contain target ranks');e[key]=expected
            else:require(_strict_equal(e[key],expected),'Exact common49 ranks after incumbent gate')


def _select(study,meta,b,features,n,*,incumbent_validator=None,expected_incumbent_orders=None,source_stage=False):
    require(incumbent_validator is None or callable(incumbent_validator),'Incumbent callback')
    require(expected_incumbent_orders is None or (isinstance(expected_incumbent_orders,list) and len(expected_incumbent_orders)==n),'Expected original order prefix')
    fs=_cached(features,meta,b,n);active=np.array([e['branch']=='available' for e in fs],dtype=bool)
    orders={};scores={};pres={};audits={};choices={}
    for k,arm in enumerate(ARMS):
        on=np.zeros(n,bool) if k==0 else active;target=np.full((n,49),np.nan)
        if k:
            field='breadth_rank' if k==1 else 'mean_rank'
            for i in np.flatnonzero(on):target[i]=fs[i][field]['targets']
        ch,post,pre,au=funding.score_and_choose(b['pre'],b['core'],b['cw'],b['base'],b['calm'],b['weak'],b['good'],b['meta']['symbols'],b['effects'],target,on)
        choices[arm]=[[int(j) for j in row] for row in ch];scores[arm]=post;pres[arm]=pre;audits[arm]=au;orders[arm]=study.rp.make_orders(ch,post,b['meta'])
        if k==0:
            if expected_incumbent_orders is not None:require(orders[arm]==expected_incumbent_orders,'Frozen incumbent ordered choices mismatch')
            if incumbent_validator is not None:
                receipt=incumbent_validator(deepcopy(orders[arm]),post.copy(),pre.copy());require(type(receipt) is dict and receipt.get('passed') is True,'Independent incumbent gate failed')
            _finalize_ranks(fs,source_stage)
    for i,e in enumerate(fs):
        if not active[i]:
            for arm in ARMS[1:]:require(orders[arm][i]==orders[ARMS[0]][i] and np.array_equal(scores[arm][i],scores[ARMS[0]][i]) and np.array_equal(pres[arm][i],pres[ARMS[0]][i]),'Exact whole-event fallback')
    cells=[c for e in fs for c in e['cells']];formal=[e for e in fs if not e['initial_partial']];sourceok=sum(c['source_available'] for c in cells)
    counts=dict(events=n,symbols_per_event=49,initial_partial_events=n-len(formal),formal_events=len(formal),calm_formal_events=sum(e['calm'] for e in formal),noncalm_formal_events=sum(not e['calm'] for e in formal),
      original_checked_cells=len(cells),original_finite_cells=sum(c['original_finite'] for c in cells),source_checked_cells=len(cells),source_available_cells=sourceok,source_missing_cells=sum(bool(c['price_gaps'] or c['volume_gaps']) for c in cells),source_zero_volume_cells=sum(bool(c['volume_zeros']) for c in cells),source_unavailable_cells=len(cells)-sourceok,
      active_events=int(active.sum()),zero_volume_events=sum(e['branch']=='zero_volume' for e in fs),missing_source_events=sum(e['branch']=='missing_source' for e in fs),known_transition_events=sum(e['branch']=='known_transition' for e in fs),
      source_price_observations=253*len(cells),source_volume_observations=315*len(cells),source_zero_volume_observations=sum(len(c['volume_zeros']) for c in cells),block_covariances=12*sourceok,ranked_cells=2*49*int(active.sum()),
      candidate_nonzero_correction_cells=audits[ARMS[1]]['counts']['nonzero_correction_cells'],control_nonzero_correction_cells=audits[ARMS[2]]['counts']['nonzero_correction_cells'],
      candidate_changed_holdings_events=sum(set(choices[ARMS[0]][i])!=set(choices[ARMS[1]][i]) for i in range(n)),control_changed_holdings_events=sum(set(choices[ARMS[0]][i])!=set(choices[ARMS[2]][i]) for i in range(n)),
      original_episode_calls=3,new_model_fits=0,predictive_model_fits=0)
    events=[dict(query_index=i,date=e['date'],branch=e['branch'],extension_active_indices=np.flatnonzero(b['ext'][i]).tolist(),original_remaining_support=b['remaining'][i].tolist(),
                 arms={arm:deepcopy(audits[arm]['events'][i]) for arm in ARMS}) for i,e in enumerate(fs)]
    require(tuple(counts)==COUNT_FIELDS and all(tuple(e)==AUDIT_EVENT_KEYS for e in events),'Decision audit schema')
    return orders,scores,pres,dict(schema='monthly_volume_breadth249_v2_decisions_v1',counts=counts,events=events)


def prepare(study,arrays,meta,panel,calendar,source_review,n,*,source_validator=None,incumbent_validator=None,expected_incumbent_orders=None):
    before=old._snapshot(arrays);before_meta=deepcopy(meta);b=old._cache(study,arrays,meta,n);require(b['width']==49,'Exactly original49 domain')
    features=_features(panel,meta,calendar,source_review,b,n)
    require(all(tuple(e)==EVENT_KEYS and all(tuple(c)==CELL_KEYS for c in e['cells']) for e in features),'Feature schema')
    if source_validator is not None:
        require(callable(source_validator),'Source callback');receipt=source_validator(deepcopy(features));require(type(receipt) is dict and receipt.get('passed') is True,'Independent original-source gate failed')
    orders,scores,pres,audit=_select(study,meta,b,features,n,incumbent_validator=incumbent_validator,expected_incumbent_orders=expected_incumbent_orders,source_stage=True)
    require(before==old._snapshot(arrays) and meta==before_meta,'Original input mutation')
    return features,orders,scores,pres,audit


def decisions(study,arrays,meta,features,n,*,incumbent_validator=None,expected_incumbent_orders=None):
    before=old._snapshot(arrays);before_meta=deepcopy(meta);before_features=deepcopy(features)
    b=old._cache(study,arrays,meta,n);require(b['width']==49,'Exactly original49 domain')
    result=_select(study,meta,b,deepcopy(features),n,incumbent_validator=incumbent_validator,expected_incumbent_orders=expected_incumbent_orders)
    require(before==old._snapshot(arrays) and meta==before_meta and features==before_features,'Cached input mutation');return result
