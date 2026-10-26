"""Direct long-momentum basket selection; caller-supplied sources only.

Source/core proof and original incumbent gates precede experimental shortlist
sorting or trio statistics. Experimental arms never invoke the old episode.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
from itertools import combinations
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd
import monthly_basket_core251_math as exact

ROOT=Path(__file__).resolve().parents[1]
OLD_PATH=ROOT/'work/monthly_calendar_clock248_v2_kernel.py'
OLD_SHA='b580b2a2389201f48d81cd78ddecb641699fdbd919c79202746665789b5a6bcb'
if hashlib.sha256(OLD_PATH.read_bytes()).hexdigest()!=OLD_SHA:raise ValueError('Frozen original core helper changed')
_spec=importlib.util.spec_from_file_location('basket251_original_core',OLD_PATH)
old=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(old)
require=old.require
INITIAL_PARTIAL='2016-08-17'
SOURCE_SCHEMA='monthly_basket_core251_metadata_source_review_v1'
ARMS=('incumbent_local67','long_momentum_basket_core','long_momentum_diagonal_core')
POLICY=dict(budget=.05,ceiling=5,gap=.02)
EVENT_KEYS=('query_index','date','initial_partial','calm','core_weight','symbols','branch','panel_first_session','query_position','next_official_session','trading_price_start','trading_price_end','risk_price_start','risk_price_end','price_start','price_end','block_boundary_dates','u0_indices','core_source_max_abs_error','original_eligible_indices','required_indices','cells','new_eligible_indices','excluded_momentum_indices','shortlist_indices','trios','common_trio_indices','winners')
CELL_KEYS=('column_index','symbol','original_finite','original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','sigma63','original_eligible','source_available','measurement')
TRIO_KEYS=('indices','symbols','statistics')
WINNER_KEYS=('branch','winning_trio_index','tied_trio_indices','selected_indices','selected_symbols')
CANDIDATE_KEYS=EVENT_KEYS[-6:]
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','calm_formal_events','noncalm_formal_events','original_checked_cells','original_finite_cells','original_eligible_cells','source_required_cells','source_available_cells','source_price_observations','momentum_estimates','block_return_estimates','new_eligible_cells','momentum_excluded_cells','shortlist_cells','enumerated_trios','common_supported_trios','zero_joint_variance_trios','zero_diagonal_variance_trios','cash_events','two_name_events','zero_variance_events','optimized_events','candidate_changed_holdings_events','diagonal_changed_holdings_events','original_episode_calls','experimental_episode_calls','new_model_fits','predictive_model_fits')
AUDIT_EVENT_KEYS=('query_index','date','branch','incumbent_choices','arms')
BRANCHES=('initial_partial','source_verified','cash','two_names','zero_variance','optimized')


def dependency_files():return (OLD_PATH,Path(exact.__file__).resolve(),ROOT/'work/test_monthly_calendar_clock248_v2_kernel.py')+old.dependency_files()


def _strict_equal(a,b,label):
    require(type(a) is type(b),label+' type')
    if isinstance(a,dict):
        require(tuple(a)==tuple(b),label+' keys')
        for k in a:_strict_equal(a[k],b[k],label+'.'+k)
    elif isinstance(a,list):
        require(len(a)==len(b),label+' length')
        for k,(x,y) in enumerate(zip(a,b)):_strict_equal(x,y,label+str(k))
    elif isinstance(a,float):require(a.hex()==b.hex(),label+' binary64')
    else:require(a==b,label+' value')


def _frame(adj):
    require(isinstance(adj,pd.DataFrame) and isinstance(adj.index,pd.DatetimeIndex),'Adjusted price frame')
    require(adj.index.tz is None and adj.index.is_unique and adj.index.is_monotonic_increasing and not adj.index.hasnans and all(x==x.normalize() for x in adj.index),'Price date axis')
    require(adj.columns.is_unique and all(type(s) is str and s and s.strip()==s for s in adj.columns),'Price symbol axis')


def _clock(i,meta,b,sessions,lookup,first):
    day=meta['dates'][i];require(day in lookup,'Query official session');t=lookup[day]
    require(t>=294 and t+1<len(sessions),'Complete clock and next official witness')
    partial=day==INITIAL_PARTIAL
    if not partial:require(sessions[t+1][:7]!=day[:7],'Formal month end')
    good=np.flatnonzero(b['good'][i]).tolist()
    return dict(query_index=i,date=day,initial_partial=partial,calm=bool(b['calm'][i]),core_weight=float(b['cw'][i,0]),symbols=list(meta['symbols'][i]),branch='initial_partial' if partial else 'source_verified',panel_first_session=first,query_position=t,next_official_session=sessions[t+1],trading_price_start=sessions[t-(252 if b['calm'][i] else 294)],trading_price_end=sessions[t-42],risk_price_start=sessions[t-63],risk_price_end=day,price_start=sessions[t-294],price_end=sessions[t-42],block_boundary_dates=[sessions[t-294+21*k] for k in range(13)],u0_indices=[],core_source_max_abs_error=None,original_eligible_indices=good,required_indices=[] if partial else good.copy(),cells=[],new_eligible_indices=None,excluded_momentum_indices=None,shortlist_indices=None,trios=None,common_trio_indices=None,winners=None)


def _review(review,events,sessions):
    require(type(review) is dict and review.get('schema')==SOURCE_SCHEMA,'Source review schema')
    n=len(events);clocks={}
    require(isinstance(review.get('clock_events'),list),'Clock events')
    for x in review['clock_events']:
        require(type(x) is dict and type(x.get('query_index')) is int and x['query_index']>=0,'Clock query index');q=x['query_index']
        if q>=n:continue
        require(q not in clocks,'Duplicate clock');clocks[q]=x
    require(set(clocks)==set(range(n)),'Exact consumed clock inventory')
    for e in events:
        t=e['query_position'];want=dict(query_index=e['query_index'],date=e['date'],initial_partial=e['initial_partial'],numeric_evaluation_required=not e['initial_partial'],symbols=e['symbols'],query_position=t,next_official_session=e['next_official_session'],price_start=e['price_start'],price_end=e['price_end'],source_start_position=t-294,source_end_position=t-42,price_count=253,return_count=252,block_count=12,block_sessions=21,block_boundary_dates=e['block_boundary_dates'])
        for k,v in want.items():_strict_equal(clocks[e['query_index']].get(k),v,'Source clock '+k)
    require(isinstance(review.get('listing_bounds'),list) and isinstance(review.get('identity_cells'),list),'Listing metadata');bounds={}
    for x in review['listing_bounds']:
        require(type(x) is dict and set(x)=={'symbol','issuer_key','not_before','available_from','evidence_id','kind'},'Listing bound fields')
        require(all(type(v) is str and v and v.strip()==v for v in x.values()),'Bound strings')
        old._iso(x['not_before'],'Not before');old._iso(x['available_from'],'Available from')
        require(x['symbol'] not in bounds,'Duplicate listing');bounds[x['symbol']]=x
    seen={}
    for x in review['identity_cells']:
        require(type(x) is dict and type(x.get('query_index')) is int and x['query_index']>=0,'Identity query');i=x['query_index']
        if i>=n:continue
        j=x.get('column_index');require(type(j) is int and 0<=j<49 and (i,j) not in seen,'Identity column/duplicate');seen[i,j]=x
    required={(i,j) for i,e in enumerate(events) for j,s in enumerate(e['symbols']) if s in bounds}
    require(set(seen)==required,'Complete listing identity inventory')
    for (i,j),x in seen.items():
        e=events[i];bound=bounds[e['symbols'][j]];t=e['query_position']
        want=dict(query_index=i,column_index=j,date=e['date'],symbol=e['symbols'][j],issuer_key=bound['issuer_key'],not_before=bound['not_before'],available_from=bound['available_from'],bound_visible=bound['available_from']<=e['date'],price_start=e['price_start'],price_end=e['price_end'],price_window_dates_before_bound=sum(d<bound['not_before'] for d in sessions[t-294:t-41]))
        require(set(x)==set(want),'Identity fields')
        for k,v in want.items():_strict_equal(x[k],v,'Identity '+k)
    return bounds


def _original(adj,events,b):
    formal=[e for e in events if not e['initial_partial']]
    if not formal:return
    past=adj.loc[:formal[-1]['date']];require(len(past)>0 and str(past.index[0].date())==formal[0]['panel_first_session'],'Full original rolling history origin')
    names=sorted(set(s for e in formal for s in e['symbols']));require(set(names)<=set(past.columns),'Original source missing symbol');px=past[names].copy()
    for s in names:
        vals=[]
        for x in px[s]:
            if x is None or x is pd.NA or (isinstance(x,Real) and not isinstance(x,(bool,np.bool_)) and pd.isna(x)):vals.append(np.nan)
            else:vals.append(old._number(x,'Original historical price',False))
        px[s]=np.asarray(vals,float)
    with np.errstate(all='ignore'):
        sigma=px.pct_change(fill_method=None).rolling(63).std();raw210=px.shift(42)/px.shift(252)-1.;raw252=(px.shift(42)/px.shift(294)-1.)/sigma.replace(0,np.nan)
    for e in formal:
        i=e['query_index'];names=e['symbols'];day=e['date'];raw=(raw210 if e['calm'] else raw252).loc[day,names].replace([np.inf,-np.inf],np.nan).to_numpy();ranks=old._rank(raw)
        rebuilt=e['core_weight']*ranks;errors=np.abs(rebuilt-b['core'][i]);require((errors<=1e-12).all(),'Original weighted-core source mismatch')
        e['core_source_max_abs_error']=float(errors.max());e['u0_indices']=np.flatnonzero(np.isfinite(raw)).tolist()
        for j,s in enumerate(names):e['cells'].append(dict(column_index=j,symbol=s,original_finite=bool(np.isfinite(raw[j])),original_raw=old._shown(raw[j]),original_rank=float(ranks[j]),frozen_core=float(b['core'][i,j]),rebuilt_core=float(rebuilt[j]),core_abs_error=float(errors[j]),sigma63=old._shown(sigma.at[pd.Timestamp(day),s]),original_eligible=bool(b['good'][i,j]),source_available=False,measurement=None))


def _features(panel,meta,calendar,review,b,n):
    require(type(panel) is dict and set(panel)=={'adj'},'Only adjusted prices');adj=panel['adj'];_frame(adj)
    first=old._iso(review.get('panel_first_session'),'Panel first session');sessions=old._calendar(calendar);lookup={d:i for i,d in enumerate(sessions)}
    events=[_clock(i,meta,b,sessions,lookup,first) for i in range(n)];bounds=_review(review,events,sessions)
    for e in events:
        if e['initial_partial']:continue
        day=pd.Timestamp(e['date']);require(day in adj.index,'Original source missing query');p=adj.index.get_loc(day);t=e['query_position']
        require(p>=294 and adj.index[p-294:p+1].strftime('%Y-%m-%d').tolist()==sessions[t-294:t+1],'Original consumed clock mismatch')
    _original(adj,events,b)
    windows=[];errors=[]
    for e in events:
        if e['initial_partial']:continue
        t=e['query_position']
        for j in e['required_indices']:
            cell=e['cells'][j];symbol=cell['symbol'];bound=bounds.get(symbol);values=[]
            for day in sessions[t-294:t-41]:
                try:
                    require(symbol in adj.columns and pd.Timestamp(day) in adj.index,'Required source axis '+symbol+' '+day)
                    v=old._number(adj.at[pd.Timestamp(day),symbol],'Required price '+symbol+' '+day);require(v>0,'Positive required price '+symbol+' '+day)
                    if bound and bound['available_from']<=e['date']:require(day>=bound['not_before'],'Visible listing contradiction '+symbol+' '+day)
                    values.append(v)
                except (ValueError,ArithmeticError) as exc:errors.append(str(exc));values.append(None)
            windows.append((cell,values))
    require(not errors,'; '.join(errors))
    for cell,values in windows:cell['measurement']=exact.measure(values);cell['source_available']=True
    return events


def _measurement(m):
    require(type(m) is dict and tuple(m)==exact.MEASURE_KEYS and m['schema']==exact.SCHEMA,'Measurement schema')
    require(type(m['price_count']) is int and m['price_count']==253 and type(m['block_count']) is int and m['block_count']==12,'Measurement counts')
    require(type(m['block_returns']) is list and len(m['block_returns'])==12,'Twelve block returns')
    gross=Fraction(1)
    for x in m['block_returns']:
        r=exact._rational(x);require(r>-1,'Positive cached block gross');gross*=1+r
    require(exact._rational(m['momentum'])==gross-1,'Exact block/annual telescoping')


def _cached(features,meta,b,n,source_stage=False):
    require(type(features) is list and len(features)>=n,'Feature prefix')
    for i,e in enumerate(features[:n]):
        require(type(e) is dict and tuple(e)==EVENT_KEYS,'Feature event schema')
        for k,v in dict(query_index=i,date=meta['dates'][i],symbols=meta['symbols'][i],initial_partial=meta['dates'][i]==INITIAL_PARTIAL,calm=bool(b['calm'][i]),core_weight=float(b['cw'][i,0]),original_eligible_indices=np.flatnonzero(b['good'][i]).tolist()).items():_strict_equal(e[k],v,'Cached '+k)
        require(e['branch'] in BRANCHES,'Feature branch')
        for k in ('panel_first_session','next_official_session','trading_price_start','trading_price_end','risk_price_start','risk_price_end','price_start','price_end'):old._iso(e[k],k)
        require(type(e['query_position']) is int and e['query_position']>=294 and e['next_official_session']>e['date'] and e['risk_price_end']==e['date'],'Cached clock')
        bd=e['block_boundary_dates'];require(type(bd) is list and len(bd)==13 and bd==sorted(set(bd)) and bd[0]==e['price_start'] and bd[-1]==e['price_end']==e['trading_price_end']<e['date'],'Block boundaries')
        for d in bd:old._iso(d,'Block date')
        if e['initial_partial']:
            require(e['branch']=='initial_partial' and e['cells']==[] and e['u0_indices']==[] and e['required_indices']==[] and e['core_source_max_abs_error'] is None and all(e[k] is None for k in CANDIDATE_KEYS),'Partial exact bypass');continue
        require(e['next_official_session'][:7]!=e['date'][:7],'Formal month end witness')
        _strict_equal(e['required_indices'],e['original_eligible_indices'],'Required exact original E')
        require(type(e['cells']) is list and len(e['cells'])==49,'All49 original source cells');raw=[];errs=[]
        for j,c in enumerate(e['cells']):
            require(type(c) is dict and tuple(c)==CELL_KEYS,'Cell schema')
            for k,v in dict(column_index=j,symbol=e['symbols'][j],frozen_core=float(b['core'][i,j]),original_eligible=bool(b['good'][i,j]),source_available=bool(b['good'][i,j])).items():_strict_equal(c[k],v,'Cell '+k)
            require(type(c['original_finite']) is bool,'Finite domain bool')
            if c['original_finite']:raw.append(old._number(c['original_raw'],'Raw'))
            else:require(c['original_raw'] is None,'Nonfinite raw marker');raw.append(np.nan)
            if c['sigma63'] is not None:old._number(c['sigma63'],'Sigma')
            if c['original_eligible']:_measurement(c['measurement'])
            else:require(c['measurement'] is None,'No Efalse new measure')
        ranks=old._rank(raw);_strict_equal(e['u0_indices'],np.flatnonzero(np.isfinite(raw)).tolist(),'Finite original domain')
        for j,c in enumerate(e['cells']):
            rebuilt=e['core_weight']*float(ranks[j]);err=abs(rebuilt-c['frozen_core']);errs.append(err)
            for k,v in dict(original_rank=float(ranks[j]),rebuilt_core=rebuilt,core_abs_error=err).items():_strict_equal(c[k],v,'Core '+k)
            require(err<=1e-12,'Original C parity')
        _strict_equal(e['core_source_max_abs_error'],max(errs),'Maximum core error')
        if source_stage:require(e['branch']=='source_verified' and all(e[k] is None for k in CANDIDATE_KEYS),'No source-stage basket selection')
    return features[:n]


def _finalize(features,source_stage):
    for e in features:
        if e['initial_partial']:continue
        expected={};M={j:exact._rational(e['cells'][j]['measurement']['momentum']) for j in e['required_indices']}
        elig=[j for j in e['required_indices'] if M[j]>0];excluded=[j for j in e['required_indices'] if M[j]<=0]
        ordered=sorted(elig,key=lambda j:(-M[j],e['symbols'][j]));short=ordered[:6];trios=[]
        for indices in combinations(sorted(short,key=lambda j:e['symbols'][j]),3):
            trios.append(dict(indices=list(indices),symbols=[e['symbols'][j] for j in indices],statistics=exact.basket([e['cells'][j]['measurement']['block_returns'] for j in indices])))
        common=[k for k,trio in enumerate(trios) if trio['statistics']['supported']]
        branch='cash' if len(elig)<2 else ('two_names' if len(elig)==2 else ('zero_variance' if not common else 'optimized'));winners={}
        for arm,var in zip(ARMS[1:],('joint_variance','diagonal_variance')):
            best=None;ties=[]
            if branch=='optimized':
                for k in common:
                    stat=trios[k]['statistics']
                    cmp=1 if best is None else exact.compare_ratios(stat['mean'],stat[var],trios[best]['statistics']['mean'],trios[best]['statistics'][var])
                    if cmp>0:best=k;ties=[k]
                    elif cmp==0:ties.append(k)
                chosen=sorted(trios[best]['indices'],key=lambda j:(-M[j],e['symbols'][j]))
            else:chosen=[] if branch=='cash' else short[:(2 if branch=='two_names' else 3)]
            winners[arm]=dict(branch=branch,winning_trio_index=best,tied_trio_indices=ties,selected_indices=chosen,selected_symbols=[e['symbols'][j] for j in chosen])
        expected=dict(new_eligible_indices=elig,excluded_momentum_indices=excluded,shortlist_indices=short,trios=trios,common_trio_indices=common,winners=winners)
        if source_stage:
            e['branch']=branch
            for k,v in expected.items():e[k]=v
        else:
            _strict_equal(e['branch'],branch,'Final branch')
            for k,v in expected.items():_strict_equal(e[k],v,'Cached candidate '+k)


def _select(study,meta,b,features,n,source_stage=False,*,incumbent_validator=None,expected_incumbent_orders=None):
    fs=_cached(features,meta,b,n,source_stage);w=b['width'];pre=b['pre'].copy();calls=[]
    source=dict(baseline=b['base'],core=b['core'],core_weight=b['cw'],eligible=b['good'],state_calm10=b['calm'],state_weak_narrow=b['weak'])
    def boundary(score,eligible,names,effects,original,policy):
        require(not calls,'Single original episode');require(np.array_equal(score,pre) and np.array_equal(eligible,b['good']) and names==meta['symbols'][:n] and np.array_equal(effects,b['effects']),'Original episode inputs')
        require(policy==POLICY and all(np.array_equal(original[k],v) for k,v in source.items() if k!='eligible'),'Original episode support/policy');calls.append(True)
        return study.ep.select_boundary(score,eligible,names,effects,original,policy)
    chosen,post=old.core_adapter.score_and_choose('incumbent66',0.,np.full((n,w),np.nan),source,b['meta'],n,pre_score=pre,effects=b['effects'],episode_policy=POLICY.copy(),boundary_selector=boundary)
    require(len(calls)==1,'Exactly one original episode');post=old._matrix(post,(n,w),n,'Original F');ch=[[int(j) for j in row] for row in chosen]
    orders={ARMS[0]:study.rp.make_orders(ch,post,b['meta'])};scores={ARMS[0]:post};pres={ARMS[0]:pre};choices={ARMS[0]:ch}
    if expected_incumbent_orders is not None:
        require(type(expected_incumbent_orders) is list and len(expected_incumbent_orders)==n,'Expected incumbent axis');_strict_equal(orders[ARMS[0]],expected_incumbent_orders,'Frozen incumbent orders')
    if incumbent_validator is not None:
        require(callable(incumbent_validator),'Incumbent validator callable');receipt=incumbent_validator(deepcopy(orders[ARMS[0]]),post.copy(),pre.copy());require(type(receipt) is dict and receipt.get('passed') is True,'Independent incumbent gate failed')
    _finalize(fs,source_stage)
    for arm in ARMS[1:]:
        display=np.zeros((n,w),float);rows=[]
        for i,e in enumerate(fs):
            if e['initial_partial']:rows.append(ch[i].copy());display[i]=post[i]
            else:
                selected=e['winners'][arm]['selected_indices'];rows.append(selected.copy());short=e['shortlist_indices'];S=len(short)
                for j in selected:display[i,j]=float(S-short.index(j))
        choices[arm]=rows;scores[arm]=display;pres[arm]=display.copy()
        for i,e in enumerate(fs):
            if e['initial_partial']:pres[arm][i]=pre[i]
        orders[arm]=study.rp.make_orders(rows,display,b['meta'])
        for i,e in enumerate(fs):
            if e['initial_partial']:_strict_equal(orders[arm][i],orders[ARMS[0]][i],'Partial ordered copy')
    formal=[e for e in fs if not e['initial_partial']];cells=[c for e in formal for c in e['cells']];trios=[t for e in formal for t in e['trios']];req=sum(len(e['required_indices']) for e in formal)
    counts=dict(events=n,symbols_per_event=w,initial_partial_events=n-len(formal),formal_events=len(formal),calm_formal_events=sum(e['calm'] for e in formal),noncalm_formal_events=sum(not e['calm'] for e in formal),original_checked_cells=len(cells),original_finite_cells=sum(c['original_finite'] for c in cells),original_eligible_cells=int(b['good'].sum()),source_required_cells=req,source_available_cells=sum(c['source_available'] for c in cells),source_price_observations=253*req,momentum_estimates=req,block_return_estimates=12*req,new_eligible_cells=sum(len(e['new_eligible_indices']) for e in formal),momentum_excluded_cells=sum(len(e['excluded_momentum_indices']) for e in formal),shortlist_cells=sum(len(e['shortlist_indices']) for e in formal),enumerated_trios=len(trios),common_supported_trios=sum(t['statistics']['supported'] for t in trios),zero_joint_variance_trios=sum(exact._rational(t['statistics']['joint_variance'])==0 for t in trios),zero_diagonal_variance_trios=sum(exact._rational(t['statistics']['diagonal_variance'])==0 for t in trios),cash_events=sum(e['branch']=='cash' for e in formal),two_name_events=sum(e['branch']=='two_names' for e in formal),zero_variance_events=sum(e['branch']=='zero_variance' for e in formal),optimized_events=sum(e['branch']=='optimized' for e in formal),candidate_changed_holdings_events=sum(set(ch[i])!=set(choices[ARMS[1]][i]) for i in range(n)),diagonal_changed_holdings_events=sum(set(ch[i])!=set(choices[ARMS[2]][i]) for i in range(n)),original_episode_calls=1,experimental_episode_calls=0,new_model_fits=0,predictive_model_fits=0)
    require(tuple(counts)==COUNT_FIELDS and all(type(x) is int for x in counts.values()),'Counts schema')
    audit=dict(schema='monthly_basket_core251_decisions_v1',counts=counts,events=[dict(query_index=i,date=e['date'],branch=e['branch'],incumbent_choices=ch[i],arms={arm:choices[arm][i] for arm in ARMS}) for i,e in enumerate(fs)])
    return orders,scores,pres,audit


def prepare(study,arrays,meta,panel,calendar,source_review,n,*,source_validator=None,incumbent_validator=None,expected_incumbent_orders=None):
    before=old._snapshot(arrays);mm=deepcopy(meta);b=old._cache(study,arrays,meta,n);require(b['width']==49,'Original49 symbol axis')
    features=_features(panel,meta,calendar,source_review,b,n)
    if source_validator is not None:
        require(callable(source_validator),'Source validator callable');receipt=source_validator(deepcopy(features));require(type(receipt) is dict and receipt.get('passed') is True,'Independent source gate failed')
    out=_select(study,meta,b,features,n,True,incumbent_validator=incumbent_validator,expected_incumbent_orders=expected_incumbent_orders)
    require(before==old._snapshot(arrays) and meta==mm,'Original input mutation');return (features,)+out


def decisions(study,arrays,meta,features,n,*,incumbent_validator=None,expected_incumbent_orders=None):
    before=old._snapshot(arrays);mm=deepcopy(meta);ff=deepcopy(features);b=old._cache(study,arrays,meta,n);require(b['width']==49,'Original49 symbol axis')
    out=_select(study,meta,b,deepcopy(features),n,False,incumbent_validator=incumbent_validator,expected_incumbent_orders=expected_incumbent_orders)
    require(before==old._snapshot(arrays) and meta==mm and features==ff,'Cached input mutation');return out
