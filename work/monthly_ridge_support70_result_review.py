"""Independent Ridge-support result review; actual execution requires root consent.

No candidate study/features, selector, model or replay-engine imports. Only scalar
source formula reconstruction plus fixed-unit replay. All actual artifact reads
are confined to audit(), gated by --evaluate-complete. Synthetic tests use none.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import copy
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
O=ROOT/'outputs';P=O/'monthly_ridge_support70'
DEST=ROOT/'work/monthly_ridge_support70_result_review.json'
PREVIOUS=O/'monthly_leadership_core70'
LOCAL=O/'monthly_local_linear70'
TARGET=O/'monthly_target_definition70'
PRICE=O/'monthly_2016_extension/PRICES.npz'
BOUNDARY=ROOT/'work/monitor_research_boundary/settlement_20261014T200412Z'
BOUND=('DATA.npz','META.json','FEATURE_AUDIT.json','INPUT_HASHES.json','CACHE_HASHES.json','PLAN.json')
DESIGN_SHA='ea680355f21c703677c7242531817d426884d8db48b7530ae21b05bceedfe03f'
ARMS=('fixed_half_support','ridge_support')
NAMES=('original65','incumbent66','incumbent_scope67','incumbent_local67',*ARMS)
NEW_KEYS=('support_h','support_ref','support_weight','support_available')
POLICY=dict(id=7,family='level',model_id=2,budget=.05,ceiling=5,gap=.02)
TEMPLATE_SHA='e72914db3e6bd1838a78a2a131efce9b1855e22993249dccac47ff1e89681765'
REVIEWED={}

def reviewed_bytes(path):
    """Bind the exact bytes inspected; fail if a file changes during review."""
    path=Path(path).resolve();content=path.read_bytes()
    try:key=str(path.relative_to(P))
    except ValueError:key=str(path)
    digest=hashlib.sha256(content).hexdigest()
    assert key not in REVIEWED or REVIEWED[key]==digest,('File changed during review',key)
    REVIEWED[key]=digest
    return content


def read(path):return json.loads(reviewed_bytes(path))


def sha(path):return hashlib.sha256(reviewed_bytes(path)).hexdigest()


def display(symbol):return 'META' if symbol=='research_equity_FB' else symbol


def check_hashes(values,base=None):
    for name,digest in values.items():
        path=Path(name)
        if not path.is_absolute():path=base/path
        assert sha(path)==digest,str(path)


def stat(dates,nav,initial=100000.):
    nav=np.asarray(nav,float);assert np.isfinite(nav).all() and (nav>0).all()
    returns=nav[1:]/nav[:-1]-1
    years=(dates[-1]-dates[0]).days/365.2425
    wealth=nav/initial;peaks=np.maximum.accumulate(np.r_[1.,wealth])[1:]
    sd=np.std(returns,ddof=1)
    return dict(start=str(dates[0].date()),end=str(dates[-1].date()),years=years,
        final_equity=float(nav[-1]),cagr=float(wealth[-1]**(1/years)-1),
        sharpe=float(np.mean(returns)/sd*np.sqrt(252)) if sd>0 else None,
        max_drawdown=float(np.min(wealth/peaks-1)),trading_sessions=len(nav),
        total_return=float(wealth[-1]-1))


def replay(close,opening,orders,dates,*,execution='close',cost=0.):
    """Scalar holdings + cash; fixed-point fees instead of engine bisection."""
    calendar=close.index;plans={}
    for row in orders:
        signal=pd.Timestamp(row['date']);j=calendar.get_loc(signal)
        day=signal if execution=='close' else calendar[j+1]
        assert day not in plans
        if day<=dates[-1]:plans[day]=row
    holdings={};cash=100000.;nav=[];trades=[];periods=[];fees=0.
    for day in dates:
        if day in plans:
            row=plans[day];chosen=list(row['symbols'])
            assert len(chosen)==len(set(chosen)) and len(chosen) in (0,1,2,3)
            prices=close.loc[day] if execution=='close' else opening.loc[day]
            symbols=sorted(set(holdings)|set(chosen))
            assert all(math.isfinite(prices[s]) and prices[s]>0 for s in symbols)
            old={s:holdings.get(s,0.)*float(prices[s]) for s in symbols}
            before=cash+sum(old.values());post=before;count=len(chosen)
            def turnover(value):
                return sum(abs((value/count if s in chosen and count else 0.)-old[s]) for s in symbols)
            for _ in range(100):
                updated=before-cost*turnover(post)
                if updated==post:break
                post=updated
            assert abs(post+cost*turnover(post)-before)<=1e-12*max(before,1.)
            fee=cost*turnover(post);fees+=fee
            new={s:post/count/float(prices[s]) for s in chosen} if count else {}
            for symbol in symbols:
                delta=new.get(symbol,0.)-holdings.get(symbol,0.)
                if abs(delta)>1e-9:
                    trades.append(dict(date=str(day.date()),symbol=display(symbol),
                        side='buy' if delta>0 else 'sell',adjusted_units=abs(delta),
                        adjusted_price=float(prices[symbol]),dollars=abs(delta)*float(prices[symbol]),
                        cost=abs(delta)*float(prices[symbol])*cost))
            holdings=new;cash=0. if count else post
            periods.append(dict(signal_date=row['signal_date'],execution_date=str(day.date()),
                symbols=[display(s) for s in chosen],scores=row['scores'],fee=fee))
        assert all(math.isfinite(close.at[day,s]) and close.at[day,s]>0 for s in holdings)
        nav.append(cash+sum(q*float(close.at[day,s]) for s,q in holdings.items()))
    nav=np.asarray(nav);metrics=stat(dates,nav)
    metrics.update(cost_per_side=cost,execution=execution,fees_paid=fees,trade_rows=len(trades),
                   dollar_turnover=sum(t['dollars'] for t in trades))
    annual=[]
    for year in sorted(set(dates.year)):
        ids=np.flatnonzero(dates.year==year)
        if len(ids)<2:continue
        base=nav[ids[0]-1] if ids[0] else 100000.
        annual.append(dict(year=int(year),return_=float(nav[ids[-1]]/base-1),partial=year in (2016,2026)))
    metrics['annual_returns']=annual
    return metrics,nav,trades,periods


def slice_stat(dates,nav,start,end):
    use=(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end))
    values=nav[use];return stat(dates[use],values/values[0]*100000.)


def measures(dates,nav,m,n=88):
    split=53 if n>=88 else n//2
    ranges=dict(training=(m['dates'][0],m['ends'][n-1]),
                early=(m['dates'][0],m['ends'][split-1]),recent=(m['dates'][split],m['ends'][n-1]))
    return {k:{t:v for t,v in slice_stat(dates,nav,*limits).items()
               if t in ('cagr','sharpe','max_drawdown','total_return')} for k,limits in ranges.items()}


def compare(actual,expected,errors,path=''):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(path,set(actual)^set(expected))
        for key in expected:compare(actual[key],expected[key],errors,path+'/'+key)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),(path,len(actual),len(expected))
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,errors,path+'/'+str(i))
    elif expected is None or isinstance(expected,(str,bool,int)):
        assert actual==expected,(path,actual,expected)
    else:
        assert math.isfinite(actual) and math.isfinite(expected),path
        delta=abs(actual-expected);errors.append((path,delta))
        assert math.isclose(actual,expected,rel_tol=2e-10,abs_tol=2e-9),(path,actual,expected,delta)


def ordinal_rank(values):
    """Average percentile ranks from scalar pair comparisons, no pandas rank."""
    n=len(values)
    return np.array([(sum(y<x for y in values)+(sum(y==x for y in values)+1)/2)/n-.5 for x in values])


def assert_array_bytes(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert actual.tobytes()==expected.tobytes()


def assert_exact_tree(actual,expected,path=''):
    """Exact output parity, including IEEE signed zero, independent of JSON layout."""
    assert type(actual) is type(expected),(path,type(actual),type(expected))
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys(),path
        for key in expected:assert_exact_tree(actual[key],expected[key],path+'/'+str(key))
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,(x,y) in enumerate(zip(actual,expected)):assert_exact_tree(x,y,path+'/'+str(i))
    elif isinstance(expected,float):assert np.float64(actual).tobytes()==np.float64(expected).tobytes(),path
    else:assert actual==expected,path


def gates(row,base):
    return dict(training_cagr=row['training']['cagr']>=base['training']['cagr']-1e-12,
        training_sharpe=row['training']['sharpe']>=base['training']['sharpe']-.05,
        training_max_drawdown=row['training']['max_drawdown']>=base['training']['max_drawdown']-.03,
        early_cagr=row['early']['cagr']>=base['early']['cagr']-.025,
        recent_cagr=row['recent']['cagr']>=base['recent']['cagr']-.025)


def winner(rows):
    assert len(rows)==6
    allowed=[3]+[i for i in (4,5) if all(gates(rows[i],rows[3]).values())]
    top=max(rows[i]['training']['cagr'] for i in allowed)
    return min(i for i in allowed if top-rows[i]['training']['cagr']<=1e-12)


def original_prefix(a,m,n=None):
    """Independently rebuild the frozen local67 pre-score before its episode."""
    shape=np.asarray(a['baseline']).shape
    assert len(shape)==2 and shape[1]>0
    if n is None:n=shape[0]
    assert not isinstance(n,(bool,np.bool_)) and isinstance(n,(int,np.integer)) and 0<=n<=shape[0]
    def matrix(key,boolean=False):
        value=np.asarray(a[key]);assert value.shape==shape
        if boolean:assert value.dtype.kind=='b';return value[:n].copy()
        return value[:n].astype(float,copy=True)
    def state(key):
        value=np.asarray(a[key]);assert value.shape==(shape[0],) and value.dtype.kind=='b'
        return value[:n].copy()
    names=m['symbols'][:n]
    assert len(names)==n and all(len(row)==shape[1] and len(set(row))==shape[1] for row in names)
    assert all(isinstance(x,str) and x for row in names for x in row)
    baseline=matrix('baseline');core=matrix('core');eligible=matrix('eligible',True)
    cw=np.asarray(a['core_weight'])
    if cw.shape==(shape[0],):cw=cw[:,None]
    assert cw.shape in ((shape[0],1),shape)
    cw=np.broadcast_to(cw[:n].astype(float),(n,shape[1])).copy()
    assert np.isfinite(baseline).all() and np.isfinite(core).all() and np.isin(cw,[.5,.52]).all()
    support=(baseline-core)/(1-cw)
    assert np.isfinite(support).all() and (np.abs(support)<=.5+1e-12).all()
    weak=state('state_weak_narrow');calm=state('state_calm10')
    original=matrix('pre_score');scope=original.copy();effect=matrix('effects')
    assert np.isfinite(original).all() and np.isfinite(effect).all()
    use=eligible&matrix('forecast_available',True);logrank=matrix('rank_log_excess')
    assert np.isfinite(logrank).all() and (np.abs(logrank)<=.5+1e-12).all()
    oldml=matrix('old_ml4');extension=matrix('extension_available',True)
    mask=weak[:,None]&use;delta=np.zeros(scope.shape)
    delta[mask]=.05*(logrank[mask]-oldml[mask]);scope[delta!=0.]+=delta[delta!=0.]
    mask=(calm&~weak)[:,None]&extension&use;delta[:]=0.
    delta[mask]=.01*(logrank[mask]-support[mask]);scope[delta!=0.]+=delta[delta!=0.]
    local=scope.copy();rank=logrank.copy();ready=np.zeros(n,bool)
    available=matrix('model_available',True)
    rawc=matrix('raw_local_constant');rawl=matrix('raw_local_linear');raws=matrix('raw_source_log')
    for i in range(n):
        ids=np.flatnonzero(use[i]);ready[i]=bool(len(ids) and available[i,ids].all()
            and np.isfinite(rawc[i,ids]).all() and np.isfinite(rawl[i,ids]).all())
        if ready[i] and not np.array_equal(rawl[i,ids],raws[i,ids]):rank[i,ids]=ordinal_rank(rawl[i,ids])
    coef=.25*(.05*weak[:,None]+.01*(calm&~weak)[:,None]*extension)
    mask=use&ready[:,None];delta[:]=0.;delta[mask]=coef[mask]*(rank[mask]-logrank[mask])
    local[delta!=0.]+=delta[delta!=0.]
    assert np.isfinite(local).all()
    return dict(shape=shape,n=n,names=names,baseline=baseline,core=core,cw=cw,eligible=eligible,
        support=support,weak=weak,calm=calm,effects=effect,original=original,scope=scope,local=local)


def episode(pre,x,*,policy=POLICY):
    """Actual original rule: lock pre top2, rerank only eligible rank3..5/gap.02."""
    assert all(policy.get(k)==v for k,v in POLICY.items())
    assert pre.shape==x['baseline'].shape and np.isfinite(pre).all()
    score=pre.copy();choices=[];records=[]
    for i,names in enumerate(x['names']):
        order=sorted(np.flatnonzero(x['eligible'][i]),key=lambda j:(-pre[i,j],names[j]))
        group=[];selected=order if len(order)==2 else []
        budget=min(.05,max(0.,.10-.025*float(x['calm'][i])-.05*float(x['weak'][i])))
        if len(order)>=3:
            group=[j for j in order[2:5] if pre[i,order[2]]-pre[i,j]<=.02]
            for j in group:
                target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
                score[i,j]+=budget*(target-x['support'][i,j])
            third=min(group,key=lambda j:(-score[i,j],names[j]))
            selected=sorted(order[:2]+[third],key=lambda j:(-score[i,j],names[j]))
        choices.append(selected)
        records.append(dict(top2=[names[j] for j in order[:2]],cohort=[names[j] for j in group],
            selected=[names[j] for j in selected],budget=budget,eligible_count=len(order)))
    return choices,score,records


def support_inputs(a,x,arm):
    """Prefix-only geometric support, never a new ranking/eligibility filter."""
    assert arm in ARMS
    n=x['n'];shape=x['shape'];values={}
    for key in NEW_KEYS:
        v=np.asarray(a[key]);assert v.shape==shape
        if key=='support_available':
            assert v.dtype.kind=='b';values[key]=v[:n].copy()
        else:values[key]=v[:n].astype(float,copy=True)
    available=values['support_available']
    for key in ('model_available','forecast_available'):
        v=np.asarray(a[key]);assert v.shape==shape and v.dtype.kind=='b'
        assert_array_bytes(available,v[:n])
    for key in ('support_h','support_ref','support_weight'):
        v=values[key]
        assert np.isfinite(v[available]).all() and np.isnan(v[~available]).all(),key
        assert (v[available]>=0.).all(),key
    assert (values['support_weight'][available]<=1.).all()
    for i,j in zip(*np.where(available)):
        h=values['support_h'][i,j];ref=values['support_ref'][i,j]
        expected=1. if ref==0. and h==0. else 0. if ref==0. else ref/(ref+h)
        assert math.isclose(values['support_weight'][i,j],expected,rel_tol=0.,abs_tol=1e-12),(i,j)
    active=available&x['eligible']
    c=np.full((n,shape[1]),np.nan)
    c[active]=.5 if arm=='fixed_half_support' else values['support_weight'][active]
    return c,active,values


def support_decisions(a,m,arm,n=None):
    x=original_prefix(a,m,n);c,active,values=support_inputs(a,x,arm)
    pre=x['local'].copy();delta=np.zeros(pre.shape)
    for i,j in zip(*np.where(active)):
        delta[i,j]=(c[i,j]-1.)*(x['local'][i,j]-x['original'][i,j])
    assert np.isfinite(delta).all()
    changed=delta!=0.;pre[changed]+=delta[changed]
    assert pre[~active].tobytes()==x['local'][~active].tobytes()
    # Core, support, states, original effects stay untouched. Anchors follow pre.
    chosen,scores,records=episode(pre,x)
    original_choices,original_scores,old_records=episode(x['local'],x)
    for i in range(x['n']):
        if not changed[i].any():
            assert scores[i].tobytes()==original_scores[i].tobytes() and chosen[i]==original_choices[i]
        records[i].update(date=m['dates'][i],available_cells=int(values['support_available'][i].sum()),
            active_cells=int(active[i].sum()),changed_pre_cells=int(changed[i].sum()),
            maximum_pre_delta=float(np.max(abs(delta[i]))),
            original_top2=old_records[i]['top2'],original_selected=old_records[i]['selected'])
    details=dict(months=records,available_cells=int(values['support_available'].sum()),active_cells=int(active.sum()),
        changed_pre_cells=int(changed.sum()),maximum_pre_delta=float(np.max(abs(delta))) if delta.size else 0.,
        same_availability_in_both_arms=True,original_support_preserved=True,
        changed_pre_top2_and_boundary_anchored=True,returned_choice_not_globally_reranked=True,
        zero_delta_updates_explicitly_skipped=True,
        peryear={year:dict(available_cells=int(values['support_available'][ids].sum()),
            active_cells=int(active[ids].sum()),
            minimum_support=float(np.min(values['support_weight'][ids][values['support_available'][ids]]))
                if values['support_available'][ids].any() else None,
            maximum_support=float(np.max(values['support_weight'][ids][values['support_available'][ids]]))
                if values['support_available'][ids].any() else None)
            for year in sorted(set(day[:4] for day in m['dates'][:x['n']]))
            for ids in [np.array([day.startswith(year) for day in m['dates'][:x['n']]])]})
    return (chosen,scores),details


def verify_clock(cached,meta,a,audit):
    """Independently reconstruct training membership, without any geometry/model fit."""
    dates=pd.DatetimeIndex(cached['dates']);ends=pd.DatetimeIndex(cached['ends'])
    event_dates=pd.DatetimeIndex(cached['event_dates']);train=np.asarray(cached['train_plain'])
    event=np.asarray(cached['event_plain']);good=np.asarray(cached['common_good'])
    assert train.shape==(len(dates),10) and event.shape==(len(event_dates),10)
    assert good.shape==(len(dates),) and good.dtype.kind=='b'
    assert not dates.hasnans and not ends.hasnans and not event_dates.hasnans
    assert len(ends)==len(dates) and (ends>dates).all()
    assert (ends.to_period('M')==dates.to_period('M')+1).all()
    assert list(event_dates)==[pd.Timestamp(d) for d in meta['dates'] for _ in range(a['baseline'].shape[1])]
    targets=('simple_excess','log_excess','vol_scaled','loss_utility')
    usable=good.copy()&np.isfinite(train).all(axis=1)
    for name in targets:
        y=np.asarray(cached['target_'+name]);assert y.shape==(len(dates),)
        usable&=np.isfinite(y)
    months=dates.to_period('M')
    for month in sorted(set(months)):
        rows=months==month
        if int(usable[rows].sum())<2:usable[rows]=False
    query=a['eligible'].reshape(-1)&np.isfinite(event).all(axis=1)
    assert audit['alpha']==10. and audit['half_life_months']==12. and audit['min_months']==18
    assert audit['lookback_years']==5 and audit['feature_count']==10 and audit['targets']==list(targets)
    assert audit['response_fit_performed'] is False and audit['event_label_filter_used'] is False
    for key in ('shared_rows_and_weights','training_only_standardization','targets_used_only_for_finite_cohort'):
        assert audit[key] is True,key
    expected_available=np.zeros(len(event_dates),bool);records=[]
    assert [z['year'] for z in audit['annual']]==sorted(set(event_dates.year))
    for row in audit['annual']:
        year=row['year'];cut=pd.Timestamp(year-1,12,31);lower=pd.Timestamp(year-6,12,31)
        selected=usable&(dates>lower)&(ends<=cut);predict=query&(event_dates.year==year)
        number=len(set(months[selected]));status='fit' if number>=18 else 'cold_start'
        expected=dict(year=year,cutoff=str(cut.date()),lookback_exclusive_start=str(lower.date()),
            training_rows=int(selected.sum()),training_months=number,prediction_rows=int(predict.sum()),
            latest_label_end=str(ends[selected].max().date()) if selected.any() else None,
            first_training_signal=str(dates[selected].min().date()) if selected.any() else None,
            latest_training_signal=str(dates[selected].max().date()) if selected.any() else None,
            status=status,training_indices=np.flatnonzero(selected).tolist(),prediction_indices=np.flatnonzero(predict).tolist())
        for key,value in expected.items():assert row[key]==value,(year,key)
        if status=='fit':expected_available[predict]=True
        records.append({k:v for k,v in expected.items() if k not in ('training_indices','prediction_indices')})
    assert_array_bytes(expected_available.reshape(a['baseline'].shape),a['support_available'])
    assert audit['available_rows']==int(expected_available.sum())
    return dict(annual=records,actual_common_cohort_and_maturity_exact=True,
        original_query_eligibility_only=True,geometry_not_recomputed=True)


def support_change_summary(a,m,details):
    """Independent recorded integration summary, using scalar reconstructed deltas."""
    x=original_prefix(a,m);out={}
    for arm in ARMS:
        _,active,values=support_inputs(a,x,arm);weight=values['support_weight'];records=[];years={}
        for i,row in enumerate(details[arm]['months']):
            w=weight[i,active[i]]
            records.append(dict(date=m['dates'][i],active_cells=int(active[i].sum()),
                support_min=float(w.min()) if len(w) else None,
                support_median=float(np.median(w)) if len(w) else None,
                support_max=float(w.max()) if len(w) else None,
                changed_score_cells=row['changed_pre_cells'],maximum_abs_delta=row['maximum_pre_delta']))
        for year in sorted({day[:4] for day in m['dates']}):
            ids=np.array([day.startswith(year) for day in m['dates']]);w=weight[ids][active[ids]]
            years[year]=dict(available_cells=len(w),weight_min=float(w.min()) if len(w) else None,
                weight_median=float(np.median(w)) if len(w) else None,weight_max=float(w.max()) if len(w) else None)
        out[arm]=dict(maximum_abs_score_delta=details[arm]['maximum_pre_delta'],
            active_cells=int(active.sum()),years=years,records=records,increment_magnitude_not_increased=True)
    return out


def source_scores(a,m):
    x=original_prefix(a,m);base=[]
    for i,names in enumerate(x['names']):
        ids=sorted(np.flatnonzero(x['eligible'][i]),key=lambda j:(-x['baseline'][i,j],names[j]))[:3]
        base.append(ids if len(ids)>=2 else [])
    result={'original65':(base,x['baseline'].copy())}
    for name,key in (('incumbent66','original'),('incumbent_scope67','scope'),('incumbent_local67','local')):
        chosen,score,_=episode(x[key],x);result[name]=(chosen,score)
    details={}
    for arm in ARMS:result[arm],details[arm]=support_decisions(a,m,arm)
    return result,details


def audit():
    """Only entry point permitted to read actual candidates and result artifacts."""
    assert sha(P/'DESIGN.json')==DESIGN_SHA==reviewed_bytes(P/'DESIGN.sha256').decode().strip()
    design=read(P/'DESIGN.json');grid=read(P/'GRID.json')
    assert len(grid)==6 and grid==design['configs']
    assert [z['id'] for z in grid]==list(range(6))
    assert [z['base'] for z in grid]==list(NAMES[:4])+['incumbent_local67']*2
    assert [z['mode'] for z in grid]==['none']*4+list(ARMS)
    assert design['boundary']==str(BOUNDARY) and design['protected_count']==21444
    for name,digest in design['source_reviews'].items():assert sha(ROOT/name)==digest,name
    frozen=read(P/'FROZEN.json')
    assert sha(P/'FROZEN.json')==reviewed_bytes(P/'FROZEN.sha256').decode().strip()
    check_hashes(frozen['hashes'],P)
    plan=read(P/'PLAN.json')
    assert plan['design_sha256']==DESIGN_SHA and plan['design']==design
    assert plan['protected_count']==21444 and plan['boundary']==str(BOUNDARY)
    assert len(plan['raw_source_hashes'])==64
    assert plan['raw_source_hashes']==read(LOCAL/'PLAN.json')['raw_source_hashes']
    for field in ('raw_source_hashes','hashes','source_hashes'):check_hashes(plan[field])
    check_hashes(read(P/'CACHE_HASHES.json'),P)
    feature_review=read(P/'ACTUAL_FEATURE_REVIEW.json');assert feature_review['passed'] is True
    for name in BOUND:assert feature_review['hashes'][name]==sha(P/name),name
    inputs=read(P/'INPUT_HASHES.json');assert inputs[str(PRICE)]==sha(PRICE)
    check_hashes(inputs)
    meta=read(P/'META.json');assert len(meta['dates'])==122
    assert len(meta['etfs'])==17 and len(set(meta['etfs']))==17
    assert all(len(row)==49 and len(set(row))==49 and len(set(row)&set(meta['etfs']))==17 for row in meta['symbols'])
    assert reviewed_bytes(P/'META.json')==reviewed_bytes(LOCAL/'META.json')
    policy=read(P/'INCUMBENT_POLICY.json');assert all(policy.get(k)==v for k,v in POLICY.items())
    assert policy==read(LOCAL/'INCUMBENT_POLICY.json')
    sha(LOCAL/'DATA.npz')
    with np.load(P/'DATA.npz',allow_pickle=False) as z:a={key:z[key] for key in z.files}
    with np.load(LOCAL/'DATA.npz',allow_pickle=False) as z:
        assert len(z.files)==27 and len(a)==31 and set(a)==set(z.files)|set(NEW_KEYS)
        inherited=set(z.files)
        for key in z.files:assert_array_bytes(a[key],z[key])
    assert a['baseline'].shape==(122,49)
    for key in NEW_KEYS:
        assert a[key].shape==(122,49)
        assert a[key].dtype==np.dtype('bool' if key=='support_available' else 'float64')
    feature_audit=read(P/'FEATURE_AUDIT.json');dv=feature_audit['driver_validation']
    assert dv['raw_source_hashes']==plan['raw_source_hashes'] and dv['raw_source_count']==64
    assert dv['inherited_array_count']==27 and dv['prepared_array_count']==31
    for key in ('inherited_arrays_byte_exact','meta_byte_exact','four_control_orders_exact',
                'source_common_good_exact','source_four_targets_unchanged','query_eligible_finiteX_only',
                'original_model_and_forecast_available_exact','no_response_refit'):
        assert dv[key] is True,key
    # The independent six-bound-file actual audit certifies annual maturity/cohort,
    # weighted scaler, Cholesky geometry, reference support and original availability.
    assert feature_review['passed'] is True and dv['new_candidate_choices'] is False
    check_hashes(feature_audit['source_bindings'],TARGET)
    assert len(dv['monthly_counts'])==122
    for i,row in enumerate(dv['monthly_counts']):
        assert row==dict(date=meta['dates'][i],available_cells=int(a['support_available'][i].sum()))
    assert str(TARGET/'TRAIN_INPUTS.npz') in inputs
    assert sha(TARGET/'TRAIN_INPUTS.npz')==inputs[str(TARGET/'TRAIN_INPUTS.npz')]
    with np.load(TARGET/'TRAIN_INPUTS.npz',allow_pickle=False) as z:cached={key:z[key] for key in z.files}
    clock_review=verify_clock(cached,meta,a,feature_audit['geometry'])
    with np.load(PRICE,allow_pickle=False) as z:
        index=pd.to_datetime(z['dates']);close=pd.DataFrame(z['close'],index=index,columns=z['symbols'])
        opening=pd.DataFrame(z['opening'],index=index,columns=z['symbols'])
    dates=close.loc[meta['dates'][0]:meta['ends'][-1]].index
    results=read(P/'RESULTS.json');validation=read(P/'VALIDATION.json');trials=read(P/'TRAIN_TRIALS.json')
    assert [row['spec'] for row in trials]==grid and frozen['trials']==6
    assert set(results)=={'primary',*NAMES,'qqq'}
    errors=[];navs={};summary={};orders={name:read(P/(name+'_ORDERS.json')) for name in results}
    assert orders['qqq']==[dict(date=meta['dates'][0],signal_date=meta['dates'][0],end=meta['ends'][-1],symbols=['QQQ'],scores=[])]
    assert all(len(row['symbols']) in (0,2,3) for name in ('primary',*NAMES) for row in orders[name])
    expected,decisions=source_scores(a,meta)
    for name,(chosen,scores) in expected.items():
        assert len(orders[name])==122
        for i,row in enumerate(orders[name]):
            assert row['date']==meta['dates'][i] and row['end']==meta['ends'][i] and row['signal_date']==row['date']
            assert row['symbols']==[meta['symbols'][i][j] for j in chosen[i]],(name,i)
            np.testing.assert_allclose(row['scores'],scores[i,chosen[i]],rtol=0.,atol=1e-14)
    source_orders={'original65':'original65','incumbent66':'incumbent66',
        'incumbent_scope67':'incumbent_scope67','incumbent_local67':'local_linear'}
    for name,source in source_orders.items():assert orders[name]==read(O/'monthly_local_linear70'/(source+'_ORDERS.json'))
    slices=dict(early2016_2020=(meta['dates'][0],'2020-12-31'),original2021_2026=('2020-12-31',meta['ends'][-1]),
        training2016_2023=(meta['dates'][0],'2023-11-30'),later34=('2023-11-30',meta['ends'][-1]))
    nav_error=0.
    for name,rows in orders.items():
        summary[name]={}
        for label,mode,cost in [('full','close',0.),('next_open_zero','open',0.),('next_open_cost','open',.0005)]:
            met,nav,trades,periods=replay(close,opening,rows,dates,execution=mode,cost=cost)
            actual=results[name] if label=='full' else results[name][label]
            compare(met,actual['metrics'],errors,name+'/'+label+'/metrics')
            computed={key:slice_stat(dates,nav,*limits) for key,limits in slices.items()}
            compare(computed,actual['slices'],errors,name+'/'+label+'/slices')
            summary[name][label]=dict(cagr=met['cagr'],sharpe=met['sharpe'],max_drawdown=met['max_drawdown'],
                fees=met['fees_paid'],turnover=met['dollar_turnover'],trade_rows=met['trade_rows'],
                independently_replayed_nav_sha256=hashlib.sha256(nav.tobytes()).hexdigest(),
                independently_replayed_nav_observations=len(nav),holding_periods=len(periods),slices=computed)
            assert len({(t['date'],t['symbol']) for t in trades})==len(trades)
            permitted_dates={r['date'] if mode=='close' else str(close.index[close.index.get_loc(pd.Timestamp(r['date']))+1].date()) for r in rows}
            assert all(t['date'] in permitted_dates for t in trades)
            assert [h['signal_date'] for h in periods]==[row['signal_date'] for row in rows
                if mode=='close' or close.index[close.index.get_loc(pd.Timestamp(row['date']))+1]<=dates[-1]]
            if label=='full':
                navs[name]=nav
                daily=read(P/(name+'_DAILY.json'))
                assert [row['date'] for row in daily]==[str(day.date()) for day in dates]
                recorded=np.array([row['equity'] for row in daily])
                nav_error=max(nav_error,float(np.max(np.abs(nav/recorded-1))))
                np.testing.assert_allclose(nav,recorded,rtol=2e-11,atol=1e-5)
                compare(trades,read(P/(name+'_TRADES.json')),errors,name+'/trades')
                compare(periods,read(P/(name+'_HOLDINGS.json')),errors,name+'/holdings')
    previous_results=read(PREVIOUS/'RESULTS.json')
    for name,oldname in [(n,n) for n in NAMES[:4]]:
        for key in ('metrics','slices','next_open_zero','next_open_cost'):
            assert_exact_tree(results[name][key],previous_results[oldname][key],name+'/'+key)
        for suffix in ('ORDERS','DAILY','TRADES','HOLDINGS'):
            assert_exact_tree(read(P/(name+'_'+suffix+'.json')),read(PREVIOUS/(oldname+'_'+suffix+'.json')),name+'/'+suffix)
    sha(P/'TRAIN_RETURNS.npz')
    with np.load(P/'TRAIN_RETURNS.npz',allow_pickle=False) as z:
        assert set(z.files)=={'returns','bounds'}
        train_end=dates.get_loc(pd.Timestamp(meta['ends'][87]))
        expected_bounds=np.array([dates.get_loc(pd.Timestamp(day)) for day in meta['dates'][:88]]+[train_end])
        np.testing.assert_array_equal(z['bounds'],expected_bounds)
        expected_returns=np.array([navs[name][1:train_end+1]/navs[name][:train_end]-1 for name in NAMES])
        assert z['returns'].shape==expected_returns.shape
        np.testing.assert_allclose(z['returns'],expected_returns,rtol=1e-10,atol=1e-12)
    measured=[measures(dates,navs[name],meta) for name in NAMES]
    for i,row in enumerate(trials):
        compare(measured[i],{key:row[key] for key in measured[i]},errors,'train/'+str(i))
        assert row['admitted']==all(gates(measured[i],measured[3]).values())
    primary=winner(measured)
    assert frozen['selected']['primary']['spec']==grid[primary]
    assert frozen['primary_source']==grid[primary] and frozen['all_diagnostics_frozen'] is True
    assert frozen['new_admitted']==sum(int(all(gates(measured[i],measured[3]).values())) for i in (4,5))
    assert orders['primary']==orders[NAMES[primary]]
    assert set(frozen['selected'])=={'primary',*NAMES}
    for i,name in enumerate(NAMES):assert frozen['selected'][name]['spec']==grid[i]
    for name,row in frozen['selected'].items():
        i=row['spec']['id'];compare(measured[i],{k:row[k] for k in measured[i]},errors,'frozen/'+name)
        assert results[name]['selection']==row
    assert results['qqq']['selection'] is None
    folds=[];joined=[];local_join=[]
    daily_returns={name:nav[1:]/nav[:-1]-1 for name,nav in navs.items()}
    for n,end in ((36,60),(60,76),(76,88)):
        stats=[measures(dates,navs[name],meta,n) for name in NAMES];selected=winner(stats)
        folds.append(dict(train_end=meta['ends'][n-1],test_end=meta['ends'][end-1],spec=grid[selected]))
        fold_mask=(dates[1:]>pd.Timestamp(meta['dates'][n]))&(dates[1:]<=pd.Timestamp(meta['ends'][end-1]))
        joined.extend(daily_returns[NAMES[selected]][fold_mask]);local_join.extend(daily_returns['incumbent_local67'][fold_mask])
    walk=read(P/'WALKFORWARD.json');assert walk['folds']==folds
    walk_dates=dates[(dates>=pd.Timestamp(meta['dates'][36]))&(dates<=pd.Timestamp(meta['ends'][87]))]
    for key,returns in [('metrics',joined),('incumbent',local_join)]:
        z=stat(walk_dates,np.r_[1.,np.cumprod(1+np.asarray(returns))]*100000.)
        compare({k:z[k] for k in ('cagr','sharpe','max_drawdown','total_return')},walk[key],errors,'walk/'+key)
    attribution=read(P/'ATTRIBUTION.json');attribution_summary={}
    comparison=read(P/'MONTH_COMPARISON.json')
    assert len(comparison)==122
    for i,(day,end) in enumerate(zip(meta['dates'],meta['ends'])):
        split='early53' if i<53 else 'recent35' if i<88 else 'later34'
        assert comparison[i]['date']==day and comparison[i]['end']==end and comparison[i]['split']==split
        assert set(comparison[i])=={'date','end','split','primary',*NAMES}
        for name in ('primary',*NAMES):
            symbols=orders[name][i]['symbols']
            r=float(np.mean([close.at[pd.Timestamp(end),x]/close.at[pd.Timestamp(day),x]-1 for x in symbols])) if symbols else 0.
            compare(comparison[i][name],dict(symbols=symbols,monthly_return=r),errors,'month/'+name+'/'+str(i))
    pairs=[(arm,'incumbent_local67') for arm in ARMS]+[('ridge_support','fixed_half_support')]
    assert set(attribution)=={name+'__vs__'+reference for name,reference in pairs}
    for name,reference in pairs:
        changes=[];split_counts=dict(early53=0,recent35=0,later34=0)
        for i,(day,end) in enumerate(zip(meta['dates'],meta['ends'])):
            split='early53' if i<53 else 'recent35' if i<88 else 'later34'
            names=orders[name][i]['symbols'];base=orders[reference][i]['symbols']
            def ret(symbols):return float(np.mean([close.at[pd.Timestamp(end),x]/close.at[pd.Timestamp(day),x]-1 for x in symbols])) if symbols else 0.
            r,b=ret(names),ret(base)
            if set(names)!=set(base):
                changes.append(dict(date=day,split=split,old=base,new=names,
                    entering=sorted(set(names)-set(base)),leaving=sorted(set(base)-set(names)),difference=r-b,
                    log_contribution=float(np.log1p(r)-np.log1p(b))))
                split_counts[split]+=1
        positive=sorted((x['log_contribution'] for x in changes if x['log_contribution']>0),reverse=True)
        full=stat(dates,navs[name])
        def undo(n):return float(((1+full['total_return'])*np.exp(-sum(positive[:n])))**(1/full['years'])-1)
        value=dict(changed_months=len(changes),changes=changes,
            entering_seats=sum(len(z['entering']) for z in changes),leaving_seats=sum(len(z['leaving']) for z in changes),
            undo_best1_diagnostic_cagr=undo(1),undo_best3_diagnostic_cagr=undo(3))
        key=name+'__vs__'+reference
        compare(value,attribution[key],errors,'attribution/'+key)
        attribution_summary[key]=dict(changed_months=len(changes),by_period=split_counts,
            entering_seats=value['entering_seats'],leaving_seats=value['leaving_seats'],
            positive_months=len(positive),negative_months=sum(z['log_contribution']<0 for z in changes),
            undo_best1=undo(1),undo_best3=undo(3))
    # All actual geometry/score changes are recomputed independently above;
    # retain full per-month support coverage without outcomes influencing scores.
    support_audit=support_change_summary(a,meta,decisions)
    compare(support_audit,read(P/'SUPPORT_CHANGE_AUDIT.json'),errors,'support_change_audit')
    protect=read(P/'PROTECTED.json');assert len(protect)==21444
    assert protect==read(BOUNDARY/'NEXT_EXPECTED_PROTECTED.json')==read(PREVIOUS/'PROTECTED.json')
    assert validation['protected_changes']==[] and validation['protected_files']==21444
    assert validation['observer_accounts_changed'] is False and validation['pristine_oos'] is False
    assert frozen['pristine_oos'] is False and frozen['author_labels_used'] is False
    assert walk['pristine_oos'] is False
    assert validation['primary_target70_reached']==(results['primary']['metrics']['cagr']>=.70)
    for key in ('original_controls_exact','independent_fixed_unit_replay','training_freeze_parity',
            'no_global_rerank','no_intramonth_transactions','annual_clock_unchanged',
            'prefix_exact','all_diagnostics_frozen','core_coefficients_unchanged','core_and_baseline_unchanged',
            'auxiliary_states_unchanged','original_forecasts_unchanged','no_response_refit',
            'boundary_anchors_and_cohort_follow_modified_pre','incremental_model_score_only_attenuated',
            'original_support_payload_exact','true_episode_boundary_rules','raw_source_and_execution_cache_hashes_verified',
            'no_new_rank','coefficient_one_zero_delta_and_missing_support_exact_identity',
            'matched_fixed_half_control','matched_pair_attribution','shared_support_availability','meta_byte_exact',
            'source_model_and_forecast_availability_exact','prepare_does_not_choose_new_candidate_portfolios',
            'independent_actual_feature_audit_required_before_train'):
        assert validation[key] is True,key
    assert validation['fixed_configs']==6 and validation['prepared_array_count']==31 and validation['inherited_array_count']==27
    assert validation['nominal_episode_budget']==.05
    out=dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),
        method='Independent old-control and local pre/support reconstruction; matched half/geometric attenuation of the exact P67-P66 increment; original support and modified-pre episode anchors; independent fixed-unit replay. No candidate/model/selector/engine imports.',
        metrics=summary,training_metrics=dict(zip(NAMES,measured)),gates={NAMES[i]:gates(measured[i],measured[3]) for i in (4,5)},
        primary_id=primary,new_admitted=frozen['new_admitted'],attribution=attribution_summary,
        order_periods=122,daily_observations=len(dates),fixed_configs=6,independent_support_decisions=decisions,
        support_change_audit=support_audit,annual_clock_review=clock_review,maximum_metric_absolute_error=max((v for k,v in errors if 'metrics/' in k),default=0.),
        metric_relative_tolerance=2e-10,metric_absolute_tolerance=2e-9,same_close_nav_max_relative_error=nav_error,
        score_absolute_tolerance=1e-14,inherited_arrays_byte_exact=27,prepared_array_count=31,source_raw_hashes_verified=64,
        geometry_verified_by_independent_bound_actual_audit=True,matched_support_available_exact=True,meta_byte_exact=True,
        four_controls_orders_and_outputs_exact=True,all_trades_on_rebalance_dates=True,no_duplicate_date_symbol_trades=True,
        monthly_fixed_total_return_units=True,no_response_fit_or_clock_change=True,source_forecasts_byte_exact=True,
        original_support_payload_used=True,modified_pre_episode_anchors=True,returned_choices_not_globally_reranked=True,
        diagnostic_frozen_before_later34=True,all_six_actual_bindings_unchanged=True,
        frozen_hashes_checked=len(frozen['hashes']),price_cache_sha256=sha(PRICE),
        protection=dict(record_count=21444,records_equal_authorized_settlement_and_previous_round=True,
            production_recorded_changes=[],all21444_files_rehashed_this_review=False,
            scope='Verified exact authorized inventory and registration/evaluate proof; production already scanned all protected files. No redundant content scan here.'),
        hashes={name:sha(P/name) for name in ('DESIGN.json','FROZEN.json','RESULTS.json','VALIDATION.json','ATTRIBUTION.json',
            'TRAIN_TRIALS.json','META.json','PROTECTED.json','ACTUAL_FEATURE_REVIEW.json','SUPPORT_CHANGE_AUDIT.json')},
        audit_script_path=str(Path(__file__).resolve()),audit_script_sha256=sha(Path(__file__)),
        adapted_prior_independent_source_sha256=TEMPLATE_SHA,pristine_out_of_sample=False,
        evidence_scope=dict(close='Persisted daily NAV, metrics, slices, trades and holdings compared.',
            next_open='Independent NAV and holdings replayed at0 and5bp/side; published metrics/slices compared. Production does not persist next-open paths.',
            scores='Entire independent scalar attenuation/selector score path reconstructed; all selected scores and choices compared against frozen orders; production unselected score matrices are not persisted.',
            factors='All27 source bytes and4 support arrays verified; independent six-hash actual review certifies matured annual geometry; c ratio and shared query availability independently rechecked here.',
            clock='No response-model refit. Original27 forecast/state arrays retain bytes; independent bound geometry review certifies prior-Dec31 maturity and strict5year/month-balanced12month clock.'),
        limitations='Repeated historical proxy universe, assumed ETF list and idealized same-close. Ridge support is relative geometric coverage, not calibrated prediction confidence. Neither a70%guarantee nor author reproduction; no observer promotion.')
    for key,digest in tuple(REVIEWED.items()):
        path=Path(key) if Path(key).is_absolute() else P/key
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,('Stale review',key)
    out['reviewed_hashes']=dict(sorted(REVIEWED.items()));out['reviewed_hashes_base']=str(P)
    DEST.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,file=str(DEST),sha256=sha(DEST),primary_id=primary,new_admitted=frozen['new_admitted'],
        metrics={arm:{k:{field:value[field] for field in ('cagr','sharpe','max_drawdown')} for k,value in summary[arm].items()} for arm in ARMS},
        attribution=attribution_summary,same_close_nav_max_relative_error=nav_error),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate-complete',action='store_true',required=True,help='Use only after root confirms evaluate/report complete')
    parser.parse_args();audit()
