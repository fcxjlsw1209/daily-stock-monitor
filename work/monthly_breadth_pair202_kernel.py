"""Pure historical other-sector weakness paired with own-sector excess blocks.

Only stored adjusted prices and query-known sector classifications enter this
measurement. No model, eligibility filter, portfolio return or source I/O.
"""
from collections import Counter
import math
import numbers
import numpy as np
import pandas as pd
from monthly_nested_trend199_kernel import _frame, _calendar, _queries

PRICE_COUNT, SKIP, BLOCK_COUNT, BLOCK_SESSIONS = 253, 42, 12, 21
BREADTH_ETFS = ('XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY')
COUNT_FIELDS = ('valid_price_observations','excess_block_observations','breadth_block_observations',
    'down_sign_comparisons','covariance_estimates','raw_score_observations','ranked_score_observations')


def empty_measure(reason, breadth_count):
    return dict(available=False,reason=reason,candidate=np.nan,control=np.nan,
        endpoint_indices=[],block_start_indices=[],block_end_indices=[],stock_log_gains=[],
        benchmark_log_gains=[],excess_log_returns=[],down_counts=[],breadth=[],
        mean_excess=None,mean_breadth=None,covariance=None,constant_branch=None,
        breadth_count=breadth_count,**{k:0 for k in COUNT_FIELDS})


def measure(stock, benchmark, breadth):
    if np.shape(stock)!=(253,) or np.shape(benchmark)!=(253,):
        raise ValueError('Exactly253 stock and benchmark prices required')
    shape=np.shape(breadth)
    if len(shape)!=2 or shape[0]!=253 or shape[1] not in (8,9):
        raise ValueError('Exactly253 by8or9 consumed breadth prices required')
    m=shape[1]; converted=[]
    for name,source in (('stock',stock),('benchmark',benchmark),('breadth',breadth)):
        raw=np.asarray(source,dtype=object)
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,numbers.Real) for x in raw.flat):
            return empty_measure('nonnumeric_'+name+'_price',m)
        try: px=np.asarray(raw,dtype=np.float64)
        except (ValueError,TypeError,OverflowError):return empty_measure('nonnumeric_'+name+'_price',m)
        if not (np.isfinite(px)&(px>0)).all():return empty_measure('invalid_'+name+'_price',m)
        converted.append(px)
    stock,benchmark,breadth=converted
    endpoints=list(range(0,253,21));starts,ends=endpoints[:-1],endpoints[1:]
    sg,bg,e,down,w=[],[],[],[],[]
    for a,b in zip(starts,ends):
        stock_gain=math.log(float(stock[b]))-math.log(float(stock[a]))
        benchmark_gain=math.log(float(benchmark[b]))-math.log(float(benchmark[a]))
        excess=stock_gain-benchmark_gain
        count=sum(bool(breadth[b,j]<breadth[a,j]) for j in range(m))
        sg.append(stock_gain);bg.append(benchmark_gain);e.append(excess);down.append(count);w.append(count/m)
    try:
        c=math.fsum(e)/12.;wbar=math.fsum(w)/12.
        branch='all_breadth_equal' if all(x==w[0] for x in w) else 'all_excess_equal' if all(x==e[0] for x in e) else None
        cov=0. if branch else math.fsum((x-wbar)*(y-c) for x,y in zip(w,e))/12.
        q=c if branch else c+cov
    except (ValueError,OverflowError) as exc:raise FloatingPointError('Nonfinite paired-block arithmetic') from exc
    if not all(math.isfinite(x) for x in sg+bg+e+w+[c,wbar,cov,q]):
        raise FloatingPointError('Nonfinite arithmetic from valid sources')
    return dict(available=True,reason='available',candidate=q,control=c,endpoint_indices=endpoints,
        block_start_indices=starts,block_end_indices=ends,stock_log_gains=sg,benchmark_log_gains=bg,
        excess_log_returns=e,down_counts=down,breadth=w,mean_excess=c,mean_breadth=wbar,
        covariance=cov,constant_branch=branch,breadth_count=m,
        valid_price_observations=253*(2+m),excess_block_observations=12,breadth_block_observations=12,
        down_sign_comparisons=12*m,covariance_estimates=1,raw_score_observations=2,ranked_score_observations=2)


def centered_rank(raw, available):
    x,a=np.asarray(raw),np.asarray(available)
    if x.ndim!=1 or x.shape!=a.shape or a.dtype!=bool or not np.isfinite(x[a]).all():
        raise ValueError('Aligned finite source and boolean mask required')
    result=np.zeros(x.shape,float);ids=np.flatnonzero(a)
    if len(ids)>1:
        result[ids]=(pd.Series(x[ids]).rank(method='average').to_numpy()-1)/(len(ids)-1)-.5
    return result


def build_features(panel,meta,calendar,benchmark_map,limit=None):
    if not isinstance(panel,dict) or set(panel)!={'adj','sectors'}:
        raise ValueError('Panel requires exactly adj and sectors')
    for frame in panel.values():_frame(frame)
    if not callable(benchmark_map):raise TypeError('Pinned benchmark mapper must be callable')
    sessions=_calendar(calendar);queries,width=_queries(meta,sessions)
    if limit is not None and (type(limit) is not int or not 0<=limit<=len(queries)):
        raise ValueError('Invalid query prefix')
    queries=queries if limit is None else queries[:limit]
    shape=len(queries),width
    values={k:np.full(shape,np.nan) for k in ('candidate','control')}
    values.update(available=np.zeros(shape,bool),candidate_rank=np.zeros(shape),control_rank=np.zeros(shape))
    adj,sectors=panel['adj'],panel['sectors'];history=[];reasons=Counter();counts=Counter()
    available_by,eights,nines=[],[],[]
    for i,(day,symbols,t) in enumerate(queries):
        iso=str(day.date());mapping=benchmark_map(iso,symbols,sectors.loc[:day],meta['etfs'])
        if (not isinstance(mapping,dict) or set(mapping)!=set(symbols) or
            any(b is not None and (not isinstance(b,str) or not b or b.strip()!=b) for b in mapping.values())):
            raise ValueError('Complete original identity mapping required')
        if any(mapping[s] is not None for s in symbols if s in meta['etfs']):
            raise ValueError('Original candidate ETFs must stay unmapped')
        dates=sessions[t-294:t-41] if t>=294 else None
        missing=dates.difference(adj.index).strftime('%Y-%m-%d').tolist() if dates is not None else []
        cells=[]
        for j,symbol in enumerate(symbols):
            b=mapping[symbol]
            basket=[s for s in BREADTH_ETFS if s!=b] if b is not None and symbol not in meta['etfs'] else []
            m=len(basket)
            if b is not None and symbol not in meta['etfs'] and len(set([symbol,b]+basket))!=2+m:
                raise ValueError('Consumed series identities must be unique')
            if symbol in meta['etfs']:detail=empty_measure('etf_neutral',m)
            elif b is None:detail=empty_measure('missing_benchmark',m)
            elif dates is None:detail=empty_measure('insufficient_history',m)
            elif symbol not in adj.columns:detail=empty_measure('missing_symbol',m)
            elif b not in adj.columns:detail=empty_measure('missing_benchmark_symbol',m)
            elif any(s not in adj.columns for s in basket):detail=empty_measure('missing_breadth_symbol',m)
            elif missing:detail=empty_measure('missing_official_session',m)
            else:
                detail=measure(adj.loc[dates,symbol].to_numpy(copy=True),adj.loc[dates,b].to_numpy(copy=True),
                    adj.loc[dates,basket].to_numpy(copy=True))
            for key in ('candidate','control','available'):values[key][i,j]=detail[key]
            cell=dict(detail,symbol=symbol,benchmark=b,date=iso,breadth_etfs=basket,candidate_rank=0.,control_rank=0.)
            for index_key,date_key in (('endpoint_indices','endpoint_dates'),('block_start_indices','block_start_dates'),('block_end_indices','block_end_dates')):
                cell[date_key]=[str(dates[k].date()) for k in detail[index_key]]
            if not detail['available']:cell['candidate']=cell['control']=None
            cells.append(cell);reasons[detail['reason']]+=1
            for key in COUNT_FIELDS:counts[key]+=detail[key]
        for key in ('candidate','control'):
            values[key+'_rank'][i]=centered_rank(values[key][i],values['available'][i])
            for j,cell in enumerate(cells):cell[key+'_rank']=float(values[key+'_rank'][i,j])
        n=int(values['available'][i].sum());eight=sum(c['available'] and c['breadth_count']==8 for c in cells)
        nine=sum(c['available'] and c['breadth_count']==9 for c in cells)
        available_by.append(n);eights.append(eight);nines.append(nine)
        history.append(dict(date=iso,symbols=symbols,session_position=t,
            price_start=str(dates[0].date()) if dates is not None else None,
            price_end=str(dates[-1].date()) if dates is not None else None,
            endpoint_dates=[str(dates[k].date()) for k in range(0,253,21)] if dates is not None else None,
            missing_official_sessions=missing,benchmarks=mapping,cells=cells,
            available=values['available'][i].tolist(),available_count=n,eight_sector_cells=eight,nine_sector_cells=nine))
    total=sum(available_by)
    return values,dict(events=len(queries),symbols_per_event=width,checked_cells=int(values['available'].size),
        price_count=253,skip_sessions=42,block_count=12,block_sessions=21,breadth_etfs=list(BREADTH_ETFS),
        available_cells=total,unavailable_cells=int(values['available'].size)-total,
        available_by_event=available_by,eight_sector_cells_by_event=eights,nine_sector_cells_by_event=nines,
        **{k:counts[k] for k in COUNT_FIELDS},reason_counts=dict(reasons),event_history=history,
        new_model_fits=0,predictive_model_fits=0,common_availability=True,ranking_performed=True,
        ranking_eligibility_filtered=False,trading_eligibility_modified=False,portfolio_returns_calculated=False)
