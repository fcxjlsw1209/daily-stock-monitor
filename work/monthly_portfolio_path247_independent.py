"""Independent source, fixed-unit portfolio and original Local67 oracle.

The new producer, its host and its math module are never imported. Historical
formation paths are descriptive; this module does not run a strategy engine.
"""
import copy
from datetime import date
import hashlib
import importlib.util
import itertools
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_core_veto241_independent.py'
PRIOR_SHA='642958e877a0ed53347c2ca21aea0027e0148591af21985268976b91d9e22fb6'
PRIOR_TEST=ROOT/'work/test_monthly_core_veto241_independent.py'
PRIOR_TEST_SHA='4fc17fdcf385f30a87baf66513a1b2e0c19db2e0549dc779b6fe41f755e64087'
ACCOUNT_TEST=ROOT/'work/test_monthly_upside_retention181_independent.py'
ACCOUNT_TEST_SHA='b597ad5d7d128625e9d5c698c007cc3a5deafc398acd9ebaf3fb8bd83df23fe0'
ARMS=('incumbent_local67','portfolio_joint_path','portfolio_diagonal_control','portfolio_structure_control')
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','active_events',
 'shortlist_cells','source_available_cells','source_price_observations','enumerated_trios','feasible_trios',
 'trio_statistics','zero_variance_trios','missing_source_events','zero_variance_events','available_events',
 'joint_changed_holdings','diagonal_changed_holdings','structure_changed_holdings','original_episode_calls',
 'new_model_fits','predictive_model_fits')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_model_rank_scope_preserved',
 'original_eligibility_unchanged','original_support_checked','original_budget_checked',
 'calm_nonweak_extension_budget_checked','single_original_episode','no_second_episode',
 'original_weighted_core_unchanged','original_top5_checked','same_feasible_domain_checked',
 'incumbent_feasible_checked','raw_core_regret_cap_checked','final_score_regret_cap_checked',
 'funded_final_auxiliary_checked','same_numerator_and_path_checked','exact_ratio_ranks_checked',
 'exact_ties_checked','structural_control_checked','common_fallback_checked',
 'named_initial_partial_unchanged','original_full_scores_and_pre_preserved',
 'exact_scores_and_choices_checked','full_audit_checked','descriptive_counts_checked','no_predictive_fits_checked')
PREPARE_FLAGS=SELECTION_FLAGS+('independent_source_rebuild','official_253_price_window_checked',
 'all_shortlist_sources_checked','no_rank_six_substitution','visible_listing_bounds_checked',
 'fixed_adjusted_units_checked','same_path_joint_diagonal_checked','sample_covariance_decomposition_checked',
 'direct_nav_identity_checked','genuine_zero_variance_gate_checked','continuous_statistics_checked',
 'independent_rank_agreement_checked','no_future_source_values')
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
NAV_ATOL=NAV_RTOL=2e-12
EVENT_KEYS=('query_index','date','symbols','initial_partial','calm','weak','core_weight','eligible_indices',
 'pre_scores','final_scores','weighted_core','incumbent_choices','pre_order','shortlist_indices','incumbent_trio',
 'reference_raw_core_mean','reference_final_mean','feasible_trio_indices','session_position','price_start',
 'price_end','source_dates','clock_sessions','source_cells','trios','branch')
TRIO_KEYS=('indices','symbols','raw_core_mean','weighted_core_mean','final_mean','auxiliary_mean',
 'raw_core_regret','final_regret','feasible','statistics')
SOURCE_KEYS=('column_index','symbol','source_available','reason','observations_checked','start_price','end_price','listing')
Q_KEYS=('joint_q','diagonal_q','joint_utility','diagonal_utility','structural_utility')
STAT_KEYS=('schema','price_count','return_count','available','reason','normalized_prices','portfolio_values',
 'prior_denominators','contributions','portfolio_returns','direct_nav_returns','mean_return','joint_square_sum',
 'contribution_means','contribution_square_sums','diagonal_square_sum','joint_variance','diagonal_variance',
 'joint_sd','diagonal_sd','joint_ratio','diagonal_ratio','pair_covariances','cross_covariance','diagnostics')


def load_pinned(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen independent dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


old=load_pinned(PRIOR,PRIOR_SHA,'portfolio_path247_original_independent')
require,exact_array,record_check=old.require,old.exact_array,old.record_check
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA

def accounting():return old.accounting()
def frozen_source_oracle():return old.frozen_source_oracle()
def dependency_files():
    for p,h in ((PRIOR_TEST,PRIOR_TEST_SHA),(ACCOUNT_TEST,ACCOUNT_TEST_SHA)):
        require(hashlib.sha256(p.read_bytes()).hexdigest()==h,'Frozen synthetic fixture changed')
    return tuple(sorted(set(old.dependency_files())|{PRIOR,PRIOR_TEST,ACCOUNT_TEST,Path(__file__).resolve()}))


def finite(x,label='arithmetic'):
    if not math.isfinite(x):raise ArithmeticError('Nonfinite '+label)
    return float(x)

def summation(values):
    try:return finite(math.fsum(values),'sum')
    except (ValueError,OverflowError) as e:raise ArithmeticError('Invalid sum') from e

def divide(a,b):
    try:r=finite(a/b,'division')
    except (ZeroDivisionError,OverflowError) as e:raise ArithmeticError('Invalid division') from e
    if a!=0. and r==0.:raise ArithmeticError('False zero division underflow')
    return r

def product(a,b):
    r=finite(a*b,'product')
    if a!=0. and b!=0. and r==0.:raise ArithmeticError('False zero product underflow')
    return r

def difference(a,b):return finite(a-b,'difference')
def close_diagnostic(a,b,atol,rtol):
    error=abs(difference(a,b));bound=finite(atol+rtol*max(abs(a),abs(b)),'diagnostic tolerance')
    if error>bound:raise ArithmeticError('Algebra diagnostic failed')
    return error


def measure_oracle(paths):
    """Scalar independent fixed-unit path, with separately checked NAV identity."""
    require(isinstance(paths,(list,tuple,np.ndarray)) and len(paths)==3,'Three source paths')
    lengths=[len(row) for row in paths];require(lengths[0]>=3 and len(set(lengths))==1,'Common source length')
    p=[]
    for row in paths:
        converted=[]
        for v in row:
            require(isinstance(v,Real) and not isinstance(v,(bool,np.bool_)),'Real nonboolean source')
            try:v=float(v)
            except (TypeError,ValueError,OverflowError) as e:raise ValueError('Unrepresentable price') from e
            require(math.isfinite(v) and v>0.,'Positive finite source');converted.append(v)
        p.append(converted)
    L=lengths[0];T=L-1
    normalized=[[divide(value,row[0]) for value in row] for row in p]
    totals=[summation([row[k] for row in normalized]) for k in range(L)]
    wealth=[divide(v,3.) for v in totals]
    columns=[[divide(difference(row[k],row[k-1]),totals[k-1]) for k in range(1,L)] for row in normalized]
    returns=[summation([columns[i][k] for i in range(3)]) for k in range(T)]
    direct=[difference(divide(wealth[k+1],wealth[k]),1.) for k in range(T)]
    nav_error=max(close_diagnostic(a,b,NAV_ATOL,NAV_RTOL) for a,b in zip(returns,direct))
    mean=divide(summation(returns),float(T))
    deviations=[difference(x,mean) for x in returns]
    joint_ss=summation(product(d,d) for d in deviations)
    means=[divide(summation(col),float(T)) for col in columns]
    centered=[[difference(value,mu) for value in col] for mu,col in zip(means,columns)]
    squares=[summation(product(d,d) for d in col) for col in centered]
    diagonal_ss=summation(squares);den=float(T-1)
    joint=divide(joint_ss,den);diagonal=divide(diagonal_ss,den)
    require(joint>=0. and diagonal>=0.,'Nonnegative sample variance')
    pairs=[];cross_sums=[]
    for i,j in ((0,1),(0,2),(1,2)):
        ss=summation(product(a,b) for a,b in zip(centered[i],centered[j]));cross_sums.append(ss)
        pairs.append(dict(indices=[i,j],square_sum=ss,covariance=divide(ss,den)))
    cross=divide(product(2.,summation(cross_sums)),den)
    observed=difference(joint,diagonal)
    covariance_error=close_diagnostic(observed,cross,CONTINUOUS_ATOL,CONTINUOUS_RTOL)
    upper=product(3.,diagonal)
    if difference(joint,upper)>CONTINUOUS_ATOL+CONTINUOUS_RTOL*max(abs(joint),abs(upper)):
        raise ArithmeticError('Joint variance upper bound failed')
    sd=finite(math.sqrt(joint));dsd=finite(math.sqrt(diagonal));available=joint>0. and diagonal>0.
    return dict(schema='monthly_portfolio_path247_math_v1',price_count=L,return_count=T,
        available=available,reason='available' if available else 'zero_variance',normalized_prices=normalized,
        portfolio_values=wealth,prior_denominators=totals[:-1],contributions=columns,portfolio_returns=returns,
        direct_nav_returns=direct,mean_return=mean,joint_square_sum=joint_ss,contribution_means=means,
        contribution_square_sums=squares,diagonal_square_sum=diagonal_ss,joint_variance=joint,
        diagonal_variance=diagonal,joint_sd=sd,diagonal_sd=dsd,
        joint_ratio=divide(mean,sd) if available else None,diagonal_ratio=divide(mean,dsd) if available else None,
        pair_covariances=pairs,cross_covariance=cross,diagnostics=dict(nav_identity_atol=NAV_ATOL,
        nav_identity_rtol=NAV_RTOL,covariance_atol=CONTINUOUS_ATOL,covariance_rtol=CONTINUOUS_RTOL,
        nav_max_absolute_error=nav_error,covariance_absolute_error=covariance_error,
        joint_minus_diagonal=observed,three_times_diagonal=upper,nav_identity_passed=True,
        covariance_identity_passed=True,variance_upper_bound_passed=True))


def original_base(original,meta,n):
    require(type(n) is int and 0<=n<=len(meta['dates']),'Consumed prefix')
    width,days,names=old.axes(original,meta,n);require(width>0,'Original width positive')
    if not n:
        return dict(width=width,days=days,names=names,P=np.empty((0,width)),F=np.empty((0,width)),
            E=np.empty((0,width),bool),core=np.empty((0,width)),cw=np.empty(0),calm=np.empty(0,bool),
            weak=np.empty(0,bool),pre_order=[],choices=[],budget_evidence=dict(minimum_remaining_support=None,
            maximum_budget_sum_error=0.,calm_nonweak_extension_count=0))
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    P=x['local'].copy();F=P.copy();cw=np.where(x['calm'],.5,.52)
    exact_array(x['cw'],np.broadcast_to(cw[:,None],(n,width)),'Original monthly core weight')
    ext=np.asarray(original['extension_available'])[:n];forecast=np.asarray(original['forecast_available'])[:n]
    ranklog=np.asarray(original['rank_log_excess'])[:n]
    require(ext.shape==forecast.shape==ranklog.shape==(n,width) and ext.dtype==forecast.dtype==np.dtype(bool),
        'Original extension axes')
    A=(x['calm']&~x['weak'])[:,None]&ext&forecast&x['eligible']&np.isfinite(ranklog)
    budgets=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    choices=[];ranks=[];minimum=1.;error=0.
    for i,ss in enumerate(names):
        rank=sorted(np.flatnonzero(x['eligible'][i]).tolist(),key=lambda j:(-P[i,j],ss[j]));ranks.append(rank)
        J=[] if len(rank)<3 else [j for j in rank[2:5] if float(P[i,rank[2]])-float(P[i,j])<=.02]
        for j in J:
            target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
            F[i,j]+=budgets[i]*(target-x['support'][i,j])
        require(np.isfinite(F[i]).all(),'Finite original final scores')
        if len(rank)<3:chosen=rank.copy() if len(rank)==2 else []
        else:
            third=min(J,key=lambda j:(-F[i,j],ss[j]))
            chosen=sorted(rank[:2]+[third],key=lambda j:(-F[i,j],ss[j]))
        choices.append(chosen)
        for j in range(width):
            weights=[float(cw[i]),.025*float(x['calm'][i]),.05*float(x['weak'][i]),.01*float(A[i,j]),
                float(budgets[i])*float(j in J)]
            remaining=1.-weights[0]
            for w in weights[1:]:remaining-=w
            err=abs(math.fsum(weights+[remaining])-1.)
            require(remaining>=.38-1e-12 and err<=1e-12,'Unchanged Local67 budget')
            minimum=min(minimum,remaining);error=max(error,err)
    return dict(width=width,days=days,names=names,P=P,F=F,E=x['eligible'],core=x['core'],cw=cw,
        calm=x['calm'],weak=x['weak'],pre_order=ranks,choices=choices,budget_evidence=dict(
        minimum_remaining_support=minimum,maximum_budget_sum_error=error,calm_nonweak_extension_count=int(A.sum())))


def base_event(base,meta,i):
    names=base['names'][i];cw=float(base['cw'][i]);partial=meta['dates'][i]=='2016-08-17'
    inc=base['choices'][i];rank=base['pre_order'][i];active=not partial and len(rank)>=3
    short=rank[:5] if active else []
    canonical=lambda ids:sorted(ids,key=lambda j:names[j])
    inc_trio=canonical(inc) if active else None
    means=lambda ids:(summation(float(base['core'][i,j])/cw for j in ids)/3.,
                      summation(float(base['core'][i,j]) for j in ids)/3.,
                      summation(float(base['F'][i,j]) for j in ids)/3.,
                      summation(float(base['F'][i,j])-float(base['core'][i,j]) for j in ids)/3.)
    reference=means(inc_trio) if active else None
    trios=[]
    for ids in itertools.combinations(sorted(short,key=lambda j:names[j]),3):
        X,C,F,A=means(ids);good=X>=reference[0]-.02 and F>=reference[2]-.02
        trios.append(dict(indices=list(ids),symbols=[names[j] for j in ids],raw_core_mean=X,
            weighted_core_mean=C,final_mean=F,auxiliary_mean=A,raw_core_regret=reference[0]-X,
            final_regret=reference[2]-F,feasible=bool(good),statistics=None))
    H=[k for k,t in enumerate(trios) if t['feasible']]
    if active:require(any(trios[k]['indices']==inc_trio for k in H),'Incumbent remains feasible in shortlist')
    return dict(query_index=i,date=meta['dates'][i],symbols=list(names),initial_partial=partial,
        calm=bool(base['calm'][i]),weak=bool(base['weak'][i]),core_weight=cw,
        eligible_indices=np.flatnonzero(base['E'][i]).tolist(),pre_scores=base['P'][i].tolist(),
        final_scores=base['F'][i].tolist(),weighted_core=base['core'][i].tolist(),incumbent_choices=inc.copy(),
        pre_order=rank.copy(),shortlist_indices=short,incumbent_trio=inc_trio,
        reference_raw_core_mean=reference[0] if active else None,reference_final_mean=reference[2] if active else None,
        feasible_trio_indices=H,session_position=None,price_start=None,price_end=None,source_dates=[],
        clock_sessions=[],source_cells=[],trios=trios,branch='initial_partial' if partial else
        ('fewer_than_three' if not active else 'available'))


def iso(s):
    require(type(s) is str,'Canonical ISO date');d=date.fromisoformat(s)
    require(d.isoformat()==s,'Canonical ISO date');return pd.Timestamp(d)

def source_axes(panel,calendar,n):
    require(isinstance(panel,dict) and set(panel)=={'adj'},'Original adjusted frame only')
    frame=panel['adj'];require(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex),'Price frame')
    ix=frame.index
    require(ix.is_unique and ix.is_monotonic_increasing and not ix.hasnans and ix.tz is None and ix.equals(ix.normalize()),'Price date axis')
    require(frame.columns.is_unique and all(type(s) is str and s and s.strip()==s for s in frame.columns),'Price column IDs')
    require(isinstance(calendar,(list,tuple)) and (len(calendar)>0 or n==0),'Official calendar')
    days=[iso(v) for v in calendar]
    require(all(d.weekday()<5 for d in days) and all(a<b for a,b in zip(days,days[1:])),'Ordered weekday calendar')
    return frame,days


def review_index(review,n):
    require(isinstance(review,dict) and isinstance(review.get('calendar'),dict),'Source review')
    out=[]
    for key,items in (('windows',review['calendar'].get('windows')),('listing_identity_cells',review.get('listing_identity_cells'))):
        require(isinstance(items,list),'Review '+key)
        table={}
        for row in items:
            require(isinstance(row,dict) and type(row.get('query_index')) is int and row['query_index']>=0,'Review query index')
            q=row['query_index']
            if q>=n:continue
            identity=q if key=='windows' else (q,row.get('column_index'))
            if key!='windows':require(type(row.get('column_index')) is int and row['column_index']>=0,'Listing column')
            require(identity not in table,'Duplicate source review identity');table[identity]=row
        out.append(table)
    require(set(out[0])==set(range(n)),'Complete consumed source clocks')
    bounds=review.get('listing_bounds');require(isinstance(bounds,list),'Listing bounds')
    return out[0],out[1],bounds


def clock_record(event,sessions,lookup):
    day=iso(event['date']);require(day in lookup,'Official query');t=lookup[day]
    require(t>=294 and t+1<len(sessions),'Complete official clock plus next session')
    if not event['initial_partial']:require(sessions[t+1].to_period('M')!=day.to_period('M'),'Official month-end')
    clock=[str(d.date()) for d in sessions[t-294:t+2]];dates=clock[:253]
    event.update(session_position=t,price_start=dates[0],price_end=dates[-1],source_dates=dates,clock_sessions=clock)
    return dict(query_index=event['query_index'],date=event['date'],initial_partial=event['initial_partial'],
        query_session_position=t,source_start_position=t-294,source_end_position=t-42,price_start=dates[0],
        price_end=dates[-1],first_return_date=dates[1],next_official_session=clock[-1],prices=253,
        return_intervals=252,source_dates_sha256=hashlib.sha256(json.dumps(dates,separators=(',',':')).encode()).hexdigest())


def listing_metadata(event,identity,bounds):
    bysymbol={}
    for b in bounds:
        require(isinstance(b,dict) and set(b)=={'symbol','issuer_key','not_before','available_from','evidence_id','kind'},'Listing bound schema')
        for key in b:require(type(b[key]) is str and b[key] and b[key].strip()==b[key],'Listing text')
        iso(b['not_before']);iso(b['available_from']);require(b['symbol'] not in bysymbol,'Unique bound symbol');bysymbol[b['symbol']]=b
    result={};q=event['query_index']
    for j,s in enumerate(event['symbols']):
        if s not in bysymbol:continue
        b=bysymbol[s];row=identity.get((q,j));require(row is not None,'Listing occurrence evidence missing')
        visible=b['available_from']<=event['date'];precedes=event['price_start']<b['not_before']
        wanted=dict(query_index=q,column_index=j,date=event['date'],symbol=s,issuer_key=b['issuer_key'],
            new_price_start=event['price_start'],new_price_end=event['price_end'],bound_visible=visible,start_precedes_bound=precedes)
        for k,v in wanted.items():record_check(row.get(k),v,'Listing '+k)
        result[j]=dict(issuer_key=b['issuer_key'],not_before=b['not_before'],available_from=b['available_from'],
            evidence_id=b['evidence_id'],visible=visible,start_precedes_bound=precedes)
    require({j for (i,j) in identity if i==q}==set(result),'No stray listing occurrence')
    return result


def source_cell(frame,event,j,listing):
    symbol=event['symbols'][j];dates=event['source_dates']
    row=dict(column_index=j,symbol=symbol,source_available=False,reason='missing_symbol',observations_checked=0,
        start_price=None,end_price=None,listing=copy.deepcopy(listing))
    if symbol not in frame.columns:return row,None
    missing=False;nan=False;values=[]
    for raw in dates:
        day=pd.Timestamp(raw)
        if day not in frame.index:missing=True;continue
        row['observations_checked']+=1;value=frame.at[day,symbol]
        require(isinstance(value,Real) and not isinstance(value,(bool,np.bool_)),'Malformed required price')
        try:v=float(value)
        except (TypeError,ValueError,OverflowError) as e:raise ValueError('Malformed required price conversion') from e
        if math.isnan(v):nan=True;continue
        require(math.isfinite(v) and v>0.,'Malformed required price value');values.append(v)
    if missing:row['reason']='missing_official_session'
    elif nan:row['reason']='missing_nan'
    else:
        require(len(values)==253,'Complete path length')
        row.update(source_available=True,reason='available',start_price=values[0],end_price=values[-1])
        if listing is not None and listing['visible'] and listing['start_precedes_bound']:
            raise ValueError('Complete path contradicts visible same-security listing bound')
    return row,values if row['source_available'] else None


def independent_features(panel,meta,calendar,original,review,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    frame,sessions=source_axes(panel,calendar,n);lookup={d:i for i,d in enumerate(sessions)}
    windows,identities,bounds=review_index(review,n);features=[]
    for i in range(n):
        event=base_event(base,meta,i);wanted=clock_record(event,sessions,lookup)
        for k,v in wanted.items():record_check(windows[i].get(k),v,'Reviewed clock/'+k)
        listings=listing_metadata(event,identities,bounds)
        if event['branch'] in ('initial_partial','fewer_than_three'):features.append(event);continue
        paths={}
        for j in event['shortlist_indices']:
            cell,path=source_cell(frame,event,j,listings.get(j));event['source_cells'].append(cell)
            if path is not None:paths[j]=path
        if len(paths)!=len(event['shortlist_indices']):event['branch']='missing_source'
        else:
            for k in event['feasible_trio_indices']:
                trio=event['trios'][k];trio['statistics']=measure_oracle([paths[j] for j in trio['indices']])
            if any(not event['trios'][k]['statistics']['available'] for k in event['feasible_trio_indices']):
                event['branch']='zero_variance'
        features.append(event)
    return features


def continuous_check(actual,expected,label='statistics'):
    if isinstance(expected,dict):
        require(type(actual) is dict and set(actual)==set(expected),label+' keys')
        for k,v in expected.items():continuous_check(actual[k],v,label+'/'+k)
    elif isinstance(expected,list):
        require(type(actual) is list and len(actual)==len(expected),label+' length')
        for i,(a,e) in enumerate(zip(actual,expected)):continuous_check(a,e,label+'/'+str(i))
    elif type(expected) is float:
        require(type(actual) is float and math.isfinite(actual) and math.isfinite(expected),label+' finite float')
        require(abs(actual-expected)<=CONTINUOUS_ATOL+CONTINUOUS_RTOL*max(abs(actual),abs(expected)),label+' tolerance')
        if label.rsplit('/',1)[-1] in ('joint_variance','diagonal_variance','joint_sd','diagonal_sd'):
            require((actual>0.)==(expected>0.) and (actual==0.)==(expected==0.),label+' exact zero gate')
    else:record_check(actual,expected,label)


def statistics_schema(stats):
    require(type(stats) is dict,'Complete feasible statistics')
    require(set(stats)==set(STAT_KEYS),'Complete statistics schema')
    record_check(stats['schema'],'monthly_portfolio_path247_math_v1','Cached math schema')
    require(type(stats['price_count']) is int and stats['price_count']==253 and
        type(stats['return_count']) is int and stats['return_count']==252,'Descriptive path lengths')
    def finite_tree(x):
        if type(x) is float:require(math.isfinite(x),'Finite cached arithmetic')
        elif type(x) is list:
            for v in x:finite_tree(v)
        elif type(x) is dict:
            for v in x.values():finite_tree(v)
        else:require(x is None or type(x) in (str,int,bool),'Plain cached scalar')
    finite_tree(stats)
    for key,length in (('normalized_prices',3),('portfolio_values',253),('prior_denominators',252),
        ('contributions',3),('portfolio_returns',252),('direct_nav_returns',252),('contribution_means',3),
        ('contribution_square_sums',3),('pair_covariances',3)):
        require(type(stats[key]) is list and len(stats[key])==length,'Cached '+key+' axis')
    require(all(len(x)==253 for x in stats['normalized_prices']) and all(len(x)==252 for x in stats['contributions']),'Cached row axes')
    v,d=stats['joint_variance'],stats['diagonal_variance']
    require(type(v) is float and type(d) is float and v>=0. and d>=0.,'Cached variance')
    available=v>0. and d>0.
    record_check(stats['available'],available,'Cached exact zero gate')
    record_check(stats['reason'],'available' if available else 'zero_variance','Cached reason')
    for key,var in (('joint_sd',v),('diagonal_sd',d)):
        record_check(stats[key],math.sqrt(var),'Cached '+key)
    for key,sd in (('joint_ratio',stats['joint_sd']),('diagonal_ratio',stats['diagonal_sd'])):
        record_check(stats[key],divide(stats['mean_return'],sd) if available else None,'Cached '+key)
    record_check(stats['diagnostics']['nav_identity_passed'],True,'NAV diagnostic')
    record_check(stats['diagnostics']['covariance_identity_passed'],True,'Covariance diagnostic')
    record_check(stats['diagnostics']['variance_upper_bound_passed'],True,'Variance bound diagnostic')


def cache_event_check(event,base,meta,i):
    require(type(event) is dict and set(event)==set(EVENT_KEYS),'Complete cached feature event')
    expected=base_event(base,meta,i)
    copied=('session_position','price_start','price_end','source_dates','clock_sessions','source_cells','branch')
    for key in expected:
        if key not in copied and key!='trios':record_check(event[key],expected[key],'Original binding/'+key)
    clock=event['clock_sessions'];require(type(clock) is list and len(clock)==296,'Cached official clock')
    days=[iso(d) for d in clock]
    require(all(d.weekday()<5 for d in days) and all(a<b for a,b in zip(days,days[1:])),'Cached official order')
    require(clock[294]==event['date'] and type(event['session_position']) is int and event['session_position']>=294,'Cached query position')
    if not event['initial_partial']:require(clock[-1][:7]!=event['date'][:7],'Cached next-month witness')
    record_check(event['source_dates'],clock[:253],'Cached source days')
    record_check(event['price_start'],clock[0],'Cached source start');record_check(event['price_end'],clock[252],'Cached source end')
    require(type(event['trios']) is list and len(event['trios'])==len(expected['trios']),'Cached trio axis')
    for got,want in zip(event['trios'],expected['trios']):
        require(type(got) is dict and set(got)==set(TRIO_KEYS),'Cached trio schema')
        for k,v in want.items():
            if k!='statistics':record_check(got[k],v,'Canonical trio/'+k)
    cells=event['source_cells'];require(type(cells) is list and len(cells)==len(event['shortlist_indices']),'All shortlist source cells')
    for cell,j in zip(cells,event['shortlist_indices']):
        require(type(cell) is dict and set(cell)==set(SOURCE_KEYS),'Source cell schema')
        record_check(cell['column_index'],j,'Source column');record_check(cell['symbol'],event['symbols'][j],'Source symbol')
        require(type(cell['source_available']) is bool and type(cell['observations_checked']) is int and 0<=cell['observations_checked']<=253,'Source observation count')
        reason=cell['reason'];require(reason in ('available','missing_symbol','missing_official_session','missing_nan'),'Source reason')
        record_check(cell['source_available'],reason=='available','Source availability')
        if reason=='missing_symbol':require(cell['observations_checked']==0,'Missing symbol observations')
        elif reason=='missing_official_session':require(cell['observations_checked']<253,'Missing date observations')
        else:require(cell['observations_checked']==253,'Complete observed calendar')
        if cell['source_available']:
            require(all(type(cell[k]) is float and math.isfinite(cell[k]) and cell[k]>0 for k in ('start_price','end_price')),'Source endpoints')
        else:require(cell['start_price'] is None and cell['end_price'] is None,'Missing source no endpoints')
        listing=cell['listing']
        if listing is not None:
            require(type(listing) is dict and set(listing)=={'issuer_key','not_before','available_from','evidence_id','visible','start_precedes_bound'},'Cached listing schema')
            for k in ('issuer_key','evidence_id'):require(type(listing[k]) is str and listing[k],'Cached listing identity')
            iso(listing['not_before']);iso(listing['available_from'])
            record_check(listing['visible'],listing['available_from']<=event['date'],'Listing visibility')
            record_check(listing['start_precedes_bound'],event['price_start']<listing['not_before'],'Listing window')
            require(not(cell['source_available'] and listing['visible'] and listing['start_precedes_bound']),'No false complete listing path')
    if expected['branch'] in ('initial_partial','fewer_than_three'):branch=expected['branch']
    elif any(not c['source_available'] for c in cells):branch='missing_source'
    else:branch='available'
    source_good=branch=='available'
    for t in event['trios']:
        if source_good and t['feasible']:statistics_schema(t['statistics'])
        else:require(t['statistics'] is None,'No statistics outside complete feasible source')
    if source_good and any(not event['trios'][k]['statistics']['available'] for k in event['feasible_trio_indices']):branch='zero_variance'
    record_check(event['branch'],branch,'Common fallback branch')


def rank_values(raw):
    require(raw and all(type(v) is float and math.isfinite(v) for v in raw),'Finite ratio rank domain')
    return [(sum(x<v for x in raw)+(sum(x==v for x in raw)+1)/2.)/len(raw)-.5 for v in raw]


def selection_oracle(original,meta,features,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    require(type(features) is list and len(features)>=n,'Feature prefix')
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=n,symbols_per_event=base['width'],original_episode_calls=1)
    result=dict(schema='monthly_portfolio_path247_decisions_v1',counts=counts,events=[])
    choices={arm:[] for arm in ARMS}
    for i,source in enumerate(features[:n]):
        cache_event_check(source,base,meta,i);event=copy.deepcopy(source)
        scale=1.-.025/(1.-event['core_weight']);event['noncore_scale']=scale
        for trio in event['trios']:
            for key in Q_KEYS:trio[key]=None
        picked={a:event['incumbent_choices'].copy() for a in ARMS}
        if event['branch']=='available':
            H=event['feasible_trio_indices'];trios=event['trios']
            jq=rank_values([trios[k]['statistics']['joint_ratio'] for k in H])
            dq=rank_values([trios[k]['statistics']['diagonal_ratio'] for k in H])
            for k,j,d in zip(H,jq,dq):
                trios[k].update(joint_q=j,diagonal_q=d,
                    joint_utility=finite(trios[k]['weighted_core_mean']+scale*trios[k]['auxiliary_mean']+.025*j),
                    diagonal_utility=finite(trios[k]['weighted_core_mean']+scale*trios[k]['auxiliary_mean']+.025*d),
                    structural_utility=trios[k]['final_mean'])
            for arm,key in zip(ARMS[1:],('joint_utility','diagonal_utility','structural_utility')):
                best=max(trios[k][key] for k in H);tied=[k for k in H if trios[k][key]==best]
                inc=[k for k in tied if trios[k]['indices']==event['incumbent_trio']]
                winner=inc[0] if inc else min(tied,key=lambda k:tuple(trios[k]['symbols']))
                ids=trios[winner]['indices']
                if set(ids)!=set(event['incumbent_choices']):
                    picked[arm]=sorted(ids,key=lambda j:(-base['F'][i,j],event['symbols'][j]))
        event['choices']=picked
        for arm in ARMS:choices[arm].append(picked[arm])
        result['events'].append(event)
        counts['eligible_cells']+=len(event['eligible_indices']);counts['initial_partial_events']+=int(event['initial_partial'])
        counts['active_events']+=int(not event['initial_partial'] and len(event['eligible_indices'])>=3)
        counts['shortlist_cells']+=len(event['shortlist_indices']);counts['source_available_cells']+=sum(c['source_available'] for c in event['source_cells'])
        counts['source_price_observations']+=sum(c['observations_checked'] for c in event['source_cells'])
        counts['enumerated_trios']+=len(event['trios']);counts['feasible_trios']+=len(event['feasible_trio_indices'])
        counts['trio_statistics']+=sum(t['statistics'] is not None for t in event['trios'])
        counts['zero_variance_trios']+=sum(t['statistics'] is not None and not t['statistics']['available'] for t in event['trios'])
        for branch,key in (('missing_source','missing_source_events'),('zero_variance','zero_variance_events'),('available','available_events')):
            counts[key]+=int(event['branch']==branch)
        for arm,key in zip(ARMS[1:],('joint_changed_holdings','diagonal_changed_holdings','structure_changed_holdings')):
            counts[key]+=int(set(picked[arm])!=set(picked[ARMS[0]]))
    scores={arm:base['F'].copy() for arm in ARMS};pres={arm:base['P'].copy() for arm in ARMS}
    return scores,choices,pres,result


def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}


def verify_selection(original,meta,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,ea=selection_oracle(original,meta,features,n)
    record_check(orders,orders_from_choices(expected,choices,meta),'Independent ordered exports')
    for got,want,label in ((scores,expected,'Final scores'),(pre_scores,pres,'Original pre')):
        require(type(got) is dict and set(got)==set(ARMS),label+' arms')
        for a in ARMS:exact_array(got[a],want[a],label+'/'+a)
    record_check(audit,ea,'Independent full decision audit')
    budget=original_base(original,meta,n)['budget_evidence']
    return dict(**{k:True for k in SELECTION_FLAGS},**budget,counts=ea['counts'],checked_events=n,
        ordered_choices_checked=4*n,producer_modules_imported=False,computes_returns=False,
        full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,
        numerical_tolerance=0.,original_source_parity_tolerance=1e-12)


def verify_prepared(panel,meta,calendar,original,review,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected=independent_features(panel,meta,calendar,original,review,n)
    require(type(features) is list and len(features)>=n,'Complete feature prefix')
    for i,(a,e) in enumerate(zip(features[:n],expected)):
        require(type(a) is dict and set(a)==set(e),'Feature schema')
        for k in e:
            if k!='trios':record_check(a[k],e[k],'Feature/'+str(i)+'/'+k)
        require(type(a['trios']) is list and len(a['trios'])==len(e['trios']),'All enumerated trios')
        for j,(ta,te) in enumerate(zip(a['trios'],e['trios'])):
            require(type(ta) is dict and set(ta)==set(te),'Trio schema')
            for k in te:
                if k=='statistics':continuous_check(ta[k],te[k],'Statistics/'+str(i)+'/'+str(j))
                else:record_check(ta[k],te[k],'Trio/'+k)
    receipt=verify_selection(original,meta,features,orders,scores=scores,pre_scores=pre_scores,audit=audit,n=n)
    es,ec,ep,ea=selection_oracle(original,meta,expected,n)
    record_check(orders,orders_from_choices(es,ec,meta),'Independent raw-source choices')
    for actual_event,expected_event in zip(audit['events'],ea['events']):
        for a,e in zip(actual_event['trios'],expected_event['trios']):
            for key in Q_KEYS:record_check(a[key],e[key],'Exact independently rebuilt rank/utility '+key)
    receipt.update({k:True for k in PREPARE_FLAGS})
    receipt.update(continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,exact_discrete=True,
        rank_tolerance=0.,full_features_supplied=True,descriptive_path_returns_checked=True)
    return receipt


def audit_outcomes(px,opening,orders,exports,results,meta):
    """Four arms, three paths and five slices; reuse only independent accounting."""
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
            if cost==0.:
                require(all(float(row['cost'])==0. for row in data['TRADES']), 'Zero-cost trade fees')
                require(all(float(row['fee'])==0. for row in data['HOLDINGS']), 'Zero-cost period fees')
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
    return dict(passed=True,checked_paths=12,checked_slices=60,checked_training_groups=24,independent_accounts=True,
        independent_of_production_model_and_engine=True,terminal_liquidation=False,
        accounting_source_sha256=ACCOUNTING_SHA,paths=reports)
