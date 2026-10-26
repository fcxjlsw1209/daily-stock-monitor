"""Independent scalar FWL residual-core source, decision and ledger verification.

Only frozen independent infrastructure is imported. New QR producer/host are
never used as an oracle. Continuous fit comparisons have prospectively fixed
bounds; all source classifications, ranks and decisions remain exact.
"""
import copy
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'work/monthly_block_sharpe242_independent.py'
PRIOR_SHA = 'd06100f847103e258ff56a4905e3b2b93401d008d6d81a58f78edfb43f07c180'
PRIOR_TEST = ROOT/'work/test_monthly_block_sharpe242_independent.py'
PRIOR_TEST_SHA = 'ad703b3d095fffdb1407f0a8b86759e551286e6b44e10b31f699a1dccf87c25e'
ARMS = ('incumbent_local67','joint_residual_halfcore','market_residual_halfcore_control')
VALUE_KEYS = ('available','candidate','control')
COUNT_FIELDS = ('events','symbols_per_event','checked_cells','stock_cells','etf_cells',
    'stock_source_available_cells','market_source_available_cells','sector_source_available_cells',
    'source_available_cells','available_cells','unavailable_cells','stock_price_observations',
    'market_price_observations','sector_price_observations','rolling_windows_checked',
    'descriptive_market_estimates','descriptive_joint_estimates','ranked_cells','new_model_fits','predictive_model_fits')
MAPPING_FIELDS = ('query_index','column_index','date','symbol','issuer_key','original_sector',
    'sector','mapping_origin','mapping_repair','mapping_evidence')
FIT_KEYS = ('market_ss','sector_ss','rho','conditioning','fit_available','beta_market_only',
    'beta_market_joint','beta_sector_joint','adjusted_market','adjusted_joint','response_ss','fit_reason')
MEASURE_KEYS = ('stock_source_available','market_source_available','sector_source_available',
    'stock_source_reason','market_source_reason','sector_source_reason','source_available','available','reason',
    'stock_start_price','stock_end_price','market_start_price','market_end_price','sector_start_price',
    'sector_end_price','stock_mean','stock_square_sum','stock_sample_vol','market_adjusted_sum',
    'joint_adjusted_sum','market_adjusted_mean','joint_adjusted_mean','raw_candidate','raw_control','fit_arrays')
CELL_KEYS = ('column_index','symbol','is_etf','mapping')+MEASURE_KEYS+('candidate','control')
EVENT_KEYS = ('query_index','date','symbols','session_position','price_start','price_end','source_dates',
    'fit_dates','missing_official_sessions','missing_vox_sessions','stock_source_available_indices','available_indices','cells')
FEATURE_FLAGS = ('passed','independent_scalar_fwl_rebuild','exact_official_505_price_window_checked',
    'skip42_checked','full_positive_source_checked','stock_source_independent_of_mapping_checked',
    'query_mapping_identity_checked','etfs_unchanged','lagged_252_fit_clock_checked',
    'application_day_excluded_from_fit','all_252_fits_checked','centered_intercept_slopes_checked',
    'intercept_not_subtracted','common_fit_support_checked','strict_conditioning_gate_checked',
    'design_scaled_slope_comparison_checked','scaled_normal_equations_checked',
    'common_total_stock_sample_vol_checked','zero_adjusted_drift_valid',
    'exact_common_stock_rank_domain_checked','average_percentile_ties_checked',
    'no_eligibility_filter','no_tolerance_created_ties','full_audit_checked',
    'descriptive_fit_counts_checked','no_future_inputs','no_predictive_fits_checked')
DECISION_COUNT_FIELDS = ('events','symbols_per_event','active_events','available_active_cells',
    'candidate_nonzero_delta_cells','control_nonzero_delta_cells','original_episode_calls',
    'new_model_fits','predictive_model_fits')
DECISION_EVENT_KEYS = ('query_index','date','initial_partial','calm','weak','active','available_indices',
    'extension_active_indices','episode_budget','candidate_delta','control_delta','original_remaining_support',
    'candidate_remaining_support','control_remaining_support')
SELECTION_FLAGS = ('passed','independent_original_pre_checked','original_model_rank_scope_preserved',
    'original_eligibility_unchanged','original_support_checked','original_budget_checked',
    'calm_nonweak_extension_budget_checked','half_core_formula_checked','core_weight_preserved',
    'missing_preserves_original_score','finite_zero_rank_active','all_states_active',
    'named_initial_partial_unchanged','etfs_unchanged','original_effects_unchanged',
    'single_original_episode','modified_pre_top_two_preserved','exact_scores_and_choices_checked',
    'no_second_rank','full_audit_checked','no_predictive_fits_checked')
RAW_ATOL = RAW_RTOL = DESIGN_ATOL = DESIGN_RTOL = 2e-10
CONDITION_MIN = 1e-8


def prior():
    if hashlib.sha256(PRIOR.read_bytes()).hexdigest()!=PRIOR_SHA:
        raise ValueError('Frozen independent source/scoring/accounting bridge changed')
    spec=importlib.util.spec_from_file_location('residual246_prior_independent',PRIOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


old=prior()
require,exact_array,record_check=old.require,old.exact_array,old.record_check
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA
finite,total,iso=old.finite,old.total,old.iso
simple_return=old.simple_return
percentile=old.percentile


def frozen_source_oracle():return old.frozen_source_oracle()
def accounting():return old.accounting()


def dependency_files():
    require(hashlib.sha256(PRIOR_TEST.read_bytes()).hexdigest()==PRIOR_TEST_SHA,'Frozen synthetic source fixture changed')
    return tuple(sorted(set(old.dependency_files())|{PRIOR,PRIOR_TEST,Path(__file__).resolve()}))


def frame_schema(frame):
    require(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex),'Price frame')
    ix=frame.index
    require(not ix.hasnans and ix.is_unique and ix.is_monotonic_increasing and ix.tz is None and
        ix.equals(ix.normalize()),'Unique ordered naive sessions')
    require(frame.columns.is_unique and all(type(s) is str and s and s.strip()==s for s in frame.columns),'Source IDs')


def axes(panel,meta,calendar,limit=None):
    require(isinstance(panel,dict) and set(panel)=={'adj','vox'},'Original adjusted and dedicated VOX frames only')
    frame_schema(panel['vox'])
    return old.axes({'adj':panel['adj']},meta,calendar,limit)


def mapping_rows(review,queries,etfs):
    require(isinstance(review,dict) and isinstance(review.get('mapping_records'),list),'Mapping-review records required')
    wanted={(i,j) for i,(_,names,_) in enumerate(queries) for j,s in enumerate(names) if s not in etfs}
    rows={};issuers={i:set() for i in range(len(queries))}
    for r in review['mapping_records']:
        require(isinstance(r,dict) and type(r.get('query_index')) is int and r['query_index']>=0,'Mapping query index')
        i=r['query_index']
        if i>=len(queries):continue
        require(set(r)==set(MAPPING_FIELDS),'Exact ten mapping fields')
        j=r['column_index'];require(type(j) is int and (i,j) in wanted and (i,j) not in rows,'Complete unique stock mapping')
        d,names,_=queries[i]
        require(r['date']==str(d.date()) and r['symbol']==names[j],'Mapping date/security identity')
        issuer=r['issuer_key'];require(type(issuer) is str and issuer and issuer.strip()==issuer and issuer not in issuers[i],
            'Nonempty unique issuer identity')
        sector=r['sector'];require(sector is None or type(sector) is str and sector and sector.strip()==sector,'Sector identifier')
        issuers[i].add(issuer);rows[i,j]=copy.deepcopy(r)
    require(set(rows)==wanted,'Every original direct stock needs mapping evidence')
    return rows


def source_path(raw):
    require(np.shape(raw)==(505,),'Exactly505 stored prices')
    if any(not isinstance(v,numbers.Real) or isinstance(v,(bool,np.bool_)) for v in raw):
        return None,'nonnumeric_adjusted_price'
    try:p=[float(v) for v in raw]
    except (TypeError,ValueError,OverflowError):return None,'nonnumeric_adjusted_price'
    if any(not math.isfinite(v) or v<=0. for v in p):return None,'invalid_adjusted_price'
    return p,'available'


def extract(frame,symbol,dates):
    reason=('insufficient_history' if not dates else 'missing_symbol' if symbol not in frame.columns else
        'missing_official_session' if any(d not in frame.index for d in dates) else None)
    return (None,reason) if reason else source_path([frame.at[d,symbol] for d in dates])


def square(v):
    out=finite(v*v,'squared deviation')
    if v!=0. and out==0.:raise FloatingPointError('Nonzero squared-deviation underflow')
    return out


def product(a,b):return finite(a*b,'cross product')


def mean(values):
    summed=total(values);value=finite(summed/float(len(values)),'mean')
    if summed!=0. and value==0.:raise FloatingPointError('Nonzero mean underflow')
    return summed,value


def center(values):
    _,mu=mean(values);z=[finite(v-mu,'centered value') for v in values]
    return z,total(square(v) for v in z)


def dot(a,b):return total(product(x,y) for x,y in zip(a,b))


def ratio(a,b):
    out=finite(a/b,'finite ratio')
    if a!=0. and out==0.:raise FloatingPointError('Nonzero ratio underflow')
    return out


def fit_fwl(y,m,s):
    """Independent scalar Frisch-Waugh-Lovell, with original centered slopes."""
    require(len(y)==len(m)==len(s)==252,'Prior252 return observations')
    yc,yy=center(y);mc,mm=center(m);sc,ss=center(s)
    out=dict(market_ss=mm,sector_ss=ss,rho=None,conditioning=None,fit_available=False,
        beta_market_only=None,beta_market_joint=None,beta_sector_joint=None,
        adjusted_market=None,adjusted_joint=None,response_ss=yy,fit_reason='available')
    if mm==0.:out['fit_reason']='zero_market_variance';return out
    if ss==0.:out['fit_reason']='zero_sector_variance';return out
    mn=finite(math.sqrt(mm));sn=finite(math.sqrt(ss));den=product(mn,sn)
    rho=ratio(dot(mc,sc),den);condition=finite(1.-product(rho,rho),'conditioning')
    out.update(rho=rho,conditioning=condition)
    if not condition>CONDITION_MIN:out['fit_reason']='collinear_factors';return out
    bm=ratio(dot(mc,yc),mm)
    projection=ratio(dot(mc,sc),mm)
    orth=[finite(b-product(projection,a),'FWL sector residual') for a,b in zip(mc,sc)]
    oo=total(square(v) for v in orth)
    require(oo>0.,'Admitted factor gate must have positive FWL residual variance')
    bs=ratio(dot(orth,yc),oo)
    bj=finite(bm-product(bs,projection),'FWL joint market slope')
    check_normal_equations(yc,mc,sc,mm,ss,bm,bj,bs)
    out.update(fit_available=True,beta_market_only=bm,beta_market_joint=bj,beta_sector_joint=bs)
    return out


def check_normal_equations(y,m,s,mm,ss,bm,bj,bs):
    norm=finite(math.sqrt(total(square(v) for v in y)))
    tol=finite(DESIGN_ATOL+DESIGN_RTOL*norm)
    u=[ratio(v,math.sqrt(mm)) for v in m]
    v=[ratio(z,math.sqrt(ss)) for z in s]
    e0=[finite(a-product(bm,b)) for a,b in zip(y,m)]
    ej=[finite(finite(a-product(bj,b))-product(bs,c)) for a,b,c in zip(y,m,s)]
    require(abs(dot(u,e0))<=tol and abs(dot(u,ej))<=tol and abs(dot(v,ej))<=tol,
        'Fixed design-scaled normal-equation verification')


def empty_measure(reason):
    out={k:None for k in MEASURE_KEYS}
    out.update(stock_source_available=False,market_source_available=False,sector_source_available=False,
        stock_source_reason=reason,market_source_reason=reason,sector_source_reason=reason,
        source_available=False,available=False,reason=reason)
    return out


def measures(stock,market,sector):
    out=empty_measure(None)
    for label,(prices,reason) in zip(('stock','market','sector'),(stock,market,sector)):
        out[label+'_source_available']=prices is not None;out[label+'_source_reason']=reason
        if prices is not None:out[label+'_start_price']=prices[0];out[label+'_end_price']=prices[-1]
    if stock[0] is None:out['reason']='stock_'+stock[1];return out
    if sector[1]=='unresolved_mapping':out['reason']='unresolved_mapping';return out
    if market[0] is None:out['reason']='market_'+market[1];return out
    if sector[0] is None:out['reason']='sector_'+sector[1];return out
    out['source_available']=True
    y,m,s=[[simple_return(p[k],p[k-1]) for k in range(1,505)] for p,_ in (stock,market,sector)]
    formation=y[252:];_,mu=mean(formation);sq=total(square(v-mu) for v in formation)
    var=finite(sq/251.,'formation sample variance')
    if sq>0. and var==0.:raise FloatingPointError('Formation variance underflow')
    sd=finite(math.sqrt(var));out.update(stock_mean=mu,stock_square_sum=sq,stock_sample_vol=sd)
    fits={k:[] for k in FIT_KEYS};first_failure=None
    for k in range(252,504):
        fit=fit_fwl(y[k-252:k],m[k-252:k],s[k-252:k])
        if fit['fit_available']:
            fit['adjusted_market']=finite(y[k]-product(fit['beta_market_only'],m[k]))
            fit['adjusted_joint']=finite(finite(y[k]-product(fit['beta_market_joint'],m[k]))-
                product(fit['beta_sector_joint'],s[k]))
        elif first_failure is None:first_failure=fit['fit_reason']
        for key in FIT_KEYS:fits[key].append(fit[key])
    out['fit_arrays']=fits
    out['reason']='zero_stock_sample_volatility' if sd==0. else first_failure or 'available'
    if first_failure:return out
    ms,mm=mean(fits['adjusted_market']);js,jm=mean(fits['adjusted_joint'])
    out.update(market_adjusted_sum=ms,joint_adjusted_sum=js,market_adjusted_mean=mm,joint_adjusted_mean=jm)
    if sd>0.:out.update(available=True,raw_candidate=ratio(jm,sd),raw_control=ratio(mm,sd))
    return out


def measure_oracle(stock,market,sector):return measures(source_path(stock),source_path(market),source_path(sector))


def independent_features(panel,meta,calendar,source_review,limit=None):
    frame,sessions,queries,width=axes(panel,meta,calendar,limit);etfs=set(meta['etfs'])
    mappings=mapping_rows(source_review,queries,etfs)
    shape=(len(queries),width);v=dict(available=np.zeros(shape,bool),candidate=np.full(shape,np.nan),control=np.full(shape,np.nan))
    rawc=np.full(shape,np.nan);rawm=np.full(shape,np.nan)
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=len(queries),symbols_per_event=width,checked_cells=len(queries)*width)
    history=[];reasons={}
    for i,(day,names,t) in enumerate(queries):
        dates=sessions[t-546:t-41] if t>=546 else [];ds=[str(d.date()) for d in dates];cells=[]
        for j,symbol in enumerate(names):
            mapping=None;is_etf=symbol in etfs
            if is_etf:
                measured=empty_measure('etf_unchanged');counts['etf_cells']+=1
            else:
                counts['stock_cells']+=1;mapping=mappings[i,j]
                stock=extract(frame,symbol,dates)
                market=extract(frame,'SPY',dates)
                sector=(None,'unresolved_mapping') if mapping['sector'] is None else extract(
                    panel['vox'] if mapping['sector']=='VOX' else frame,mapping['sector'],dates)
                measured=measures(stock,market,sector)
                for label in ('stock','market','sector'):
                    counts[label+'_source_available_cells']+=int(measured[label+'_source_available'])
                counts['source_available_cells']+=int(measured['source_available'])
                if measured['fit_arrays'] is not None:
                    fits=sum(measured['fit_arrays']['fit_available'])
                    counts['descriptive_market_estimates']+=fits;counts['descriptive_joint_estimates']+=fits
            good=measured['available'];v['available'][i,j]=good
            if good:rawc[i,j]=measured['raw_candidate'];rawm[i,j]=measured['raw_control'];counts['available_cells']+=1
            reasons[measured['reason']]=reasons.get(measured['reason'],0)+1
            cells.append(dict(column_index=j,symbol=symbol,is_etf=is_etf,mapping=mapping,**measured,candidate=None,control=None))
        history.append(dict(query_index=i,date=str(day.date()),symbols=names,session_position=t,
            price_start=ds[0] if ds else None,price_end=ds[-1] if ds else None,source_dates=ds,
            fit_dates=[dict(training_price_start=ds[d-253],training_return_start=ds[d-252],
                training_return_end=ds[d-1],application_date=ds[d]) for d in range(253,505)] if ds else [],
            missing_official_sessions=[str(d.date()) for d in dates if d not in frame.index],
            missing_vox_sessions=[str(d.date()) for d in dates if d not in panel['vox'].index],
            stock_source_available_indices=[j for j,c in enumerate(cells) if c['stock_source_available']],
            available_indices=[j for j,c in enumerate(cells) if c['available']],cells=cells))
    v['candidate']=percentile(rawc,v['available']);v['control']=percentile(rawm,v['available'])
    for i,event in enumerate(history):
        for j,c in enumerate(event['cells']):
            if c['available']:c.update(candidate=float(v['candidate'][i,j]),control=float(v['control'][i,j]))
    counts.update(unavailable_cells=counts['checked_cells']-counts['available_cells'],
        rolling_windows_checked=252*counts['source_available_cells'],ranked_cells=2*counts['available_cells'])
    for label in ('stock','market','sector'):counts[label+'_price_observations']=505*counts[label+'_source_available_cells']
    return v,dict(schema='monthly_residual_core246_features_v1',events=len(queries),counts=counts,
        reason_counts=reasons,event_history=history)


def close_stat(actual,expected,label):
    require(isinstance(actual,numbers.Real) and not isinstance(actual,(bool,np.bool_)) and math.isfinite(float(actual)),
        'Finite reported '+label)
    require(abs(float(actual)-expected)<=RAW_ATOL+RAW_RTOL*abs(expected),'Independent continuous statistic '+label)


def verify_fit(actual,expected,stock,market,sector,label):
    require(isinstance(actual,dict) and set(actual)==set(FIT_KEYS),'Full rolling fit arrays '+label)
    require(all(isinstance(actual[k],list) and len(actual[k])==252 for k in FIT_KEYS),'Every252 fit retained '+label)
    y,m,s=[[simple_return(p[k],p[k-1]) for k in range(1,505)] for p in (stock,market,sector)]
    for h in range(252):
        for key in ('fit_available','fit_reason'):
            record_check(actual[key][h],expected[key][h],label+'/'+key)
        for key in ('market_ss','sector_ss','rho','conditioning','response_ss','adjusted_market','adjusted_joint'):
            a,e=actual[key][h],expected[key][h]
            if e is None:require(a is None,label+'/'+key+' absent when undefined')
            else:close_stat(a,e,label+'/'+key)
        # A tolerated numeric difference may NEVER change a source/fit gate.
        mm,ss=actual['market_ss'][h],actual['sector_ss'][h]
        require(mm>=0. and ss>=0. and actual['response_ss'][h]>=0.,'Nonnegative centered sums of squares')
        require((mm>0.)==(expected['market_ss'][h]>0.) and (ss>0.)==(expected['sector_ss'][h]>0.),
            'Exact positive factor variance classifications')
        if mm==0.:reason='zero_market_variance'
        elif ss==0.:reason='zero_sector_variance'
        elif not actual['conditioning'][h]>CONDITION_MIN:reason='collinear_factors'
        else:reason='available'
        require(reason==expected['fit_reason'][h],'Exact producer/oracle factor gate '+label)
        keys=('beta_market_only','beta_market_joint','beta_sector_joint')
        if not expected['fit_available'][h]:
            require(all(actual[k][h] is None for k in keys),'Unsupported fit has no slopes')
            continue
        k=h+252;yc,yy=center(y[h:k]);mc,ms=center(m[h:k]);sc,sss=center(s[h:k])
        tol=DESIGN_ATOL+DESIGN_RTOL*math.sqrt(yy)
        for key,norm in zip(keys,(math.sqrt(ms),math.sqrt(ms),math.sqrt(sss))):
            a,e=actual[key][h],expected[key][h]
            require(isinstance(a,numbers.Real) and not isinstance(a,(bool,np.bool_)) and math.isfinite(float(a)),
                'Finite fitted slope '+label+'/'+key)
            require(abs(float(a)-e)*norm<=tol,'Fixed design-scaled independent slope '+label+'/'+key)
        bm,bj,bs=[float(actual[key][h]) for key in keys]
        check_normal_equations(yc,mc,sc,ms,sss,bm,bj,bs)
        # Canonical application order is checked against reported slopes, not merely close FWL results.
        em=finite(y[k]-product(bm,m[k]));ej=finite(finite(y[k]-product(bj,m[k]))-product(bs,s[k]))
        require(actual['adjusted_market'][h]==em and actual['adjusted_joint'][h]==ej,
            'Reported slopes must reproduce application in declared order')


def verify_features(panel,meta,calendar,source_review,values,audit,limit=None):
    wanted,report=independent_features(panel,meta,calendar,source_review,limit)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Three feature arrays')
    for key in VALUE_KEYS:exact_array(values[key],wanted[key],'Exact independent source/rank '+key)
    require(isinstance(audit,dict) and set(audit)==set(report),'Full feature audit schema')
    for key in ('schema','events','counts','reason_counts'):record_check(audit[key],report[key],'Audit '+key)
    require(isinstance(audit['event_history'],list) and len(audit['event_history'])==len(report['event_history']),'Full event evidence')
    continuous={'stock_mean','stock_square_sum','stock_sample_vol','market_adjusted_sum','joint_adjusted_sum',
        'market_adjusted_mean','joint_adjusted_mean','raw_candidate','raw_control'}
    for i,(ae,ee) in enumerate(zip(audit['event_history'],report['event_history'])):
        require(isinstance(ae,dict) and set(ae)==set(ee),'Event schema')
        for key in ee:
            if key!='cells':record_check(ae[key],ee[key],'Event '+key)
        require(isinstance(ae['cells'],list) and len(ae['cells'])==len(ee['cells']),'Full cell inventory')
        for j,(ac,ec) in enumerate(zip(ae['cells'],ee['cells'])):
            label=str(i)+'/'+str(j)
            require(isinstance(ac,dict) and set(ac)==set(ec),'Full cell schema '+label)
            for key in ec:
                if key=='fit_arrays':continue
                if key in continuous and ec[key] is not None:
                    close_stat(ac[key],ec[key],label+'/'+key)
                    if key in ('stock_square_sum','stock_sample_vol'):
                        require(ac[key]>=0. and (ac[key]>0.)==(ec[key]>0.),'Exact stock risk zero/positive classification')
                else:record_check(ac[key],ec[key],label+'/'+key)
            if ec['fit_arrays'] is None:require(ac['fit_arrays'] is None,'No fits without common source')
            else:
                dates=[pd.Timestamp(d) for d in ee['source_dates']];symbol=ec['symbol'];sector=ec['mapping']['sector']
                stock=extract(panel['adj'],symbol,dates)[0];market=extract(panel['adj'],'SPY',dates)[0]
                sp=extract(panel['vox'] if sector=='VOX' else panel['adj'],sector,dates)[0]
                verify_fit(ac['fit_arrays'],ec['fit_arrays'],stock,market,sp,label)
            # Ranks must also follow the producer's own raw numbers without tolerant ties.
    mask=values['available'];pc=np.full(mask.shape,np.nan);pm=pc.copy()
    for i,ev in enumerate(audit['event_history']):
        for j,c in enumerate(ev['cells']):
            if c['available']:pc[i,j]=c['raw_candidate'];pm[i,j]=c['raw_control']
    exact_array(values['candidate'],percentile(pc,mask),'Producer raw candidate ranks exact')
    exact_array(values['control'],percentile(pm,mask),'Producer raw control ranks exact')
    return dict(**{k:True for k in FEATURE_FLAGS},**report['counts'],counts=report['counts'],
        numerical_tolerance=RAW_ATOL,relative_tolerance=RAW_RTOL,continuous_atol=RAW_ATOL,continuous_rtol=RAW_RTOL,
        exact_discrete=True,design_absolute_tolerance=DESIGN_ATOL,
        design_relative_tolerance=DESIGN_RTOL,rank_tolerance=0.,producer_modules_imported=False,computes_returns=False)


def selection_values(values,n,width,meta):
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Three feature arrays')
    mask=np.asarray(values['available']);require(mask.ndim==2 and mask.dtype==np.dtype(bool) and len(mask)>=n and
        (mask.shape[1]==width or n==0 and mask.shape==(0,0)),'Feature availability shape')
    if n==0:return dict(available=np.empty((0,width),bool),candidate=np.empty((0,width)),control=np.empty((0,width)))
    out={'available':mask[:n].copy()}
    for key in ('candidate','control'):
        a=np.asarray(values[key]);require(a.shape==mask.shape and a.dtype==np.dtype(np.float64),'Float64 feature shape')
        a=a[:n];require(np.isfinite(a[out['available']]).all() and np.isnan(a[~out['available']]).all(),'Common finite support')
        require((np.abs(a[out['available']])<=.5).all(),'Original percentile bounds');out[key]=a.copy()
    etfs=set(meta['etfs'])
    for i,ss in enumerate(meta['symbols'][:n]):
        require(not any(out['available'][i,j] for j,s in enumerate(ss) if s in etfs),'ETFs keep original core')
    return out


def _selection(original,meta,values,n=None):
    n=len(meta['dates']) if n is None else n;require(type(n) is int and 0<=n<=len(meta['dates']),'Selection prefix')
    width,days,names=old.old.axes(original,meta,n);v=selection_values(values,n,width,meta)
    dc={k:0 for k in DECISION_COUNT_FIELDS};dc.update(events=n,symbols_per_event=width,original_episode_calls=3)
    audit=dict(schema='monthly_residual_core246_decisions_v1',counts=dc,events=[])
    if n==0:
        empty={arm:np.empty((0,width)) for arm in ARMS}
        return empty,{arm:[] for arm in ARMS},{arm:a.copy() for arm,a in empty.items()},audit,dict(
            minimum_remaining_support=None,maximum_budget_sum_error=0.,calm_nonweak_extension_count=0,
            maximum_absolute_delta=0.,changed_cells={arm:0 for arm in ARMS})
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    exact_array(x['cw'],np.broadcast_to(np.where(x['calm'],.5,.52)[:,None],(n,width)),'Original core weight')
    require((np.abs(x['core'])<=.5*x['cw']).all(),'Original weighted core bounds')
    ext=np.asarray(original['extension_available'])[:n];forecast=np.asarray(original['forecast_available'])[:n]
    ranklog=np.asarray(original['rank_log_excess'])[:n]
    require(ext.shape==forecast.shape==ranklog.shape==(n,width) and ext.dtype==forecast.dtype==np.dtype(bool),'Original extension masks')
    A=(x['calm']&~x['weak'])[:,None]&ext&forecast&x['eligible']&np.isfinite(ranklog)
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    active=np.array([str(d.date())!='2016-08-17' for d in days],bool)
    reserve=(1.-x['cw'])-(.025*x['calm'][:,None]+.05*x['weak'][:,None]+.01*A)-budget[:,None]
    require((reserve>=.38-1e-12).all(),'Original conservative support reserve')
    scores={};choices={};pres={};deltas={};changed={};maxdelta=0.;sum_error=0.
    for arm,key in zip(ARMS,(None,'candidate','control')):
        delta=np.zeros((n,width));pre=x['local'].copy()
        if key is not None:
            for i in range(n):
                if active[i]:
                    for j in range(width):
                        if v['available'][i,j]:
                            delta[i,j]=.5*(float(x['cw'][i,j])*float(v[key][i,j])-float(x['core'][i,j]))
        require(np.isfinite(delta).all() and (np.abs(delta)<=.26).all(),'Finite half-core correction')
        nz=delta!=0.;pre[nz]+=delta[nz];require(np.isfinite(pre).all(),'Finite original modified pre')
        deltas[arm]=delta;pres[arm]=pre.copy();changed[arm]=int(nz.sum());maxdelta=max(maxdelta,float(np.max(np.abs(delta))))
        score=pre.copy();rows=[]
        for i,ss in enumerate(names):
            order=sorted([j for j in range(width) if x['eligible'][i,j]],key=lambda j:(-pre[i,j],ss[j]))
            if len(order)<3:rows.append(order if len(order)==2 else []);continue
            boundary=[j for j in order[2:5] if float(pre[i,order[2]])-float(pre[i,j])<=.02]
            for j in boundary:
                target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
                score[i,j]+=budget[i]*(target-x['support'][i,j])
            winner=min(boundary,key=lambda j:(-score[i,j],ss[j]))
            rows.append(sorted(order[:2]+[winner],key=lambda j:(-score[i,j],ss[j])))
        require(np.isfinite(score).all(),'Finite single-episode scores');scores[arm]=score;choices[arm]=rows
    for i in range(n):
        for j in range(width):
            err=abs(math.fsum([float(x['cw'][i,j]),.025*float(x['calm'][i]),.05*float(x['weak'][i]),
                .01*float(A[i,j]),float(budget[i]),float(reserve[i,j])])-1.)
            require(err<=1e-12,'Original complete score budget');sum_error=max(sum_error,err)
        audit['events'].append(dict(query_index=i,date=meta['dates'][i],initial_partial=not bool(active[i]),
            calm=bool(x['calm'][i]),weak=bool(x['weak'][i]),active=bool(active[i]),
            available_indices=np.flatnonzero(v['available'][i]).tolist(),extension_active_indices=np.flatnonzero(A[i]).tolist(),
            episode_budget=float(budget[i]),candidate_delta=deltas[ARMS[1]][i].tolist(),control_delta=deltas[ARMS[2]][i].tolist(),
            original_remaining_support=reserve[i].tolist(),candidate_remaining_support=reserve[i].tolist(),
            control_remaining_support=reserve[i].tolist()))
    dc.update(active_events=int(active.sum()),available_active_cells=int((active[:,None]&v['available']).sum()),
        candidate_nonzero_delta_cells=changed[ARMS[1]],control_nonzero_delta_cells=changed[ARMS[2]])
    return scores,choices,pres,audit,dict(minimum_remaining_support=float(np.min(reserve)),maximum_budget_sum_error=sum_error,
        calm_nonweak_extension_count=int(A.sum()),maximum_absolute_delta=maxdelta,changed_cells=changed)


def selection_oracle(original,meta,values,n=None):return _selection(original,meta,values,n)[:4]


def verify_selection(original,meta,values,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,expected_audit,stats=_selection(original,meta,values,n)
    wanted={arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(expected[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders,wanted,'Independent ordered choices')
    for got,ref,label in ((scores,expected,'full scores'),(pre_scores,pres,'full pre-scores')):
        require(isinstance(got,dict) and set(got)==set(ARMS),'Three '+label)
        for arm in ARMS:exact_array(got[arm],ref[arm],label+'/'+arm)
    record_check(audit,expected_audit,'Full decision evidence')
    return dict(**{k:True for k in SELECTION_FLAGS},**stats,counts=expected_audit['counts'],checked_events=n,
        ordered_choices_checked=3*n,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,
        numerical_tolerance=0.,original_source_parity_tolerance=1e-12,producer_modules_imported=False,computes_returns=False)


def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arms')
    mapping=dict(zip(old.ARMS,ARMS))
    report=old.audit_outcomes(px,opening,*[{k:data[v] for k,v in mapping.items()} for data in (orders,exports,results)],meta)
    report['paths']={mapping[k]:v for k,v in report['paths'].items()};return report
