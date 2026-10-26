"""Independent251 raw-source, direct-basket and incumbent verification.

Only pinned OLD independent original-core/accounting code is reused. No new251
producer math, kernel or host is imported. Basket choices never invoke episode.
"""
import copy
from fractions import Fraction
import hashlib
import importlib.util
from itertools import combinations
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_calendar_clock248_v2_independent.py'
PRIOR_SHA='c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c'
PRIOR_TEST=ROOT/'work/test_monthly_reference_gain250_independent.py'
PRIOR_TEST_SHA='8b15f23ba83f36f2d063a87a54a9b76093c703df2c9a1f1f2a7846a4afc4e106'
ACCOUNT_BASE=ROOT/'work/monthly_reference_gain250_independent.py'
ACCOUNT_BASE_SHA='d61b3243af183da3e65fd1cbe4052e35d8933fb5c2472fc236e29e92832729a4'
MATH=ROOT/'work/monthly_basket_core251_math_independent.py'
MATH_SHA='39ea19982e43129da1a3c72f1e283840779aa33dc333036608d1cff2e9f58233'
MATH_TEST=ROOT/'work/test_monthly_basket_core251_math_independent.py'
MATH_TEST_SHA='9f0d4e17cde9f6f6e07a6f79fe9802fbc6a30b2c169dee28e980a36f70769f4f'
ARMS=('incumbent_local67','long_momentum_basket_core','long_momentum_diagonal_core')
C_SOURCE_ATOL=1e-12
C_SOURCE_RTOL=0.
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10


def load_pinned(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen independent dependency changed '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


old=load_pinned(PRIOR,PRIOR_SHA,'basket251_old_independent')
exactmath=load_pinned(MATH,MATH_SHA,'basket251_exact_independent')
require,exact_array,iso=old.require,old.exact_array,old.iso
record_check=exactmath.exact
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA

def accounting():return old.accounting()
def frozen_source_oracle():return old.frozen_source_oracle()
def dependency_files():
    for p,h in ((PRIOR_TEST,PRIOR_TEST_SHA),(ACCOUNT_BASE,ACCOUNT_BASE_SHA),(MATH_TEST,MATH_TEST_SHA)):
        require(hashlib.sha256(p.read_bytes()).hexdigest()==h,'Frozen source/test dependency changed')
    fixture_bridge=load_pinned(ACCOUNT_BASE,ACCOUNT_BASE_SHA,'basket251_fixture_dependencies')
    return tuple(sorted(set(old.dependency_files())|set(fixture_bridge.dependency_files())|{PRIOR,PRIOR_TEST,ACCOUNT_BASE,MATH,MATH_TEST,Path(__file__).resolve()}))


def original_base(original,meta,n):
    base=old.original_base(original,meta,n);require(base['width']==49,'Exactly49 original members');return base


def clock_row(i,base,meta,sessions):
    day=base['days'][i];require(day in sessions,'Official query');t=sessions.index(day)
    require(t>=294 and t+1<len(sessions),'Complete formation clock and witness')
    partial=meta['dates'][i]=='2016-08-17'
    if not partial:require(sessions[t+1].to_period('M')!=day.to_period('M'),'Official month end')
    ds=lambda k:str(sessions[k].date())
    return dict(query_index=i,date=meta['dates'][i],initial_partial=partial,numeric_evaluation_required=not partial,
      symbols=list(base['names'][i]),query_position=t,next_official_session=ds(t+1),price_start=ds(t-294),price_end=ds(t-42),
      source_start_position=t-294,source_end_position=t-42,price_count=253,return_count=252,block_count=12,block_sessions=21,
      block_boundary_dates=[ds(t-294+21*b) for b in range(13)])


def review_index(review,n):
    require(type(review) is dict and review.get('schema')=='monthly_basket_core251_metadata_source_review_v1','Source review schema')
    origin=review.get('panel_first_session');iso(origin)
    require(type(review.get('listing_bounds')) is list,'Listing bounds');bounds={}
    for b in review['listing_bounds']:
        require(type(b) is dict and set(b)=={'symbol','issuer_key','not_before','available_from','kind','evidence_id'},'Listing bound keys')
        require(all(type(v) is str and v and v.strip()==v for v in b.values()),'Listing metadata strings')
        iso(b['not_before']);iso(b['available_from']);require(b['symbol'] not in bounds,'Duplicate bound');bounds[b['symbol']]=b
    ids={};clocks={}
    for key,target,with_column in (('identity_cells',ids,True),('clock_events',clocks,False)):
        require(type(review.get(key)) is list,key+' list')
        for item in review[key]:
            require(type(item) is dict and type(item.get('query_index')) is int and item['query_index']>=0,key+' query index')
            q=item['query_index']
            if q>=n:continue
            if with_column:
                j=item.get('column_index');require(type(j) is int and 0<=j<49,'Identity column');index=(q,j)
            else:index=q
            require(index not in target,'Duplicate '+key);target[index]=item
    require(set(clocks)==set(range(n)),'Every consumed clock')
    return origin,bounds,ids,clocks


def metadata_checks(clock,sessions,bounds,ids,clocks):
    q=clock['query_index'];t=clock['query_position']
    for k,v in clock.items():record_check(clocks[q].get(k),v,'Clock/'+k)
    found={}
    for j,symbol in enumerate(clock['symbols']):
        if symbol not in bounds:continue
        b=bounds[symbol];item=ids.get((q,j));require(item is not None,'Missing identity occurrence')
        fields=dict(query_index=q,column_index=j,date=clock['date'],symbol=symbol,issuer_key=b['issuer_key'],
          not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=clock['date'],
          price_start=clock['price_start'],price_end=clock['price_end'],
          price_window_dates_before_bound=sum(str(d.date())<b['not_before'] for d in sessions[t-294:t-41]))
        for k,v in fields.items():record_check(item.get(k),v,'Identity/'+k)
        found[j]=b
    require({j for i,j in ids if i==q}==set(found),'Exact identity inventory');return found


def required_prices(frame,symbol,days,bound,query):
    require(symbol in frame.columns,'Required source column missing');out=[];errors=[]
    for offset,d in enumerate(days):
        if d not in frame.index:errors.append((offset,'missing_row'));continue
        v=frame.at[d,symbol]
        if not isinstance(v,Real) or isinstance(v,(bool,np.bool_)):errors.append((offset,'real_nonboolean'));continue
        try:f=float(v)
        except (TypeError,ValueError,OverflowError):errors.append((offset,'conversion'));continue
        if not math.isfinite(f) or f<=0.:errors.append((offset,'positive_finite'));continue
        if bound and bound['available_from']<=query and str(d.date())<bound['not_before']:errors.append((offset,'visible_listing_conflict'))
        out.append(f)
    require(not errors,'Required positive finite source or visible listing failure '+symbol+': '+repr(errors))
    return out

EVENT_KEYS=('query_index','date','initial_partial','calm','core_weight','symbols','branch','panel_first_session',
 'query_position','next_official_session','trading_price_start','trading_price_end','risk_price_start','risk_price_end',
 'price_start','price_end','block_boundary_dates','u0_indices','core_source_max_abs_error','original_eligible_indices',
 'required_indices','cells','new_eligible_indices','excluded_momentum_indices','shortlist_indices','trios','common_trio_indices','winners')
CELL_KEYS=('column_index','symbol','original_finite','original_raw','original_rank','frozen_core','rebuilt_core',
 'core_abs_error','sigma63','original_eligible','source_available','measurement')
TRIO_KEYS=('indices','symbols','statistics')
WINNER_KEYS=('branch','winning_trio_index','tied_trio_indices','selected_indices','selected_symbols')
CANDIDATE_KEYS=('new_eligible_indices','excluded_momentum_indices','shortlist_indices','trios','common_trio_indices','winners')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','calm_formal_events','noncalm_formal_events',
 'original_checked_cells','original_finite_cells','original_eligible_cells','source_required_cells','source_available_cells',
 'source_price_observations','momentum_estimates','block_return_estimates','new_eligible_cells','momentum_excluded_cells',
 'shortlist_cells','enumerated_trios','common_supported_trios','zero_joint_variance_trios','zero_diagonal_variance_trios',
 'cash_events','two_name_events','zero_variance_events','optimized_events','candidate_changed_holdings_events',
 'diagonal_changed_holdings_events','original_episode_calls','experimental_episode_calls','new_model_fits','predictive_model_fits')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_core_checked','original_eligibility_preserved_as_input',
 'exact_incumbent_copy_checked','new_annual_momentum_domain_checked','exact_top6_shortlist_checked','complete_trio_enumeration_checked',
 'common_positive_variance_domain_checked','exact_rational_statistics_checked','exact_ratio_comparison_checked','lexicographic_ties_checked',
 'direct_basket_orders_checked','display_scores_only_checked','no_experimental_episode','named_initial_partial_unchanged',
 'small_domain_and_zero_variance_fallback_checked','exact_scores_and_choices_checked','full_audit_checked','descriptive_counts_checked','no_predictive_fits_checked')
PREPARE_FLAGS=SELECTION_FLAGS+('independent_source_rebuild','original_full_history_pandas_checked','original_official_row_clock_checked',
 'original_raw_rank_parity_checked','original_weighted_core_tolerance_checked','price_clocks_checked','all_Eold_required_sources_checked',
 'Eold_false_new_source_unconsumed','positive_annual_mean_invariant_checked','visible_listing_bounds_checked','no_future_source_values')
SOURCE_GATE_FLAGS=('passed','full_features_supplied','original_core_checked','price_clocks_checked','original_raw_rank_parity_checked',
 'all_Eold_required_sources_checked','Eold_false_new_source_unconsumed','visible_listing_bounds_checked','exact_discrete')
INCUMBENT_GATE_FLAGS=old.INCUMBENT_GATE_FLAGS


def independent_features(panel,meta,calendar,original,review,n=None,*,include_selection=False):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    frame,sessions=old.source_axes(panel,calendar,n);origin,bounds,ids,clocks=review_index(review,n);features=[]
    for i in range(n):
        clock=clock_row(i,base,meta,sessions);t=clock['query_position'];day=base['days'][i];names=list(base['names'][i])
        bound=metadata_checks(clock,sessions,bounds,ids,clocks);partial=clock['initial_partial']
        eligible=np.flatnonzero(base['E'][i]).tolist()
        e=dict(query_index=i,date=meta['dates'][i],initial_partial=partial,calm=bool(base['calm'][i]),
          core_weight=float(base['cw'][i]),symbols=names,branch='initial_partial' if partial else 'source_verified',panel_first_session=origin,
          query_position=t,next_official_session=clock['next_official_session'],
          trading_price_start=str(sessions[t-(252 if base['calm'][i] else 294)].date()),trading_price_end=clock['price_end'],
          risk_price_start=str(sessions[t-63].date()),risk_price_end=meta['dates'][i],price_start=clock['price_start'],price_end=clock['price_end'],
          block_boundary_dates=clock['block_boundary_dates'],u0_indices=[],core_source_max_abs_error=None,
          original_eligible_indices=eligible,required_indices=[] if partial else eligible.copy(),cells=[],
          new_eligible_indices=None,excluded_momentum_indices=None,shortlist_indices=None,trios=None,common_trio_indices=None,winners=None)
        if not partial:
            require(all(s in frame.columns for s in names),'All original price columns required')
            prefix=frame.loc[:day]
            require(len(prefix)>=295 and prefix.index[-295:].equals(pd.DatetimeIndex(sessions[t-294:t+1])),'Original last295 official row parity')
            pairs,sigmas=old.original_values(frame,day,names,origin)
            raw=np.array([p[0 if base['calm'][i] else 1] for p in pairs]);raw[~np.isfinite(raw)]=np.nan
            ranks,U=old.independent_rank(raw);e['u0_indices']=U
            source_windows={};source_errors=[]
            # All Eold members are inspected before exact new M or basket work.
            for j in eligible:
                try:source_windows[j]=required_prices(frame,names[j],sessions[t-294:t-41],bound.get(j),e['date'])
                except ValueError as error:source_errors.append(str(error))
            require(not source_errors,'Required Eold sources: '+'; '.join(source_errors))
            for j,s in enumerate(names):
                rebuilt=float(base['cw'][i]*ranks[j]);frozen=float(base['core'][i,j]);err=abs(rebuilt-frozen)
                require(err<=C_SOURCE_ATOL,'Original core source parity')
                measurement=exactmath.measure_oracle(source_windows[j]) if j in source_windows else None
                e['cells'].append(dict(column_index=j,symbol=s,original_finite=j in U,original_raw=float(raw[j]) if j in U else None,
                  original_rank=float(ranks[j]),frozen_core=frozen,rebuilt_core=rebuilt,core_abs_error=err,
                  sigma63=float(sigmas[j]) if math.isfinite(sigmas[j]) else None,original_eligible=bool(base['E'][i,j]),
                  source_available=j in source_windows,measurement=measurement))
            e['core_source_max_abs_error']=max(c['core_abs_error'] for c in e['cells'])
            if include_selection:e=candidate_event(e)
        features.append(e)
    return features


def validate_measurement(m):
    require(type(m) is dict and set(m)==set(exactmath.MEASURE_KEYS),'Exact measure fields')
    for k,v in dict(schema=exactmath.MEASURE_SCHEMA,price_count=253,block_count=12).items():record_check(m[k],v,'Measure/'+k)
    require(type(m['block_returns']) is list and len(m['block_returns'])==12,'Twelve blocks')
    gross=Fraction(1)
    for p in m['block_returns']:
        r=exactmath.rational(p);require(r>-1,'Positive block gross');gross*=1+r
    require(gross==1+exactmath.rational(m['momentum']),'Exact momentum/block clock parity')


def candidate_event(source):
    """Complete only post-incumbent fields; no source or original score reads."""
    e=copy.deepcopy(source)
    if e['initial_partial']:return e
    momentum={c['column_index']:exactmath.rational(c['measurement']['momentum']) for c in e['cells'] if c['original_eligible']}
    included=sorted(j for j,m in momentum.items() if m>0);excluded=sorted(j for j,m in momentum.items() if m<=0)
    shortlist=sorted(included,key=lambda j:(-momentum[j],e['symbols'][j]))[:6]
    triples=[]
    for ids in combinations(sorted(shortlist,key=lambda j:e['symbols'][j]),3):
        statistics=exactmath.basket_oracle([e['cells'][j]['measurement']['block_returns'] for j in ids])
        triples.append(dict(indices=list(ids),symbols=[e['symbols'][j] for j in ids],statistics=statistics))
    common=[i for i,trio in enumerate(triples) if trio['statistics']['supported']]
    branch='cash' if len(included)<2 else 'two_names' if len(included)==2 else 'zero_variance' if not common else 'optimized'
    winners={}
    for arm,key in ((ARMS[1],'joint_variance'),(ARMS[2],'diagonal_variance')):
        index=None;ties=[]
        if branch=='cash':selected=[]
        elif branch=='two_names':selected=shortlist.copy()
        elif branch=='zero_variance':selected=shortlist[:3]
        else:
            best=None
            for t in common:
                st=triples[t]['statistics']
                cmp=1 if best is None else exactmath.compare_ratios(st['mean'],st[key],triples[best]['statistics']['mean'],triples[best]['statistics'][key])
                if cmp>0:best=t;ties=[t]
                elif cmp==0:ties.append(t)
            # Canonical lexicographic enumeration makes the first exact tie win.
            index=ties[0];selected=sorted(triples[index]['indices'],key=lambda j:(-momentum[j],e['symbols'][j]))
        winners[arm]=dict(branch=branch,winning_trio_index=index,tied_trio_indices=ties,
          selected_indices=selected,selected_symbols=[e['symbols'][j] for j in selected])
    e.update(branch=branch,new_eligible_indices=included,excluded_momentum_indices=excluded,shortlist_indices=shortlist,
      trios=triples,common_trio_indices=common,winners=winners)
    return e


def validate_source_cache(e,base,meta,i):
    require(type(e) is dict and set(e)==set(EVENT_KEYS),'Feature event schema')
    for k,v in dict(query_index=i,date=meta['dates'][i],initial_partial=meta['dates'][i]=='2016-08-17',
      calm=bool(base['calm'][i]),core_weight=float(base['cw'][i]),symbols=list(base['names'][i])).items():record_check(e[k],v,'Cached/'+k)
    t=e['query_position'];require(type(t) is int and t>=294,'Cached query position');iso(e['panel_first_session'])
    require(iso(e['next_official_session'])>iso(e['date']),'Cached next witness')
    if not e['initial_partial']:require(iso(e['next_official_session']).to_period('M')!=iso(e['date']).to_period('M'),'Cached month end')
    for k in ('trading_price_start','trading_price_end','risk_price_start','risk_price_end','price_start','price_end'):iso(e[k])
    require(e['risk_price_end']==e['date'] and e['trading_price_end']==e['price_end'] and e['price_start']<=e['trading_price_start']<e['price_end']<e['date'],'Cached clocks')
    dates=e['block_boundary_dates'];require(type(dates) is list and len(dates)==13 and dates==sorted(set(dates)),'Block clock inventory')
    for d in dates:iso(d)
    require(dates[0]==e['price_start'] and dates[-1]==e['price_end'],'Block first/last boundaries')
    eligible=np.flatnonzero(base['E'][i]).tolist();record_check(e['original_eligible_indices'],eligible,'Original E domain')
    record_check(e['required_indices'],[] if e['initial_partial'] else eligible,'Source E domain')
    if e['initial_partial']:
        for k,v in dict(branch='initial_partial',cells=[],u0_indices=[],core_source_max_abs_error=None).items():record_check(e[k],v,'Partial/'+k)
        for k in CANDIDATE_KEYS:record_check(e[k],None,'Partial/'+k)
        return
    require(type(e['cells']) is list and len(e['cells'])==49,'All original49 cells')
    raw=[]
    for j,c in enumerate(e['cells']):
        require(type(c) is dict and set(c)==set(CELL_KEYS),'Cell schema')
        for k,v in dict(column_index=j,symbol=base['names'][i][j],original_eligible=bool(base['E'][i,j]),source_available=bool(base['E'][i,j]),frozen_core=float(base['core'][i,j])).items():record_check(c[k],v,'Cached cell/'+k)
        require(type(c['original_finite']) is bool,'Original finite bool')
        if c['original_finite']:
            require(type(c['original_raw']) is float and math.isfinite(c['original_raw']),'Finite original raw');raw.append(c['original_raw'])
        else:record_check(c['original_raw'],None,'Missing original raw');raw.append(np.nan)
        if c['sigma63'] is not None:require(type(c['sigma63']) is float and math.isfinite(c['sigma63']) and c['sigma63']>=0,'Original sigma')
        if c['original_eligible']:validate_measurement(c['measurement'])
        else:record_check(c['measurement'],None,'Efalse new numeric bypass')
    ranks,U=old.independent_rank(raw);record_check(e['u0_indices'],U,'Original rank domain');errs=[]
    for j,c in enumerate(e['cells']):
        rebuilt=float(base['cw'][i]*ranks[j]);error=abs(rebuilt-float(base['core'][i,j]));require(error<=C_SOURCE_ATOL,'Cached original core source parity')
        for k,v in dict(original_rank=float(ranks[j]),rebuilt_core=rebuilt,core_abs_error=error).items():record_check(c[k],v,'Cached core/'+k)
        errs.append(error)
    record_check(e['core_source_max_abs_error'],max(errs),'Maximum original core error')


def selection_oracle(original,meta,features,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    require(type(features) is list and len(features)>=n,'Feature prefix list')
    inc,incchoices=old.episode(base['P'],base)
    scores={ARMS[0]:inc};pres={ARMS[0]:base['P'].copy()};choices={ARMS[0]:incchoices}
    for arm in ARMS[1:]:scores[arm]=np.zeros((n,49));pres[arm]=np.zeros((n,49));choices[arm]=[]
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=n,symbols_per_event=49,original_episode_calls=1);events=[]
    for i,e in enumerate(features[:n]):
        validate_source_cache(e,base,meta,i)
        if not e['initial_partial']:
            expected=candidate_event(e)
            record_check(e,expected,'Complete direct-basket evidence')
        for arm in ARMS[1:]:
            if e['initial_partial']:
                scores[arm][i]=inc[i];pres[arm][i]=base['P'][i];choices[arm].append(incchoices[i].copy())
            else:
                selected=e['winners'][arm]['selected_indices'];choices[arm].append(selected.copy());S=len(e['shortlist_indices'])
                for j in selected:
                    display=float(S-e['shortlist_indices'].index(j));scores[arm][i,j]=display;pres[arm][i,j]=display
        formal=not e['initial_partial'];cells=e['cells'];required=len(e['required_indices']);trios=e['trios'] or []
        counts['initial_partial_events']+=int(not formal);counts['formal_events']+=int(formal)
        counts['calm_formal_events']+=int(formal and e['calm']);counts['noncalm_formal_events']+=int(formal and not e['calm'])
        counts['original_checked_cells']+=len(cells);counts['original_finite_cells']+=len(e['u0_indices']);counts['original_eligible_cells']+=len(e['original_eligible_indices'])
        counts['source_required_cells']+=required;counts['source_available_cells']+=required;counts['source_price_observations']+=253*required
        counts['momentum_estimates']+=required;counts['block_return_estimates']+=12*required
        counts['new_eligible_cells']+=len(e['new_eligible_indices'] or []);counts['momentum_excluded_cells']+=len(e['excluded_momentum_indices'] or [])
        counts['shortlist_cells']+=len(e['shortlist_indices'] or []);counts['enumerated_trios']+=len(trios);counts['common_supported_trios']+=len(e['common_trio_indices'] or [])
        counts['zero_joint_variance_trios']+=sum(exactmath.rational(t['statistics']['joint_variance'])==0 for t in trios)
        counts['zero_diagonal_variance_trios']+=sum(exactmath.rational(t['statistics']['diagonal_variance'])==0 for t in trios)
        for branch,key in (('cash','cash_events'),('two_names','two_name_events'),('zero_variance','zero_variance_events'),('optimized','optimized_events')):counts[key]+=int(e['branch']==branch)
        counts['candidate_changed_holdings_events']+=int(set(choices[ARMS[1]][i])!=set(incchoices[i]));counts['diagonal_changed_holdings_events']+=int(set(choices[ARMS[2]][i])!=set(incchoices[i]))
        events.append(dict(query_index=i,date=meta['dates'][i],branch=e['branch'],incumbent_choices=incchoices[i],arms={a:choices[a][i] for a in ARMS}))
    return scores,choices,pres,dict(schema='monthly_basket_core251_decisions_v1',counts=counts,events=events)


def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
      symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids]) for i,ids in enumerate(choices[arm])] for arm in ARMS}


def _checked_source_features(panel,meta,calendar,original,review,features,n,*,include_selection):
    expected=independent_features(panel,meta,calendar,original,review,n,include_selection=include_selection)
    require(type(features) is list and len(features)>=n,'Complete supplied features')
    for i,(a,e) in enumerate(zip(features[:n],expected)):
        require(type(a) is dict and set(a)==set(e),'Full feature event keys')
        for k,v in e.items():
            if k!='cells':record_check(a[k],v,'Full feature/'+str(i)+'/'+k)
        require(type(a['cells']) is list and len(a['cells'])==len(e['cells']),'Full cells')
        for j,(ac,ec) in enumerate(zip(a['cells'],e['cells'])):
            require(type(ac) is dict and set(ac)==set(ec),'Full cell keys')
            for k,v in ec.items():
                if k in ('original_raw','sigma63') and v is not None:old.continuous_check(ac[k],v,'Source '+k)
                else:record_check(ac[k],v,'Full cell/'+str(i)+'/'+str(j)+'/'+k)
    return expected


def verify_source_features(panel,meta,calendar,original,review,features,n=None):
    n=len(meta['dates']) if n is None else n
    _checked_source_features(panel,meta,calendar,original,review,features,n,include_selection=False)
    return dict(**{k:True for k in SOURCE_GATE_FLAGS},checked_events=n,C_source_atol=C_SOURCE_ATOL,C_source_rtol=C_SOURCE_RTOL,
      continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,producer_modules_imported=False,
      computes_returns=False,candidate_selection_computed=False,episode_calls=0,new_target_ranks_computed=False,new_basket_selection_computed=False)


def verify_incumbent(original,meta,orders,scores,pre_scores,n=None):
    n=len(meta['dates']) if n is None else n;original_base(original,meta,n)
    return old.verify_incumbent(original,meta,orders,scores,pre_scores,n)


def verify_selection(original,meta,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,wanted=selection_oracle(original,meta,features,n)
    record_check(orders,orders_from_choices(expected,choices,meta),'Ordered direct choices')
    for got,ref,label in ((scores,expected,'Scores'),(pre_scores,pres,'Pres')):
        require(type(got) is dict and set(got)==set(ARMS),'Full '+label+' arms')
        for arm in ARMS:exact_array(got[arm],ref[arm],label+'/'+arm)
    record_check(audit,wanted,'Full independent decision audit')
    return dict(**{k:True for k in SELECTION_FLAGS},counts=wanted['counts'],checked_events=n,ordered_choices_checked=3*n,
      producer_modules_imported=False,computes_returns=False,full_features_supplied=True,full_scores_supplied=True,
      full_pre_scores_supplied=True,full_audit_supplied=True,numerical_tolerance=0.,C_source_atol=C_SOURCE_ATOL,C_source_rtol=C_SOURCE_RTOL)


def verify_prepared(panel,meta,calendar,original,review,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected=_checked_source_features(panel,meta,calendar,original,review,features,n,include_selection=True)
    receipt=verify_selection(original,meta,expected,orders,scores=scores,pre_scores=pre_scores,audit=audit,n=n)
    # The supplied features were fully compared above; decisions use independent rebuilt evidence.
    receipt.update({k:True for k in PREPARE_FLAGS})
    receipt.update(continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True)
    return receipt
def audit_outcomes(px,opening,orders,exports,results,meta):
    """Three arms, three paths and five slices; reuse only independent accounting."""
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
    return dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,
        independent_of_production_model_and_engine=True,terminal_liquidation=False,
        accounting_source_sha256=ACCOUNTING_SHA,paths=reports)
