"""Independent250 reference-gain source checks: unresolved native zero volume is explicit unavailability.

The new producer kernel/math/adapter are never imported. Only SHA-pinned old
independent source/accounting and independently implemented exact math are reused.
"""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_calendar_clock248_v2_independent.py'
PRIOR_SHA='c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c'
PRIOR_TEST=ROOT/'work/test_monthly_calendar_clock248_v2_independent.py'
PRIOR_TEST_SHA='1213e1bad7836293547a76c9d27c60c6392b573c7eeeabd45ff60225c4885b84'
SOURCE_BASE=ROOT/'work/monthly_volume_breadth249_v2_independent.py'
SOURCE_BASE_SHA='2d6ff743765b83e0fbfa0f6a6bce76893ba1f5104b54b19768d063032aed23f6'
MATH=ROOT/'work/monthly_reference_gain250_math_independent.py'
MATH_SHA='4f2a4619711ad8ead687f5d0b26952ba5cb39d94f23e80f256a8e970b854450b'
MATH_TEST=ROOT/'work/test_monthly_reference_gain250_math_independent.py'
MATH_TEST_SHA='da802e82dc866e4e2b20a8e0c6b04f28b20ee29c076a554ef723cda4fbb82a36'
ARMS=('incumbent_local67','lagged_volume_reference_gain','lagged_equal_reference_gain_control')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events',
 'calm_formal_events','noncalm_formal_events','original_checked_cells','original_finite_cells',
 'source_checked_cells','source_available_cells','source_missing_cells','source_zero_volume_cells',
 'source_unavailable_cells','active_events','zero_volume_events','missing_source_events','known_transition_events','source_price_observations',
 'source_volume_observations','source_zero_volume_observations','weighted_reference_estimates','equal_reference_estimates','ranked_cells','candidate_nonzero_correction_cells',
 'control_nonzero_correction_cells','candidate_changed_holdings_events','control_changed_holdings_events',
 'original_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','calm','core_weight','symbols','branch',
 'panel_first_session','query_position','next_official_session','trading_price_start',
 'trading_price_end','risk_price_start','risk_price_end','price_start','price_end',
 'volume_start','volume_end','u0_indices','core_source_max_abs_error','missing_indices',
 'zero_indices','transition_ids','continuity_unknown_ids','cells','volume_rank','equal_rank')
CELL_KEYS=('column_index','symbol','original_finite','original_raw','original_rank',
 'frozen_core','rebuilt_core','core_abs_error','sigma63','source_available','price_gaps',
 'volume_gaps','volume_zeros','measurement')
ADAPTER_EVENT_KEYS=('query_index','calm','weak','active','target_consumed','core_weight',
 'funding_budget','funding_coefficient','alpha','original_episode_budget',
 'requested_episode_budget','effective_episode_budget','target','correction',
 'original_support','boundary_indices','chosen_indices')
DECISION_KEYS=('query_index','date','branch','extension_active_indices','original_remaining_support','arms')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_model_rank_scope_preserved',
 'original_eligibility_unchanged','original_core_checked','original_support_checked',
 'original_effects_unchanged','all_auxiliary_funding_checked','episode_funding_checked',
 'calm_nonweak_extension_budget_checked','whole_event_fallback_checked','exact49_domain_checked',
 'pre_eligibility_ranking_checked','exact_rational_audit_checked','average_percentile_ties_checked',
 'single_original_episode','no_second_episode','no_final_global_rerank',
 'named_initial_partial_unchanged','exact_incumbent_copy_checked',
 'exact_scores_and_choices_checked','full_audit_checked','descriptive_counts_checked','no_predictive_fits_checked',
 'unresolved_zero_volume_fallback_checked','zero_and_missing_evidence_separate')
PREPARE_FLAGS=SELECTION_FLAGS+('independent_source_rebuild','original_full_history_pandas_checked',
 'original_official_row_clock_checked','original_raw_rank_parity_checked',
 'original_weighted_core_tolerance_checked','volume_clocks_checked','historical_reference_window_checked',
 'exact_weighted_equal_references_checked','reference_day_volume_excluded','source_missing_not_imputed','invalid_source_not_hidden_by_missing',
 'visible_listing_bounds_checked','metadata_transition_policy_checked','no_future_source_values')
SOURCE_GATE_FLAGS=('passed','full_features_supplied','original_core_checked','volume_clocks_checked',
 'original_raw_rank_parity_checked','exact49_domain_checked','whole_event_fallback_checked',
 'visible_listing_bounds_checked','metadata_transition_policy_checked','exact_discrete',
 'unresolved_zero_volume_fallback_checked','zero_and_missing_evidence_separate')
INCUMBENT_GATE_FLAGS=('passed','full_scores_supplied','full_pre_scores_supplied',
 'independent_original_pre_checked','exact_incumbent_copy_checked','original_eligibility_unchanged','single_original_episode')
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
C_SOURCE_ATOL=1e-12
C_SOURCE_RTOL=0.

def load_pinned(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen independent dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

old=load_pinned(PRIOR,PRIOR_SHA,'reference250_old_independent')
exactmath=load_pinned(MATH,MATH_SHA,'reference250_exact_independent')
require,exact_array=old.require,old.exact_array
record_check=exactmath._exact
iso=old.iso
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA
def accounting():return old.accounting()
def frozen_source_oracle():return old.frozen_source_oracle()
def dependency_files():
    for path,digest in ((PRIOR_TEST,PRIOR_TEST_SHA),(MATH_TEST,MATH_TEST_SHA),(SOURCE_BASE,SOURCE_BASE_SHA)):
        require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Frozen synthetic dependency changed')
    return tuple(sorted(set(old.dependency_files())|{PRIOR,PRIOR_TEST,MATH,MATH_TEST,SOURCE_BASE,Path(__file__).resolve()}))

def original_base(original,meta,n):
    base=old.original_base(original,meta,n)
    require(base['width']==49,'Exactly49 original members including ETFs')
    return base

def source_axes(panel,calendar,n):
    require(type(panel) is dict and set(panel)=={'adj','volume'},'Price/volume panel')
    adj,sessions=old.source_axes({'adj':panel['adj']},calendar,n)
    volume,_=old.source_axes({'adj':panel['volume']},calendar,n)
    return adj,volume,sessions

def clock_row(i,base,meta,sessions):
    day=base['days'][i];require(day in sessions,'Query official session')
    t=sessions.index(day);require(t>=294 and t+1<len(sessions),'Complete original and volume clock')
    partial=meta['dates'][i]=='2016-08-17'
    if not partial:require(sessions[t+1].to_period('M')!=day.to_period('M'),'Official month-end witness')
    dates=[str(x.date()) for x in sessions]
    return dict(query_index=i,date=meta['dates'][i],initial_partial=partial,
      symbols=list(base['names'][i]),query_position=t,next_official_session=dates[t+1],
      price_start=dates[t-294],price_end=dates[t-42],volume_start=dates[t-294],volume_end=dates[t-43],
      numeric_evaluation_required=not partial,price_count=253,volume_count=252,historical_count=252,
      historical_price_start=dates[t-294],historical_price_end=dates[t-43],
      reference_date=dates[t-42],reference_volume_consumed=False)


def _q_index(rows,n,label):
    require(type(rows) is list,label+' list');out={}
    for row in rows:
        require(type(row) is dict and type(row.get('query_index')) is int and row['query_index']>=0,label+' query index')
        q=row['query_index']
        if q>=n:continue
        require(q not in out,label+' duplicate query');out[q]=row
    return out

def review_index(review,n):
    require(type(review) is dict and review.get('schema')=='monthly_reference_gain250_metadata_source_review_v1','Source review schema')
    origin=review.get('panel_first_session');iso(origin)
    require(type(review.get('listing_bounds')) is list,'Listing bounds')
    bounds={}
    for b in review['listing_bounds']:
        require(type(b) is dict and set(b)=={'symbol','issuer_key','not_before','available_from','kind','evidence_id'},'Listing bound schema')
        require(all(type(v) is str and v and v.strip()==v for v in b.values()),'Listing bound strings')
        iso(b['not_before']);iso(b['available_from']);require(b['symbol'] not in bounds,'Unique bound');bounds[b['symbol']]=b
    require(type(review.get('identity_cells')) is list,'Identity cells')
    identities={}
    for row in review['identity_cells']:
        require(type(row) is dict and type(row.get('query_index')) is int and row['query_index']>=0,'Identity query')
        if row['query_index']>=n:continue
        j=row.get('column_index');require(type(j) is int and 0<=j<49,'Identity column')
        key=(row['query_index'],j);require(key not in identities,'Duplicate identity');identities[key]=row
    clocks=_q_index(review.get('clock_events'),n,'Clock')
    require(set(clocks)==set(range(n)),'Complete consumed clock rows')
    require(type(review.get('transition_records')) is list and type(review.get('undated_loader_rules')) is list,'Transition ledger lists')
    records={}
    for row in review['transition_records']:
        require(type(row) is dict and type(row.get('id')) is str and row['id'] and row['id'] not in records,'Transition id')
        records[row['id']]=row
    auto=review.get('automatic_known_unresolved_transition_fallback_ids')
    require(type(auto) is list and all(type(v) is str for v in auto) and len(set(auto))==len(auto),'Automatic transition ids')
    for key in auto:
        require(key in records,'Missing automatic transition metadata');r=records[key]
        require(r.get('known_unresolved_fallback_eligible') is True and r.get('unresolved_unit_transition') is True,'Automatic unresolved declaration')
        iso(r.get('effective_date'))
        require(type(r.get('symbols')) is list and all(type(v) is str and v for v in r['symbols']),'Transition symbols')
        require(type(r.get('affected_fields')) is list and any(v in ('adj','volume') for v in r['affected_fields']),'Transition source fields')
        require(r.get('retrospective_integrity_basis') is True or r.get('available_from') is not None,'Transition evidence visibility')
        if r.get('available_from') is not None:iso(r['available_from'])
    unknown={}
    for row in review['undated_loader_rules']:
        require(type(row) is dict and type(row.get('id')) is str and row['id'] and row['id'] not in unknown,'Unknown continuity id')
        require(type(row.get('symbols')) is list and all(type(v) is str for v in row['symbols']),'Unknown continuity symbols')
        unknown[row['id']]=row
    return origin,bounds,identities,clocks,records,auto,unknown

def metadata_checks(clock,sessions,bounds,identities,clocks):
    i=clock['query_index'];t=clock['query_position'];got=clocks[i]
    for k,v in clock.items():record_check(got.get(k),v,'Clock/'+k)
    wanted={}
    for j,s in enumerate(clock['symbols']):
        if s not in bounds:continue
        b=bounds[s];r=identities.get((i,j));require(r is not None,'Missing listing occurrence')
        fields=dict(query_index=i,column_index=j,date=clock['date'],symbol=s,issuer_key=b['issuer_key'],
          not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=clock['date'],
          price_start=clock['price_start'],price_end=clock['price_end'],volume_start=clock['volume_start'],volume_end=clock['volume_end'],
          price_window_dates_before_bound=sum(str(d.date())<b['not_before'] for d in sessions[t-294:t-41]),
          volume_window_dates_before_bound=sum(str(d.date())<b['not_before'] for d in sessions[t-294:t-42]))
        for k,v in fields.items():record_check(r.get(k),v,'Listing/'+k)
        wanted[j]=b
    require({j for q,j in identities if q==i}==set(wanted),'No stray listing occurrence')
    return wanted

def transition_ids(clock,records,auto,unknown):
    symbols=set(clock['symbols']);matched=[]
    for key in auto:
        r=records[key]
        visible=r.get('retrospective_integrity_basis') is True or r['available_from']<=clock['date']
        if not visible or not symbols.intersection(r['symbols']):continue
        d=r['effective_date'];fields=r['affected_fields']
        if ('adj' in fields and clock['price_start']<d<=clock['price_end']) or ('volume' in fields and clock['volume_start']<d<=clock['volume_end']):
            matched.append(key)
    return sorted(matched),sorted(k for k,r in unknown.items() if symbols.intersection(r['symbols']))

def extract(frame,symbol,days,bound,query,*,is_volume=False):
    """Keep native zeros distinct from missing; never feed them to positive math."""
    values=[];gaps=[];zeros=[]
    for offset,day in enumerate(days):
        label=str(day.date());native_zero=False
        if symbol not in frame.columns:reason='missing_symbol';value=None
        elif day not in frame.index:reason='missing_row';value=None
        else:
            original=frame.at[day,symbol];value=original;reason=None
            if value is None or value is pd.NA:reason='missing_value';value=None
            else:
                require(isinstance(original,Real) and not isinstance(original,(bool,np.bool_)),'Required real nonboolean observation')
                native_zero=bool(original==0)
                try:value=float(original)
                except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Required binary64 conversion') from exc
                if math.isnan(value):reason='missing_value';value=None
                else:
                    require(math.isfinite(value),'Required finite observation')
                    if value==0.:
                        require(native_zero,'Nonzero source underflow is not native zero')
                        require(is_volume,'Required positive finite price')
                        zeros.append(dict(offset=offset,date=label,value=value,reason='unresolved_zero_volume'))
                    else:
                        require(value>0.,'Required positive finite observation')
                        if bound and bound['available_from']<=query:require(label>=bound['not_before'],'Source predates visible listing bound')
        values.append(value)
        if reason:gaps.append(dict(offset=offset,date=label,reason=reason))
    return values,gaps,zeros

def independent_features(panel,meta,calendar,original,review,n=None,*,include_ranks=False):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    adj,volume,sessions=source_axes(panel,calendar,n)
    origin,bounds,identities,clocks,records,auto,unknown=review_index(review,n)
    features=[]
    for i in range(n):
        clock=clock_row(i,base,meta,sessions);t=clock['query_position'];day=base['days'][i];names=list(base['names'][i])
        bound=metadata_checks(clock,sessions,bounds,identities,clocks)
        transition,unk=transition_ids(clock,records,auto,unknown)
        e=dict(query_index=i,date=meta['dates'][i],initial_partial=clock['initial_partial'],calm=bool(base['calm'][i]),
          core_weight=float(base['cw'][i]),symbols=names,branch='initial_partial',panel_first_session=origin,
          query_position=t,next_official_session=clock['next_official_session'],
          trading_price_start=str(sessions[t-(252 if base['calm'][i] else 294)].date()),
          trading_price_end=str(sessions[t-42].date()),risk_price_start=str(sessions[t-63].date()),risk_price_end=meta['dates'][i],
          price_start=clock['price_start'],price_end=clock['price_end'],volume_start=clock['volume_start'],volume_end=clock['volume_end'],
          u0_indices=[],core_source_max_abs_error=None,missing_indices=[],zero_indices=[],transition_ids=transition,
          continuity_unknown_ids=unk,cells=[],volume_rank=None,equal_rank=None)
        if clock['initial_partial']:
            e['transition_ids']=[];e['continuity_unknown_ids']=[];features.append(e);continue
        require(all(s in adj.columns for s in names),'Original required price columns')
        prefix=adj.loc[:day]
        require(len(prefix)>=295 and prefix.index[-295:].equals(pd.DatetimeIndex(sessions[t-294:t+1])),'Original last295 official row parity')
        pairs,sigmas=old.original_values(adj,day,names,origin)
        raw=np.array([pair[0 if base['calm'][i] else 1] for pair in pairs]);raw[~np.isfinite(raw)]=np.nan
        ranks,U=old.independent_rank(raw);e['u0_indices']=U
        for j,s in enumerate(names):
            rebuilt=float(base['cw'][i]*ranks[j]);frozen=float(base['core'][i,j]);err=abs(rebuilt-frozen)
            require(err<=C_SOURCE_ATOL,'Original core source parity')
            if j in U and j in bound and bound[j]['available_from']<=e['date']:
                require(e['trading_price_start']>=bound[j]['not_before'],'Original finite core predates visible bound')
            prices,pg,_=extract(adj,s,sessions[t-294:t-41],bound.get(j),e['date'])
            volumes,vg,vz=extract(volume,s,sessions[t-294:t-42],bound.get(j),e['date'],is_volume=True)
            measurement=None if vz else exactmath.measure_oracle(prices,volumes)
            available=not vz and measurement['available']
            if pg or vg:e['missing_indices'].append(j)
            if vz:e['zero_indices'].append(j)
            e['cells'].append(dict(column_index=j,symbol=s,original_finite=j in U,original_raw=float(raw[j]) if j in U else None,
              original_rank=float(ranks[j]),frozen_core=frozen,rebuilt_core=rebuilt,core_abs_error=err,
              sigma63=float(sigmas[j]) if math.isfinite(sigmas[j]) else None,source_available=available,
              price_gaps=pg,volume_gaps=vg,volume_zeros=vz,measurement=measurement))
        e['core_source_max_abs_error']=max(c['core_abs_error'] for c in e['cells'])
        e['branch']='zero_volume' if e['zero_indices'] else 'missing_source' if e['missing_indices'] else 'known_transition' if transition else 'available'
        if e['branch']=='available' and include_ranks:
            e['volume_rank']=exactmath.rank49_oracle([c['measurement']['raw_volume_gain'] for c in e['cells']])
            e['equal_rank']=exactmath.rank49_oracle([c['measurement']['raw_equal_gain'] for c in e['cells']])
        features.append(e)
    return features

def _indices(values,maximum,label):
    require(type(values) is list and all(type(x) is int and 0<=x<maximum for x in values) and values==sorted(set(values)),label)

def validate_measurement(measurement):
    """Recheck exact reference algebra without source reads or raw float casts."""
    m=measurement
    require(type(m) is dict and set(m)==set(exactmath.MEASURE_KEYS),'Full measure schema')
    for k,v in dict(schema=exactmath.SCHEMA,price_count=253,volume_count=252,historical_count=252).items():
        record_check(m[k],v,'Measure/'+k)
    _indices(m['missing_price_indices'],253,'Missing price indices')
    _indices(m['missing_volume_indices'],252,'Missing volume indices')
    missing=bool(m['missing_price_indices'] or m['missing_volume_indices'])
    record_check(m['available'],not missing,'Measure availability')
    record_check(m['reason'],'missing_source' if missing else 'available','Measure reason')
    keys=exactmath.MEASURE_KEYS[8:]
    if missing:
        for k in keys:record_check(m[k],None,'Unavailable '+k)
        return
    v={k:exactmath.rational(m[k]) for k in keys}
    for k in ('reference_price','historical_price_sum','volume_sum','weighted_price_sum','weighted_reference','equal_reference'):
        require(v[k]>0,'Positive exact reference aggregate')
    weighted=v['weighted_price_sum']/v['volume_sum'];equal=v['historical_price_sum']/252
    # Identity from audited sufficient statistics; source reconstruction above
    # independently uses per-observation gains to establish the raw values.
    for k,x in dict(weighted_reference=weighted,equal_reference=equal,
      raw_volume_gain=1-weighted/v['reference_price'],raw_equal_gain=1-equal/v['reference_price']).items():
        record_check(m[k],exactmath.pair(x),'Measure/'+k)


def validate_cached_features(features,base,meta,n):
    require(type(features) is list and len(features)>=n,'Full feature prefix')
    targets={arm:np.full((n,49),np.nan) for arm in ARMS}
    active=np.zeros(n,bool)
    for i,e in enumerate(features[:n]):
        require(type(e) is dict and set(e)==set(EVENT_KEYS),'Feature event schema')
        for k,v in dict(query_index=i,date=meta['dates'][i],initial_partial=meta['dates'][i]=='2016-08-17',
          calm=bool(base['calm'][i]),core_weight=float(base['cw'][i]),symbols=list(base['names'][i])).items():
            record_check(e[k],v,'Cached '+k)
        iso(e['panel_first_session']);require(type(e['query_position']) is int and e['query_position']>=294,'Cached official position')
        require(iso(e['next_official_session'])>iso(e['date']),'Cached next witness')
        for key in ('trading_price_start','trading_price_end','risk_price_start','risk_price_end','price_start','price_end','volume_start','volume_end'):iso(e[key])
        require(e['risk_price_end']==e['date'] and e['price_end']==e['trading_price_end'],'Clock shared ends')
        require(e['volume_start']==e['price_start']<e['volume_end']<e['price_end']<e['date'],'Cached clock ordering')
        require(e['trading_price_start']>=e['price_start'],'Original trading start bound')
        for key in ('transition_ids','continuity_unknown_ids'):
            vals=e[key];require(type(vals) is list and all(type(x) is str and x for x in vals) and vals==sorted(set(vals)),'Canonical ledger ids')
        if e['initial_partial']:
            require(e['branch']=='initial_partial','Initial branch')
            for key in ('u0_indices','missing_indices','zero_indices','transition_ids','continuity_unknown_ids','cells'):record_check(e[key],[],'Partial '+key)
            for key in ('core_source_max_abs_error','volume_rank','equal_rank'):record_check(e[key],None,'Partial '+key)
            continue
        require(type(e['cells']) is list and len(e['cells'])==49,'Full49 cells before eligibility')
        raw=[];missing=[];zero_cells=[];errors=[]
        for j,c in enumerate(e['cells']):
            require(type(c) is dict and set(c)==set(CELL_KEYS),'Cell schema')
            record_check(c['column_index'],j,'Cell index');record_check(c['symbol'],base['names'][i][j],'Cell symbol')
            require(type(c['original_finite']) is bool and type(c['source_available']) is bool,'Cell booleans')
            record_check(c['frozen_core'],float(base['core'][i,j]),'Frozen C')
            if c['original_finite']:
                require(type(c['original_raw']) is float and math.isfinite(c['original_raw']),'Original finite raw')
                raw.append(c['original_raw'])
            else:record_check(c['original_raw'],None,'Original missing raw');raw.append(np.nan)
            if c['sigma63'] is not None:require(type(c['sigma63']) is float and math.isfinite(c['sigma63']) and c['sigma63']>=0,'Original finite sigma')
            zeros=c['volume_zeros'];require(type(zeros) is list,'Explicit zero list');zero_offsets=[]
            for z in zeros:
                require(type(z) is dict and tuple(z)==('offset','date','value','reason'),'Zero evidence schema')
                require(type(z['offset']) is int and 0<=z['offset']<252,'Zero offset')
                require(type(z['value']) is float and z['value']==0.,'Native binary64 zero evidence')
                record_check(z['reason'],'unresolved_zero_volume','Zero reason')
                iso(z['date']);require(e['volume_start']<=z['date']<=e['volume_end'],'Zero source clock')
                zero_offsets.append(z['offset'])
            _indices(zero_offsets,252,'Unique ordered zero offsets')
            m=c['measurement']
            if zeros:
                record_check(m,None,'Zero cell has no fabricated measurement');record_check(c['source_available'],False,'Zero unavailable')
                zero_cells.append(j)
            else:
                validate_measurement(m);record_check(c['source_available'],m['available'],'Source measure availability')
            any_gap=False
            for key,size,mkey in (('price_gaps',253,'missing_price_indices'),('volume_gaps',252,'missing_volume_indices')):
                gaps=c[key];require(type(gaps) is list,'Gaps list');offsets=[]
                for gap in gaps:
                    require(type(gap) is dict and set(gap)=={'offset','date','reason'},'Gap schema')
                    require(type(gap['offset']) is int and 0<=gap['offset']<size,'Gap index')
                    require(gap['reason'] in ('missing_symbol','missing_row','missing_value'),'Gap reason')
                    iso(gap['date']);start=e['price_start'] if key=='price_gaps' else e['volume_start']
                    end=e['price_end'] if key=='price_gaps' else e['volume_end']
                    require(start<=gap['date']<=end,'Gap within required clock')
                    offsets.append(gap['offset'])
                _indices(offsets,size,'Unique ordered missing offsets');any_gap=any_gap or bool(offsets)
                if key=='volume_gaps':require(not set(offsets).intersection(zero_offsets),'Zero is not missing placeholder')
                if not zeros:record_check(offsets,m[mkey],'Gap offsets vs measure')
            if any_gap:missing.append(j)

        ranks,U=old.independent_rank(raw)
        record_check(e['u0_indices'],U,'Original U0')
        for j,c in enumerate(e['cells']):
            rebuilt=float(base['cw'][i]*ranks[j]);err=abs(rebuilt-float(base['core'][i,j]))
            require(err<=C_SOURCE_ATOL,'Cached original C source')
            for k,v in dict(original_rank=float(ranks[j]),rebuilt_core=rebuilt,core_abs_error=err).items():record_check(c[k],v,'Cached '+k)
            errors.append(err)
        record_check(e['core_source_max_abs_error'],max(errors),'Max core error');record_check(e['missing_indices'],missing,'Whole missing cells')
        record_check(e['zero_indices'],zero_cells,'Whole zero cells')
        branch='zero_volume' if zero_cells else 'missing_source' if missing else 'known_transition' if e['transition_ids'] else 'available'
        record_check(e['branch'],branch,'Common event branch')
        if branch=='available':
            active[i]=True
            for arm,key,rawkey in ((ARMS[1],'volume_rank','raw_volume_gain'),(ARMS[2],'equal_rank','raw_equal_gain')):
                wanted=exactmath.rank49_oracle([c['measurement'][rawkey] for c in e['cells']])
                record_check(e[key],wanted,'Exact49 '+key);targets[arm][i]=wanted['targets']
        else:
            record_check(e['volume_rank'],None,'Fallback rank');record_check(e['equal_rank'],None,'Fallback rank')
    return targets,active

def funded_episode(base,target,active):
    """Independent scalar episode, independent funding arithmetic; no new adapter."""
    n=len(base['P']);pre=base['P'].copy();correction=np.zeros((n,49))
    coefficient=np.zeros(n);alpha=np.ones(n);requested=np.full(n,.05)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        for i in range(n):
            if active[i]:
                coefficient[i]=.02/(1.-base['cw'][i]);alpha[i]=1.-coefficient[i]
                require(np.isfinite(target[i]).all() and (np.abs(target[i])<=.5).all(),'Bounded active targets')
                for j in range(49):
                    correction[i,j]=.02*target[i,j]-coefficient[i]*(base['P'][i,j]-base['core'][i,j])
                    if correction[i,j]!=0.:pre[i,j]+=correction[i,j]
                requested[i]=alpha[i]*base['budget'][i]
        cap=np.maximum(0.,.10-(.025*base['calm']+.05*base['weak']))
        effective=np.minimum(requested,cap)
        runbase=dict(base,budget=effective);scores,choices=old.episode(pre,runbase)
    require(np.isfinite(pre).all() and np.isfinite(scores).all(),'Finite funded scores')
    events=[]
    for i,names in enumerate(base['names']):
        rank=sorted(np.flatnonzero(base['E'][i]).tolist(),key=lambda j:(-pre[i,j],names[j]))
        hook=[] if len(rank)<3 else [j for j in rank[2:5] if pre[i,rank[2]]-pre[i,j]<=.02]
        events.append(dict(query_index=i,calm=bool(base['calm'][i]),weak=bool(base['weak'][i]),
          active=bool(active[i]),target_consumed=bool(active[i]),core_weight=float(base['cw'][i]),
          funding_budget=.02 if active[i] else 0.,funding_coefficient=float(coefficient[i]),alpha=float(alpha[i]),
          original_episode_budget=float(base['budget'][i]),requested_episode_budget=float(requested[i]),
          effective_episode_budget=float(effective[i]),target=target[i].tolist() if active[i] else None,
          correction=correction[i].tolist(),original_support=base['support'][i].tolist(),
          boundary_indices=hook,chosen_indices=choices[i]))
    return scores,choices,pre,events

def selection_oracle(original,meta,features,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    targets,active=validate_cached_features(features,base,meta,n)
    scores={};choices={};pres={};arm_events={}
    for arm in ARMS:
        on=np.zeros(n,bool) if arm==ARMS[0] else active
        scores[arm],choices[arm],pres[arm],arm_events[arm]=funded_episode(base,targets[arm],on)
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=n,symbols_per_event=49,original_episode_calls=3)
    events=[]
    for i,e in enumerate(features[:n]):
        partial=e['initial_partial'];formal=not partial;cells=e['cells']
        counts['initial_partial_events']+=int(partial);counts['formal_events']+=int(formal)
        counts['calm_formal_events']+=int(formal and e['calm']);counts['noncalm_formal_events']+=int(formal and not e['calm'])
        counts['original_checked_cells']+=len(cells);counts['original_finite_cells']+=len(e['u0_indices'])
        counts['source_checked_cells']+=len(cells);good=sum(c['source_available'] for c in cells)
        counts['source_available_cells']+=good;counts['source_missing_cells']+=sum(bool(c['price_gaps'] or c['volume_gaps']) for c in cells)
        counts['source_zero_volume_cells']+=sum(bool(c['volume_zeros']) for c in cells)
        counts['source_unavailable_cells']+=len(cells)-good
        counts['source_zero_volume_observations']+=sum(len(c['volume_zeros']) for c in cells)
        counts['source_price_observations']+=253*len(cells);counts['source_volume_observations']+=252*len(cells)
        counts['weighted_reference_estimates']+=good;counts['equal_reference_estimates']+=good;counts['ranked_cells']+=98*int(active[i])
        for branch,key in (('available','active_events'),('zero_volume','zero_volume_events'),('missing_source','missing_source_events'),('known_transition','known_transition_events')):
            counts[key]+=int(e['branch']==branch)
        counts['candidate_nonzero_correction_cells']+=sum(v!=0. for v in arm_events[ARMS[1]][i]['correction'])
        counts['control_nonzero_correction_cells']+=sum(v!=0. for v in arm_events[ARMS[2]][i]['correction'])
        counts['candidate_changed_holdings_events']+=int(set(choices[ARMS[1]][i])!=set(choices[ARMS[0]][i]))
        counts['control_changed_holdings_events']+=int(set(choices[ARMS[2]][i])!=set(choices[ARMS[0]][i]))
        remaining=(1.-base['cw'][i])-(.025*base['calm'][i]+.05*base['weak'][i]+.01*base['A'][i])-base['budget'][i]
        require(np.min(remaining)>=.38-1e-12,'Original auxiliary budget including extension')
        events.append(dict(query_index=i,date=meta['dates'][i],branch=e['branch'],
          extension_active_indices=np.flatnonzero(base['A'][i]).tolist(),
          original_remaining_support=remaining.tolist(),arms={arm:arm_events[arm][i] for arm in ARMS}))
    return scores,choices,pres,dict(schema='monthly_reference_gain250_decisions_v1',counts=counts,events=events)

def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
      symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
      for i,ids in enumerate(choices[arm])] for arm in ARMS}

def verify_selection(original,meta,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,wanted=selection_oracle(original,meta,features,n)
    record_check(orders,orders_from_choices(expected,choices,meta),'Ordered choices')
    for got,ref,label in ((scores,expected,'Final scores'),(pre_scores,pres,'Pre scores')):
        require(type(got) is dict and set(got)==set(ARMS),label+' arms')
        for arm in ARMS:exact_array(got[arm],ref[arm],label+'/'+arm)
    record_check(audit,wanted,'Full independent decision audit')
    return dict(**{k:True for k in SELECTION_FLAGS},counts=wanted['counts'],checked_events=n,
      ordered_choices_checked=3*n,producer_modules_imported=False,computes_returns=False,
      full_features_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,
      full_audit_supplied=True,numerical_tolerance=0.,C_source_atol=C_SOURCE_ATOL,C_source_rtol=C_SOURCE_RTOL)

def _checked_source_features(panel,meta,calendar,original,review,features,n,*,include_ranks=True):
    expected=independent_features(panel,meta,calendar,original,review,n,include_ranks=include_ranks)
    require(type(features) is list and len(features)>=n,'Complete supplied features')
    for i,(got,wanted) in enumerate(zip(features[:n],expected)):
        require(type(got) is dict and set(got)==set(wanted),'Full feature schema')
        for key,value in wanted.items():
            if key!='cells':record_check(got[key],value,'Feature/'+str(i)+'/'+key)
        require(type(got['cells']) is list and len(got['cells'])==len(wanted['cells']),'Full cell axis')
        for j,(a,e) in enumerate(zip(got['cells'],wanted['cells'])):
            require(type(a) is dict and set(a)==set(e),'Full cell schema')
            for key,value in e.items():
                if key in ('original_raw','sigma63') and value is not None:old.continuous_check(a[key],value,'Source continuous/'+key)
                else:record_check(a[key],value,'Cell/'+str(i)+'/'+str(j)+'/'+key)
    return expected

def verify_source_features(panel,meta,calendar,original,review,features,n=None):
    """Pre-candidate source/core gate; no episode or candidate selection."""
    n=len(meta['dates']) if n is None else n
    _checked_source_features(panel,meta,calendar,original,review,features,n,include_ranks=False)
    return dict(**{k:True for k in SOURCE_GATE_FLAGS},checked_events=n,
      C_source_atol=C_SOURCE_ATOL,C_source_rtol=C_SOURCE_RTOL,
      continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,
      producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False,episode_calls=0,
      new_target_ranks_computed=False)

def verify_incumbent(original,meta,orders,scores,pre_scores,n=None):
    n=len(meta['dates']) if n is None else n
    original_base(original,meta,n)  # Enforce this study's exact49 axis as well.
    return old.verify_incumbent(original,meta,orders,scores,pre_scores,n)

def verify_prepared(panel,meta,calendar,original,review,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected=_checked_source_features(panel,meta,calendar,original,review,features,n)
    receipt=verify_selection(original,meta,features,orders,scores=scores,pre_scores=pre_scores,audit=audit,n=n)
    es,ec,ep,ea=selection_oracle(original,meta,expected,n)
    record_check(orders,orders_from_choices(es,ec,meta),'Raw-source choices')
    for arm in ARMS:
        exact_array(scores[arm],es[arm],'Raw-source final');exact_array(pre_scores[arm],ep[arm],'Raw-source pre')
    record_check(audit,ea,'Raw-source audit')
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
