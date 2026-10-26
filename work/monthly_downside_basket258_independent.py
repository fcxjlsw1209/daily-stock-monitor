"""Independent258 canonical downside moments, full-source and direct-basket proof.

Only pinned older independent source/original/accounting helpers are imported.
No new producer, host, strategy execution or market-data I/O is used here.
"""
import copy
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_portfolio_path247_independent.py'
PRIOR_SHA='23211381f2b6ca6f0f85aec685439eedc2897aac022f88592d35c2a96c6c7920'
PRIOR_TEST=ROOT/'work/test_monthly_portfolio_path247_independent.py'
PRIOR_TEST_SHA='c121570d9f47b96611245e191aaaf0fd0e1f152cc1f2f1e6b987de80ec2fa47e'

def load_pinned(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen independent dependency changed '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

old=load_pinned(PRIOR,PRIOR_SHA,'downside258_old_independent')
require,exact_array,record_check=old.require,old.exact_array,old.record_check
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA
original_base,base_event=old.original_base,old.base_event
EVENT_KEYS,TRIO_KEYS,SOURCE_KEYS=old.EVENT_KEYS,old.TRIO_KEYS,old.SOURCE_KEYS
ARMS=('incumbent_local67','basket_downside_moment','basket_symmetric_moment')
STAT_KEYS=('schema','price_count','return_count','normalized_prices','wealth','prior_totals','contributions',
 'portfolio_returns','direct_returns','downside_square_sum','symmetric_square_sum','upside_square_sum',
 'downside_moment','symmetric_moment','upside_moment','negative_returns','zero_returns','positive_returns',
 'zero_downside','zero_symmetric')
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','active_events','shortlist_cells',
 'source_available_cells','source_price_observations','enumerated_trios','feasible_trios','trio_statistics',
 'formation_return_observations','zero_downside_trios','zero_symmetric_trios','missing_source_events','available_events',
 'downside_changed_holdings','symmetric_changed_holdings','candidate_control_changed_holdings','original_episode_calls',
 'new_model_fits','predictive_model_fits')
FEATURE_FLAGS=('passed','independent_original_pre_checked','original_support_checked','original_budget_checked',
 'original_top5_checked','same_feasible_domain_checked','incumbent_feasible_checked','raw_core_regret_cap_checked',
 'final_score_regret_cap_checked','official_253_price_window_checked','all_shortlist_sources_checked',
 'no_rank_six_substitution','visible_listing_bounds_checked','fixed_adjusted_units_checked',
 'canonical_moments_exact','canonical_signs_exact','direct_nav_identity_checked','zero_moments_valid',
 'common_fallback_checked','no_future_source_values','no_predictive_fits_checked')
INCUMBENT_FLAGS=('passed','independent_original_pre_checked','exact_incumbent_copy_checked',
 'original_eligibility_unchanged','single_original_episode')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_support_checked','original_budget_checked',
 'original_top5_checked','same_feasible_domain_checked','incumbent_feasible_checked','raw_core_regret_cap_checked',
 'final_score_regret_cap_checked','canonical_moments_exact','exact_ties_checked','common_fallback_checked',
 'named_initial_partial_unchanged','original_full_scores_and_pre_preserved','exact_scores_and_choices_checked',
 'full_audit_checked','descriptive_counts_checked','no_predictive_fits_checked','no_second_episode')
PREPARE_FLAGS=tuple(dict.fromkeys(FEATURE_FLAGS+SELECTION_FLAGS+('independent_source_rebuild',)))
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
NAV_ATOL=NAV_RTOL=2e-12

def accounting():return old.accounting()
def dependency_files():
    require(hashlib.sha256(PRIOR_TEST.read_bytes()).hexdigest()==PRIOR_TEST_SHA,'Frozen synthetic fixture changed')
    return tuple(sorted(set(old.dependency_files())|{PRIOR,PRIOR_TEST,Path(__file__).resolve()}))

def finite(v):
    if not math.isfinite(v):raise ArithmeticError('Nonfinite canonical arithmetic')
    return float(v)
def total(xs):
    try:return finite(math.fsum(xs))
    except (OverflowError,ValueError) as e:raise ArithmeticError('Invalid canonical sum') from e
def quotient(a,b):
    try:v=finite(a/b)
    except (ZeroDivisionError,OverflowError) as e:raise ArithmeticError('Invalid canonical division') from e
    if a!=0. and v==0.:raise ArithmeticError('Nonzero division underflow')
    return v
def multiply(a,b):
    v=finite(a*b)
    if a!=0. and b!=0. and v==0.:raise ArithmeticError('Nonzero multiplication underflow')
    return v
def close(a,b,atol,rtol):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b)),'Continuous algebra diagnostic')

def measure_oracle(paths):
    require(isinstance(paths,(list,tuple,np.ndarray)) and len(paths)==3,'Three paths')
    require(all(isinstance(p,(list,tuple,np.ndarray)) and len(p)==253 for p in paths),'253 prices per path')
    rows=[]
    for p in paths:
        row=[]
        for raw in p:
            require(isinstance(raw,Real) and not isinstance(raw,(bool,np.bool_)),'Real nonboolean price')
            try:v=float(raw)
            except (ValueError,TypeError,OverflowError) as e:raise ValueError('Invalid price conversion') from e
            require(math.isfinite(v) and v>0.,'Positive finite price');row.append(v)
        rows.append(row)
    x=[[quotient(v,p[0]) for v in p] for p in rows]
    return normalized_statistics(x)

def normalized_statistics(x):
    """Rewritten scalar reduction order, also used to verify cached algebra."""
    require(type(x) is list and len(x)==3 and all(type(row) is list and len(row)==253 for row in x),'Normalized paths')
    require(all(type(v) is float and math.isfinite(v) and v>0. for row in x for v in row),'Positive normalized prices')
    require(all(row[0]==1. for row in x),'Unit initial prices')
    denominators=[total(row[k] for row in x) for k in range(253)]
    wealth=[quotient(d,3.) for d in denominators]
    contributions=[[quotient(finite(row[k]-row[k-1]),denominators[k-1]) for k in range(1,253)] for row in x]
    q=[total(contributions[i][k] for i in range(3)) for k in range(252)]
    direct=[finite(quotient(wealth[k+1],wealth[k])-1.) for k in range(252)]
    for a,b in zip(q,direct):close(a,b,NAV_ATOL,NAV_RTOL)
    minus=total(multiply(min(r,0.),min(r,0.)) for r in q)
    symmetric=total(multiply(r,r) for r in q)
    plus=total(multiply(max(r,0.),max(r,0.)) for r in q)
    lm,ls,lp=[quotient(s,252.) for s in (minus,symmetric,plus)]
    close(ls,lm+lp,CONTINUOUS_ATOL,CONTINUOUS_RTOL)
    require((lm==0.)==all(r>=0. for r in q),'Operational zero downside')
    require((ls==0.)==all(r==0. for r in q),'Operational zero symmetric')
    return dict(schema='monthly_downside_basket258_moments_v1',price_count=253,return_count=252,
      normalized_prices=copy.deepcopy(x),wealth=wealth,prior_totals=denominators[:-1],contributions=contributions,
      portfolio_returns=q,direct_returns=direct,downside_square_sum=minus,symmetric_square_sum=symmetric,
      upside_square_sum=plus,downside_moment=lm,symmetric_moment=ls,upside_moment=lp,
      negative_returns=sum(r<0. for r in q),zero_returns=sum(r==0. for r in q),positive_returns=sum(r>0. for r in q),
      zero_downside=lm==0.,zero_symmetric=ls==0.)

def statistics_check(actual,expected=None):
    require(type(actual) is dict and set(actual)==set(STAT_KEYS),'Complete258 statistics')
    expected=normalized_statistics(actual['normalized_prices']) if expected is None else expected
    for k in STAT_KEYS:
        if k=='direct_returns':
            require(type(actual[k]) is list and len(actual[k])==252,'Direct NAV axis')
            for a,e,q in zip(actual[k],expected[k],actual['portfolio_returns']):
                require(type(a) is float,'Direct NAV float');close(a,e,NAV_ATOL,NAV_RTOL);close(a,q,NAV_ATOL,NAV_RTOL)
        else:record_check(actual[k],expected[k],'Canonical258/'+k)

def independent_features(panel,meta,calendar,arrays,review,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(arrays,meta,n)
    frame,sessions=old.source_axes(panel,calendar,n);lookup={d:i for i,d in enumerate(sessions)}
    windows,identities,bounds=old.review_index(review,n);features=[]
    for i in range(n):
        event=base_event(base,meta,i);clock=old.clock_record(event,sessions,lookup)
        for k,v in clock.items():record_check(windows[i].get(k),v,'Source clock/'+k)
        listings=old.listing_metadata(event,identities,bounds)
        if event['branch'] in ('initial_partial','fewer_than_three'):features.append(event);continue
        paths={}
        for j in event['shortlist_indices']:
            cell,path=old.source_cell(frame,event,j,listings.get(j));event['source_cells'].append(cell)
            if path is not None:paths[j]=path
        if len(paths)!=len(event['shortlist_indices']):event['branch']='missing_source'
        else:
            for k in event['feasible_trio_indices']:
                t=event['trios'][k];t['statistics']=measure_oracle([paths[j] for j in t['indices']])
        features.append(event)
    return features

def feature_compare(actual,expected):
    require(type(actual) is dict and set(actual)==set(expected),'Feature keys')
    for k,v in expected.items():
        if k!='trios':record_check(actual[k],v,'Feature/'+k)
    require(type(actual['trios']) is list and len(actual['trios'])==len(expected['trios']),'Trio length')
    for a,e in zip(actual['trios'],expected['trios']):
        require(type(a) is dict and set(a)==set(TRIO_KEYS),'Trio keys')
        for k,v in e.items():
            if k=='statistics' and v is not None:statistics_check(a[k],v)
            else:record_check(a[k],v,'Trio/'+k)

def feature_counts(features,width):
    c={k:0 for k in COUNT_FIELDS};c.update(events=len(features),symbols_per_event=width,original_episode_calls=1)
    for e in features:
        c['eligible_cells']+=len(e['eligible_indices']);c['initial_partial_events']+=int(e['initial_partial'])
        c['active_events']+=int(not e['initial_partial'] and len(e['eligible_indices'])>=3)
        c['shortlist_cells']+=len(e['shortlist_indices']);c['source_available_cells']+=sum(x['source_available'] for x in e['source_cells'])
        c['source_price_observations']+=sum(x['observations_checked'] for x in e['source_cells'])
        c['enumerated_trios']+=len(e['trios']);c['feasible_trios']+=len(e['feasible_trio_indices'])
        for t in e['trios']:
            s=t['statistics']
            if s is not None:
                c['trio_statistics']+=1;c['formation_return_observations']+=252
                c['zero_downside_trios']+=int(s['zero_downside']);c['zero_symmetric_trios']+=int(s['zero_symmetric'])
        c['missing_source_events']+=int(e['branch']=='missing_source');c['available_events']+=int(e['branch']=='available')
    return c

def feature_receipt(features,width,source_rebuilt):
    c=feature_counts(features,width)
    return dict(**{k:True for k in FEATURE_FLAGS},checked_events=len(features),feature_counts=c,
      full_features_supplied=True,source_rebuilt=source_rebuilt,descriptive_path_evaluations=c['trio_statistics'] if source_rebuilt else 0,
      cached_path_validations=0 if source_rebuilt else c['trio_statistics'],original_episode_calls=1,
      candidate_selection_computed=False,computes_returns=False,producer_modules_imported=False,
      continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,nav_atol=NAV_ATOL,nav_rtol=NAV_RTOL,
      rank_tolerance=0.,exact_discrete=True)

def verify_features(panel,meta,calendar,arrays,review,features,n=None):
    n=len(meta['dates']) if n is None else n
    expected=independent_features(panel,meta,calendar,arrays,review,n)
    require(type(features) is list and len(features)>=n,'Feature prefix')
    for a,e in zip(features[:n],expected):feature_compare(a,e)
    return feature_receipt(expected,np.asarray(arrays['eligible']).shape[1],True)

def verify_features_cached(arrays,meta,features,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(arrays,meta,n)
    require(type(features) is list and len(features)>=n,'Cached feature prefix')
    for i in range(n):cache_event_check(features[i],base,meta,i)
    return feature_receipt(features[:n],base['width'],False)

def orders_from_choices(scores,choices,meta):
    return {a:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
      symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[a][i,j]) for j in ids])
      for i,ids in enumerate(choices[a])] for a in ARMS}

def incumbent_receipt(n):
    return dict(**{k:True for k in INCUMBENT_FLAGS},checked_events=n,ordered_choices_checked=n,
      full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.,original_episode_calls=1,
      descriptive_path_evaluations=0,cached_path_validations=0,
      candidate_selection_computed=False,producer_modules_imported=False,computes_returns=False)

def verify_incumbent(arrays,meta,orders,scores,pre_scores,n=None):
    n=len(meta['dates']) if n is None else n;b=original_base(arrays,meta,n)
    expected=orders_from_choices({a:b['F'] for a in ARMS},{a:b['choices'] for a in ARMS},meta)[ARMS[0]]
    record_check(orders,expected,'Exact incumbent orders');exact_array(scores,b['F'],'Exact incumbent final');exact_array(pre_scores,b['P'],'Exact incumbent pre')
    return incumbent_receipt(n)

def selection_oracle(arrays,meta,features,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(arrays,meta,n)
    require(type(features) is list and len(features)>=n,'Feature prefix')
    counts=feature_counts(features[:n],base['width']);events=[];choices={a:[] for a in ARMS}
    for i in range(n):
        cache_event_check(features[i],base,meta,i);e=copy.deepcopy(features[i])
        picked={a:e['incumbent_choices'].copy() for a in ARMS};winners={a:None for a in ARMS[1:]};ties={a:[] for a in ARMS[1:]}
        if e['branch']=='available':
            for arm,objective in zip(ARMS[1:],('downside_moment','symmetric_moment')):
                H=e['feasible_trio_indices'];trios=e['trios'];best=min(trios[k]['statistics'][objective] for k in H)
                tied=[k for k in H if trios[k]['statistics'][objective]==best]
                inc=[k for k in tied if trios[k]['indices']==e['incumbent_trio']]
                winner=inc[0] if inc else min(tied,key=lambda k:tuple(trios[k]['symbols']))
                winners[arm]=winner;ties[arm]=tied;ids=trios[winner]['indices']
                if set(ids)!=set(e['incumbent_choices']):picked[arm]=sorted(ids,key=lambda j:(-base['F'][i,j],e['symbols'][j]))
        e.update(choices=picked,winner_trio_indices=winners,tied_trio_indices=ties);events.append(e)
        for a in ARMS:choices[a].append(picked[a])
        for a,key in zip(ARMS[1:],('downside_changed_holdings','symmetric_changed_holdings')):
            counts[key]+=int(set(picked[a])!=set(picked[ARMS[0]]))
        counts['candidate_control_changed_holdings']+=int(set(picked[ARMS[1]])!=set(picked[ARMS[2]]))
    audit=dict(schema='monthly_downside_basket258_decisions_v1',counts=counts,events=events,
      incumbent_gate=dict(passed=True,synthetic_no_callback=True),source_gate=dict(passed=True,synthetic_no_callback=True))
    return {a:base['F'].copy() for a in ARMS},choices,{a:base['P'].copy() for a in ARMS},audit

def verify_selection(arrays,meta,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    es,ch,ep,ea=selection_oracle(arrays,meta,features,n)
    record_check(orders,orders_from_choices(es,ch,meta),'Exact ordered choices')
    for actual,expected,label in ((scores,es,'Final'),(pre_scores,ep,'Pre')):
        require(type(actual) is dict and set(actual)==set(ARMS),label+' arms')
        for a in ARMS:exact_array(actual[a],expected[a],label+'/'+a)
    require(type(audit) is dict and set(audit)==set(ea),'Complete decision audit')
    if audit['incumbent_gate']!=ea['incumbent_gate']:
        record_check(audit['incumbent_gate'],incumbent_receipt(n),'Incumbent gate proof');ea['incumbent_gate']=audit['incumbent_gate']
    if audit['source_gate']!=ea['source_gate']:
        proof=audit['source_gate'];require(type(proof) is dict and type(proof.get('source_rebuilt')) is bool,'Source gate proof')
        record_check(proof,feature_receipt(features[:n],es[ARMS[0]].shape[1],proof['source_rebuilt']),'Source gate proof');ea['source_gate']=proof
    record_check(audit,ea,'Full decision audit')
    return dict(**{k:True for k in SELECTION_FLAGS},checked_events=n,ordered_choices_checked=3*n,counts=ea['counts'],
      full_features_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,
      numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,
      descriptive_path_evaluations=0,cached_path_validations=ea['counts']['trio_statistics'],original_episode_calls=1)

def verify_prepared(panel,meta,calendar,arrays,review,features,orders,*,scores,pre_scores,audit,n=None):
    source=verify_features(panel,meta,calendar,arrays,review,features,n)
    result=verify_selection(arrays,meta,features,orders,scores=scores,pre_scores=pre_scores,audit=audit,n=n)
    result.update({k:True for k in PREPARE_FLAGS})
    result.update(continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,nav_atol=NAV_ATOL,nav_rtol=NAV_RTOL,
      exact_discrete=True,rank_tolerance=0.,descriptive_path_returns_checked=True,
      descriptive_path_evaluations=source['descriptive_path_evaluations'],original_episode_calls=2)
    return result


def cache_event_check(event,base,meta,i):
    require(type(event) is dict and set(event)==set(EVENT_KEYS),'Complete cached feature event')
    expected=base_event(base,meta,i)
    copied=('session_position','price_start','price_end','source_dates','clock_sessions','source_cells','branch')
    for key in expected:
        if key not in copied and key!='trios':record_check(event[key],expected[key],'Original binding/'+key)
    clock=event['clock_sessions'];require(type(clock) is list and len(clock)==296,'Cached official clock')
    days=[old.iso(d) for d in clock]
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
            old.iso(listing['not_before']);old.iso(listing['available_from'])
            record_check(listing['visible'],listing['available_from']<=event['date'],'Listing visibility')
            record_check(listing['start_precedes_bound'],event['price_start']<listing['not_before'],'Listing window')
            require(not(cell['source_available'] and listing['visible'] and listing['start_precedes_bound']),'No false complete listing path')
    if expected['branch'] in ('initial_partial','fewer_than_three'):branch=expected['branch']
    elif any(not c['source_available'] for c in cells):branch='missing_source'
    else:branch='available'
    source_good=branch=='available'
    for t in event['trios']:
        if source_good and t['feasible']:statistics_check(t['statistics'])
        else:require(t['statistics'] is None,'No statistics outside complete feasible source')
    if source_good:
        shared={};bycol={c['column_index']:c for c in cells}
        for t in event['trios']:
            if t['statistics'] is None:continue
            for j,path in zip(t['indices'],t['statistics']['normalized_prices']):
                if j in shared:record_check(path,shared[j],'Identical security path across trios')
                shared[j]=path;cell=bycol[j]
                record_check(path[-1],quotient(cell['end_price'],cell['start_price']),'Normalized source endpoint')
    record_check(event['branch'],branch,'Common fallback branch')



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
                require(float(results[arm][name]['metrics']['fees_paid'])==0. and
                    float(results[arm][name]['slices']['full122']['fees_paid'])==0.,'Zero-cost aggregate fees')
            audited=helper.audit_path(px,opening,orders[arm],*[data[k] for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')],execution,cost)
            if cost==0.:require(float(audited['fees_paid'])==0.,'Independent zero-cost aggregate')
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
