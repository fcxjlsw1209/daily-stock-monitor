"""Independent round149 accounting and selection audit; no producer imports.

Only the frozen prior independent reviewer and its accounting implementation are
reused. Actual files require a registered PLAN and completed diagnostic marker.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
import monthly_stock_comp149_review_contract as contract

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'work/monthly_same_report145_outcome_review.py'
OLD_SHA = '968038244802fa5ea18253adbc99786979bd5e40890c96d04f859721ed728d8e'
if hashlib.sha256(OLD.read_bytes()).hexdigest() != OLD_SHA:
    raise ValueError('Frozen independent reviewer changed')
import monthly_same_report145_outcome_review as old
h, require = old.h, old.require
Reader, replay, compare_stats = old.Reader, old.replay, old.compare_stats
normalize, rr_statistics = old.normalize, old.rr_statistics
STUDY = 'outputs/monthly_stock_comp149'
PRICE = old.PRICE
CONCEPT = contract.CONCEPT
CONCEPT_SHA = contract.CONCEPT_SHA
NAMES = ('incumbent_local67', 'earnings_control', 'cashflow_control', 'comp_adjusted_cashflow')
HANDS = ('earnings_bounded', 'cash_flow_bounded', 'comp_adjusted_cash_flow_bounded')
SBC_REFS = ('sbc_dataset', 'sbc_admission', 'sbc_policy', 'sbc_review')
METHOD = 'current_annual_cash_quality_v1'
MEASUREMENT = contract.MEASUREMENT
PATHS = old.PATHS
CACHE_FILES = old.CACHE_FILES
FROZEN_FILES = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json',
                'TRAIN_OPEN_TRIALS.json', 'WALKFORWARD.json', 'INDEPENDENT_PREPARE.json')
PREPARE_CHECKS = ('original27_arrays_byte_identical', 'all_features_independently_reconstructed',
    'all_four_score_matrices_and_ordered_choices_independently_verified',
    'one_episode_and_prefix_isolation_verified', 'zero_and_missing_exact_fallback')
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'sessions',
          'protected_manifest', 'admitted_dataset', 'source_admission')


def source_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    return [root/'work/monthly_stock_comp149_outcome_review.py',
            root/'work/test_monthly_stock_comp149_outcome_review.py',
            root/'work/monthly_stock_comp149_review_contract.py',
            root/'work/stock_comp_source_admission_v1_independent.py',
            root/'work/test_stock_comp_source_admission_v1_independent.py',
            root/'work/monthly_same_report145_outcome_review.py',
            root/'work/monthly_ridge_support70_result_review.py']


def proposal_contract():
    return dict(total_months=122,training_months=88,early_months=53,recent_months=35,later_months=34,
        walkforward=[[36,60],[60,76],[76,88]],
        execution_paths=[dict(execution=x,cost=c,key=k) for x,c,k in PATHS],
        same_close_cagr='strict_greater_than_all_three_controls',
        next_open_cost_cagr='noninferior_incumbent_strict_greater_than_two_controls',
        strict_cagr_epsilon=1e-12,max_sharpe_degradation=.05,
        max_drawdown_degradation=.03,max_early_cagr_degradation=.025,
        max_recent_cagr_degradation=.025,new_model_fits=0,
        intramonth_trades=False,later34_for_selection=False)


def verify_proposal(reader,plan,pins):
    return contract.verify_proposal(reader,plan,pins)


def select(rows, opened):
    fields = (('training','cagr'), ('training','sharpe'), ('training','max_drawdown'),
              ('early','cagr'), ('recent','cagr'))
    def checks(group, at_open):
        require(type(group) is list and len(group)==4 and [r.get('name') for r in group]==list(NAMES), 'Four exact selection arms')
        values = [[r.get(a, {}).get(b) for a,b in fields] for r in group]
        ans = {}
        for i in range(3):
            candidate, control = values[3], values[i]
            valid = all(type(x) in (int,float) and math.isfinite(x) for x in candidate+control)
            if not valid:
                ans[NAMES[i]] = False
                continue
            growth = candidate[0] >= control[0]-1e-12 if at_open and i==0 else candidate[0] > control[0]+1e-12
            ans[NAMES[i]] = bool(growth and all(c >= b-t for c,b,t in zip(candidate[1:],control[1:],(.05,.03,.025,.025))))
        return ans
    closed, op = checks(rows, False), checks(opened, True)
    admitted = all(closed.values()) and all(op.values())
    return dict(primary=NAMES[3] if admitted else NAMES[0], admitted=admitted,
                guards_against=closed, next_open_guards_against=op, mechanism_confirmed=all(op.values()))


def validate_orders(close, opening, orders, meta):
    dates = old.daily_axis(close.index, 'prices')
    require(opening.index.equals(dates) and opening.columns.equals(close.columns) and close.columns.is_unique, 'Same price axes')
    days, ends = old.daily_axis(meta['dates'], 'signals'), pd.DatetimeIndex(meta['ends'])
    require(len(days)==len(ends)==len(meta['symbols'])==122 and days.isin(dates).all() and ends.isin(dates).all() and (ends>days).all(), '122 ordered holding periods')
    require(days[1:].equals(ends[:-1]), 'Adjacent holding periods')
    require(set(orders)==set(NAMES), 'Four exact order arms')
    for name in NAMES:
        require(len(orders[name])==122, '122 orders per arm')
        for i,row in enumerate(orders[name]):
            require(set(row)=={'date','end','signal_date','symbols','scores'}, 'Exact order fields')
            require(row['date']==row['signal_date']==meta['dates'][i] and row['end']==meta['ends'][i], 'Order period')
            symbols = row['symbols']
            require(type(symbols) is list and len(symbols)==len(set(symbols)) and len(symbols) in (0,2,3), 'Cash or two/three positions')
            require(all(type(s) is str and s in meta['symbols'][i] and s in close.columns for s in symbols), 'Original security identifiers')
            require(len(row['scores'])==len(symbols) and all(type(x) in (int,float) and math.isfinite(x) for x in row['scores']), 'Aligned finite scores')
            require(dates.get_loc(days[i])+1<len(dates), 'Next session exists')
    return dates[(dates>=days[0]) & (dates<=ends[-1])]


def walkforward(dates, navs, open_navs, meta):
    folds, selected, control = [], [], []
    for n,end in ((36,60),(60,76),(76,88)):
        opened = [dict(name=name, **h.measures(dates,open_navs[name],meta,n)) for name in NAMES]
        rows = [dict(name=name, **h.measures(dates,navs[name],meta,n)) for name in NAMES]
        pick = select(rows, opened)
        folds.append(dict(n=n,end=end,selection=pick,open_training=opened))
        lo,hi = dates.get_loc(pd.Timestamp(meta['dates'][n])), dates.get_loc(pd.Timestamp(meta['ends'][end-1]))
        for dest,name in ((selected,pick['primary']),(control,NAMES[0])):
            nav = navs[name]
            dest.extend((nav[1:]/nav[:-1]-1.)[lo:hi])
    return dict(folds=folds,pristine_oos=False,
        metrics=rr_statistics(selected,meta['dates'][36],meta['ends'][87]),
        incumbent=rr_statistics(control,meta['dates'][36],meta['ends'][87]))


def validate_cells(cells, meta):
    expected = [(i,j,day,s) for i,(day,pool) in enumerate(zip(meta['dates'],meta['symbols'])) for j,s in enumerate(pool)]
    require(type(cells) is list and len(cells)==len(expected), 'Complete feature population')
    keys = {'signal_index','pool_index','signal_date','symbol','available','reason','method','current_event_id','current_available_from','current_period_end','prior_period_end','current_days','prior_days','currency_key','measurement','original_cash_quality_available','stock_compensation_event_id',*HANDS}
    def day(x):
        require(type(x) is str, 'Explicit ISO day')
        d = pd.Timestamp(x)
        require(not pd.isna(d) and d.tzinfo is None and str(d.date())==x, 'Canonical ISO day')
        return d
    groups = [[] for _ in meta['dates']]
    for cell, identity in zip(cells, expected):
        require(type(cell) is dict and set(cell)==keys, 'Exact cash-quality cell schema')
        require(type(cell['signal_index']) is int and type(cell['pool_index']) is int and tuple(cell[k] for k in ('signal_index','pool_index','signal_date','symbol'))==identity, 'Exact cell identity/order')
        require(type(cell['available']) is bool and type(cell['reason']) is str and cell['reason'] and cell['method']==METHOD and cell['measurement']==MEASUREMENT, 'Cell status/method')
        require(type(cell['original_cash_quality_available']) is bool, 'Original availability boolean')
        values = [cell[k] for k in HANDS]
        if cell['available']:
            require(cell['reason']=='available' and all(type(x) in (int,float) and math.isfinite(x) and abs(x)<=1. for x in values), 'Three common bounded features')
            require(cell['original_cash_quality_available'] and cell['stock_compensation_event_id']==cell['current_event_id'], 'Same currently eligible event')
            require(type(cell['current_event_id']) is str and cell['current_event_id'] and cell['currency_key']=='same-report-dollar:'+cell['current_event_id'], 'Same-report identity and currency')
            cur,prev,available,signal = [day(cell[k]) for k in ('current_period_end','prior_period_end','current_available_from','signal_date')]
            require(prev<cur<available<=signal and 0<=(signal-cur).days<=550, 'Current availability and staleness')
            require(all(type(cell[k]) is int and 330<=cell[k]<=400 for k in ('current_days','prior_days')) and (cur-prev).days==cell['current_days'], 'Annual contiguous periods')
        else:
            require(all(x is None for x in values), 'Three features unavailable together')
            require(cell['stock_compensation_event_id'] is None, 'No admitted SBC event for unavailable cell')
        if cell['symbol'] in meta['etfs']:
            require(not cell['available'], 'No ETF accounting values')
        groups[identity[0]].append(cell)
    return groups


def verify_feature_matrices(arrays, feature, meta):
    shape=(122,49)
    want={k:np.full(shape,np.nan) for k in HANDS}
    want['common_available']=np.zeros(shape,bool)
    validate_cells(feature['cells'],meta)
    for cell in feature['cells']:
        i,j=cell['signal_index'],cell['pool_index']
        want['common_available'][i,j]=cell['available']
        if cell['available']:
            for k in HANDS: want[k][i,j]=cell[k]
    for key,value in want.items():
        require(key in arrays and arrays[key].shape==shape and arrays[key].dtype==value.dtype, 'Exact feature matrix schema '+key)
        np.testing.assert_array_equal(arrays[key],value)
    return len(want)


def feature_checks(feature, validation, meta):
    for obj in (feature,validation): require(type(obj.get('new_model_fits')) is int and obj['new_model_fits']==0, 'No new model fits')
    for key in ('original_arrays_unchanged','original_auxiliary_states_and_core_coefficients_preserved','latest_event_before_validity_filter','same_current_release_for_all_cells'):
        require(feature.get(key) is True,'Feature claim '+key)
    for key in ('returns_or_nav_computed','annual_duration_normalization','comparison_flow_values_used'):
        require(feature.get(key) is False,'Preparation scope '+key)
    for key in ('earnings_and_cashflow_controls_identical_to146_on_common_mask','common_mask_subset_of146','same_selected_current_event_for_all_fields','signed_stock_compensation_preserved','stock_compensation_raw_flags_do_not_substitute_new_admission','adjusted_cashflow_is_research_measure_not_reported_GAAP_cashflow'):
        require(feature.get(key) is True,'Stock-compensation claim '+key)
    require(feature.get('measurement')==MEASUREMENT and feature.get('stock_compensation_definition')=='current_annual_reported_stock_compensation_addback_v1','Stock-compensation definition')
    require(feature.get('original_months')==122 and feature.get('original_width')==49, 'Original dimensions')
    require(feature.get('arithmetic_method')=='exact_fraction_scaled_cashflow_minus_signed_sbc_over_average_assets_then_bounded_float', 'Exact declared arithmetic')
    validate_cells(feature['cells'],meta)
    require(type(feature.get('common_available_cells')) is int and feature['common_available_cells']==sum(x['available'] for x in feature['cells']), 'Available count')
    require(type(feature.get('original_cash_quality_available_cells')) is int and feature['original_cash_quality_available_cells']==sum(x['original_cash_quality_available'] for x in feature['cells']), 'Original available count')
    require(feature.get('reasons')==dict(Counter(x['reason'] for x in feature['cells'])), 'Complete feature reasons')


def monthly(orders,close,meta,cells):
    groups=validate_cells(cells,meta); out=[]
    for i,day in enumerate(meta['dates']):
        held={name:orders[name][i]['symbols'] for name in NAMES}
        ret={name:(math.fsum(float(close.at[pd.Timestamp(meta['ends'][i]),s])/float(close.at[pd.Timestamp(day),s])-1. for s in held[name])/len(held[name]) if held[name] else 0.) for name in NAMES}
        for name,key in zip(NAMES[1:],HANDS):
            if not any(x[key] is not None and x[key]!=0. for x in groups[i]): h.assert_exact_tree(orders[name][i],orders[NAMES[0]][i])
        out.append(dict(date=day,end=meta['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34',holdings=held,returns=ret))
    return out


def check_outcomes(close,opening,orders,meta,read):
    dates=validate_orders(close,opening,orders,meta)
    results=read('RESULTS.json');require(set(results)==set(NAMES),'Four result arms')
    bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),original2021_2026=(meta['dates'][53],meta['ends'][-1]),training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
    errors=[];navs={};open_navs={};metrics={};details={};maxnav=0.
    for name in NAMES:
        require(set(results[name])=={x[2] for x in PATHS},'Three paths per arm')
        for execution,cost,key in PATHS:
            met,nav,trades,hold=replay(close,opening,orders[name],dates,execution,cost)
            prefix=name+'_'+key
            require(set(results[name][key])=={'metrics','slices'},'Exact result fields')
            compare_stats(results[name][key]['metrics'],met,errors,prefix+'/metrics')
            compare_stats(results[name][key]['slices'],{k:h.slice_stat(dates,nav,*span) for k,span in bounds.items()},errors,prefix+'/slices')
            daily=read(prefix+'_DAILY.json')
            require(all(set(x)=={'date','equity'} for x in daily) and [x['date'] for x in daily]==[str(d.date()) for d in dates],'Exact daily NAV dates')
            actual=np.asarray([x['equity'] for x in daily],float)
            require(np.isfinite(actual).all() and (actual>0).all(),'Finite positive NAV')
            np.testing.assert_allclose(actual,nav,atol=1e-5,rtol=2e-11)
            maxnav=max(maxnav,float(np.max(np.abs(actual-nav))))
            actual_trades,actual_hold=read(prefix+'_TRADES.json'),read(prefix+'_HOLDINGS.json')
            h.compare(normalize(actual_trades),normalize(trades),errors,prefix+'/trades')
            h.compare(normalize(actual_hold),normalize(hold),errors,prefix+'/holdings')
            allowed={row['date'] if execution=='close' else str(close.index[close.index.get_loc(pd.Timestamp(row['date']))+1].date()) for row in orders[name]}
            require(all(x['date'] in allowed for x in actual_trades),'No intramonth trades')
            require([x['year'] for x in met['annual_returns']]==sorted({int(x.year) for x in dates}) and all(x['partial'] is (x['year'] in (2016,2026)) for x in met['annual_returns']),'Annual coverage and partial flags')
            details[prefix]=dict(trades=len(trades),holding_periods=len(hold),fees_paid=met['fees_paid'])
            if execution=='close':navs[name]=nav;metrics[name]=met
            if key=='next_open_cost':open_navs[name]=nav
        td=dates[dates<=pd.Timestamp(meta['ends'][87])]
        for execution,cost,full in (('close',0.,navs[name]),('open',.0005,open_navs[name])):
            _,tn,_,_=replay(close.loc[:td[-1]],opening.loc[:td[-1]],orders[name][:88],td,execution,cost)
            np.testing.assert_allclose(tn,full[:len(td)],atol=1e-5,rtol=2e-11)
    rows=[dict(name=name,**h.measures(dates,navs[name],meta)) for name in NAMES]
    opened=[dict(name=name,**h.measures(dates,open_navs[name],meta)) for name in NAMES]
    compare_stats(read('TRAIN_TRIALS.json'),rows,errors,'training')
    compare_stats(read('TRAIN_OPEN_TRIALS.json'),opened,errors,'open_training')
    pick=select(rows,opened);frozen=read('FROZEN.json');h.assert_exact_tree(frozen['selection'],pick)
    compare_stats(read('WALKFORWARD.json'),walkforward(dates,navs,open_navs,meta),errors,'walkforward')
    validation=read('VALIDATION.json');h.assert_exact_tree(validation['selection'],pick)
    require(validation.get('pristine_oos') is False and validation.get('independent_outcome_review_pending') is True,'Diagnostic scope')
    require(validation.get('no_observer_writes') is True and validation.get('no_intramonth_trades') is True,'Scope claims')
    feature=read('FEATURE_AUDIT.json');feature_checks(feature,validation,meta)
    months=monthly(orders,close,meta,feature['cells'])
    return dict(passed=True,stage='evaluate',execution_paths_verified=12,total_monthly_orders_verified=488,
        stock_compensation_feature_matrices_verified=4,daily_nav_trades_holdings_metrics_slices=True,
        first88_and_three_fold_selection_verified=True,first88_dual_execution_selection_verified=True,
        annual_partial_flags_verified=True,first88_causal_replay_verified=True,
        selected_name=pick['primary'],primary_cagr=metrics[pick['primary']]['cagr'],primary_target_reached=metrics[pick['primary']]['cagr']>=.7,
        max_compared_absolute_difference=max((x for _,x in errors),default=0.),max_nav_absolute_difference=maxnav,
        replay_details=details,independent_monthly_comparison=months,
        producer_modules_imported=False,models_fit=0,producer_model_fits=0,zero_model_fit_count_verified=True,
        protected_member_scan_repeated=False,available_feature_cells=sum(x['available'] for x in feature['cells']),
        unavailable_feature_cells=sum(not x['available'] for x in feature['cells']))


def verify_admission(reader,plan,pins):
    refs=plan['inputs'];data=reader.json(refs['admitted_dataset']['path']);receipt=reader.json(refs['source_admission']['path'])
    require(receipt.get('stage')=='conditional_cash_quality_source_admission' and receipt.get('passed') is True,'Conditional cash-quality admission')
    require(receipt.get('data_sha256')==refs['admitted_dataset']['sha256']==reader.sha(refs['admitted_dataset']['path']),'Dataset admission digest')
    require(data.get('method')==receipt.get('method')==METHOD and data.get('conditional_research_only') is True,'New field-specific method')
    policy=plan['source_policy'];policy_path=reader.path(policy['path'])
    require(set(policy)=={'path','sha256'} and reader.sha(policy_path)==policy['sha256']==data.get('policy_sha256')==receipt.get('policy_sha256')==pins.get(str(policy_path)), 'Exact source policy binding')
    closure=reader.verify(receipt['source_hashes']);require(all(pins.get(p)==sha for p,sha in closure.items()),'Admitted source closure')
    require(closure.get(str(reader.path(refs['sessions']['path'])))==refs['sessions']['sha256'],'Shared source calendar')
    for key in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):
        require(receipt.get(key) is False,'No upgraded source certification')
    for key in ('financial_ratios','factor_calculations','portfolio_calculations','return_calculations','network_requests','observer_changes'):
        require(type(receipt.get(key)) is int and receipt[key]==0,'Admission before financial calculations')
    for key in ('original_raw_events_preserved','original_rejections_preserved'):
        require(receipt.get(key) is True,'Original source preserved')
    events=data.get('events');require(type(events) is list and len(events)==44 and len({e['event_id'] for e in events})==44,'All44 events retained')
    admitted=[];holds=[]
    for e in events:
        a=e['cash_quality_admission'];require(set(a)=={'admitted','reasons','definition'} and type(a['admitted']) is bool and type(a['reasons']) is list,'Exact cash admission')
        require(a['definition']=='consolidated_gaap_cfo_net_income_average_assets_v1','Exact field definition')
        require(all(e.get(k) is False for k in ('fact_admitted','model_admitted','strategy_admitted')),'Raw flags preserved')
        if a['admitted']:
            require(not a['reasons'] and e.get('prior_field_admitted') is True and not e.get('inherited_rejection_codes'),'Cannot clear old holds')
            admitted.append(e['event_id'])
        else:holds.append(dict(event_id=e['event_id'],**a))
    require(receipt.get('events')==44 and receipt.get('admitted_events')==len(admitted) and receipt.get('held_events')==len(holds) and receipt.get('admitted_event_ids')==admitted and receipt.get('holds')==holds,'Exact admission census')
    contract.verify_stock_comp_source(reader,plan,pins,data)
    return receipt


def verify_diagnostic_start(reader,directory,plan_sha,audit_started):
    path=directory/'DIAGNOSE_STARTED.json';require(path.is_file(),'Diagnostic start marker required')
    marker=reader.json(path)
    require(set(marker)=={'plan_sha256','started_utc'} and marker['plan_sha256']==plan_sha,'Diagnostic marker binding')
    value=marker['started_utc'];require(type(value) is str and value.endswith(('+00:00','Z')),'UTC diagnostic time')
    try: start=datetime.fromisoformat(value[:-1]+'+00:00' if value.endswith('Z') else value)
    except ValueError: raise ValueError('UTC diagnostic time') from None
    require(start.tzinfo is not None and start.utcoffset().total_seconds()==0 and start<audit_started,'Diagnostic must precede audit in UTC')


def bindings(reader,plan_sha,audit_started=None):
    require(type(plan_sha) is str and len(plan_sha)==64 and all(c in '0123456789abcdef' for c in plan_sha),'Explicit PLAN SHA')
    p=reader.root/STUDY;require(reader.sha(p/'PLAN.json')==plan_sha,'Exact registered PLAN')
    plan=reader.json(p/'PLAN.json')
    require(plan.get('round')==149 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES) and plan.get('primary_challenger')==NAMES[3],'Registered four-arm protocol')
    require(reader.path(plan['output_directory'])==p and plan.get('dose')==.01,'Output and dose')
    require(plan.get('runtime')==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Pinned runtime')
    pins=reader.verify(plan['hashes'])
    for file in source_files():require(pins.get(str(file))==reader.sha(file),'Outcome reviewer source in PLAN')
    for key in ('raw_source_hashes','independent_prepare_source_hashes'):
        require(all(pins.get(k)==v for k,v in reader.verify(plan[key]).items()),'PLAN source closure '+key)
    for key in ('actual_feature_calculations_before_registration','actual_returns_before_registration','actual_model_fits_before_registration'):
        require(type(plan.get(key)) is int and plan[key]==0,'Prospective registration')
    require(set(plan['inputs'])==set(INPUTS),'Exact named inputs')
    for name,ref in plan['inputs'].items():require(reader.sha(ref['path'])==ref['sha256']==pins.get(str(reader.path(ref['path']))),'Named input '+name)
    verify_proposal(reader,plan,pins);verify_admission(reader,plan,pins)
    require(reader.path(plan['inputs']['prices']['path'])==reader.root/PRICE and reader.path(plan['inputs']['original_orders']['path'])==reader.root/'outputs/monthly_local_linear70/local_linear_ORDERS.json','Original prices/orders')
    cache=reader.json(p/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Exact CACHE artifacts');cached=reader.verify(cache,p)
    prep=reader.json(p/'INDEPENDENT_PREPARE.json')
    extra=('source_admission_bound','source_union_preservation_verified','raw_prices_independently_verified','calendar_overlay_independently_verified','all_signal_prefixes_verified','exact_integer_closed_form_arithmetic','stock_comp_source_admission_verified','common_mask_subset_of146_verified')
    require(prep.get('passed') is True and prep.get('stage')=='prepare' and all(prep.get(k) is True for k in PREPARE_CHECKS+extra),'Independent prepare passed')
    for key,value in (('orders_checked',488),('source_cells_checked',5978),('stock_cells_checked',3904),('new_model_fits',0),('monthly_label_models_fit',0)):
        require(type(prep.get(key)) is int and prep[key]==value,'Exact prepare counts')
    require(prep.get('producer_modules_imported') is False and prep.get('returns_or_nav_computed') is False,'Prepare isolation')
    for key,name in (('plan_sha256','PLAN.json'),('cache_sha256','CACHE.json'),('orders_sha256','PRETRAIN_ORDERS.json')):
        require(prep.get(key)==reader.sha(p/name),'Prepare binding '+key)
    require(reader.verify(prep['source_hashes'])==reader.verify(plan['independent_prepare_source_hashes']),'Prepare source closure')
    artifacts=reader.verify(prep['artifact_hashes']);required=dict(cached,**{str(p/n):reader.sha(p/n) for n in ('PLAN.json','CACHE.json')})
    require(all(artifacts.get(k)==v for k,v in required.items()),'Complete prepare artifacts')
    allowed=dict(pins,**required);require(all(allowed.get(k)==v for k,v in artifacts.items()),'No unbound prepare artifacts')
    frozen=reader.json(p/'FROZEN.json');require(set(frozen['hashes'])==set(FROZEN_FILES),'Exact FROZEN input set');reader.verify(frozen['hashes'],p)
    require(frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,'Frozen diagnostic scope')
    verify_diagnostic_start(reader,p,plan_sha,audit_started or datetime.now(timezone.utc))
    orders=reader.json(p/'PRETRAIN_ORDERS.json');h.assert_exact_tree(orders[NAMES[0]],reader.json(plan['inputs']['original_orders']['path']))
    return p


def audit(root,plan_sha):
    started=datetime.now(timezone.utc);reader=Reader(root);p=bindings(reader,plan_sha,started)
    plan=reader.json(p/'PLAN.json');meta=reader.json(p/'META.json');orders=reader.json(p/'PRETRAIN_ORDERS.json')
    require(meta==reader.json(plan['inputs']['meta']['path']) and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Original full horizon/meta')
    data=reader.npz(PRICE);require(set(data)=={'dates','symbols','close','opening'},'Price payload')
    dates=pd.to_datetime(data['dates']);close=pd.DataFrame(data['close'],index=dates,columns=data['symbols']);opening=pd.DataFrame(data['opening'],index=dates,columns=data['symbols'])
    report=check_outcomes(close,opening,orders,meta,lambda n:reader.json(p/n))
    feature=reader.json(p/'FEATURE_AUDIT.json')
    report['stock_compensation_feature_matrices_verified']=verify_feature_matrices(reader.npz(p/'DATA.npz'),feature,meta)
    for field,key in (('source_admission_sha256','source_admission'),('admitted_dataset_sha256','admitted_dataset')):
        require(feature.get(field)==reader.sha(plan['inputs'][key]['path']),'Feature admission binding')
    for key in SBC_REFS:
        require(feature.get(key+'_sha256')==plan[key]['sha256'],'Feature stock-compensation source binding')
    report.update(schema_version=1,proposal_sha256=plan['inputs']['proposal']['sha256'],plan_sha256=plan_sha,cache_sha256=reader.sha(p/'CACHE.json'),orders_sha256=reader.sha(p/'PRETRAIN_ORDERS.json'),frozen_sha256=reader.sha(p/'FROZEN.json'),
        source_admission_bound=True,common_availability_verified=True,baseline_local67_full_orders_unchanged=True,
        stock_comp_source_admission_verified=True,common_mask_subset_of146_verified=True,
        source_admission_sha256=feature['source_admission_sha256'],admitted_dataset_sha256=feature['admitted_dataset_sha256'],
        diagnostic_start_marker_verified=True,diagnose_started_sha256=reader.sha(p/'DIAGNOSE_STARTED.json'),audit_started_utc=started.isoformat(),
        source_hashes={str(f):reader.sha(f) for f in source_files()},first_public_version_verified=False,source_admission_is_conditional=True,
        limitations=['Repeatedly reused history; not pristine out of sample.', 'Conditional four-issuer annual accounting scope; no certified first-public history.', 'No protected inventory rescan in this outcome-only audit.'])
    for path,sha in list(reader.seen.items()):require(reader.sha(path)==sha,'Post-review binding')
    report['artifact_hashes']=dict(reader.seen)
    return report



def synthetic_receipt(result):
    reviewed=[ROOT/'work/monthly_stock_comp149_runner.py',ROOT/'work/monthly_stock_comp149_factor.py',ROOT/CONCEPT]
    reviewed.extend(ROOT/'outputs/monthly_stock_comp_source_admission_v2'/name for name in ('DATASET.json','ADMISSION.json','POLICY.json','INDEPENDENT_REVIEW.json'))
    proposal=ROOT/'work/monthly_stock_comp149_proposal.json'
    if proposal.exists(): reviewed.append(proposal)
    return dict(schema_version=1,stage='synthetic_outcome_review_only',passed=result.wasSuccessful(),
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()},
        reviewed_contract_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in reviewed},
        runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),
        actual_evaluation_checked=False,actual_factors_calculated=0,actual_returns_calculated=0,
        producer_modules_imported=False,arms=list(NAMES),selection_contract=proposal_contract(),
        expected_execution_paths=12,expected_monthly_orders=488,expected_source_cells=5978,
        actual_api='./run_local.sh work/monthly_stock_comp149_outcome_review.py --plan-sha <SHA256> --evaluate-complete',
        actual_output=STUDY+'/INDEPENDENT_OUTCOME.json',
        pending_requirements=['Registered PLAN and completed independent preparation, first88 selection, and full diagnostics.',
          'Exact registered proposal digest, unchanged fixed concept and selection semantics, and bound source admission.'],
        scope_limits=['Only generated prices and metadata were used in tests.',
          'Actual outcomes are independently replayed; producer factor/runner modules are never imported.',
          'All earlier frozen round146 evidence remains unchanged.'])


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan-sha',required=True);cli.add_argument('--evaluate-complete',action='store_true');cli.add_argument('--output',type=Path)
    args=cli.parse_args();require(args.evaluate_complete,'Actual audit only after completed evaluation')
    report=audit(ROOT,args.plan_sha);dest=args.output or ROOT/STUDY/'INDEPENDENT_OUTCOME.json'
    with dest.open('x') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,output=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),execution_paths=12)))

if __name__=='__main__':main()
