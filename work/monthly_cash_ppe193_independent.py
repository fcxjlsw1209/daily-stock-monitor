"""Independent exact-decimal cash-after-PPE oracle; no producer calculation imports.

Only independent accounting is reused through a pinned, isolated frozen module.
No actual-source CLI, training, replay, file mutation or network operations.
"""
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
METHOD='current_annual_cash_after_ppe_v1'
OLD_DEFINITION='consolidated_gaap_cfo_net_income_average_assets_v1'
DEFINITION='current_annual_cash_ppe_outflow_v1'
KEYS=('candidate_bounded','control_bounded','common_available')
ARMS=('incumbent_local67','same_support_cashflow_control','cash_after_ppe')
RAW={'same_support_cashflow_control':'control_bounded','cash_after_ppe':'candidate_bounded'}
DOSE=.01
FROZEN_HELPER=ROOT/'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA='f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'


def require(ok, message):
    if not ok: raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest()==FROZEN_HELPER_SHA,'Frozen192 accounting adapter changed')
    spec=importlib.util.spec_from_file_location('cash193_independent_accounts',FROZEN_HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def accounting():return frozen_helper().accounting()


def dependency_files():return tuple(sorted(set(frozen_helper().dependency_files())|{Path(__file__).resolve()}))


def audit_outcomes(px,opening,orders,exports,results,meta):
    helper=frozen_helper();helper.ARMS=ARMS
    receipt=helper.audit_outcomes(px,opening,orders,exports,results,meta)
    require(receipt['checked_paths']==9 and receipt['checked_slices']==45 and set(receipt['paths'])==set(ARMS),'Incomplete 193 account audit')
    receipt['cash_ppe_account_adapter_sha256']=FROZEN_HELPER_SHA
    return receipt


def text(value):return type(value) is str and bool(value) and value.strip()==value


def iso(value):
    require(type(value) is str,'ISO date required');result=date.fromisoformat(value)
    require(result.isoformat()==value,'Canonical ISO required');return result


def decimal_fraction(value, plain=False):
    pattern=r'-?\d+(?:\.\d+)?' if plain else r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    require(text(value) and re.fullmatch(pattern,value),'Explicit decimal required')
    sign,digits,exponent=Decimal(value).as_tuple()
    integer=0
    for digit in digits:integer=10*integer+digit
    if sign:integer=-integer
    return Fraction(integer*10**max(exponent,0),10**max(-exponent,0))


def blank(reason,event=None):
    return dict(available=False,reason=reason,method=METHOD,
        current_event_id=None if event is None else event.get('event_id'),
        current_available_from=None if event is None else event.get('available_from'),
        current_period_end=None,prior_period_end=None,current_days=None,prior_days=None,currency_key=None,
        candidate_bounded=None,control_bounded=None)


def period(value):
    if type(value) is not dict:return None,'invalid_period_dates'
    try:start,end=iso(value['period_start']),iso(value['period_end'])
    except (KeyError,TypeError,ValueError):return None,'invalid_period_dates'
    days=(end-start).days+1
    if not 330<=days<=400:return None,'nonannual_duration'
    if type(value.get('days_inclusive')) is not int or value['days_inclusive']!=days:return None,'duration_mismatch'
    return (start,end,days),None


def amount(cell,event,side):
    require(type(cell) is dict,'Missing amount')
    require(cell.get('source')==event['source'] and cell.get('period')==event['periods'][side]
            and cell.get('available_from')==event['available_from'] and cell.get('currency_literal')=='$','Cash-source binding differs')
    value=decimal_fraction(cell['value_decimal']);scale=decimal_fraction(cell['scale'])
    require(scale>0,'Positive scale required');return value*scale


def ppe_amount(event):
    p=event.get('cash_ppe');evidence=event.get('cash_ppe_source_evidence')
    require(type(p) is dict and type(evidence) is dict,'PPE missing')
    require(all(evidence.get(k)==event.get(k) for k in ('event_id','symbol','release_date','available_from','source')),'PPE identity/clock/source')
    require(evidence.get('inherited_admitted') is True and evidence.get('extraction_verified') is True
            and evidence.get('status')=='verified' and type(evidence.get('hold_reasons')) is list
            and not evidence['hold_reasons'],'PPE evidence held')
    require(not ('field' in evidence and 'capex' in evidence),'Ambiguous original cell')
    original=evidence.get('capex') if 'capex' in evidence else evidence.get('field')
    require(type(original) is dict and all(k in p and p[k]==v for k,v in original.items()),'Raw PPE not preserved')
    require(p.get('source')==event['source'] and p.get('period')==event['periods']['current']
            and p.get('available_from')==event['available_from'] and p.get('fact_admitted') is False,'PPE normalization')
    loc=p.get('locator');cfo=event['fields']['current']['operating_cash_flow']
    require(type(loc) is dict and loc.get('period_end')==event['periods']['current']['period_end'],'PPE annual column')
    require(p.get('currency_iso') in (None,'USD') and p.get('statement_currency_literal') in (None,'$'),'Currency conflict')
    if p.get('currency_literal')!='$':
        require(p.get('currency_literal') is None and p.get('statement_currency_literal')=='$','Dollar evidence missing')
        cfo_loc=cfo.get('locator')
        require(type(cfo_loc) is dict and loc.get('original_cfo_locator')==cfo_loc
                and loc.get('format')==cfo_loc.get('format'),'Statement identity differs')
        if loc.get('format')=='html':
            number=loc.get('table_index_zero_based')
            require(p.get('currency_scope')=='same formal cash-flow table; ISO unverified'
                    and type(number) is int and number>=0 and number==cfo_loc.get('table_index_zero_based')
                    and loc.get('statement_segment')==cfo_loc.get('statement_segment'),'Cash-flow table differs')
        elif loc.get('format')=='pdf':
            number=loc.get('page_1based')
            require(p.get('currency_scope')=='same formal cash-flow page; ISO unverified'
                    and type(number) is int and number>0 and number==cfo_loc.get('page_1based'),'Cash-flow page differs')
        else:raise ValueError('Statement format missing')
    outflow=decimal_fraction(p['positive_outflow_decimal'],True)
    signed=decimal_fraction(p['signed_value_decimal'],True);scale=decimal_fraction(p['scale'],True)
    require(outflow>=0 and signed<=0 and signed+outflow==0 and scale>0
            and scale==decimal_fraction(cfo['scale']),'PPE sign/scale')
    require(text(p.get('signed_literal')),'Displayed signed cell missing')
    row=p.get('row_text')
    require(type(row) is str and hashlib.sha256(row.encode()).hexdigest()==p.get('row_text_sha256'),'PPE evidence row hash')
    label=p.get('row_label',loc.get('row_label',''))
    require(type(label) is str and re.search(r'propert\w*.*(?:plant|equipment)',label,re.I),'PPE formal label')
    # The acquisition of the PP&E itself is not an acquisition of businesses.
    remainder=re.sub(r'\bacquisition\s+of\s+property(?:,?\s+plant)?\s+and\s+equipment\b','',label,flags=re.I)
    require(not re.search(r'intangible|acquisit|business|companies|subsidiar|principal|finance lease|net investing|free cash flow',remainder,re.I),'Mixed/noncash scope')
    return outflow*scale


def evaluate_oracle(event,signal):
    day=iso(signal)
    if type(event) is not dict:return blank('invalid_event')
    result=blank('unavailable',event)
    def fail(reason):result['reason']=reason;return result
    for name,definition in [('cash_quality',OLD_DEFINITION),('cash_ppe',DEFINITION)]:
        gate=event.get(name+'_admission')
        if type(gate) is not dict or set(gate)!={'admitted','definition','reasons'}:return fail(name+'_admission_required')
        if gate['admitted'] is not True:return fail('current_not_'+name+'_admitted')
        if gate['definition']!=definition:return fail('wrong_'+name+'_definition')
        if type(gate['reasons']) is not list or gate['reasons']:return fail(name+'_admitted_event_has_rejection_reasons')
    if type(event.get('inherited_rejection_codes')) is not list or event['inherited_rejection_codes']:return fail('inherited_rejection_not_cleared')
    if not all(text(event.get(k)) for k in ('event_id','symbol')) or event['symbol'] not in ('AAPL','MSFT','AMD','NVDA'):return fail('unsupported_or_missing_identity')
    identity=event.get('identity')
    if type(identity) is not dict or not all(text(identity.get(k)) for k in ('issuer_id','security_id')):return fail('missing_identity')
    if identity['security_id']!=event['symbol']:return fail('security_symbol_mismatch')
    source=event.get('source')
    if type(source) is not dict or not text(source.get('path')) or type(source.get('sha256')) is not str or not re.fullmatch('[0-9a-f]{64}',source['sha256']):return fail('missing_source_reference')
    try:
        known,released=iso(event['available_from']),iso(event['release_date'])
        start,end=iso(identity['identity_valid_from']),iso(identity['identity_valid_through'])
    except (KeyError,ValueError,TypeError):return fail('invalid_dates_or_identity_bounds')
    if 'current_release_available_from' in event and event['current_release_available_from']!=event['available_from']:return fail('current_release_availability_mismatch')
    if known>day:return fail('report_not_yet_available')
    if released>=known:return fail('availability_not_after_release')
    if start!=known:return fail('identity_start_not_current_availability')
    periods,fields=event.get('periods'),event.get('fields')
    if type(periods) is not dict or set(periods)!={'current','comparison'}:return fail('two_annual_periods_required')
    if type(fields) is not dict or set(fields)!={'current','comparison'} or any(type(fields[s]) is not dict or set(fields[s])!={'operating_cash_flow','net_income','total_assets'} for s in ('current','comparison')):return fail('raw_field_schema_required')
    current,why=period(periods['current'])
    if why:return fail('current_'+why)
    prior,why=period(periods['comparison'])
    if why:return fail('comparison_'+why)
    result.update(current_period_end=str(current[1]),prior_period_end=str(prior[1]),current_days=current[2],prior_days=prior[2],currency_key='same-report-dollar:'+event['event_id'])
    if current[1]>released:return fail('release_before_period_end')
    if prior[1]+timedelta(days=1)!=current[0]:return fail('noncontiguous_annual_periods')
    if end!=current[1]+timedelta(days=550):return fail('identity_end_not_causal_expiry')
    if (day-current[1]).days>550:return fail('stale_current_report')
    if not start<=day<=end:return fail('identity_outside_interval')
    numbers={}
    for side,field in [('current','operating_cash_flow'),('current','net_income'),('current','total_assets'),('comparison','total_assets')]:
        cell=fields[side][field]
        if cell is None:return fail('missing_'+side+'_'+field)
        try:numbers[side,field]=amount(cell,event,side)
        except (KeyError,TypeError,ValueError,ArithmeticError):return fail('invalid_'+side+'_'+field)
    ac,ap=numbers['current','total_assets'],numbers['comparison','total_assets']
    if min(ac,ap)<=0:return fail('nonpositive_asset_endpoint')
    try:ppe=ppe_amount(event)
    except (KeyError,TypeError,ValueError,ArithmeticError):return fail('invalid_or_missing_cash_ppe_evidence')
    # Independent closed form: h(2*n/(ac+ap))=2*n/(ac+ap+|2*n|).
    cash=numbers['current','operating_cash_flow']
    bounded=lambda numerator:float(2*numerator/(ac+ap+abs(2*numerator)))
    result.update(available=True,reason='available',candidate_bounded=bounded(cash-ppe),control_bounded=bounded(cash))
    return result


def select_oracle(events,signal):
    day=iso(signal);known={}
    for event in events:
        require(type(event) is dict and text(event.get('event_id')),'Event identity required')
        clock=iso(event.get('available_from'))
        if clock>day:continue
        key=event['event_id']
        if key in known and known[key]!=event:return blank('conflicting_event_id')
        known[key]=event
    if not known:return blank('current_missing_event')
    latest=max(e['available_from'] for e in known.values())
    newest=[e for e in known.values() if e['available_from']==latest]
    return evaluate_oracle(newest[0],signal) if len(newest)==1 else blank('current_ambiguous_latest_event')


def independent_features(meta,dataset):
    require(type(dataset) is dict and type(dataset.get('schema_version')) is int and dataset['schema_version']==1
            and dataset.get('method')==METHOD and dataset.get('conditional_research_only') is True,'Conditional dataset schema')
    require(type(dataset.get('events')) is list,'Event list required')
    groups={}
    for event in dataset['events']:
        require(type(event) is dict and text(event.get('symbol')),'Event issuer required')
        groups.setdefault(event['symbol'],[]).append(event)
    dates=pd.DatetimeIndex(meta['dates']);etfs=meta['etfs']
    require(dates.tz is None and not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing
            and dates.equals(dates.normalize()) and len(meta['symbols'])==len(dates),'Query axis')
    require(len(etfs)==17 and len(set(etfs))==17 and all(text(x) for x in etfs),'ETF identity axis')
    values={k:np.full((len(dates),49),np.nan) for k in KEYS[:-1]};values[KEYS[-1]]=np.zeros((len(dates),49),bool)
    cells=[]
    for i,(day,names) in enumerate(zip(dates,meta['symbols'])):
        require(len(names)==49 and len(set(names))==49 and set(etfs)<=set(names) and all(text(x) for x in names),'49-candidate identities')
        for j,symbol in enumerate(names):
            detail=blank('etf_not_applicable') if symbol in etfs else select_oracle(groups.get(symbol,[]),str(day.date()))
            if detail['available']:
                values['common_available'][i,j]=True
                for key in KEYS[:-1]:values[key][i,j]=detail[key]
            cells.append(dict(signal_index=i,pool_index=j,signal_date=str(day.date()),symbol=symbol,**detail))
    audit=dict(original_arrays_unchanged=True,original_months=len(dates),original_width=49,
        common_available_cells=int(values['common_available'].sum()),reasons=dict(Counter(c['reason'] for c in cells)),cells=cells,
        new_model_fits=0,returns_or_nav_computed=False,latest_event_before_validity_filter=True,
        same_current_release_for_all_cells=True,both_admissions_required=True,original_ni_gate_preserved=True,
        net_income_used_in_new_ratios=False,comparison_flow_values_used=False,annual_duration_normalization=False,
        original_auxiliary_states_and_core_coefficients_preserved=True,
        source_verification='caller_verified_cash_ppe_source_review_and_complete_hash_closure_required',
        arithmetic_method='exact_scaled_decimal_fraction_ratios_then_single_bounded_float',
        currency_scope='same_report_dollar_literal_assumption_not_ISO_certification',
        interpretation='annual_as_reported_CFO_less_cash_PPE_not_GAAP_FCF')
    return dict(arrays=values,audit=audit)


def checked_arrays(values):
    require(type(values) is dict and set(values)==set(KEYS),'Feature keys')
    a={k:np.asarray(v) for k,v in values.items()};m=a['common_available']
    require(m.ndim==2 and m.shape[1]==49 and m.dtype==bool,'Availability axis')
    for key in KEYS[:-1]:
        v=a[key];require(v.shape==m.shape and np.issubdtype(v.dtype,np.floating),'Value shape/dtype')
        require(not np.isinf(v).any() and np.array_equal(np.isfinite(v),m) and np.all(np.abs(v[m])<=1),'Common bounded support')
    require(np.all(a['candidate_bounded'][m]<=a['control_bounded'][m]),'PPE cannot increase same-support cash quality')
    return a


def record_check(actual,expected,path='audit'):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual)==set(expected),path+' keys')
        for k,v in expected.items():record_check(actual[k],v,path+'/'+k)
    elif type(expected) is list:
        require(type(actual) is list and len(actual)==len(expected),path+' list')
        for i,(a,b) in enumerate(zip(actual,expected)):record_check(a,b,path+'/'+str(i))
    else:require(type(actual) is type(expected) and actual==expected,path+' exact scalar')


def verify_features(meta,dataset,built):
    require(type(built) is dict and set(built)=={'arrays','audit'},'Build schema')
    actual=checked_arrays(built['arrays']);expected=independent_features(meta,dataset)
    for key in KEYS:
        require(actual[key].shape==expected['arrays'][key].shape and np.array_equal(actual[key],expected['arrays'][key],equal_nan=True),'Independent exact feature differs: '+key)
    record_check(built['audit'],expected['audit'])
    n,width=actual['common_available'].shape;count=int(actual['common_available'].sum())
    return dict(passed=True,events=n,width=width,symbols_per_event=width,checked_cells=n*width,available_cells=count,unavailable_cells=n*width-count,
        independent_scalar_rebuild=True,exact_common_availability=True,latest_report_selected_before_validity=True,
        no_older_report_fallback=True,current_annual_only=True,same_report_assets_checked=True,
        exact_rational_arithmetic_checked=True,cash_ppe_scope_checked=True,no_future_or_outcome_inputs=True,audit_checked=True,
        both_admissions_checked=True,original_net_income_gate_preserved=True,statement_currency_binding_checked=True,
        new_model_fits=0,predictive_model_fits=0,computes_returns=False,portfolio_returns_calculated=False)


def independent_deltas(values,eligibility,dose=DOSE):
    require(type(dose) in (int,float) and dose==DOSE,'Fixed dose .01');a=checked_arrays(values);mask=a['common_available']
    eligible=np.asarray(eligibility)
    require(eligible.dtype==bool and eligible.shape==mask.shape,'Original eligibility axis')
    deltas={arm:np.zeros(mask.shape) for arm in RAW}
    for i,j in zip(*np.nonzero(mask & eligible)):
        for arm,key in RAW.items():deltas[arm][i,j]=dose*float(a[key][i,j])
    return deltas


def verify_deltas(values,deltas,eligibility,dose=DOSE):
    expected=independent_deltas(values,eligibility,dose)
    require(type(deltas) is dict and set(deltas)==set(RAW),'Delta arms')
    for arm,wanted in expected.items():
        got=np.asarray(deltas[arm]);require(got.shape==wanted.shape and np.issubdtype(got.dtype,np.floating)
            and np.isfinite(got).all() and np.array_equal(got,wanted) and np.all(np.abs(got)<=DOSE),'Delta exact mapping differs: '+arm)
    return dict(passed=True,dose=DOSE,max_absolute_delta=DOSE,checked_events=expected[ARMS[1]].shape[0],
        common_available_only=True,direct_linear_mapping=True,eligibility_preserved=True,eligible_only=True,
        no_cross_sectional_ranking=True,no_new_availability_filter=True,
        changed_cells={arm:int(np.count_nonzero(v)) for arm,v in expected.items()})
