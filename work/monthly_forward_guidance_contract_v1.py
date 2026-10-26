"""Pure source-contract validation for next-quarter reported revenue guidance.

No raw-document parser, price loader, score, return or strategy admission.
Booleans asserted by extractors require independent source verification.
"""
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
import re
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')
WINDOW = 21
METRIC = 'reported_total_revenue'
SCOPE = 'consolidated_reported_revenue'
BASIS = 'reported'
AMOUNT_FIELDS = ('point', 'low', 'high', 'center', 'spread')
KINDS = {'point': ('point',), 'closed_range': ('low', 'high'),
         'plus_minus_amount': ('center', 'spread'), 'plus_minus_percent': ('center', 'spread')}


def source_files():
    here = Path(__file__).resolve()
    return [here, here.with_name('test_monthly_forward_guidance_contract_v1.py'),
            here.parent.parent/'outputs/monthly_forward_guidance_preflight_v1/CONTRACT.json']


def _decimal(value):
    if not isinstance(value, str) or re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?', value) is None:
        raise ValueError('Amounts require finite unsigned decimal strings')
    if len(value.replace('.', '')) > 40:
        raise ValueError('More than40 decimal digits is not supported')
    return Decimal(value)


def _text(value):
    out = format(value, 'f')
    return out.rstrip('0').rstrip('.') if '.' in out else out


def normalize_amount(amount):
    """Normalize a printed point/closed interval, not a forecast growth score."""
    if not isinstance(amount, dict) or amount.get('kind') not in KINDS:
        raise ValueError('Unknown amount representation')
    if set(amount) - {'kind', 'unit_scale', *AMOUNT_FIELDS}:
        raise ValueError('Unexpected amount fields')
    scale = amount.get('unit_scale')
    if scale not in ('1', '1000', '1000000', '1000000000'):
        raise ValueError('Explicit supported unit scale required')
    kind = amount['kind']; needed = KINDS[kind]
    if any(amount.get(k) is not None for k in AMOUNT_FIELDS if k not in needed):
        raise ValueError('Conflicting amount representations')
    values = {k: _decimal(amount.get(k)) for k in needed}
    with localcontext() as context:
        context.prec = 100
        if kind == 'point': low = high = values['point']
        elif kind == 'closed_range': low, high = values['low'], values['high']
        elif kind == 'plus_minus_amount': low, high = values['center']-values['spread'], values['center']+values['spread']
        else:
            offset = values['center']*values['spread']/Decimal(100)
            low, high = values['center']-offset, values['center']+offset
        if not Decimal(0) < low <= high:
            raise ValueError('Positive ordered closed interval required')
        low *= Decimal(scale); high *= Decimal(scale)
        middle = (low+high)/Decimal(2)
    return {'kind': kind, 'lower': _text(low), 'upper': _text(high), 'midpoint': _text(middle),
            'unit_scale_applied': scale, 'factor_computed': False}


def _day(value):
    if not isinstance(value, str) or re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is None:
        raise ValueError('ISO calendar date required')
    return date.fromisoformat(value)


def _instant(value):
    if not isinstance(value, str): raise ValueError('Aware timestamp string required')
    out = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if out.tzinfo is None or out.utcoffset() is None: raise ValueError('Timestamp lacks offset')
    return out.astimezone(timezone.utc)


def clock_info(clock):
    """Original release day and independent version bound; unknown is not zero."""
    reasons=[]; pub=None; instant=None; versions=[]
    if not isinstance(clock, dict): clock={}
    try: pub=_day(clock.get('visible_date'))
    except (ValueError, TypeError): reasons.append('missing_or_invalid_publication_date')
    if clock.get('publication_date_verified') is not True: reasons.append('publication_date_not_verified')
    kind=clock.get('time_kind')
    if kind not in ('aware_release', 'date_only'): reasons.append('publication_precision_unresolved')
    raw=clock.get('published_at')
    if kind=='aware_release' and raw is None: reasons.append('aware_publication_missing')
    if kind=='date_only' and raw is not None: reasons.append('date_only_must_not_invent_timestamp')
    if raw is not None:
        try:
            instant=_instant(raw)
            if pub is not None and instant.astimezone(NY).date()!=pub: reasons.append('publication_date_timestamp_conflict')
        except (ValueError, TypeError, OverflowError): reasons.append('invalid_publication_timestamp')
    for key in ('modified_at', 'known_version_not_before'):
        if clock.get(key) is not None:
            try: versions.append(_instant(clock[key]))
            except (ValueError, TypeError, OverflowError): reasons.append('unresolved_'+key)
    state=clock.get('version_evidence_status')
    if state not in ('checked_no_known_late_version','known_version_bound'): reasons.append('version_evidence_unresolved')
    if state=='known_version_bound' and not versions: reasons.append('version_bound_missing')
    if state=='checked_no_known_late_version' and pub is not None and any(v.astimezone(NY).date()>pub for v in versions):
        reasons.append('known_late_version_misclassified')
    if type(clock.get('modification_history_complete')) is not bool: reasons.append('modification_history_status_missing')
    version_day=max([pub]+[v.astimezone(NY).date() for v in versions]) if pub is not None else None
    # A dated but unresolved version remains a barrier in select_latest_observed.
    dated=pub is not None and clock.get('publication_date_verified') is True
    publication_errors=[r for r in reasons if r.startswith(('missing_or_invalid_publication','publication_','aware_publication','date_only_','invalid_publication'))]
    if publication_errors: dated=False
    return {'eligible':not reasons,'publication_date':pub.isoformat() if dated else None,
            'publication_instant':instant.isoformat() if dated and instant else None,
            'version_date':version_day.isoformat() if version_day is not None and not reasons else None,
            'modification_history_complete':clock.get('modification_history_complete') is True,
            'source_time_kind':kind,'known_version_instants':sorted({v.isoformat() for v in versions}),
            'reasons':sorted(set(reasons))}


def _source_reasons(source):
    if not isinstance(source,dict):return ['source_missing']
    reasons=[]
    for k in ('path','url','source_class'):
        if not isinstance(source.get(k),str) or not source[k].strip():reasons.append('source_'+k+'_missing')
    if re.fullmatch('[0-9a-f]{64}',str(source.get('sha256',''))) is None:reasons.append('source_hash_missing')
    refs=source.get('receipt_refs')
    if not isinstance(refs,list) or not refs or any(not isinstance(x,dict) or not x.get('path') or re.fullmatch('[0-9a-f]{64}',str(x.get('sha256',''))) is None for x in refs):
        reasons.append('source_receipt_refs_missing')
    for k in ('identity_verified','receipt_verified','body_complete'):
        if source.get(k) is not True:reasons.append('source_'+k+'_not_verified')
    return reasons


def _period_key(period):
    if not isinstance(period,dict) or period.get('quarter_identity_verified') is not True:
        raise ValueError('Quarter identity not verified')
    y,q=period.get('fiscal_year'),period.get('fiscal_quarter')
    if type(y) is not int or not 1900<=y<=2200 or type(q) is not int or not 1<=q<=4:
        raise ValueError('Fiscal year/quarter missing')
    if not isinstance(period.get('label'),str) or not period['label'].strip():raise ValueError('Printed fiscal label required')
    start,end=period.get('start'),period.get('end')
    if start is not None:_day(start)
    if end is not None:_day(end)
    if start is not None and end is not None and start>end:raise ValueError('Inverted fiscal dates')
    return y,q


def _target_reasons(obj):
    reasons=[]
    if not isinstance(obj,dict):return ['target_missing']
    for key,value in (('metric',METRIC),('scope',SCOPE),('accounting_basis',BASIS)):
        if obj.get(key)!=value:reasons.append('unsupported_'+key)
    if re.fullmatch('[A-Z]{3}',str(obj.get('currency',''))) is None:reasons.append('currency_not_verified')
    for key in ('scope_verified','currency_verified','amount_verified'):
        if obj.get(key) is not True:reasons.append(key+'_missing')
    for key in ('quote','locator'):
        if not isinstance(obj.get(key),str) or not obj[key].strip():reasons.append(key+'_missing')
    return reasons


def _actual_asof_reasons(obj,event,actual_clock,event_clock):
    """Prove the original announcement information set, never just same-day order.

    Same-byte evidence shares the event clock and downstream version gate. Distinct
    evidence must have been known by the original announcement, not next session.
    """
    if not actual_clock['eligible'] or not event_clock['eligible']:
        return ['original_publication_information_set_unverified']
    actual_source=obj.get('source') if isinstance(obj.get('source'),dict) else {}
    event_source=event.get('source') if isinstance(event.get('source'),dict) else {}
    same_bytes=(actual_source.get('sha256') is not None and actual_source.get('sha256')==event_source.get('sha256'))
    if same_bytes:
        return [] if actual_clock==event_clock else ['same_source_clock_conflict']
    event_day=event_clock['publication_date']
    last_day=max(actual_clock['publication_date'],actual_clock['version_date'])
    if last_day<event_day:return []
    if last_day>event_day:return ['actual_not_known_by_event_date']
    # At least one required evidence bound is on the announcement day. An exact
    # guide clock is indispensable; a same-day date-only actual cannot precede it.
    guide=event_clock['publication_instant']
    actual=actual_clock['publication_instant']
    if guide is None or (actual_clock['publication_date']==event_day and actual is None):
        return ['same_day_actual_order_unproved']
    instants=actual_clock['known_version_instants'][:]
    if actual is not None:instants.append(actual)
    if any(_instant(x)>_instant(guide) for x in instants):
        return ['actual_after_guidance_instant']
    return []


def _actual(slot,obj,expected,event,event_clock,forecast):
    reasons=[]; normalized=None
    if not isinstance(obj,dict):return None,[slot+':missing']
    reasons+=_target_reasons(obj);reasons+=_source_reasons(obj.get('source'))
    if obj.get('issuer_id')!=event.get('issuer_id'):reasons.append('issuer_mismatch')
    if obj.get('quarter_only_verified') is not True:reasons.append('not_verified_single_quarter')
    try:
        if _period_key(obj.get('period'))!=expected:reasons.append('wrong_fiscal_quarter')
    except (ValueError,TypeError):reasons.append('fiscal_quarter_unverified')
    if obj.get('currency')!=forecast.get('currency'):reasons.append('currency_mismatch')
    try:
        normalized=normalize_amount(obj.get('amount'))
        if normalized['kind']!='point':reasons.append('actual_not_point')
    except (ValueError,TypeError):reasons.append('invalid_actual_amount')
    c=clock_info(obj.get('clock'));reasons.extend('clock_'+r for r in c['reasons'])
    reasons+=_actual_asof_reasons(obj,event,c,event_clock)
    period=obj.get('period')
    if isinstance(period,dict) and period.get('end') is not None and c['publication_date'] is not None:
        try:
            if _day(period['end'])>_day(c['publication_date']):reasons.append('actual_period_after_publication')
        except (ValueError,TypeError):pass  # Already rejected as an invalid fiscal period.
    required=forecast.get('required_prior_basis_id') if slot=='target_prior_year' and forecast.get('comparison_basis')=='explicit_adjusted_prior_base' else 'as_reported'
    if not required or obj.get('basis_id')!=required:reasons.append('comparison_basis_mismatch')
    return normalized,[slot+':'+r for r in reasons]


def assess_event(event):
    """Source readiness only. All missing/invalid evidence keeps flags false."""
    if not isinstance(event,dict):raise ValueError('Event object required')
    reasons=[]
    if type(event.get('schema_version')) is not int or event.get('schema_version')!=1:reasons.append('schema_version')
    for key in ('event_id','symbol','issuer_id'):
        if not isinstance(event.get(key),str) or not event[key].strip():reasons.append(key+'_missing')
    if event.get('event_kind')!='earnings_release' or event.get('earnings_event_verified') is not True:reasons.append('earnings_event_not_verified')
    reasons+=_source_reasons(event.get('source'))
    clock=clock_info(event.get('clock'));reasons+=['clock:'+r for r in clock['reasons']]
    if clock['publication_date'] is not None and not '2016-01-01'<=clock['publication_date']<='2026-08-31':
        reasons.append('outside_fixed_forecast_window')
    forecast=event.get('forecast') if isinstance(event.get('forecast'),dict) else {}
    if forecast.get('status')!='numeric':reasons.append('latest_earnings_has_no_numeric_target')
    reasons+=_target_reasons(forecast)
    if forecast.get('target_relation_verified') is not True:reasons.append('next_quarter_relation_not_verified')
    if forecast.get('comparison_basis') not in ('as_reported','explicit_adjusted_prior_base'):reasons.append('comparison_basis_unresolved')
    if forecast.get('comparison_basis')=='explicit_adjusted_prior_base' and not forecast.get('required_prior_basis_id'):reasons.append('required_prior_basis_missing')
    if forecast.get('comparison_basis')=='as_reported' and forecast.get('required_prior_basis_id') is not None:reasons.append('conflicting_prior_basis')
    current=target=None
    try:
        current=_period_key(event.get('reported_period'));target=_period_key(forecast.get('target_period'))
        expected=(current[0]+1,1) if current[1]==4 else (current[0],current[1]+1)
        if target!=expected:reasons.append('target_not_next_fiscal_quarter')
        if clock['publication_date'] is not None:
            pub=_day(clock['publication_date'])
            if event['reported_period'].get('end') is not None and _day(event['reported_period']['end'])>pub:
                reasons.append('reported_period_after_publication')
            if forecast['target_period'].get('end') is not None and _day(forecast['target_period']['end'])<=pub:
                reasons.append('target_period_already_ended')
    except (ValueError,TypeError):reasons.append('fiscal_quarter_unverified')
    normalized=None
    try:normalized=normalize_amount(forecast.get('amount'))
    except (ValueError,TypeError):reasons.append('invalid_guidance_amount')
    source_ok=not reasons
    actual_reasons=[];actual_amounts={}
    slots=event.get('actuals') if isinstance(event.get('actuals'),dict) else {}
    if current is not None and target is not None:
        expected={'target_prior_year':(target[0]-1,target[1]),'current_quarter':current,'current_prior_year':(current[0]-1,current[1])}
        for key,period in expected.items():
            actual_amounts[key],errors=_actual(key,slots.get(key),period,event,clock,forecast);actual_reasons+=errors
    else:actual_reasons.append('actuals:target_period_unresolved')
    return {'event_id':event.get('event_id'),'source_eligible':source_ok,'comparison_eligible':source_ok and not actual_reasons,
            'source_reasons':sorted(set(reasons)),'comparison_reasons':sorted(set(actual_reasons)),
            'normalized_forecast':normalized,'normalized_actuals':actual_amounts,'clock':clock,
            'semantic_admission':False,'trading_admission':False,'factor_computed':False}


def select_latest_observed(events,symbol,signal_date,sessions,window=WINDOW):
    """One signal/source decision. Caller must bind the official session list."""
    if window!=WINDOW:raise ValueError('Fixed21-session source window')
    days=[_day(x) for x in sessions]
    if not days or days!=sorted(set(days)):raise ValueError('Unique sorted official sessions required')
    signal=_day(signal_date);pos=bisect_left(days,signal)
    if pos==len(days) or days[pos]!=signal:raise ValueError('Signal must be an official session')
    ids=[e.get('event_id') for e in events]
    if any(not isinstance(x,str) or not x for x in ids) or len(ids)!=len(set(ids)):raise ValueError('Unique event IDs required')
    seen=[];unlocated=[]
    for e in events:
        if e.get('symbol')!=symbol:continue
        c=clock_info(e.get('clock'))
        if c['publication_date'] is None:unlocated.append(e['event_id']);continue
        pub=_day(c['publication_date'])
        if pub<days[0]:raise ValueError('Calendar begins after an observed source')
        first=bisect_right(days,pub)
        if first<=pos:seen.append((pub,c['publication_instant'],first,c,e))
    out={'symbol':symbol,'signal_date':signal_date,'event_id':None,'comparison_available':False,
         'unlocated_event_ids':sorted(unlocated),'unlocated_events_present':bool(unlocated),
         'selection_scope':'latest_located_observed_event_only','complete_event_stream_signal':False,
         'source_corpus_complete':False,'trading_admission':False,'factor_computed':False}
    if not seen:return dict(out,status='no_observed_earnings')
    latest=max(x[0] for x in seen);ties=[x for x in seen if x[0]==latest]
    if len(ties)>1 and all(x[1] is not None for x in ties):
        last=max(_instant(x[1]) for x in ties);ties=[x for x in ties if _instant(x[1])==last]
    if len(ties)!=1:return dict(out,status='ambiguous_same_day_earnings',tied_event_ids=sorted(x[4]['event_id'] for x in ties))
    pub,instant,first,c,e=ties[0];age=pos-first;assessment=assess_event(e)
    out.update(event_id=e['event_id'],publication_age_sessions=age,first_release_session=days[first].isoformat(),assessment=assessment)
    if c['version_date'] is None:return dict(out,status='latest_version_unresolved')
    available=bisect_right(days,_day(c['version_date']))
    out['first_version_session']=days[available].isoformat() if available<len(days) else None
    if available>pos:return dict(out,status='latest_version_not_available')
    if not assessment['comparison_eligible']:return dict(out,status='latest_earnings_not_comparable')
    if age>=WINDOW:return dict(out,status='original_release_outside_window')
    return dict(out,status='observed_comparison_in_window',comparison_available=True)
