"""Causal conditional-report selection and pure ROA-stability features.

No I/O, fitting or portfolio outcomes. Actual invocation requires host PLAN.
Source metadata uses the registered document-date/next-session assumption;
this module does not certify first-publication vintage or missing corrections.
"""
from bisect import bisect_right
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import numpy as np
import roa_stability_preflight_math as arithmetic

METHOD = 'as_reported_original_NI_assets_three_annual_reports_v1'
FIELDS = ('annual_net_income', 'opening_total_assets', 'closing_total_assets')
COUNT_FIELDS = ('checked_cells', 'available_cells', 'unavailable_cells',
                'annual_ratio_evaluations', 'mean_estimates', 'sample_dispersion_estimates', 'bounded_scores')
VERIFIED = 'verified_calendar_rule_only'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def day(value):
    require(type(value) is str, 'Canonical date string required')
    result = date.fromisoformat(value)
    require(result.isoformat() == value, 'Canonical ISO day required')
    return result


def visible(row, cutoff, *, catalog=False):
    if cutoff is None:
        return False
    if catalog:
        if row.get('clock_status') == VERIFIED:
            return day(row['assumed_available_session']) <= day(cutoff)
        publication = row.get('publication_date')
        # Unknown blocker timing cannot establish historical availability.
        # Retain it conservatively for a subsequent explicit unavailable result.
        return publication is None or day(publication) < day(cutoff)
    known = row.get('known_at')
    return known is None or day(known) <= day(cutoff)


def truncate_sources(dataset, cutoff):
    """Remove future catalog/value/evidence/chain records, retaining static pins."""
    result = dict(dataset)
    for key in ('catalog', 'reports', 'edges', 'chains'):
        result[key] = [deepcopy(r) for r in dataset[key]
                       if visible(r, cutoff, catalog=(key == 'catalog'))]
    return result


def unique_index(rows, key):
    ids = [r[key] for r in rows]
    require(len(ids) == len(set(ids)) and all(type(x) is str and x for x in ids), 'Duplicate/invalid '+key)
    return {r[key]: r for r in rows}


def unavailable(symbol, signal, reason, latest=None):
    return dict(symbol=symbol, date=signal, available=False, reason=reason,
        latest_event_id=latest, report_version_ids=[], edge_version_ids=[], chain_version_id=None)


def exact_decimal(text):
    require(type(text) is str, 'Reported decimal string required')
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError('Invalid reported decimal') from exc
    require(number.is_finite(), 'Finite source amount required')
    return Fraction(number)


def _clock(row, calendar):
    require(row.get('clock_status') == VERIFIED, 'unverified_report_clock')
    publication, available = day(row['publication_date']), day(row['assumed_available_session'])
    require(day(calendar[0]) <= publication < day(calendar[-1]), 'publication_outside_calendar')
    index = bisect_right(calendar, publication.isoformat())
    require(index < len(calendar) and calendar[index] == available.isoformat(), 'wrong_next_session')
    require(row.get('clock', {}).get('next_session_rule_verified') is True,
            'missing_clock_receipt')


def _report_amounts(report, catalog, signal, calendar, latest):
    _clock(catalog, calendar)
    require(catalog.get('original_field_state') == 'admitted' and
            report.get('inherited_single_report_fields_admitted') is True, 'held_report')
    require(report['report_version_id'] == catalog['report_version_id'] and
            report['event_id'] == catalog['event_id'] and
            report['issuer_id'] == catalog['issuer_id'] and report['security_id'] == catalog['security_id'] and
            report['fiscal_year'] == catalog['fiscal_year'], 'report_catalog_binding')
    available = day(report['known_at'])
    require(report['known_at'] == report['available_from'] == catalog['assumed_available_session'] and
            available <= day(signal), 'report_evidence_clock')
    period = report['period']; start, end = day(period['period_start']), day(period['period_end'])
    duration = (end-start).days + 1
    require(330 <= duration <= 400 and period['days_inclusive'] == duration and
            catalog['period_start'] == start.isoformat() and catalog['period_end'] == end.isoformat(),
            'report_period')
    require(end <= day(catalog['publication_date']) < available, 'report_release_period')
    age_day = day(signal) if latest else available
    require(0 <= (age_day-end).days <= 550, 'report_age')
    require(report.get('corrected_original_values_substituted') is False, 'replaced_original_values')
    source = report['source']; require(source.get('path') and source.get('sha256'), 'original_source_required')
    require(set(report['fields']) == set(FIELDS), 'nine_field_schema')
    amounts, currencies = [], []
    for name in FIELDS:
        field = report['fields'][name]; cell = field['source_cell']
        expected = start-timedelta(days=1) if name == 'opening_total_assets' else end
        require(field['as_of'] == expected.isoformat() and
                cell.get('as_of', expected.isoformat()) == expected.isoformat(), 'field_period_endpoint')
        require(cell['source']['path'] == source['path'] and
                cell['source']['sha256'] == source['sha256'], 'field_original_source')
        require(cell.get('available_from', available.isoformat()) == available.isoformat(), 'field_clock')
        require(field.get('original_field_path') and cell.get('locator'), 'field_locator')
        scale = exact_decimal(cell['scale']); require(scale > 0, 'positive_scale')
        value = exact_decimal(cell['value_decimal'])*scale
        if name != 'annual_net_income': require(value > 0, 'positive_asset_endpoint')
        if name == 'annual_net_income':
            require(field.get('ni_attribution') == 'reported_consolidated_total_NI_before_NCI_deduction', 'NI_attribution')
            ni_period = cell.get('period')
            if ni_period is not None:
                require(ni_period.get('period_start') == start.isoformat() and
                        ni_period.get('period_end') == end.isoformat(), 'NI_annual_period')
        require(cell['currency_literal'] == '$', 'currency_literal')
        currencies.append(cell.get('currency_iso')); amounts.append(value)
    require(len(set(currencies)) == 1, 'same_report_currency')
    return tuple(amounts), currencies[0]


def select_chain(dataset, symbol, signal, calendar):
    """No ratios here. Full known catalog chooses the latest fiscal anchor first."""
    out = unavailable(symbol, signal, 'no_known_annual_event')
    registered = [x for x in dataset['identity_registry'] if symbol in x['symbol_aliases']]
    if not registered: return unavailable(symbol, signal, 'unsupported_issuer')
    if len(registered) != 1: return unavailable(symbol, signal, 'ambiguous_issuer_registry')
    identity = (registered[0]['issuer_id'], registered[0]['security_id'])
    known = [x for x in dataset['catalog'] if x['canonical_symbol'] == symbol and visible(x, signal, catalog=True)]
    unique_index(known, 'event_id')
    if not known: return out
    if any(x.get('publication_date') is None for x in known):
        return unavailable(symbol, signal, 'unknown_catalog_timing')
    if any(type(x.get('fiscal_year')) is not int for x in known):
        return unavailable(symbol, signal, 'unknown_fiscal_anchor')
    originals = [x for x in known if x.get('kind') == 'original_annual_event']
    corrections = [x for x in known if x.get('kind') == 'correction']
    if len(originals)+len(corrections) != len(known):
        return unavailable(symbol, signal, 'unknown_event_kind')
    if not originals: return out
    anchors=[(x['fiscal_year'],day(x['period_end'])) for x in originals if x.get('period_end') is not None]
    if any(fa < fb and ea >= eb for fa,ea in anchors for fb,eb in anchors):
        return unavailable(symbol,signal,'inconsistent_fiscal_anchors')
    latest_fy = max(x['fiscal_year'] for x in originals)
    latest_rows = [x for x in originals if x['fiscal_year'] == latest_fy]
    if len(latest_rows) != 1: return unavailable(symbol, signal, 'ambiguous_latest_fiscal_anchor')
    anchor = latest_rows[0]; latest = anchor['event_id'];out['latest_event_id'] = latest
    if anchor.get('clock_status') != VERIFIED:
        return unavailable(symbol, signal, 'latest_clock_unverified', latest)
    if (anchor.get('issuer_id'), anchor.get('security_id')) != identity:
        return unavailable(symbol, signal, 'latest_identity_unverified', latest)
    if anchor.get('original_field_state') != 'admitted' or not anchor.get('report_version_id'):
        return unavailable(symbol, signal, 'latest_report_held_or_unprocessed', latest)
    if anchor.get('period_end') is None:
        return unavailable(symbol, signal, 'latest_period_unknown', latest)
    if not 0 <= (day(signal)-day(anchor['period_end'])).days <= 550:
        return unavailable(symbol, signal, 'latest_report_stale', latest)
    annual = []
    for fy in range(latest_fy-2, latest_fy+1):
        matches = [x for x in originals if x['fiscal_year'] == fy]
        if len(matches) != 1:
            return unavailable(symbol, signal, 'missing_or_ambiguous_preceding_year', latest)
        row=matches[0]
        if (row.get('issuer_id'),row.get('security_id')) != identity:
            return unavailable(symbol, signal, 'mixed_or_unknown_issuer', latest)
        annual.append(row)
    ids = [x.get('report_version_id') for x in annual]
    if None in ids or len(set(ids)) != 3:
        return unavailable(symbol, signal, 'three_distinct_originals_required', latest)
    events = {x['event_id'] for x in annual}
    for correction in corrections:
        target = correction.get('corrects_report_id'); event = correction.get('corrects_event_id')
        if not target and not event:
            return unavailable(symbol, signal, 'unlinked_known_correction', latest)
        if target in ids or event in events:
            return unavailable(symbol, signal, 'known_correction_requires_new_chain_review', latest)
    reports = unique_index([x for x in dataset['reports'] if visible(x,signal)],'report_version_id')
    if not all(x in reports for x in ids):
        return unavailable(symbol, signal, 'report_version_not_known', latest)
    chains = [x for x in dataset['chains'] if x['report_version_ids'] == ids and visible(x,signal)]
    if len(chains) != 1:
        return unavailable(symbol, signal, 'missing_or_ambiguous_chain_evidence', latest)
    chain = chains[0]
    if chain.get('conditional_allowed') is not True or chain.get('known_at') is None:
        return unavailable(symbol, signal, 'chain_not_conditionally_admitted', latest)
    if (chain['issuer_id'],chain['security_id']) != identity:
        return unavailable(symbol, signal, 'chain_identity_mismatch', latest)
    if chain['fiscal_years'] != list(range(latest_fy-2,latest_fy+1)):
        return unavailable(symbol, signal, 'chain_fiscal_years_mismatch', latest)
    edges = unique_index([x for x in dataset['edges'] if visible(x,signal)], 'edge_version_id')
    edge_ids = chain['edge_version_ids']
    if len(edge_ids) != 2 or not all(x in edges for x in edge_ids):
        return unavailable(symbol, signal, 'edge_version_not_known', latest)
    for i, eid in enumerate(edge_ids):
        edge=edges[eid]
        if edge.get('conditional_allowed') is not True or edge.get('known_at') is None:
            return unavailable(symbol, signal, 'edge_not_conditionally_admitted', latest)
        if any(e.get('supersedes_version_id') == eid for e in edges.values()):
            return unavailable(symbol, signal, 'edge_version_superseded', latest)
        if ((edge['issuer_id'],edge['security_id']) != identity or
            edge['earlier_report_version_id'] != ids[i] or edge['later_report_version_id'] != ids[i+1]):
            return unavailable(symbol, signal, 'edge_binding_mismatch', latest)
        refs=edge.get('evidence_refs')
        if not isinstance(refs,list) or len(refs)<2:
            return unavailable(symbol,signal,'edge_original_evidence_binding',latest)
        for identifier in ids[i:i+2]:
            linked=[r for r in refs if r.get('report_version_id')==identifier]
            original=reports[identifier]
            if (len(linked)!=1 or linked[0].get('event_id')!=original['event_id'] or
                linked[0].get('source',{}).get('path')!=original['source']['path'] or
                linked[0].get('source',{}).get('sha256')!=original['source']['sha256'] or
                linked[0].get('assumed_available_from')!=original['known_at']):
                return unavailable(symbol,signal,'edge_original_evidence_binding',latest)
        if any(not r.get('source',{}).get('path') or not r.get('source',{}).get('sha256') for r in refs):
            return unavailable(symbol,signal,'edge_original_evidence_binding',latest)
        clocks=[reports[ids[i]]['known_at'], reports[ids[i+1]]['known_at']]
        clocks += [r.get('assumed_available_from') for r in refs]
        try:
            parsed=[day(x) for x in clocks]
            valid_clock=bool(parsed) and day(edge['known_at'])>=max(parsed) and day(chain['known_at'])>=day(edge['known_at'])
        except (TypeError,ValueError):
            valid_clock=False
        if not valid_clock:
            return unavailable(symbol, signal, 'edge_evidence_clock', latest)
    try:
        amounts, currencies = [], []
        for i, (identifier,row) in enumerate(zip(ids,annual)):
            report=reports[identifier]
            if any(r.get('supersedes_version_id') == identifier for r in reports.values()):
                return unavailable(symbol,signal,'report_version_superseded',latest)
            amount,currency = _report_amounts(report,row,signal,calendar,i==2)
            amounts.append(amount);currencies.append(currency)
            if i:
                require(day(report['period']['period_start']) == day(reports[ids[i-1]]['period']['period_end'])+timedelta(days=1), 'annual_boundaries_not_contiguous')
        require(len({x for x in currencies if x is not None}) <= 1, 'cross_report_currency')
        refs={(r['report_version_id'],r['field']):r for r in chain['nine_required_fields']}
        require(len(chain['nine_required_fields']) == len(refs) == 9, 'nine_field_bindings')
        for identifier in ids:
            report=reports[identifier]
            for name in FIELDS:
                ref=refs[(identifier,name)]
                require(ref['source_sha256'] == report['source']['sha256'] and
                        ref['event_id'] == report['event_id'] and
                        ref['original_field_path'] == report['fields'][name]['original_field_path'], 'nine_field_binding_mismatch')
    except (KeyError, TypeError, ValueError) as exc:
        return unavailable(symbol,signal,'invalid_required_source_field:'+str(exc),latest)
    out.update(available=True,reason='available',report_version_ids=ids,edge_version_ids=edge_ids,
               chain_version_id=chain['chain_version_id'])
    return out, amounts


def build_features(dataset, meta, calendar):
    require(dataset.get('method') == METHOD and dataset.get('conditional_research_only') is True,
            'Exact conditional source policy required')
    require(calendar and calendar == sorted(set(calendar)), 'Sorted unique calendar')
    for x in calendar: require(day(x).weekday()<5, 'Weekday sessions required')
    dates,symbols = meta['dates'],meta['symbols']
    require(isinstance(meta['etfs'],(list,tuple,set,frozenset)) and all(type(x) is str and x and x.strip()==x for x in meta['etfs']), 'ETF identity labels')
    etfs=set(meta['etfs']);require(len(etfs)==len(meta['etfs']), 'Unique ETF identities')
    require(len(dates) == len(symbols) and dates == sorted(set(dates)), 'Ordered decision metadata')
    require(all(isinstance(row,(list,tuple)) and row and all(type(x) is str and x and x.strip()==x for x in row) for row in symbols), 'Candidate identity labels')
    width=len(symbols[0]) if dates else 0
    require(all(len(row)==width and len(row)==len(set(row)) for row in symbols), 'Fixed unique candidate axes')
    values={k:np.full((len(dates),width),np.nan) for k in ('candidate','control')}
    values['available']=np.zeros((len(dates),width),bool)
    history=[]
    for i, (signal, names) in enumerate(zip(dates,symbols)):
        require(signal in calendar,'Decision outside calendar')
        index=bisect_right(calendar,signal)-1
        if not (i==0 and signal=='2016-08-17'):
            require(index+1<len(calendar) and calendar[index+1][:7] != signal[:7], 'Official month-end required')
        cells=[]
        for j,symbol in enumerate(names):
            result=unavailable(symbol,signal,'ETF_no_financial_feature') if symbol in etfs else select_chain(dataset,symbol,signal,calendar)
            if isinstance(result,tuple):
                cell,amounts=result
                measured=arithmetic.measure(amounts)
                values['candidate'][i,j]=measured['candidate'];values['control'][i,j]=measured['control']
                values['available'][i,j]=True
                cell['measurement']=dict(ratios_exact=[str(x) for x in measured['roa']],
                    mean_exact=str(measured['mean']),variance_exact=str(measured['sample_variance']),
                    sample_std=measured['sample_std'],candidate=measured['candidate'],control=measured['control'])
            else: cell=result
            cells.append(cell)
        history.append(dict(date=signal,symbols=list(names),available_count=sum(c['available'] for c in cells),cells=cells))
    count=int(values['available'].sum());total=len(dates)*width
    audit=dict(schema_version=1,events=len(dates),symbols_per_event=width,checked_cells=total,
        available_cells=count,unavailable_cells=total-count,annual_ratio_evaluations=3*count,
        mean_estimates=count,sample_dispersion_estimates=count,bounded_scores=2*count,
        new_model_fits=0,predictive_model_fits=0,event_history=history)
    return values,audit
