"""New CAPEX193 source-only gate; preserves every original financial event.

This module does not load market prices, compute factors, train, or trade.
Old source-extraction runtime failures are not reclassified as passes.
"""
from copy import deepcopy
from datetime import date
from bisect import bisect_right
from decimal import Decimal
import hashlib
import re

DEFINITION = 'current_annual_cash_ppe_outflow_v1'
META = ('event_id', 'symbol', 'release_date', 'available_from')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def indexed(rows, ids):
    require(isinstance(rows, list), 'Record list required')
    require(all(isinstance(r, dict) and isinstance(r.get('event_id'), str) for r in rows), 'Event IDs required')
    out = {r['event_id']: r for r in rows}
    require(len(out) == len(rows) and set(out) == set(ids), 'Exact unique source census required')
    return out


def amount(text):
    require(isinstance(text, str) and re.fullmatch(r'-?\d+(?:\.\d+)?', text), 'Explicit decimal string required')
    d = Decimal(text)
    require(d.is_finite(), 'Finite amount required')
    return d


def check_record(original, target, producer, independent, sessions):
    """Checks metadata and source cells only; no financial ratios or scoring."""
    for key in META:
        require(producer.get(key) == original.get(key), 'Producer metadata changed: ' + key)
    eid = original['event_id']
    require(target['event_id'] == independent.get('event_id') == eid, 'Event binding mismatch')
    require(producer.get('inherited_admitted') is original['cash_quality_admission']['admitted'], 'Inherited gate changed')
    require(target['inherited_admitted'] is original['cash_quality_admission']['admitted'], 'Scope inherited gate changed')
    require(producer.get('source') == target['source'], 'Scope source changed')
    require(target['inherited_available_from'] == original['available_from'], 'Scope clock changed')
    require(independent.get('source_sha256') == target['source']['sha256'], 'Independent source binding absent')
    require(independent.get('review_passed') is True, 'Completed independent event review required')
    require(producer.get('status') in ('verified', 'held'), 'Unknown producer status')
    require(type(producer.get('extraction_verified')) is bool, 'Explicit extraction result required')
    require(isinstance(producer.get('hold_reasons'), list), 'Explicit hold list required')
    require(sessions == sorted(set(sessions)), 'Sorted unique session calendar required')
    released, available = original['release_date'], original['available_from']
    for d in (released, available):require(date.fromisoformat(d).isoformat() == d, 'Canonical clock dates required')
    ix = bisect_right(sessions, released)
    require(ix < len(sessions) and sessions[ix] == available, 'Not first session after release')
    reasons = []
    if not original['cash_quality_admission']['admitted']:
        reasons.append('preserved_original_cash_quality_hold')
    field = producer.get('field', producer.get('capex'))
    if field is None:
        require(producer['extraction_verified'] is False and producer['hold_reasons'], 'Missing cell must remain an explained hold')
        require(independent.get('cash_ppe_verified') is False, 'Independent/producer missing cell disagreement')
        reasons.extend(['cash_ppe_missing_or_ambiguous'] + producer['hold_reasons'])
    else:
        require(producer['extraction_verified'] is True and independent.get('cash_ppe_verified') is True, 'Cell not independently verified')
        signed = amount(field['signed_value_decimal'])
        positive = amount(field['positive_outflow_decimal'])
        scale = amount(field['scale'])
        require(signed <= 0 and positive >= 0 and positive == -signed and scale > 0, 'Cash outflow sign/scale mismatch')
        currency = field.get('currency_literal')
        require(field.get('statement_currency_literal') in (None, '$'), 'Conflicting statement currency')
        if currency is None:
            cfo = original['fields']['current']['operating_cash_flow']
            loc = field['locator']; original_loc = None if cfo is None else cfo.get('locator')
            require(field.get('statement_currency_literal') == '$' and isinstance(original_loc, dict) and loc.get('original_cfo_locator') == original_loc, 'Cash statement currency evidence absent')
            require(cfo['currency_literal'] == '$' and loc.get('format') == original_loc.get('format'), 'Cash statement currency context differs')
            if loc.get('format') == 'html':
                require(field.get('currency_scope') == 'same formal cash-flow table; ISO unverified' and type(loc.get('table_index_zero_based')) is int and loc['table_index_zero_based'] >= 0 and loc['table_index_zero_based'] == original_loc.get('table_index_zero_based') and loc.get('statement_segment') == original_loc.get('statement_segment'), 'Cash statement table differs')
            elif loc.get('format') == 'pdf':
                require(field.get('currency_scope') == 'same formal cash-flow page; ISO unverified' and type(loc.get('page_1based')) is int and loc['page_1based'] > 0 and loc['page_1based'] == original_loc.get('page_1based'), 'Cash statement page differs')
            else:
                raise ValueError('Cash statement format absent')
            currency = field['statement_currency_literal']
        require(currency == '$' and field.get('currency_iso') in (None, 'USD'), 'Cash currency mismatch')
        require(isinstance(field.get('signed_literal'), str) and field['signed_literal'], 'Displayed signed literal absent')
        require(hashlib.sha256(field['row_text'].encode()).hexdigest() == field['row_text_sha256'], 'Row hash mismatch')
        loc = field['locator']
        label = field.get('row_label', loc.get('row_label', ''))
        require(bool(re.search(r'propert\w*.*(?:plant|equipment)', label, re.I)), 'Explicit PP&E label required')
        require(not re.search(r'intangible|acquisitions?(?! of property)|principal|finance lease|net investing|free cash flow', label, re.I), 'Excluded mixed or noncash scope')
        require(loc['period_end'] == original['periods']['current']['period_end'], 'Annual period mismatch')
        if 'source' in field:require(field['source'] == target['source'], 'Cell source changed')
        if 'period' in field:require(field['period'] == original['periods']['current'], 'Cell annual duration changed')
        require(independent.get('positive_outflow_decimal') == field['positive_outflow_decimal'], 'Independent cash amount differs')
        require(independent.get('signed_value_decimal') == field['signed_value_decimal'], 'Independent signed amount differs')
        require(independent.get('scale') == field['scale'] and independent.get('annual_period_end') == loc['period_end'], 'Independent unit/annual period differs')
        cfo = original['fields']['current']['operating_cash_flow']
        if cfo is not None:
            require(cfo['source'] == target['source'], 'Paired CFO release differs')
            require(amount(cfo['scale']) == scale and cfo['currency_literal'] == currency, 'Paired CFO unit differs')
            require(independent.get('paired_cfo_verified') is True, 'Paired CFO current-source review absent')
        elif original['cash_quality_admission']['admitted']:
            raise ValueError('Admitted legacy report has missing CFO')
        if original['cash_quality_admission']['admitted']:
            require(independent.get('paired_assets_verified') is True, 'Paired annual assets current-source review absent')
        if producer['hold_reasons']:
            reasons.extend(producer['hold_reasons'])
    reasons = list(dict.fromkeys(reasons))
    require(bool(reasons) == (producer['status'] == 'held'), 'Producer status/reasons disagree')
    result = deepcopy(original)
    normalized = deepcopy(field)
    if normalized is not None:
        normalized.setdefault('source', deepcopy(target['source']))
        normalized.setdefault('period', deepcopy(original['periods']['current']))
        normalized.setdefault('available_from', original['available_from'])
        require(normalized['source'] == target['source'] and normalized['period'] == original['periods']['current'] and normalized['available_from'] == original['available_from'], 'Normalized cell evidence mismatch')
        require(normalized.get('fact_admitted', False) is False, 'Raw field admission changed')
        normalized['fact_admitted'] = False
    result['cash_ppe'] = normalized
    result['cash_ppe_source_evidence'] = deepcopy(producer)
    result['cash_ppe_admission'] = {'admitted':not reasons, 'reasons':reasons, 'definition':DEFINITION}
    require({k:v for k,v in result.items() if k not in ('cash_ppe','cash_ppe_source_evidence','cash_ppe_admission')} == original, 'Original event was changed')
    return result
