"""Pure same-report SBC adjustment with the frozen146 causal eligibility gate.

The caller verifies both source-admission closures. This module performs no
I/O, source admission, model fitting, scoring, ranking or portfolio operations.
"""
from collections import Counter, defaultdict

import numpy as np
import monthly_cash_quality146_factor as old

METHOD = old.METHOD
MEASUREMENT = 'current_annual_stock_compensation_adjusted_cash_flow_v1'
DEFINITION = 'current_annual_reported_stock_compensation_addback_v1'
EXTRA = ('earnings_bounded', 'cash_flow_bounded',
         'comp_adjusted_cash_flow_bounded', 'common_available')
METADATA = ('event_id', 'symbol', 'fiscal_year', 'release_date', 'available_from',
            'identity', 'periods', 'source', 'inherited_rejection_codes',
            'cash_quality_admission')
require = old._require


def _index(events):
    require(type(events) is list, 'Complete event list required')
    indexed = defaultdict(list)
    for event in events:
        require(type(event) is dict and
                all(old._text(event.get(k)) for k in ('event_id', 'symbol')),
                'Declared source event identity required')
        indexed[(event['symbol'], event['event_id'])].append(event)
    return indexed


def _adjusted(original, candidate):
    """Validate the new field gate and return a bounded exact-rational result."""
    admission = candidate.get('stock_compensation_admission')
    require(type(admission) is dict and
            set(admission) == {'admitted', 'reasons', 'definition'},
            'stock_compensation_admission_required')
    require(admission['admitted'] is True, 'current_not_stock_compensation_admitted')
    require(admission['definition'] == DEFINITION, 'wrong_stock_compensation_definition')
    require(type(admission['reasons']) is list and not admission['reasons'],
            'stock_compensation_admission_has_reasons')
    require(all(k in original and k in candidate and candidate[k] == original[k] for k in METADATA),
            'stock_compensation_event_metadata_mismatch')
    if 'current_release_available_from' in candidate:
        require(candidate['current_release_available_from'] == original['available_from'],
                'stock_compensation_event_availability_mismatch')
    result = candidate.get('stock_compensation')
    require(type(result) is dict and result.get('field') == 'stock_compensation_addback',
            'stock_compensation_raw_field_required')
    require(type(result.get('gaps')) is list and not result['gaps'],
            'stock_compensation_raw_field_has_gaps')
    value = result.get('value')
    require(type(value) is dict, 'stock_compensation_value_missing')
    require(value.get('source_sha256') == original['source']['sha256'],
            'stock_compensation_source_sha_mismatch')
    try:
        amount = old._amount(value, original, 'current')
        current_cfo = original['fields']['current']['operating_cash_flow']
        require(old._fraction(value['scale']) == old._fraction(current_cfo['scale']),
                'stock_compensation_cashflow_scale_mismatch')
        require(value['currency_literal'] == current_cfo['currency_literal'],
                'stock_compensation_cashflow_currency_mismatch')
        cash = old._amount(current_cfo, original, 'current')
        current_assets = old._amount(original['fields']['current']['total_assets'], original, 'current')
        prior_assets = old._amount(original['fields']['comparison']['total_assets'], original, 'comparison')
    except (KeyError, TypeError, ArithmeticError) as exc:
        raise ValueError('stock_compensation_invalid_monetary_cell') from exc
    require(current_assets > 0 and prior_assets > 0, 'stock_compensation_nonpositive_assets')
    average_assets = (current_assets+prior_assets)/2
    return old.bounded((cash-amount)/average_assets)


def build_features(meta, original_dataset, sbc_dataset):
    """Return (months,49) arrays; one shared causal mask, missing values NaN.

    Original146 selects the latest known event before eligibility filtering.
    Its selection is never replaced by an older report with an available SBC
    value. Both control values are copied exactly on the narrowed shared mask.
    """
    original = old.build_features(meta, original_dataset)
    require(type(sbc_dataset) is dict and type(sbc_dataset.get('schema_version')) is int
            and sbc_dataset['schema_version'] == 1
            and sbc_dataset.get('method') == MEASUREMENT, 'Explicit SBC dataset schema required')
    original_events = _index(original_dataset['events'])
    sbc_events = _index(sbc_dataset.get('events'))
    shape = original['arrays']['common_available'].shape
    arrays = {key: np.full(shape, np.nan) for key in EXTRA[:-1]}
    arrays['common_available'] = np.zeros(shape, dtype=bool)
    cells = []
    for original_cell in original['audit']['cells']:
        cell = {k: v for k, v in original_cell.items() if k != 'low_accrual_bounded'}
        cell.update(measurement=MEASUREMENT,
                    original_cash_quality_available=original_cell['available'],
                    comp_adjusted_cash_flow_bounded=None,
                    stock_compensation_event_id=None)
        for key in EXTRA[:-1]:
            cell[key] = None
        cell['available'] = False
        if original_cell['available']:
            key = (cell['symbol'], cell['current_event_id'])
            prior = [event for event in original_events[key]
                     if event['available_from'] == cell['current_available_from']]
            require(prior and all(event == prior[0] for event in prior),
                    'Unique original146 selected report required')
            candidates = sbc_events.get(key, [])
            if not candidates:
                cell['reason'] = 'current_stock_compensation_event_missing'
            elif any(event != candidates[0] for event in candidates):
                cell['reason'] = 'conflicting_stock_compensation_event'
            else:
                try:
                    adjusted = _adjusted(prior[0], candidates[0])
                except ValueError as exc:
                    cell['reason'] = str(exc)
                else:
                    i, j = cell['signal_index'], cell['pool_index']
                    cell.update(available=True, reason='available',
                                stock_compensation_event_id=cell['current_event_id'],
                                comp_adjusted_cash_flow_bounded=adjusted)
                    arrays['common_available'][i, j] = True
                    arrays['comp_adjusted_cash_flow_bounded'][i, j] = adjusted
                    for name in ('earnings_bounded', 'cash_flow_bounded'):
                        arrays[name][i, j] = original['arrays'][name][i, j]
                        cell[name] = original_cell[name]
        cells.append(cell)
    audit = dict(original['audit'])
    audit.update(
        cells=cells, measurement=MEASUREMENT,
        common_available_cells=int(arrays['common_available'].sum()),
        original_cash_quality_available_cells=original['audit']['common_available_cells'],
        reasons=dict(Counter(cell['reason'] for cell in cells)),
        source_verification='caller_verified_original146_and_new_stock_compensation_admission_closures_required',
        arithmetic_method='exact_fraction_scaled_cashflow_minus_signed_sbc_over_average_assets_then_bounded_float',
        earnings_and_cashflow_controls_identical_to146_on_common_mask=True,
        common_mask_subset_of146=True,
        same_selected_current_event_for_all_fields=True,
        signed_stock_compensation_preserved=True,
        stock_compensation_definition=DEFINITION,
        stock_compensation_raw_flags_do_not_substitute_new_admission=True,
        adjusted_cashflow_is_research_measure_not_reported_GAAP_cashflow=True)
    return dict(arrays=arrays, audit=audit)
