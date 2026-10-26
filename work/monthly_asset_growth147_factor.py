"""Pure asset-expansion feature; preserves frozen146 E/Q and availability."""
from collections import defaultdict
import numpy as np
import monthly_cash_quality146_factor as old

METHOD = old.METHOD
MEASUREMENT = 'same_report_total_asset_expansion_v1'
EXTRA = ('earnings_bounded', 'low_accrual_bounded',
         'low_asset_growth_bounded', 'common_available')
require = old._require


def low_asset_growth(event):
    """Exact scaled endpoints and rational mapping; no intermediate floats."""
    current = old._amount(event['fields']['current']['total_assets'], event, 'current')
    prior = old._amount(event['fields']['comparison']['total_assets'], event, 'comparison')
    require(current > 0 and prior > 0, 'Positive asset endpoints required')
    growth = (current-prior)/prior
    return old.bounded(-growth)


def build_features(meta, admitted_dataset):
    original = old.build_features(meta, admitted_dataset)
    arrays = {key: original['arrays'][key].copy()
              for key in ('earnings_bounded', 'low_accrual_bounded', 'common_available')}
    arrays['low_asset_growth_bounded'] = np.full(arrays['earnings_bounded'].shape, np.nan)
    by_key = defaultdict(list)
    for event in admitted_dataset['events']:
        by_key[(event['symbol'], event['event_id'])].append(event)
    cells = []
    for original_cell in original['audit']['cells']:
        cell = {key: value for key, value in original_cell.items() if key != 'cash_flow_bounded'}
        cell['measurement'] = MEASUREMENT
        cell['low_asset_growth_bounded'] = None
        if cell['available']:
            candidates = [e for e in by_key[(cell['symbol'], cell['current_event_id'])]
                          if e['available_from'] == cell['current_available_from']
                          and e['available_from'] <= cell['signal_date']]
            require(candidates and all(e == candidates[0] for e in candidates),
                    'Unique previously selected current report required')
            value = low_asset_growth(candidates[0])
            arrays['low_asset_growth_bounded'][cell['signal_index'], cell['pool_index']] = value
            cell['low_asset_growth_bounded'] = value
        cells.append(cell)
    audit = dict(original['audit'])
    audit.update(
        cells=cells, measurement=MEASUREMENT,
        source_verification='caller_verified_147_source_use_review_and_frozen146_source_closure_required',
        arithmetic_method='exact_fraction_scaled_endpoints_negative_asset_change_over_prior_assets_then_bounded_float',
        earnings_and_low_accrual_controls_identical_to146=True,
        common_mask_identical_to146=True,
        growth_denominator='comparison_total_assets',
        exact_feature_range='(-1,0.5); float rounding may reach an endpoint',
        dose_0_01_exact_score_range='(-0.01,0.005)',
        expansion_negative_contraction_positive_zero_change_zero=True,
        asset_growth_not_organic_investment_or_CMA_replication=True)
    return dict(arrays={key: arrays[key] for key in EXTRA}, audit=audit)
