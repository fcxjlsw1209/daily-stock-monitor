"""Independent prepared-feature audit; never constructs orders or reads returns.

Run only after the registered prepare phase has completed. The reference uses
ordinary log-price differences and a two-column least-squares design with an
intercept, independently of features.py's centered/scaled covariance code.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
O = P.parent
LOCAL = O / 'monthly_local_linear70'
DESIGN_SHA = '970d3dbc04b0442e13fcee16c81e7ee7bc75aa7ecb60ba49d314488541e42e86'
NEW_KEYS = {'raw_beta_gap', 'rank_beta_gap', 'beta_gap_available'}
REASONS = ('available', 'insufficient_history', 'invalid_market_path',
           'invalid_security_path', 'insufficient_group', 'zero_market_variance')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def assert_hashes(mapping, root=None):
    """Hash opaque files only; no performance or order documents are parsed."""
    for filename, expected in mapping.items():
        path = Path(filename)
        if root is not None and not path.is_absolute():
            path = root / path
        assert sha(path) == expected, 'Changed fingerprint: ' + str(path)
    return len(mapping)


def valid_prices(values):
    return values.shape == (253,) and bool((np.isfinite(values) & (values > 0)).all())


def independent_reference(px, meta):
    shape = (len(meta['dates']), 49)
    raw = np.full(shape, np.nan)
    ranks = np.full(shape, np.nan)
    available = np.zeros(shape, dtype=bool)
    details = []
    etfs = set(meta['etfs'])
    assert len(etfs) == len(meta['etfs']) == 17
    assert px.index.is_monotonic_increasing and px.index.is_unique
    assert len(meta['symbols']) == shape[0]
    previous = None
    for i, (date, symbols) in enumerate(zip(meta['dates'], meta['symbols'])):
        date = pd.Timestamp(date)
        assert previous is None or date > previous
        previous = date
        assert len(symbols) == len(set(symbols)) == 49
        assert etfs <= set(symbols) and len(set(symbols) - etfs) == 32
        t = int(px.index.get_loc(date))
        left, right = t - 294, t - 42
        detail = dict(date=str(date.date()), first_price_date=None, first_return_date=None,
                      last_price_date=None, up_count=None, down_count=None, zero_count=None,
                      beta_up=[None] * 49, beta_down=[None] * 49, status=[None] * 49)
        if left < 0:
            detail['status'] = ['insufficient_history'] * 49
        elif 'SPY' not in px.columns:
            detail['status'] = ['invalid_market_path'] * 49
        else:
            detail.update(first_price_date=str(px.index[left].date()),
                          first_return_date=str(px.index[left + 1].date()),
                          last_price_date=str(px.index[right].date()))
            # Numeric conversion occurs strictly after the intended slice.
            market_prices = px['SPY'].iloc[left:right + 1].to_numpy(dtype=float)
            if not valid_prices(market_prices):
                detail['status'] = ['invalid_market_path'] * 49
            else:
                market = np.diff(np.log(market_prices))
                assert market.shape == (252,) and np.isfinite(market).all()
                up, down = market > 0, market < 0
                detail.update(up_count=int(up.sum()), down_count=int(down.sum()),
                              zero_count=int((market == 0).sum()))
                for j, symbol in enumerate(symbols):
                    prices = px[symbol].iloc[left:right + 1].to_numpy(dtype=float)
                    if not valid_prices(prices):
                        detail['status'][j] = 'invalid_security_path'
                        continue
                    security = np.diff(np.log(prices))
                    assert np.isfinite(security).all()
                    if min(int(up.sum()), int(down.sum())) < 40:
                        detail['status'][j] = 'insufficient_group'
                        continue
                    betas = []
                    for group in (up, down):
                        x, y = market[group], security[group]
                        if np.ptp(x) == 0:
                            betas.append(np.nan)
                            continue
                        design = np.column_stack((np.ones(len(x)), x))
                        fit, _, matrix_rank, _ = np.linalg.lstsq(design, y, rcond=None)
                        assert matrix_rank == 2, 'Numerically singular independent OLS'
                        assert np.isfinite(fit).all(), 'Nonfinite independent OLS'
                        beta = float(fit[1])
                        # These exact mathematical identities prevent a fictitious
                        # last-bit beta gap from moving a genuine zero's rank.
                        if np.ptp(y) == 0:
                            assert abs(beta) <= 1e-10
                            beta = 0.
                        elif symbol == 'SPY':
                            assert abs(beta - 1.) <= 1e-10
                            beta = 1.
                        betas.append(beta)
                    detail['beta_up'][j] = betas[0] if np.isfinite(betas[0]) else None
                    detail['beta_down'][j] = betas[1] if np.isfinite(betas[1]) else None
                    if not np.isfinite(betas).all():
                        detail['status'][j] = 'zero_market_variance'
                        continue
                    value = betas[0] - betas[1]
                    assert np.isfinite(value), 'Nonfinite independent beta difference'
                    raw[i, j] = value
                    available[i, j] = True
                    detail['status'][j] = 'available'
        ranks[i] = pd.Series(raw[i]).rank(method='average', pct=True).to_numpy() - .5
        details.append(detail)
    return raw, ranks, available, details


def run():
    review = dict(created_at=datetime.now(timezone.utc).isoformat(), status='running',
        reference='Independent np.diff(log(adjusted prices)) and np.linalg.lstsq([1, SPY], security), separately by market sign.',
        production_features_imported=False, candidate_returns_read=False,
        orders_read=False, strategy_selection_performed=False, script_sha256=sha(__file__))
    try:
        plan = read(P / 'PLAN.json')
        assert sha(P / 'DESIGN.json') == DESIGN_SHA == (P / 'DESIGN.sha256').read_text().strip()
        assert plan['design_sha256'] == DESIGN_SHA
        assert plan['design'] == read(P / 'DESIGN.json')
        assert read(P / 'GRID.json') == plan['design']['configs']
        # PLAN is not self-hashed, so record its digest alongside verified
        # registered source and implementation maps for subsequent audits.
        review['fingerprints'] = dict(design=DESIGN_SHA, plan=sha(P / 'PLAN.json'),
            cache_manifest=sha(P / 'CACHE_HASHES.json'),
            implementation_files=assert_hashes(plan['hashes']),
            registered_source_files=assert_hashes(plan['source_hashes']),
            cached_files=assert_hashes(read(P / 'CACHE_HASHES.json'), P),
            input_files=assert_hashes(read(P / 'INPUT_HASHES.json')),
            raw_sources=assert_hashes(plan['raw_source_hashes']))
        assert len(plan['raw_source_hashes']) == 64
        assert plan['raw_source_hashes'] == read(LOCAL / 'PLAN.json')['raw_source_hashes']
        meta = read(P / 'META.json')
        assert meta == read(LOCAL / 'META.json')
        assert read(P / 'INCUMBENT_POLICY.json') == read(LOCAL / 'INCUMBENT_POLICY.json')
        with np.load(LOCAL / 'DATA.npz', allow_pickle=False) as source, np.load(P / 'DATA.npz', allow_pickle=False) as prepared:
            assert len(source.files) == 27
            assert set(prepared.files) == set(source.files) | NEW_KEYS
            for key in source.files:
                assert prepared[key].dtype == source[key].dtype
                np.testing.assert_array_equal(prepared[key], source[key], err_msg=key)
            stored_raw = prepared['raw_beta_gap'].copy()
            stored_rank = prepared['rank_beta_gap'].copy()
            stored_available = prepared['beta_gap_available'].copy()
        review['inherited_arrays_exact'] = 27
        assert stored_raw.shape == stored_rank.shape == stored_available.shape == (122, 49)
        assert stored_available.dtype == bool
        loader_path = O / 'monthly_stability_monitor/inputs.py'
        spec = importlib.util.spec_from_file_location('beta_gap_independent_raw_inputs', loader_path)
        inputs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inputs)
        panel, info = inputs.load_panel(meta['etfs'])
        loaded_hashes = dict(info['source_hashes'])
        loaded_hashes[str(loader_path)] = sha(loader_path)
        assert loaded_hashes == plan['raw_source_hashes']
        review['reloaded_raw_source_hashes_exact'] = len(loaded_hashes)
        raw, rank, available, details = independent_reference(panel['adj'], meta)
        np.testing.assert_array_equal(available, stored_available)
        np.testing.assert_array_equal(np.isfinite(stored_raw), stored_available)
        np.testing.assert_array_equal(np.isfinite(stored_rank), stored_available)
        np.testing.assert_allclose(raw, stored_raw, atol=1e-10, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(rank, stored_rank)
        errors = np.abs(raw[available] - stored_raw[available])
        review.update(cells=int(raw.size), available_cells=int(available.sum()),
            missing_cells=int((~available).sum()), mask_exact=True, rank_exact=True,
            raw_atol=1e-10, raw_rtol=0, max_raw_absolute_error=float(errors.max(initial=0)),
            candidate_identity_exact=True, inherited_metadata_exact=True,
            eligible_not_used_for_reference=True, fixed_etfs=sorted(meta['etfs']))
        audit = read(P / 'FEATURE_AUDIT.json')
        assert audit['events'] == 122 and audit['candidate_count'] == 49
        assert audit['etfs'] == sorted(meta['etfs']) and audit['min_group'] == 40
        assert audit['price_count'] == 253 and audit['return_count'] == 252
        assert audit['first_price_offset'] == -294 and audit['last_price_offset'] == -42
        for key in ('eligibility_read', 'state_read', 'recent_42_prices_read',
                    'future_labels_used', 'beta_clipped', 'beta_shrinkage'):
            assert audit[key] is False
        assert len(audit['months']) == len(details)
        beta_error = 0.
        for actual, expected in zip(audit['months'], details):
            for key in ('date', 'first_price_date', 'first_return_date', 'last_price_date',
                        'up_count', 'down_count', 'zero_count', 'status'):
                assert actual[key] == expected[key], (key, expected['date'])
            for key in ('beta_up', 'beta_down'):
                x = np.array([np.nan if v is None else v for v in actual[key]])
                y = np.array([np.nan if v is None else v for v in expected[key]])
                np.testing.assert_allclose(x, y, atol=1e-10, rtol=0, equal_nan=True)
                finite = np.isfinite(x)
                beta_error = max(beta_error, float(np.abs(x[finite]-y[finite]).max(initial=0)))
        counts = {reason: sum(month['status'].count(reason) for month in details) for reason in REASONS}
        assert sum(counts.values()) == 5978
        for reason, count in counts.items():
            assert audit[reason + '_cells'] == count
        validation = read(P / 'PREPARE_VALIDATION.json')
        for key in ('four_controls_exact', 'prefix_exact', 'missing_beta_gap_returns_local_control',
                    'original_core_and_auxiliary_states_preserved', 'no_refit',
                    'whole_pool_ranks_independently_recomputed', 'adjusted_price_and_opening_scale_parity'):
            assert validation[key] is True
        assert validation['source_raw_inputs'] == 64
        assert validation['finite_features'] == int(available.sum())
        parity = read(P / 'PRICE_PARITY.json')
        for key in ('adjusted_levels_constant_scale', 'adjusted_opening_parity',
                    'missing_price_masks_exact', 'positive_price_masks_exact'):
            assert parity[key] is True
        review.update(status='passed', independent_month_logs_exact=True,
            max_individual_beta_absolute_error=beta_error, status_counts=counts,
            prepare_validation_claims_consistent=True,
            control_and_execution_note='Original 27 arrays independently checked exactly; existing control/execution logs checked for consistency, without executing selectors or reading orders.')
    except Exception as exc:
        review.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (P / 'ACTUAL_FEATURE_REVIEW.json').write_text(json.dumps(review, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: review[key] for key in ('status', 'cells', 'available_cells',
        'inherited_arrays_exact', 'reloaded_raw_source_hashes_exact', 'mask_exact',
        'rank_exact', 'max_raw_absolute_error', 'max_individual_beta_absolute_error')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', action='store_true', help='Confirm root authorized auditing completed prepare output.')
    args = parser.parse_args()
    if not args.prepared:
        parser.error('Run only after root confirms prepare completed: --prepared')
    run()
