"""Read-only audit of completed round172 exports; no feature or replay imports."""
from pathlib import Path
from datetime import date
import hashlib
import json
import math
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_operating_margin172'
EXPECTED_PLAN = 'a98233242b31a41005179331849e866662100a0552edc8d293a5313e62973c6f'
EXPECTED_RESULTS = '0ac1621ffa9fcb7e920c42a6d50ca01c0bd7c56fe490d2d48f90af83a2201ea3'
ARMS = ('incumbent_local67', 'annual_margin_confirmation', 'annual_income_confirmation')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
SLICES = ('full122', 'first88', 'early53', 'recent35', 'later34')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def metric(days, values, initial):
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.2425
    changes = [b / a - 1 for a, b in zip(values, values[1:])]
    peak, drawdown = initial, 0.0
    for value in values:
        assert value > 0 and math.isfinite(value)
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1)
    return dict(cagr=(values[-1] / initial) ** (1 / years) - 1,
                sharpe=statistics.mean(changes) / statistics.stdev(changes) * math.sqrt(252),
                max_drawdown=drawdown)


def symbol(value):
    return {'research_equity_FB': 'META'}.get(value, value)


def main():
    assert sha(OUT / 'PLAN.json') == EXPECTED_PLAN
    assert sha(OUT / 'results/COMPLETION.json') == EXPECTED_RESULTS
    plan = read(OUT / 'PLAN.json')
    prepared = read(OUT / 'prepared/COMPLETION.json')
    completed = read(OUT / 'results/COMPLETION.json')
    frozen = read(OUT / 'selected/FROZEN.json')
    training = read(OUT / 'selected/TRAIN_RESULTS.json')
    results = read(OUT / 'results/RESULTS.json')
    independent = read(OUT / 'results/INDEPENDENT_OUTCOME.json')
    prep_audit = read(OUT / 'prepared/INDEPENDENT_PREPARE.json')
    meta = read(OUT / 'prepared/META.json')
    orders = read(OUT / 'prepared/ORDERS.json')
    features_audit = read(OUT / 'prepared/FEATURE_AUDIT.json')
    assert prepared['plan_sha256'] == completed['plan_sha256'] == frozen['plan_sha256'] == EXPECTED_PLAN
    assert not prepared['new_returns_computed'] and prepared['new_model_fits'] == 0
    for directory, completion in (('prepared', prepared), ('results', completed)):
        for name, digest in completion['artifacts_sha256'].items():
            assert sha(OUT / directory / name) == digest, name
    assert completed['frozen_sha256'] == sha(OUT / 'selected/FROZEN.json')
    for key, name in (('prepared_sha256', 'prepared/COMPLETION.json'),
                      ('orders_sha256', 'prepared/ORDERS.json'),
                      ('training_sha256', 'selected/TRAIN_RESULTS.json')):
        assert frozen[key] == sha(OUT / name)
    assert frozen['selection'] == results['selection']
    assert prepared['created_utc'] < frozen['created_utc'] < completed['created_utc']
    assert independent['passed'] and independent['checked_paths'] == 9 and independent['checked_slices'] == 45
    assert independent['independent_accounts'] and independent['independent_of_production_model_and_engine']
    assert not independent['terminal_liquidation']
    assert prep_audit['source']['checked_records'] == 35 and prep_audit['source']['checked_annual_cells'] == 140
    assert all(prep_audit['source'][k] is True for k in ('passed', 'exact_source_cells', 'exact_source_clocks',
                                                      'exact_source_durations', 'source_hold_preservation'))
    assert prep_audit['source']['unavailable_records'] == [['AAPL', 2016]]
    assert prep_audit['feature']['checked_cells'] == 122 * 49
    assert prep_audit['prefixes'] == 123 and prep_audit['original_arrays_unchanged'] and prep_audit['original_orders_exact']
    assert orders[ARMS[0]] == read(ROOT / 'outputs/monthly_local_linear70/local_linear_ORDERS.json')
    with np.load(OUT / 'prepared/FEATURES.npz') as npz:
        values = {k: npz[k] for k in npz.files}
    with np.load(OUT / 'prepared/RANKS_DELTAS.npz') as npz:
        arrays = {k: npz[k] for k in npz.files}
    mask = values['available']
    assert mask.shape == (122, 49) and mask.dtype == np.dtype(bool)
    for arm, key in zip(ARMS[1:], ('candidate', 'control')):
        expected = np.where(mask, values[key] * 0.01, 0.0)
        expected[0] = 0.0
        np.testing.assert_array_equal(arrays['delta_' + arm], expected)
        assert np.all(np.isnan(values[key][~mask]))
        assert np.all(np.isin(values[key][mask], [-1.0, 0.0, 1.0]))
    assert len(meta['dates']) == 122 and all(len(row) == 122 for row in orders.values())
    bounds = dict(full122=(meta['dates'][0], meta['ends'][-1]),
                  first88=(meta['dates'][0], meta['ends'][87]),
                  early53=(meta['dates'][0], meta['ends'][52]),
                  recent35=(meta['dates'][53], meta['ends'][87]),
                  later34=(meta['dates'][88], meta['ends'][-1]))
    computed, daily, position, prices = {}, {}, {}, {}
    for arm in ARMS:
        computed[arm] = {}
        for path in PATHS:
            rows = read(OUT / 'results' / (arm + '_' + path + '_DAILY.json'))
            daily[arm, path] = {r['date']: r['equity'] for r in rows}
            assert len(daily[arm, path]) == len(rows) and list(daily[arm, path]) == sorted(daily[arm, path])
            computed[arm][path] = {}
            for slice_name, (start, end) in bounds.items():
                sliced = [r for r in rows if start <= r['date'] <= end]
                assert sliced[0]['date'] == start and sliced[-1]['date'] == end
                numbers = [r['equity'] for r in sliced]
                initial = 100000.0 if slice_name == 'full122' else numbers[0]
                current = metric([r['date'] for r in sliced], numbers, initial)
                computed[arm][path][slice_name] = current
                for k, value in current.items():
                    near(value, results['arms'][arm][path]['slices'][slice_name][k])
                    near(value, independent['paths'][arm][path]['slices'][slice_name][k])
            if path in ('same_close_zero', 'next_open_cost'):
                for a, b in (('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')):
                    for key, value in computed[arm][path][b].items():
                        near(value, training[arm][path][a][key])
        rows = read(OUT / 'results' / (arm + '_same_close_zero_DAILY_POSITIONS.json'))
        position[arm] = {r['date']: r for r in rows}
        for row in rows:
            for name, units in row['adjusted_units'].items():
                if units:
                    prices[row['date'], symbol(name)] = row['market_values'][name] / units
        for row in read(OUT / 'results' / (arm + '_same_close_zero_TRADES.json')):
            prices[row['date'], symbol(row['symbol'])] = row['adjusted_price']
    changes = {arm: [i for i in range(122) if set(orders[arm][i]['symbols']) != set(orders[ARMS[0]][i]['symbols'])]
               for arm in ARMS[1:]}
    assert [i for i in changes[ARMS[1]] if i < 88] == frozen['selection']['changed_months'] == [23, 45, 49, 69]

    def guard_details(path, reference, gain):
        c, r = training[ARMS[1]][path], training[reference][path]
        return dict(cagr=c['training']['cagr'] >= r['training']['cagr'] + gain - 1e-12,
                    sharpe=c['training']['sharpe'] >= r['training']['sharpe'] - .05,
                    drawdown=c['training']['max_drawdown'] >= r['training']['max_drawdown'] - .03,
                    early=c['early']['cagr'] >= r['early']['cagr'] - .025,
                    recent=c['recent']['cagr'] >= r['recent']['cagr'] - .025)

    guards = dict(close_vs_incumbent=guard_details('same_close_zero', ARMS[0], .005),
                  open_vs_incumbent=guard_details('next_open_cost', ARMS[0], 0.),
                  close_vs_income_only=guard_details('same_close_zero', ARMS[2], 0.),
                  open_vs_income_only=guard_details('next_open_cost', ARMS[2], 0.))
    assert all(all(v.values()) == frozen['selection']['guards'][k] for k, v in guards.items())
    attributed = []
    for i in frozen['selection']['changed_months']:
        start, end = meta['dates'][i], meta['ends'][i]
        row = dict(index=i, signal=start, end=end, arms={}, source_cells=[])
        for arm in ARMS:
            names = [symbol(s) for s in orders[arm][i]['symbols']]
            returns = {s: prices[end, s] / prices[start, s] - 1 for s in names}
            actual = daily[arm, 'same_close_zero'][end] / daily[arm, 'same_close_zero'][start] - 1
            near(statistics.mean(returns.values()), actual)
            row['arms'][arm] = dict(symbols=names, scores=orders[arm][i]['scores'], stock_returns=returns,
                                    same_close_period_return=actual,
                                    next_open_cost_signal_to_signal_return=daily[arm, 'next_open_cost'][end] / daily[arm, 'next_open_cost'][start] - 1)
        for cell in features_audit['event_history'][i]['cells']:
            if cell['symbol'] in ('AAPL', 'MSFT', 'ABBV'):
                row['source_cells'].append({k: cell[k] for k in ('symbol', 'selected_report_id', 'known_session', 'candidate', 'control', 'available')})
        row['candidate_minus_incumbent_pp'] = 100 * (row['arms'][ARMS[1]]['same_close_period_return'] - row['arms'][ARMS[0]]['same_close_period_return'])
        attributed.append(row)
    compounded = 1.0
    extra_control = []
    for i, (start, end) in enumerate(zip(meta['dates'], meta['ends'])):
        growth = {a: daily[a, 'same_close_zero'][end] / daily[a, 'same_close_zero'][start] for a in ARMS}
        if i in changes[ARMS[1]]:
            compounded *= growth[ARMS[1]] / growth[ARMS[0]]
        else:
            near(growth[ARMS[1]], growth[ARMS[0]])
        if i in set(changes[ARMS[2]]) - set(changes[ARMS[1]]):
            extra_control.append(dict(index=i, signal=start, end=end,
                                      incumbent_symbols=orders[ARMS[0]][i]['symbols'],
                                      income_symbols=orders[ARMS[2]][i]['symbols'],
                                      incumbent_return=growth[ARMS[0]] - 1,
                                      income_return=growth[ARMS[2]] - 1))
    terminal_ratio = daily[ARMS[1], 'same_close_zero'][meta['ends'][-1]] / daily[ARMS[0], 'same_close_zero'][meta['ends'][-1]]
    near(compounded, terminal_ratio)
    summary = dict(verification=dict(plan_sha256=EXPECTED_PLAN, results_completion_sha256=EXPECTED_RESULTS,
                                     bound_prepared_artifacts=len(prepared['artifacts_sha256']), bound_result_artifacts=len(completed['artifacts_sha256']),
                                     independently_recalculated_slices=45, original_orders_exact=True,
                                     feature_available=int(mask.sum()), sign_disagreements=int(np.count_nonzero(values['candidate'][mask] != values['control'][mask]))),
                   selection=frozen['selection'], guard_details=guards, changed_sets=changes,
                   metrics={arm: {path: computed[arm][path] for path in ('same_close_zero', 'next_open_cost')} for arm in ARMS},
                   attribution=attributed, extra_income_changes=extra_control,
                   candidate_terminal_wealth_ratio=terminal_ratio,
                   four_changed_periods_compounded_wealth_ratio=compounded,
                   unchanged_118_periods_same_close_return_equal=True)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
