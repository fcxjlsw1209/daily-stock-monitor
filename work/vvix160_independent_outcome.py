"""Independent160 support, one-time selection and nine-path accounting.

No producer/kernel imports, no market reads and no executable real-data entry
point. Accounting accepts supplied objects and reuses only the old independently
implemented fixed-unit replay. A registered host must bind all inputs first.
"""
import math
import numpy as np
import pandas as pd

from vvix160_independent_prepare import NAMES, require, independent_imports

PATHS = (('close', 0., 'same_close'), ('open', 0., 'next_open_zero'),
         ('open', .0005, 'next_open_cost'))


def accounting_helper():
    independent_imports()
    import monthly_paired_target143_outcome_review as accounting
    return accounting


def support_oracle(monthly, meta):
    require(len(monthly) == len(meta['dates']) == 122 and meta['dates'][0] == '2016-08-17', 'Original122 source axis')
    valid = []
    for i, row in enumerate(monthly):
        require(type(row['signal_index']) is int and (row['signal_index'], row['date']) == (i,meta['dates'][i]), 'Source axis')
        require(type(row['initial_partial']) is bool and row['initial_partial'] == (i == 0), 'Initial source marker')
        n = 0 if i == 0 else 252
        require(type(row['window_size']) is int and row['window_size'] == n, 'Exact252 source window')
        passed = []
        for name in ('VIX','VVIX'):
            value = row[name]
            require(type(value['passed']) is bool and type(value['observations']) is int, 'Explicit source status')
            passed.append(value['passed'] and value['observations'] == n)
        if i == 0:
            require(all(passed) and row['window_start'] is None and row['window_end'] is None and
                    all(row[name]['m'] == 0 for name in ('VIX','VVIX')), 'Neutral initial source row')
        elif all(passed):
            valid.append(i)
    return dict(passed=len(valid)==121, full_window_months=len(valid), required_complete_source_months=121,
                initial_partial_exempt=True, first88=sum(i<88 for i in valid), early53=sum(i<53 for i in valid),
                recent35=sum(53<=i<88 for i in valid), later34=sum(i>=88 for i in valid),
                missing_months=[i for i in range(1,122) if i not in valid])


def changed_support_oracle(orders, name):
    require(name in NAMES[1:], 'Fixed challenger')
    months = []
    for i in range(88):
        rows = [orders[k][i]['symbols'] for k in (NAMES[0],name)]
        require(all(type(x) is list and len(x)==len(set(x)) and all(type(y) is str and y for y in x) for x in rows), 'Unique held symbols')
        if set(rows[0]) != set(rows[1]): months.append(i)
    early = sum(i<53 for i in months); recent = len(months)-early
    return dict(passed=len(months)>=3 and early>=1 and recent>=1, months=months,
                early_months=early, recent_months=recent, rule='first88 unordered held-set changes; later34 excluded')


def metric_guard(candidate, baseline, gain):
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    values = [[row.get(a, {}).get(b) for a, b in fields] for row in (candidate, baseline)]
    if not all(type(x) in (float, int) and math.isfinite(x) for group in values for x in group):
        return False
    c, b = values
    return c[0] >= b[0]+gain-1e-12 and all(x >= y-limit for x, y, limit in zip(c[1:], b[1:], (.05, .03, .025, .025)))


def select_first88_oracle(rows, opened, support, changed):
    for group in (rows, opened):
        require(type(group) is list and [r['name'] for r in group] == list(NAMES), 'Fixed three-arm order')
    guards, admitted = {}, []
    for i in (1, 2):
        name = NAMES[i]
        guards[name] = dict(source_support=support['passed'] is True,
                            changed_holdings=changed[name]['passed'] is True,
                            close=metric_guard(rows[i], rows[0], .005),
                            cost_open=metric_guard(opened[i], opened[0], 0.))
        if all(guards[name].values()):
            admitted.append(i)
    # First obtain the true maximum, then apply the fixed tie rule once.
    winner = 0
    if admitted:
        best = max(rows[i]['training']['cagr'] for i in admitted)
        winner = min(i for i in admitted if not best > rows[i]['training']['cagr']+1e-12)
    return dict(primary=NAMES[winner], admitted=winner != 0, selection_count=1,
                guards=guards, later34_used_for_selection=False)


def validate_orders(close, opening, orders, meta):
    a = accounting_helper()
    dates = a.daily_axis(close.index, 'prices')
    require(opening.index.equals(dates) and opening.columns.equals(close.columns) and close.columns.is_unique,
            'Same unique close/open axes')
    signals = a.daily_axis(meta['dates'], 'signals')
    ends = pd.DatetimeIndex(meta['ends'])
    require(len(signals) == len(ends) == len(meta['symbols']) == 122 and signals.isin(dates).all() and
            ends.isin(dates).all() and (ends > signals).all() and signals[1:].equals(ends[:-1]),
            'Full122 adjacent holding intervals')
    require(set(orders) == set(NAMES), 'Exactly three order arms')
    for name in NAMES:
        require(len(orders[name]) == 122, 'Exactly122 orders')
        for i, row in enumerate(orders[name]):
            require(set(row) == {'date', 'signal_date', 'end', 'symbols', 'scores'}, 'Exact order schema')
            require(row['date'] == row['signal_date'] == meta['dates'][i] and row['end'] == meta['ends'][i],
                    'Original order interval')
            held = row['symbols']
            require(type(held) is list and len(held) == len(set(held)) and len(held) in (0, 2, 3), 'Cash/EW2/EW3')
            require(all(s in meta['symbols'][i] and s in close.columns for s in held), 'Original candidate identity')
            require(len(row['scores']) == len(held) and all(type(x) in (float, int) and math.isfinite(x) for x in row['scores']),
                    'Aligned finite scores')
    return dates[(dates >= signals[0]) & (dates <= ends[-1])]


def nine_path_oracle(close, opening, orders, meta):
    """Pure supplied-data replay; caller must enforce the registered phase boundary."""
    a = accounting_helper()
    dates = validate_orders(close, opening, orders, meta)
    result, navs, opened, details = {}, {}, {}, {}
    bounds = dict(early2016_2020=(meta['dates'][0], meta['ends'][52]),
                  original2021_2026=(meta['dates'][53], meta['ends'][-1]),
                  training2016_2023=(meta['dates'][0], meta['ends'][87]),
                  later34=(meta['dates'][88], meta['ends'][-1]))
    for name in NAMES:
        result[name] = {}
        for execution, cost, key in PATHS:
            metrics, nav, trades, holdings = a.replay(close, opening, orders[name], dates, execution, cost)
            allowed_dates = {row['date'] if execution == 'close' else
                             str(close.index[close.index.get_loc(pd.Timestamp(row['date']))+1].date()) for row in orders[name]}
            require(all(row['date'] in allowed_dates for row in trades), 'Intramonth trade')
            require(all(row['partial'] is (row['year'] in (2016, 2026)) for row in metrics['annual_returns']),
                    'Original partial-year convention')
            result[name][key] = dict(metrics=metrics,
                slices={label: a.h.slice_stat(dates, nav, *span) for label, span in bounds.items()})
            details[name+'_'+key] = dict(daily=[dict(date=str(d.date()), equity=float(v)) for d, v in zip(dates, nav)],
                                         trades=trades, holdings=holdings)
            if execution == 'close':
                navs[name] = nav
            if key == 'next_open_cost':
                opened[name] = nav
        # Training is causally replayed independently, without any later34 orders.
        train_dates = dates[dates <= pd.Timestamp(meta['ends'][87])]
        for execution, cost, expected in (('close', 0., navs[name]), ('open', .0005, opened[name])):
            _, prefix, _, _ = a.replay(close.loc[:train_dates[-1]], opening.loc[:train_dates[-1]],
                                      orders[name][:88], train_dates, execution, cost)
            np.testing.assert_allclose(prefix, expected[:len(prefix)], atol=1e-5, rtol=2e-11)
    return dict(results=result, details=details,
                training=[dict(name=name, **a.h.measures(dates, navs[name], meta)) for name in NAMES],
                opened=[dict(name=name, **a.h.measures(dates, opened[name], meta)) for name in NAMES],
                paths=9, orders=366, prefix_reselection_count=0)


def compare_nine_paths(expected, artifacts):
    """artifacts is a supplied filename->decoded object mapping; never a reader."""
    a = accounting_helper()
    errors = []
    a.compare_stats(artifacts['RESULTS.json']['arms'], expected['results'], errors, 'RESULTS/arms')
    for prefix, values in expected['details'].items():
        actual = artifacts[prefix+'_DAILY.json']
        require(all(set(row) == {'date', 'equity'} for row in actual) and
                [r['date'] for r in actual] == [r['date'] for r in values['daily']], 'Exact daily NAV dates')
        observed = np.asarray([r['equity'] for r in actual], float)
        wanted = np.asarray([r['equity'] for r in values['daily']], float)
        require(np.isfinite(observed).all() and (observed > 0).all(), 'Positive finite daily NAV')
        np.testing.assert_allclose(observed, wanted, atol=1e-5, rtol=2e-11)
        errors.append((prefix+'/daily', float(np.max(np.abs(observed-wanted)))))
        for key in ('trades', 'holdings'):
            a.h.compare(a.normalize(artifacts[prefix+'_'+key.upper()+'.json']), a.normalize(values[key]),
                        errors, prefix+'/'+key)
    for filename, key in (('TRAIN_TRIALS.json', 'training'), ('TRAIN_OPEN_TRIALS.json', 'opened')):
        a.compare_stats(artifacts[filename], expected[key], errors, filename)
    return dict(paths=9, orders=366, max_absolute_error=max((x for _, x in errors), default=0.))


