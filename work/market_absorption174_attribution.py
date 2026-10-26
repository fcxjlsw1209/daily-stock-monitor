"""Post-selection attribution of frozen174 exports; no prices, models or replays."""
from pathlib import Path
from datetime import date, datetime, timezone
import hashlib
import json
import math
import statistics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/monthly_market_absorption174'
OUT = SOURCE / 'attribution'
COMPLETION_SHA = '00a6b33b27976d06f1f4125fbf4b8dbd0d58c0fc597c0ca1d827ab6f6d913828'
ARMS = ('incumbent_local67', 'market_adjusted_recovery', 'raw_recovery_contrast')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def equal(a, b, message, atol=2e-10, rtol=1e-10):
    require(math.isclose(a, b, rel_tol=rtol, abs_tol=atol), message)


def display(symbol):
    return {'research_equity_FB': 'META'}.get(symbol, symbol)


def statistics_from_export(rows, normalize):
    dates = [row['date'] for row in rows]
    values = [float(row['equity']) for row in rows]
    require(len(rows) >= 2 and dates == sorted(set(dates)), 'Invalid daily export dates')
    require(all(math.isfinite(v) and v > 0. for v in values), 'Invalid NAV')
    initial = values[0] if normalize else 100000.
    returns = [values[i]/values[i-1]-1 for i in range(1, len(values))]
    years = (date.fromisoformat(dates[-1])-date.fromisoformat(dates[0])).days/365.2425
    peak, drawdown = initial, 0.
    for value in values:
        peak = max(peak, value)
        drawdown = min(drawdown, value/peak-1.)
    sd = statistics.stdev(returns)
    return dict(start=dates[0], end=dates[-1], cagr=(values[-1]/initial)**(1/years)-1.,
                sharpe=statistics.mean(returns)/sd*math.sqrt(252) if sd else None,
                max_drawdown=drawdown, total_return=values[-1]/initial-1.)


def main():
    require(not OUT.exists(), 'Attribution is write-once')
    results_dir = SOURCE / 'results'
    completion_path = results_dir / 'COMPLETION.json'
    require(sha(completion_path) == COMPLETION_SHA, 'Result completion differs')
    complete = read(completion_path)
    require(complete['status'] == 'COMPLETE' and complete['paths'] == 9 and complete['slices'] == 45,
            'Results are not complete')
    sources = {str(completion_path.relative_to(ROOT)): COMPLETION_SHA}
    for name, digest in complete['artifacts_sha256'].items():
        path = results_dir / name
        require(path.parent == results_dir and sha(path) == digest, 'Result artifact differs: '+name)
        sources[str(path.relative_to(ROOT))] = digest
    require(len(complete['artifacts_sha256']) == 38, 'Expected all38 result artifacts')
    frozen_path = SOURCE / 'selected/FROZEN.json'
    require(sha(frozen_path) == complete['frozen_sha256'], 'Frozen selection differs')
    frozen = read(frozen_path)
    for relative, key in (('prepared/COMPLETION.json', 'prepared_sha256'),
                          ('prepared/ORDERS.json', 'orders_sha256'),
                          ('selected/TRAIN_RESULTS.json', 'training_sha256')):
        path = SOURCE / relative
        require(sha(path) == frozen[key], 'Frozen input differs: '+relative)
        sources[str(path.relative_to(ROOT))] = sha(path)
    prepared = read(SOURCE / 'prepared/COMPLETION.json')
    for name, digest in prepared['artifacts_sha256'].items():
        path = SOURCE / 'prepared' / name
        require(path.parent == SOURCE/'prepared' and sha(path) == digest, 'Prepared input differs: '+name)
        sources[str(path.relative_to(ROOT))] = digest
    sources[str(frozen_path.relative_to(ROOT))] = sha(frozen_path)
    plan = SOURCE / 'PLAN.json'
    require(sha(plan) == complete['plan_sha256'] == frozen['plan_sha256'] == prepared['plan_sha256'],
            'Plan bindings differ')
    sources[str(plan.relative_to(ROOT))] = sha(plan)
    outcome = read(results_dir / 'INDEPENDENT_OUTCOME.json')
    require(outcome['passed'] is True and outcome['checked_paths'] == 9 and outcome['checked_slices'] == 45,
            'Independent outcome did not pass')
    result = read(results_dir / 'RESULTS.json')
    require(result['selection'] == frozen['selection'], 'Selection changed after freeze')
    orders = read(SOURCE / 'prepared/ORDERS.json')
    meta = read(SOURCE / 'prepared/META.json')
    require(set(orders) == set(ARMS) and all(len(orders[a]) == 122 for a in ARMS), 'Order coverage')
    dates, ends = meta['dates'], meta['ends']
    require(len(dates) == len(ends) == 122 and all(ends[i] == dates[i+1] for i in range(121)),
            'Monthly boundaries do not tile history')
    daily, positions, compact = {}, {}, {}
    unchanged_units_checks = 0
    for arm in ARMS:
        daily[arm], positions[arm], compact[arm] = {}, {}, {}
        for path in PATHS:
            d = read(results_dir / (arm+'_'+path+'_DAILY.json'))
            pos = read(results_dir / (arm+'_'+path+'_DAILY_POSITIONS.json'))
            hold = read(results_dir / (arm+'_'+path+'_HOLDINGS.json'))
            require(len(hold) == 122 and len(pos) == len(d), 'Export coverage')
            nav = {r['date']: r['equity'] for r in d}
            pmap = {r['date']: r for r in pos}
            require(len(nav) == len(d) and set(nav) == set(pmap), 'Export dates differ')
            for day, row in pmap.items():
                equal(nav[day], row['equity'], 'NAV vs daily position differs', atol=5e-8)
            for i, h in enumerate(hold):
                require(h['signal_date'] == dates[i], 'Holding signal differs')
                require(set(h['symbols']) == {display(s) for s in orders[arm][i]['symbols']}, 'Holding/order selection differs')
                start = h['execution_date']
                stop = hold[i+1]['execution_date'] if i+1 < len(hold) else '9999-12-31'
                window = [r for r in pos if start <= r['date'] < stop]
                require(window, 'No position window')
                start_units = window[0]['adjusted_units']
                require(set(start_units) == set(orders[arm][i]['symbols']), 'Executed positions differ')
                for r in window[1:]:
                    require(r['adjusted_units'] == start_units, 'Intramonth share change')
                    unchanged_units_checks += 1
            bounds = dict(full122=(dates[0], ends[-1]), first88=(dates[0], ends[87]),
                          early53=(dates[0], ends[52]), recent35=(dates[53], ends[87]),
                          later34=(dates[88], ends[-1]))
            compact[arm][path] = {}
            for segment, (lo, hi) in bounds.items():
                rows = [r for r in d if lo <= r['date'] <= hi]
                require(rows[0]['date'] == lo and rows[-1]['date'] == hi, 'Split boundary missing')
                stats = statistics_from_export(rows, normalize=segment != 'full122')
                for key in ('cagr', 'sharpe', 'max_drawdown', 'total_return'):
                    equal(stats[key], result['arms'][arm][path]['slices'][segment][key], 'Recomputed '+segment+'/'+key)
                    equal(stats[key], outcome['paths'][arm][path]['slices'][segment][key], 'Independent receipt '+segment+'/'+key)
                compact[arm][path][segment] = stats
            daily[arm][path], positions[arm][path] = nav, pmap
    comparisons = {}
    for comparison, arm, reference in ((ARMS[1], ARMS[1], ARMS[0]),
                                        (ARMS[2], ARMS[2], ARMS[0]),
                                        ('candidate_vs_matched_raw', ARMS[1], ARMS[2])):
        months = []
        for i, (start, end) in enumerate(zip(dates, ends)):
            base_symbols, arm_symbols = orders[reference][i]['symbols'], orders[arm][i]['symbols']
            changed = set(base_symbols) != set(arm_symbols)
            row = dict(index=i, signal_date=start, end=end, era='early53' if i < 53 else 'recent35' if i < 88 else 'later34',
                       base_symbols=base_symbols, arm_symbols=arm_symbols, changed_set=changed,
                       rank_only_change=not changed and base_symbols != arm_symbols,
                       added=sorted(set(arm_symbols)-set(base_symbols)), removed=sorted(set(base_symbols)-set(arm_symbols)), paths={})
            for path in PATHS:
                b, a = daily[reference][path], daily[arm][path]
                br, ar = b[end]/b[start]-1., a[end]/a[start]-1.
                row['paths'][path] = dict(base_return=br, arm_return=ar, excess_percentage_points=(ar-br)*100,
                    relative_log_growth=math.log1p(ar)-math.log1p(br))
            months.append(row)
        audits = {}
        for path in PATHS:
            logs = [r['paths'][path]['relative_log_growth'] for r in months]
            b, a = daily[reference][path], daily[arm][path]
            equal(math.fsum(logs), math.log((a[ends[-1]]/a[dates[0]])/(b[ends[-1]]/b[dates[0]])), 'Attribution does not telescope')
            unchanged = [r for r in months if not r['changed_set']]
            equal_sets_different_units = sum(
                positions[reference][path][r['signal_date']]['adjusted_units'] != positions[arm][path][r['signal_date']]['adjusted_units']
                for r in unchanged)
            audits[path] = dict(relative_terminal_wealth=math.expm1(math.fsum(logs)),
                log_growth_from_changed_set_months=math.fsum(r['paths'][path]['relative_log_growth'] for r in months if r['changed_set']),
                log_growth_from_unchanged_set_months=math.fsum(r['paths'][path]['relative_log_growth'] for r in unchanged),
                equal_selection_months_with_different_signal_close_units=equal_sets_different_units,
                maximum_abs_unchanged_set_month_excess_pp=max(abs(r['paths'][path]['excess_percentage_points']) for r in unchanged))
        comparisons[comparison] = dict(arm=arm, reference=reference,
            changed_set_months=[r for r in months if r['changed_set']],
            rank_only_months=[r['signal_date'] for r in months if r['rank_only_change']],
            counts={s: sum(r['changed_set'] and (s == 'full122' or r['era'] == s or (s == 'first88' and r['index'] < 88)) for r in months)
                    for s in ('full122', 'first88', 'early53', 'recent35', 'later34')},
            reconciliation=audits, monthly_intervals=months)
    expected_changed = [r['index'] for r in comparisons[ARMS[1]]['changed_set_months'] if r['index'] < 88]
    require(expected_changed == result['selection']['changed_months'], 'Selection changed-month record differs')
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        result_completion_sha256=COMPLETION_SHA, result_artifacts_verified=38,
        independently_recomputed_paths=9, independently_recomputed_slices=45,
        unchanged_intramonth_units_checks=unchanged_units_checks, selection=result['selection'],
        metrics=compact, comparisons=comparisons,
        interpretation=[
            'Post-frozen-selection diagnostics only; no new prices, factors, models, orders, strategy replays or parameter search.',
            'Monthly attribution is exported closing NAV(end)/NAV(signal date)-1. Same-close zero-cost allocation makes it the exact new holding-period portfolio return.',
            'Next-open paths include the old holdings overnight into execution and fees. Signal-close intervals remain telescoping but are not pure new-holding return attribution.',
            'Same symbols with different unit counts across arms are not additional selection changes: prior performance changes account equity and hence next rebalance unit quantities.',
            'Within each actual execution interval, adjusted units are checked constant. Intramonth weights can drift with prices; no daily rebalance is inferred.',
            'Rank-only changes among the same equal-weight holdings are recorded separately from selection set changes.',
        ], source_sha256=sources, script_sha256=sha(Path(__file__)))
    for relative, digest in sources.items():
        require(sha(ROOT/relative) == digest, 'Source changed during attribution')
    OUT.mkdir()
    target = OUT/'REPORT.json'
    with target.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    with (OUT/'COMPLETION.json').open('x') as stream:
        json.dump(dict(status='COMPLETE', report_sha256=sha(target), script_sha256=sha(Path(__file__)),
                       source_result_completion_sha256=COMPLETION_SHA), stream, indent=2)
        stream.write('\n')
    brief = dict(metrics={a: {p: {s: v for s, v in z.items() if s in ('full122', 'first88', 'later34')}
                              for p, z in paths.items() if p != 'next_open_zero'} for a, paths in compact.items()},
        selection=result['selection'], comparisons={a:{'counts':v['counts'],
        'changed_months':v['changed_set_months'], 'reconciliation':v['reconciliation']} for a,v in comparisons.items()},
        report=str(target), report_sha256=sha(target))
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
