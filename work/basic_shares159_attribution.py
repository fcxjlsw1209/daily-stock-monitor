"""Post-audit descriptive attribution of the already fixed round159 portfolios.

No selection, features, counterfactual portfolio construction or compounding.
Real input decoding is gated by exact PLAN/FROZEN/RESULTS/outcome-receipt hashes.
"""
import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import sys

NAMES = ('incumbent_local67', 'centered_negative_basic_share_growth_magnitude',
         'centered_negative_basic_share_growth_rank')
ISSUERS = ('AAPL', 'AMD', 'MSFT', 'NVDA')
OUT = 'outputs/monthly_basic_shares159'
PRICE = 'outputs/monthly_2016_extension/PRICES.npz'
SLICES = {'all122': (0, 122), 'first88': (0, 88), 'early53': (0, 53),
          'recent35': (53, 88), 'later34': (88, 122)}
OUTCOME_CHECKS = ('all9_daily_nav_trade_holdings_metrics', 'first88_once_selection',
    'causal_first88_replay', 'source_and_changed_support_verified', 'diagnostic_start_bound',
    'baseline_full_orders_unchanged', 'all_original_inputs_and_frozen_artifacts_unchanged')
FROZEN_FILES = {'PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'INDEPENDENT_PREPARE.json',
                'TRAIN_TRIALS.json', 'TRAIN_OPEN_TRIALS.json', 'SUPPORT_AUDIT.json'}
CACHE_FILES = {'DATA.npz', 'SCORES.npz', 'META.json', 'FEATURE_AUDIT.json',
               'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json', 'SUPPORT_AUDIT.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contract():
    return dict(schema='basic_shares159_fixed_holdings_attribution_v1',
        arms=list(NAMES), issuer_groups=list(ISSUERS),
        formula='(candidate_start_equal_weight-incumbent_start_equal_weight)*(end_close/start_close-1)',
        execution='same-close, gross, fixed units throughout each original holding interval',
        event_mapping='this month available FEATURE_AUDIT cell only; otherwise unattributed',
        summaries='sum and mean of arithmetic monthly excess contributions; not CAGR or wealth',
        leave_one_group='subtract only that group arithmetic contribution from fixed portfolio excess',
        no_new_portfolios=True, no_reselection=True, no_new_features=True,
        no_changes_to_nine_formal_paths=True, costs_or_open_execution_attributed=False,
        causal_effect_claimed=False, strict_source_admission_changed=False,
        slices={k: list(v) for k, v in SLICES.items()},
        reconciliation_absolute_tolerance=1e-12, reconciliation_relative_tolerance=1e-12)


def finite(value, label):
    require(not isinstance(value, bool), 'Boolean numeric '+label)
    value = float(value)
    require(math.isfinite(value), 'Nonfinite '+label)
    return value


def month_contributions(candidate, incumbent, start, end, event_by_symbol):
    """Pure fixed-units endpoint arithmetic; missing held prices fail, never fill."""
    for names in (candidate, incumbent):
        require(isinstance(names, list) and len(names) == len(set(names)) and
                len(names) in (0, 2, 3), 'Fixed cash/EW2/EW3 holdings required')
        require(all(isinstance(s, str) and s for s in names), 'Exact security labels required')
    rows = []
    for symbol in sorted(set(candidate) | set(incumbent)):
        first, last = finite(start[symbol], 'start close'), finite(end[symbol], 'end close')
        require(first > 0 and last > 0, 'Positive held endpoint prices required')
        wc = 1./len(candidate) if symbol in candidate else 0.
        wi = 1./len(incumbent) if symbol in incumbent else 0.
        ret = last/first-1.
        event = event_by_symbol.get(symbol)
        require(event is None or (symbol in ISSUERS and isinstance(event, str) and event),
                'Available fixed-issuer event identity required')
        rows.append(dict(symbol=symbol, candidate_start_weight=wc, incumbent_start_weight=wi,
            weight_difference=wc-wi, start_close=first, end_close=last, holding_return=ret,
            contribution=(wc-wi)*ret, issuer_group=symbol if event is not None else None,
            event_id=event))
    candidate_return = math.fsum(r['candidate_start_weight']*r['holding_return'] for r in rows)
    incumbent_return = math.fsum(r['incumbent_start_weight']*r['holding_return'] for r in rows)
    excess = candidate_return-incumbent_return
    summed = math.fsum(r['contribution'] for r in rows)
    require(math.isclose(excess, summed, abs_tol=1e-12, rel_tol=1e-12),
            'Stock contributions do not reconcile to fixed-holdings gross return difference')
    by_issuer = {s: math.fsum(r['contribution'] for r in rows if r['issuer_group'] == s) for s in ISSUERS}
    unattributed = math.fsum(r['contribution'] for r in rows if r['event_id'] is None)
    require(math.isclose(math.fsum([*by_issuer.values(), unattributed]), excess,
                        abs_tol=1e-12, rel_tol=1e-12), 'Issuer/unattributed reconciliation failed')
    return dict(stocks=rows, candidate_gross_return=candidate_return,
        incumbent_gross_return=incumbent_return, arithmetic_excess_return=excess,
        sum_stock_contributions=summed, reconciliation_error=summed-excess,
        issuer_contributions=by_issuer, unattributed_contribution=unattributed)


def arithmetic_summary(values):
    total = math.fsum(values)
    return dict(months=len(values), sum_arithmetic_monthly_excess=total,
                mean_arithmetic_monthly_excess=total/len(values) if values else 0.)


def grouped_summary(months, available_events):
    baseline = [m['arithmetic_excess_return'] for m in months]
    def group(values):
        return dict(contribution=arithmetic_summary(values),
                    leave_one_group_arithmetic=arithmetic_summary([v-g for v, g in zip(baseline, values)]))
    issuers = {s: group([m['issuer_contributions'][s] for m in months]) for s in ISSUERS}
    events = []
    for symbol, event in sorted(available_events):
        values = [math.fsum(r['contribution'] for r in m['stocks']
                           if r['symbol'] == symbol and r['event_id'] == event) for m in months]
        events.append(dict(symbol=symbol, event_id=event, **group(values)))
    unattributed = group([m['unattributed_contribution'] for m in months])
    combined = math.fsum([*(v['contribution']['sum_arithmetic_monthly_excess'] for v in issuers.values()),
                          unattributed['contribution']['sum_arithmetic_monthly_excess']])
    require(math.isclose(combined, math.fsum(baseline), abs_tol=1e-12, rel_tol=1e-12),
            'Slice grouped contributions do not reconcile')
    event_total = math.fsum(e['contribution']['sum_arithmetic_monthly_excess'] for e in events)
    require(math.isclose(event_total, math.fsum(v['contribution']['sum_arithmetic_monthly_excess']
                        for v in issuers.values()), abs_tol=1e-12, rel_tol=1e-12), 'Event grouping mismatch')
    return dict(fixed_portfolio_excess=arithmetic_summary(baseline), by_issuer=issuers,
                by_event=events, unattributed=unattributed)


def decompose(meta, orders, feature_audit, close_at):
    """Pure supplied-input decomposition on the fixed original122x49 grid."""
    require(len(meta['dates']) == len(meta['ends']) == len(meta['symbols']) == 122,
            'Original122 months required')
    require(meta['train_months'] == 88 and meta['early_months'] == 53, 'Original slices required')
    dates, ends = meta['dates'], meta['ends']
    require(all(date.fromisoformat(d).isoformat() == d for d in dates+ends), 'Exact ISO session dates')
    require(all(a < b for a, b in zip(dates, ends)) and dates[1:] == ends[:-1], 'Adjacent fixed intervals')
    require(set(orders) == set(NAMES) and all(len(orders[n]) == 122 for n in NAMES), 'Fixed three arms/366 orders')
    cells = feature_audit['cells']
    require(len(cells) == 5978, 'Full5978 audited feature cells required')
    event_maps = []
    event_owners = {}
    for i, names in enumerate(meta['symbols']):
        require(len(names) == len(set(names)) == 49, 'Original49 unique candidates')
        event_map = {}
        for j, symbol in enumerate(names):
            c = cells[i*49+j]
            require((c['signal_index'], c['candidate_index'], c['symbol'], c['date']) ==
                    (i, j, symbol, dates[i]), 'Original ordered feature axis')
            require(type(c['available']) is bool, 'Explicit available flag')
            if c['available']:
                event = c['event_id']
                require(symbol in ISSUERS and symbol not in meta['etfs'] and
                        isinstance(event, str) and event, 'Available source identity')
                require(c['original_eligible'] is True, 'Available original-eligible source required')
                require(event not in event_owners or event_owners[event] == symbol, 'Event issuer conflict')
                event_owners[event] = symbol
                event_map[symbol] = event
        event_maps.append(event_map)
        for name in NAMES:
            o = orders[name][i]
            require(o['date'] == o['signal_date'] == dates[i] and o['end'] == ends[i], 'Original order interval')
            require(all(s in names for s in o['symbols']), 'Order outside original candidate grid')
    arms = {}
    for name in NAMES[1:]:
        months = []
        for i in range(122):
            candidate, incumbent = orders[name][i]['symbols'], orders[NAMES[0]][i]['symbols']
            union = set(candidate) | set(incumbent)
            start = {s: close_at(dates[i], s) for s in union}
            end = {s: close_at(ends[i], s) for s in union}
            monthly = month_contributions(candidate, incumbent, start, end, event_maps[i])
            months.append(dict(signal_index=i, start_date=dates[i], end_date=ends[i], **monthly))
        summaries = {}
        for label, (lo, hi) in SLICES.items():
            events = {(s, e) for m in event_maps[lo:hi] for s, e in m.items()}
            summaries[label] = grouped_summary(months[lo:hi], events)
        arms[name] = dict(months=months, summaries=summaries)
    return dict(contract=contract(), arms=arms, months_checked=122, candidate_cells_checked=5978,
                orders_used=366, new_strategy_paths=0, new_feature_calculations=0)


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.seen = {}
    def path(self, value):
        p = Path(value)
        p = (p if p.is_absolute() else self.root/p).resolve()
        require(not any(s in str(p) for s in ('CloudStorage', 'OneDrive', 'stock-observer-runtime')),
                'Forbidden input path')
        return p
    def sha(self, value):
        p = self.path(value)
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        digest = h.hexdigest()
        require(str(p) not in self.seen or self.seen[str(p)] == digest, 'Input changed during attribution: '+str(p))
        self.seen[str(p)] = digest
        return digest
    def bound(self, value, digest):
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest), 'Explicit SHA256 required')
        require(self.sha(value) == digest, 'Input hash mismatch: '+str(value))
    def json(self, value):
        self.sha(value)
        def unique(items):
            out = {}
            for k, v in items:
                require(k not in out, 'Duplicate JSON key')
                out[k] = v
            return out
        def bad(value):
            raise ValueError('Nonfinite JSON token '+value)
        return json.loads(self.path(value).read_text(), object_pairs_hook=unique, parse_constant=bad)
    def verify(self, pins):
        require(type(pins) is dict and pins, 'Nonempty input hash closure required')
        normalized = {}
        for p, h in pins.items():
            key = str(self.path(p))
            require(key not in normalized or normalized[key] == h, 'Conflicting normalized input hash')
            normalized[key] = h
            self.bound(p, h)
        return normalized
    def refresh(self):
        for p, digest in list(self.seen.items()):
            self.bound(p, digest)


def phase_gate(reader, plan_sha, frozen_sha, results_sha, outcome_sha):
    """Read only gate documents first; invalid outcome receipts never read prices."""
    directory = reader.path(OUT)
    docs = {}
    for name, digest in (('PLAN.json', plan_sha), ('FROZEN.json', frozen_sha),
                         ('RESULTS.json', results_sha), ('INDEPENDENT_OUTCOME.json', outcome_sha)):
        reader.bound(directory/name, digest)
        docs[name] = reader.json(directory/name)
    plan, frozen, results, review = [docs[n] for n in
        ('PLAN.json', 'FROZEN.json', 'RESULTS.json', 'INDEPENDENT_OUTCOME.json')]
    require(plan['round'] == 159 and plan['status'] == 'REGISTERED' and plan['arms'] == list(NAMES), 'Registered fixed round159')
    require(reader.path(plan['output_directory']) == directory, 'Fixed output directory')
    require(review.get('schema') == 'basic_shares159_independent_outcome_v1' and
            review.get('passed') is True and review.get('stage') == 'outcome' and
            review.get('producer_modules_imported') is False, 'Passed independent outcome required')
    require(all(review.get('checks', {}).get(k) is True for k in OUTCOME_CHECKS), 'Complete nine-path audit checks')
    require(all(type(review.get(k)) is int and review[k] == v for k, v in
                (('paths_checked', 9), ('orders_checked', 366), ('selection_count', 1))), 'Audited counts')
    require(review['plan_sha256'] == frozen['plan_sha256'] == results['plan_sha256'] == plan_sha and
            review['frozen_sha256'] == frozen_sha and review['results_sha256'] == results_sha, 'Exact phase bindings')
    require(review['selection'] == frozen['selection'] == results['selection'], 'Frozen selection changed')
    require(frozen['selection_count'] == 1 and frozen['later34_diagnostic_only'] is True and
            set(frozen['hashes']) == FROZEN_FILES, 'Original single selection/frozen closure')
    require(set(results['arms']) == set(NAMES) and all(set(a) == {'same_close', 'next_open_zero', 'next_open_cost'}
            for a in results['arms'].values()), 'Exactly nine formal paths remain unchanged')
    # Only after the passed gate is established are source/cache/price hashes read.
    pins = reader.verify(plan['hashes'])
    artifacts = reader.verify(review['artifact_hashes'])
    sources = reader.verify(review['source_hashes'])
    require(all(pins.get(k) == v for k, v in sources.items()), 'Outcome implementation must be registered')
    for name, h in frozen['hashes'].items():
        reader.bound(directory/name, h)
        require(artifacts.get(str(directory/name)) == h, 'Frozen file absent from outcome audit')
    for name, h in (('PLAN.json', plan_sha), ('FROZEN.json', frozen_sha), ('RESULTS.json', results_sha)):
        require(artifacts.get(str(directory/name)) == h, 'Outcome audit lacks current '+name)
    cache = reader.json(directory/'CACHE.json')
    require(set(cache) == CACHE_FILES, 'Complete prepared cache')
    for name, h in cache.items():
        require(artifacts.get(str(directory/name)) == h, 'Cache differs from outcome audit')
        reader.bound(directory/name, h)
    paths = {}
    for key, fixed in (('prices', PRICE), ('meta', 'outputs/monthly_local_linear70/META.json'),
                       ('original_orders', 'outputs/monthly_local_linear70/local_linear_ORDERS.json')):
        ref = plan['inputs'][key]
        p = reader.path(ref['path'])
        require(p == reader.path(fixed) and pins.get(str(p)) == ref['sha256'] == artifacts.get(str(p)),
                'Original audited input required: '+key)
        reader.bound(p, ref['sha256'])
        paths[key] = p
    return plan, paths


def load_prices(path):
    """The sole numerical file decoder, called after phase_gate succeeds."""
    import numpy as np
    with np.load(path, allow_pickle=False) as values:
        require(set(values.files) == {'dates', 'symbols', 'close', 'opening'}, 'Original price schema')
        days = [str(x)[:10] for x in values['dates']]
        symbols = [str(x) for x in values['symbols']]
        close = values['close'].copy()
    require(len(days) == len(set(days)) and days == sorted(days) and
            len(symbols) == len(set(symbols)) and close.shape == (len(days), len(symbols)), 'Price axes')
    require(all(date.fromisoformat(d).isoformat() == d for d in days), 'ISO price dates')
    di, si = {d: i for i, d in enumerate(days)}, {s: i for i, s in enumerate(symbols)}
    return lambda d, s: close[di[d], si[s]]


def write_once(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def run(root, plan_sha, frozen_sha, results_sha, outcome_sha):
    reader = Reader(root)
    directory = reader.path(OUT)
    manifest_path, report_path = directory/'ATTRIBUTION_MANIFEST.json', directory/'ATTRIBUTION.json'
    require(not manifest_path.exists() and not report_path.exists(), 'Write-once attribution already exists')
    plan, paths = phase_gate(reader, plan_sha, frozen_sha, results_sha, outcome_sha)
    meta, orders, feature = [reader.json(directory/n) for n in ('META.json', 'PRETRAIN_ORDERS.json', 'FEATURE_AUDIT.json')]
    require(meta == reader.json(paths['meta']) and orders[NAMES[0]] == reader.json(paths['original_orders']),
            'Original metadata/incumbent orders changed')
    implementation = {str(reader.path(p)): reader.sha(p) for p in
        ('work/basic_shares159_attribution.py', 'work/test_basic_shares159_attribution.py',
         'run_local.sh', 'local_run.py', 'local.sandbox.sb', sys.executable)}
    import numpy as np
    import pandas as pd
    runtime = dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__)
    require(runtime == plan['runtime'], 'Registered runtime required')
    close_at = load_prices(paths['prices'])
    report = decompose(meta, orders, feature, close_at)
    reader.refresh()
    manifest = dict(schema='basic_shares159_attribution_manifest_v1', contract=contract(),
        plan_sha256=plan_sha, frozen_sha256=frozen_sha, results_sha256=results_sha,
        independent_outcome_sha256=outcome_sha, runtime=runtime,
        implementation_hashes=implementation, input_hashes=dict(reader.seen))
    write_once(manifest_path, manifest)
    report.update(plan_sha256=plan_sha, independent_outcome_sha256=outcome_sha,
                  manifest_sha256=reader.sha(manifest_path))
    reader.refresh()
    write_once(report_path, report)
    return dict(report=str(report_path), sha256=reader.sha(report_path),
                manifest=str(manifest_path), manifest_sha256=reader.sha(manifest_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    for name in ('plan', 'frozen', 'results', 'outcome'):
        parser.add_argument('--'+name+'-sha')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(contract(), indent=2))
    else:
        print(json.dumps(run(Path(__file__).resolve().parents[1], args.plan_sha,
                             args.frozen_sha, args.results_sha, args.outcome_sha)))


if __name__ == '__main__':
    main()
