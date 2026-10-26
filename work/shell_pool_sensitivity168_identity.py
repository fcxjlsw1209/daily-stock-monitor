"""Identity-only proof for the registered Shell pool sensitivity.

No strategy imports, numerical features, model fits, portfolio selection or
returns. Read only date/identity arrays from NPZ files; never price arrays.
"""
import argparse
import bisect
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_sensitivity168'
LOCAL = 'outputs/monthly_local_linear70/'
PINS = {
    'outputs/monthly_shell_pool_sensitivity168/PLAN.json':
        'c50e7a4edd9b83a23efe7a290f6fa82d5f920141b02aff1aed34520191c7d2a6',
    'outputs/monthly_shell_pool_sensitivity168/DATA_COMPARISON.json':
        '6b6afc5fa84a3abcbeb05e108e2aecb35c6fc30e5c85c274c0b2455ac6980c32',
    'outputs/monthly_shell_pool_sensitivity168/CONDITIONAL_POOLS.json':
        '6efd9c7e4d47b0480453fde1f549ac942205e58bacbb7a9a6ae26b96c6a1132e',
    'outputs/monthly_sector_diffusion167/PLAN.json':
        '807bffe7ae4c1e758e516613b15ff36a71703d3ae2e2a7761a756cb3909f5501',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def path(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require(p.is_relative_to(ROOT) and p.resolve() == p and
            'CloudStorage' not in p.parts, 'Nonlocal/symlink input: ' + str(p))
    return p


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(p):
    return json.loads(path(p).read_text())


def actual(symbol):
    return {'META': 'research_equity_FB', 'BRK-B': 'BRK.B'}.get(symbol, symbol)


def normalized_pool(pool, etfs):
    require(len(pool) == 32 and len(set(pool)) == 32, 'Not 32 unique issuer names')
    names = set(map(actual, pool))
    require(len(names) == 32 and not names.intersection(etfs), 'Alias/ETF collision')
    return sorted(names | set(etfs))


def member_table(pools, columns):
    dates = sorted(pools)
    require(len(columns) == len(set(columns)), 'Duplicate price columns')
    member = pd.DataFrame(False, index=pd.to_datetime(dates), columns=columns)
    for day in dates:
        names = [actual(s) for s in pools[day]]
        require(len(names) == len(set(names)) == 32, 'Invalid mapped membership')
        require(set(names).issubset(columns), 'Member absent from price axis: ' + day)
        member.loc[pd.Timestamp(day), names] = True
    return member


def training_rows(signals, ends, member, legacy, etfs):
    """Independently reproduce latest-known membership, never rank values."""
    require(len(signals) == len(ends), 'Signal/end count differs')
    known = member.index.strftime('%Y-%m-%d').tolist()
    rows = []
    for signal, end in zip(signals, ends):
        j = bisect.bisect_right(known, signal) - 1
        if j >= 0:
            origin = known[j]
            flags = member.iloc[j]
        else:
            origin = 'inherited_pre2016_historical_proxy'
            require(pd.Timestamp(signal) in legacy.index, 'Missing legacy date: ' + signal)
            flags = legacy.loc[signal]
        names = sorted(set(flags.index[flags]) | set(etfs))
        require(len(names) == len(set(names)) == 49, 'Training width/identity: ' + signal)
        rows.append(dict(date=signal, end=end, symbols=names, pool_origin=origin))
    return rows


def cached_rows(meta):
    require(len({len(meta[k]) for k in ('dates', 'ends', 'symbols', 'pool_origins')}) == 1,
            'Cached metadata arrays have different lengths')
    return [dict(date=d, end=e, symbols=s, pool_origin=p) for d, e, s, p in
            zip(meta['dates'], meta['ends'], meta['symbols'], meta['pool_origins'])]


def report_rows(old, new, cached):
    require(old == new, 'Old/new training identity differs')
    require(old == cached, 'Reconstructed training identity differs from existing cache')
    return dict(periods=len(old), security_rows=sum(len(r['symbols']) for r in old),
                pre_archive_periods=sum(r['pool_origin'].startswith('inherited_') for r in old),
                old_sha256=digest(old), new_sha256=digest(new), cached_sha256=digest(cached),
                records=[dict(date=r['date'], end=r['end'], pool_origin=r['pool_origin'],
                              names=len(r['symbols']), symbols_sha256=digest(r['symbols']),
                              old_new_equal=True, cache_equal=True) for r in old])


def run():
    require(not (OUT / 'INPUT_IDENTITY.json').exists(), 'Identity receipt already exists')
    require(platform.python_version() == '3.9.6' and np.__version__ == '2.0.2' and
            pd.__version__ == '2.3.3', 'Registered local runtime required')
    checked = {}

    def verify(mapping):
        for name, expected in mapping.items():
            p = path(name)
            key = str(p.relative_to(ROOT))
            require(key not in checked or checked[key] == expected, 'Conflicting pin: ' + key)
            require(p.is_file(), 'Required input missing: ' + str(p))
            require(sha(p) == expected, 'Input SHA mismatch: ' + key)
            checked[key] = expected

    verify(PINS)
    plan = read(OUT / 'PLAN.json')
    comparison = read(OUT / 'DATA_COMPARISON.json')
    require(comparison['plan_sha256'] == PINS['outputs/monthly_shell_pool_sensitivity168/PLAN.json'],
            'Data comparison not bound to registered plan')
    verify(plan['inputs'])
    verify({'work/shell_pool_sensitivity168.py': plan['script_sha256']})
    prior = read('outputs/monthly_sector_diffusion167/PLAN.json')
    prior_pins = {str(path(k)): v for k, v in prior['hashes'].items()
                  if str(k).startswith(str(ROOT) + '/')}
    verify({LOCAL + k: prior_pins[str(path(LOCAL + k))] for k in
            ('META.json', 'ROW_IDENTITIES.npz', 'CACHE_HASHES.json', 'INPUT_HASHES.json',
             'local_linear_ORDERS.json')})
    verify(read(LOCAL + 'INPUT_HASHES.json'))
    verify({LOCAL + k: v for k, v in read(LOCAL + 'CACHE_HASHES.json').items()})
    verify(prior['raw_source_hashes'])
    protected = prior['inputs']['protected_manifest']
    verify({protected['path']: protected['sha256']})
    archive = read(protected['path'])
    named = [
        'outputs/monthly_2016_extension/INPUTS.json',
        'outputs/monthly_2016_extension/PRICES.npz',
        'outputs/monthly_2016_extension/EARLY_RANKS.json',
        'outputs/monthly_2016_extension/POOLS.json',
        'outputs/monthly_top35_extension/POOLS.json',
        'outputs/monthly_reconstruction_v2/foreign/membership.csv.gz',
        'outputs/monthly_architecture_restart/META.json',
        'outputs/monthly_architecture_restart/DATA.npz',
        'outputs/monthly_blend_optimization/MONTHLY_TRAIN.json',
        'outputs/monthly_blend_optimization/MONTHLY_TRAIN.npz',
        'outputs/monthly_state_rank_research/TRAIN_DATA.json',
        'outputs/monthly_state_rank_research/TRAIN_DATA.npz',
        'outputs/monthly_state_rank_research/state_rank.py',
        'outputs/monthly_blend_optimization/blend_research.py',
        'outputs/monthly_stability_monitor/inputs.py',
    ]
    for name in named:
        p = str(path(name))
        expected = prior_pins.get(p, archive.get(p))
        require(expected is not None, 'Required metadata has no existing source pin: ' + name)
        verify({name: expected})

    old_pools = read('outputs/monthly_2016_extension/POOLS.json')
    new_pools = read(OUT / 'CONDITIONAL_POOLS.json')
    meta = read(LOCAL + 'META.json')
    extension_meta = read('outputs/monthly_2016_extension/INPUTS.json')
    require(meta == extension_meta, 'Current and extension event metadata differ')
    etfs = meta['etfs']
    require(len(etfs) == len(set(etfs)) == 17, 'Exact 17 ETF identities required')
    dates = sorted(old_pools)
    require(dates == sorted(new_pools) == meta['dates'] and len(dates) == 122,
            'Pool dates differ or are not the current 122-event axis')
    old_names = [normalized_pool(old_pools[d], etfs) for d in dates]
    new_names = [normalized_pool(new_pools[d], etfs) for d in dates]
    require(old_names == new_names == meta['symbols'], 'Mapped event identities differ')
    require(meta['ends'] == dates[1:] + ['2026-09-24'], 'Event end dates differ')
    early = read('outputs/monthly_2016_extension/EARLY_RANKS.json')
    rows = comparison['rows']
    require(len(early) == len(rows) == 53, '53 early records required')
    for source, item, day in zip(early, rows, dates[:53]):
        require(source['date'] == item['signal_date'] == day and source['asof'] == item['asof'],
                'Early date/asof misalignment')
        require([r['symbol'] for r in source['top40'][:32]] == item['old_top32'] == old_pools[day],
                'Old ranked list differs from original archive')
        require(item['new_top32'] == new_pools[day], 'Conditional ranked list differs')
        require(set(item['old_top32']) == set(item['new_top32']), 'Changed raw issuer set')
        require(item['added'] == item['removed'] == [], 'Changed membership in comparison')
    require(all(old_pools[d] == new_pools[d] for d in dates[53:]), 'Later pool changed')
    order_changes = sum(old_pools[d] != new_pools[d] for d in dates[:53])
    require(comparison['summary']['membership_changed_events'] == 0 and
            comparison['summary']['ordered_list_changed_events'] == order_changes,
            'Summary inconsistent with independently compared lists')

    # NPZ accesses below explicitly select axes only, never close/open/features/labels.
    with np.load(path('outputs/monthly_2016_extension/PRICES.npz'), allow_pickle=False) as z:
        price_dates = z['dates'].astype(str).tolist()
        columns = z['symbols'].astype(str).tolist()
    raw = path('outputs/index_union/alias_resolved_inputs/adj.csv.gz')
    sessions = pd.read_csv(raw, usecols=[0], index_col=0, parse_dates=True).index
    require(sessions.is_unique and sessions.is_monotonic_increasing, 'Invalid raw session axis')
    require(sessions.strftime('%Y-%m-%d').tolist() == price_dates, 'Raw/cached price date axis differs')
    later = read('outputs/monthly_top35_extension/POOLS.json')['32']
    merged_old = {**{d: s for d, s in old_pools.items() if d < '2020-12-31'}, **later}
    merged_new = {**{d: s for d, s in new_pools.items() if d < '2020-12-31'}, **later}
    old_member, new_member = [member_table(p, columns) for p in (merged_old, merged_new)]
    pd.testing.assert_frame_equal(old_member, new_member)
    require(all(int(n) == 32 for n in old_member.sum(axis=1)), 'Member width is not 32')
    legacy = pd.read_csv(path('outputs/monthly_reconstruction_v2/foreign/membership.csv.gz'),
                         index_col=0, parse_dates=True).fillna(False).astype(bool)
    require(legacy.index.is_unique, 'Duplicate legacy membership date')

    monthends = sessions.to_series().groupby(sessions.to_period('M')).last().tolist()
    monthly_dates, monthly_ends = [], []
    for i, signal in enumerate(monthends[:-1]):
        if pd.Timestamp('2011-01-01') <= signal <= pd.Timestamp('2025-12-31'):
            monthly_dates.append(str(signal.date()))
            monthly_ends.append(str(monthends[i + 1].date()))
    weekly_axis = sessions[(sessions >= '2011-01-01') & (sessions <= '2025-12-31')]
    weekly_signals = weekly_axis.to_series().groupby(weekly_axis.to_period('W-FRI')).last()
    weekly_dates = [str(d.date()) for d in weekly_signals]
    weekly_ends = []
    for signal in weekly_signals:
        pos = sessions.get_loc(signal)
        require(pos + 21 < len(sessions), 'Missing weekly label-end session')
        weekly_ends.append(str(sessions[pos + 21].date()))
    tm = read('outputs/monthly_blend_optimization/MONTHLY_TRAIN.json')
    wm = read('outputs/monthly_state_rank_research/TRAIN_DATA.json')
    arch = read('outputs/monthly_architecture_restart/META.json')
    require(arch['monthly'] == meta and arch['training'] == tm, 'Architecture identity differs')
    reports = {}
    for name, signals, ends, cache in [('monthly', monthly_dates, monthly_ends, tm),
                                      ('weekly', weekly_dates, weekly_ends, wm)]:
        original = training_rows(signals, ends, old_member, legacy, etfs)
        updated = training_rows(signals, ends, new_member, legacy, etfs)
        reports[name] = report_rows(original, updated, cached_rows(cache))
        offsets = np.r_[0, np.cumsum([len(r['symbols']) for r in original])]
        cache_file = ('outputs/monthly_blend_optimization/MONTHLY_TRAIN.npz' if name == 'monthly'
                      else 'outputs/monthly_state_rank_research/TRAIN_DATA.npz')
        with np.load(path(cache_file), allow_pickle=False) as z:
            np.testing.assert_array_equal(offsets, z['offsets'])
        if name == 'monthly':
            with np.load(path(LOCAL + 'ROW_IDENTITIES.npz'), allow_pickle=False) as z:
                require(set(z.files) == {'dates', 'ends', 'symbols', 'event_dates', 'event_symbols', 'offsets'},
                        'Unexpected current row identity fields')
                np.testing.assert_array_equal(z['offsets'], offsets)
                for key, values in [('dates', signals), ('ends', ends)]:
                    require(z[key].astype('datetime64[D]').astype(str).tolist() ==
                            [d for d, r in zip(values, original) for _ in r['symbols']],
                            'Current flattened ' + key + ' differs')
                require(z['symbols'].astype(str).tolist() == [s for r in original for s in r['symbols']],
                        'Current flattened training symbols differ')
                require(z['event_dates'].astype('datetime64[D]').astype(str).tolist() ==
                        [d for d, ss in zip(dates, new_names) for _ in ss], 'Current event dates differ')
                require(z['event_symbols'].astype(str).tolist() == [s for ss in new_names for s in ss],
                        'Current flattened event symbols differ')
            with np.load(path('outputs/monthly_architecture_restart/DATA.npz'), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['offsets'], offsets)

    verify(dict(checked))  # Detect source mutation during this proof.
    code = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for name in ('run_local.sh', 'local_run.py', 'local.sandbox.sb'):
        code[name] = sha(path(name))
    payload = dict(schema='shell_pool_sensitivity168_input_identity_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        plan_sha256=PINS['outputs/monthly_shell_pool_sensitivity168/PLAN.json'],
        scope='Identity equivalence under registered conditional proxy; not net market-cap/PIT certification',
        summary=dict(early_periods=53, raw_ranked_list_changes=order_changes,
            mapped_member_set_changes=0, event_periods=122, event_security_rows=sum(map(len, new_names)),
            monthly_training_periods=reports['monthly']['periods'],
            weekly_training_periods=reports['weekly']['periods'],
            mapped_events_equal=True, boolean_member_equal=True,
            monthly_identity_and_origin_equal=True, weekly_identity_and_origin_equal=True,
            current_row_identities_exact=True, numerical_recomputation_needed=False),
        events=dict(old_sha256=digest(old_names), new_sha256=digest(new_names),
            cached_sha256=digest(meta['symbols']), records=[dict(date=d, end=e,
                original_rank_order_changed=old_pools[d] != new_pools[d],
                symbols=ss, same_mapped_identity=True) for d, e, ss in zip(dates, meta['ends'], new_names)]),
        member=dict(dates=old_member.index.strftime('%Y-%m-%d').tolist(), columns=columns,
            shape=list(old_member.shape), dtype='bool',
            old_values_sha256=hashlib.sha256(old_member.to_numpy().tobytes()).hexdigest(),
            new_values_sha256=hashlib.sha256(new_member.to_numpy().tobytes()).hexdigest(),
            construction='early dates<2020-12-31 then later32 override; actual aliases; 32 true columns per date'),
        training=reports, input_sha256=checked, proof_code_sha256=code,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        inference=dict(strategy='monthly_local_linear70 id4 / local_linear',
            statement='Only discarded market-cap ordering differs. All consumed identity axes and source bytes remain identical, so deterministic existing features, fitted models, scores and orders are unchanged.',
            numerical_replay_performed=False, original_orders_sha256=checked.get(LOCAL + 'local_linear_ORDERS.json'),
            limits=['Does not certify current net shares, market cap, ADR or quote assumptions.',
                    'Does not repair pre-first-archive membership proxy or historical price-vintage limitations.',
                    'Equivalence is for these bound inputs and this exact conditional pool; not a generic claim for changed membership.']),
        source_assumptions=plan['assumptions'], features_computed=0, model_fits=0,
        orders_computed=0, returns_computed=0, strategy_trials_added=0, frozen_files_modified=False)
    with (OUT / 'INPUT_IDENTITY.json').open('x') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(passed=True, **payload['summary'],
                          receipt_sha256=sha(OUT / 'INPUT_IDENTITY.json'))))


def self_test():
    import unittest

    class IdentityTests(unittest.TestCase):
        def test_rank_permutation_erased(self):
            pool = ['S' + str(i) for i in range(32)]
            self.assertEqual(normalized_pool(pool, ['ETF']), normalized_pool(pool[::-1], ['ETF']))

        def test_alias_collision_rejected(self):
            with self.assertRaises(ValueError):
                normalized_pool(['META', 'research_equity_FB'] + ['S' + str(i) for i in range(30)], [])

        def test_asof_is_not_future(self):
            ss = ['S' + str(i) for i in range(32)]
            etfs = ['E' + str(i) for i in range(17)]
            member = member_table({'2016-08-17': ss}, ss)
            legacy = pd.DataFrame(True, index=pd.to_datetime(['2016-08-16']), columns=ss)
            rows = training_rows(['2016-08-16', '2016-08-17'], ['2016-09-16', '2016-09-17'], member, legacy, etfs)
            self.assertEqual(rows[0]['pool_origin'], 'inherited_pre2016_historical_proxy')
            self.assertEqual(rows[1]['pool_origin'], '2016-08-17')

        def test_missing_legacy_fails(self):
            member = member_table({'2016-08-17': ['S' + str(i) for i in range(32)]}, ['S' + str(i) for i in range(32)])
            with self.assertRaises(ValueError):
                training_rows(['2016-08-16'], ['2016-09-16'], member, pd.DataFrame(), [])

        def test_cached_identity_change_fails(self):
            row = dict(date='2016-08-17', end='2016-09-17', symbols=['A'], pool_origin='2016-08-17')
            with self.assertRaises(ValueError):
                report_rows([row], [row], [dict(row, symbols=['B'])])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IdentityTests))
    require(result.wasSuccessful(), 'Synthetic identity tests failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run()
