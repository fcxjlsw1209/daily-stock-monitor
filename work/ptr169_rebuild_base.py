"""Pure recalculation of the frozen local67 base for an explicitly supplied pool.

No prepare/load/prices entry point, old computed cache, model cache, or output
writer is called. Real-data execution belongs to the separately registered
host. The only standalone command is a synthetic test suite.
"""
from functools import lru_cache
from pathlib import Path
from types import FunctionType, ModuleType
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
FIELDS = ('adj', 'open', 'close', 'high', 'low', 'volume')
PRIOR = [dict(family='residual', key='resid_126_210_sum', amount=.025, state='calm10'),
         dict(key='ml_4', state='weak_narrow', amount=.05)]
WEEKLY_CONFIG = dict(target='downside_rank', alpha=10., half_life=12)
EPISODE_CONFIG = dict(id=2, horizon=5, match='signal', target='utility', mode='level')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@lru_cache(None)
def dependencies():
    # Importing definitions does not call their file-writing CLI entry points.
    state = module('ptr169_frozen_state', OUTPUTS / 'monthly_state_rank_research/state_rank.py')
    episode = module('ptr169_frozen_episode', OUTPUTS / 'monthly_daily_episodes/episodes.py')
    base = module('ptr169_frozen_base', OUTPUTS / 'monthly_stability_monitor/base_strategy.py')
    require(state.MODEL_CONFIGS[4] == WEEKLY_CONFIG, 'Frozen weekly configuration drift')
    require(episode.models()[2] == EPISODE_CONFIG, 'Frozen episode configuration drift')
    return state, episode, base


def dependency_files():
    """Actual imported local source graph for the host's pre-execution pins."""
    pending, seen, files = list(dependencies()), set(), {
        Path(__file__).resolve(),
        OUTPUTS / 'monthly_stability_monitor/strategy.py',
    }
    while pending:
        value = pending.pop()
        if id(value) in seen: continue
        seen.add(id(value))
        name = value.__code__.co_filename if isinstance(value, FunctionType) else getattr(value, '__file__', None)
        if name is None: continue
        path = Path(name).resolve()
        if OUTPUTS not in path.parents: continue
        files.add(path)
        namespace = value.__globals__ if isinstance(value, FunctionType) else vars(value)
        pending.extend(v for v in namespace.values() if isinstance(v, (ModuleType, FunctionType)))
    return tuple(sorted(files))


def required_source_files():
    return (OUTPUTS / 'monthly_state_rank_research/TRAIN_DATA.json',
            OUTPUTS / 'monthly_reconstruction_v2/foreign/membership.csv.gz',
            OUTPUTS / 'monthly_stability_research/LOCAL_FROZEN.json',
            OUTPUTS / 'monthly_2016_extension/FROZEN.json')


def validate_meta(m):
    require(len(m['dates']) == len(m['ends']) == len(m['symbols']) > 0, 'Unaligned event metadata')
    require(m['dates'] == sorted(set(m['dates'])), 'Event dates must be unique and sorted')
    width = len(m['symbols'][0])
    for date, end, ss in zip(m['dates'], m['ends'], m['symbols']):
        require(len(ss) == width and ss == sorted(set(ss)), 'Sorted unique equal-width event symbols required')
        require(date < end, 'Event label end must follow signal')
        require(set(m['etfs']).issubset(ss), 'Each event must include the frozen ETFs')
    return len(m['dates']), width


def validate_panel(panel):
    px = panel['adj']
    require(px.index.is_unique and px.index.is_monotonic_increasing and px.columns.is_unique,
            'Invalid panel axes')
    require({'SPY', 'QQQ'}.issubset(px.columns), 'Market reference histories missing')
    for key in FIELDS:
        require(panel[key].index.equals(px.index) and panel[key].columns.equals(px.columns),
                'Raw field axes differ: ' + key)
    member = panel['member']
    require(member.index.is_unique and member.index.is_monotonic_increasing and member.columns.is_unique,
            'Invalid membership axes')
    require(all(dtype == bool for dtype in member.dtypes) and not member.isna().any().any(),
            'Membership must be explicit boolean data')


def members_at(member, date, etfs, legacy=None):
    date = pd.Timestamp(date)
    previous = member.index[member.index <= date]
    if len(previous):
        row = member.loc[previous[-1]]
        origin = str(previous[-1].date())
    else:
        require(legacy is not None and date in legacy.index, 'Missing pre-archive membership at ' + str(date.date()))
        row = legacy.loc[date]
        origin = 'inherited_pre2016_historical_proxy'
    require(row.dtype == bool and not row.isna().any(), 'Invalid membership row')
    return sorted(set(row.index[row]) | set(etfs)), origin


def base_features(panel, m, px):
    """Rebuild original anchor arithmetic plus robust_control industry mixture.

    The anchor snapshot receives only its used raw fields. Unused historical
    calendar/sector60 features are deliberately not substituted or evaluated.
    Missing values have exactly the original per-column percentile semantics.
    """
    state, _, base = dependencies()
    n, width = validate_meta(m)
    require(px.index.is_unique and px.columns.is_unique, 'Invalid baseline price axes')
    raw = {key: panel[key].reindex(index=px.index, columns=px.columns) for key in FIELDS if key != 'adj'}
    values, eligible, regime = base.factors(px, raw['close'], raw['high'], raw['low'], raw['volume'])
    momenta = {length: px.shift(42) / px.shift(length) - 1 for length in (126, 252)}
    feature_subset = dict(risk294=values['momentum'], vol126=values['volatility'],
                          acceleration=values['acceleration'], neg_cmf63=values['regime_cmf'],
                          raw252=momenta[252])
    r = px.pct_change(fill_method=None)
    state_frame = pd.DataFrame({'high_vol': regime > 0})
    warm = px.notna().rolling(295).sum().eq(295) & px.gt(0)
    corr = {b: (r.rolling(63).corr(r[b]), r.rolling(252).corr(r[b]))
            for b in state.ex.t.strict.SECTOR.values()}
    beta = r.rolling(63).cov(r.SPY).div(r.SPY.rolling(63).var(), axis=0).shift(1)
    residual = r - beta.mul(r.SPY, axis=0)
    extreme = r.SPY.lt(r.SPY.rolling(252).quantile(.1).shift(1))
    tail = residual.where(extreme, axis=0).rolling(252, min_periods=10).mean()
    logret = np.log(px / px.shift(1))
    efficiency = np.log(px.shift(42) / px.shift(294)) / logret.abs().rolling(252).sum().shift(42).replace(0, np.nan)
    calm = r.SPY.rolling(63).std() * np.sqrt(252) < .10
    a = dict(eligible=np.empty((n, width), bool), state_weak_narrow=np.empty(n, bool))
    d = {key: [] for key in ('core', 'support')}
    x = {key: [] for key in ('vol126', 'core252', 'sector126', 'sector252', 'stability', 'tail')}
    f = {key: [] for key in ('252_raw', '294_v63', 'path', 'vol', 'accel', 'flow', 'sector', 'stability', 'tail')}
    risk252 = momenta[252] / r.rolling(63).std().replace(0, np.nan)
    states = []
    for i, (date, ss) in enumerate(zip(m['dates'], m['symbols'])):
        day = pd.Timestamp(date)
        expected, _ = members_at(panel['member'], day, m['etfs'])
        require(ss == expected, 'Event symbols do not match passed membership at ' + date)
        snap = state.ex.t.anchor.snapshot(day, ss, feature_subset, state_frame, warm)
        q = snap['q']
        original_core = snap['comp']['momentum'].to_numpy()
        support = snap['score'].to_numpy() - original_core
        # The historical stability feature builder recovered CMF this way.
        flow = (support - .219375 * q.vol126.to_numpy() - .185625 * q.acceleration.to_numpy()) / .075
        mapping = state.ex.t.strict.benchmark_map(day, ss, panel['sectors'], set(m['etfs']))
        rel126 = state.prior.ranks(state.ex.t.strict.relative_row(momenta[126], day, ss, mapping))
        rel252 = state.prior.ranks(state.ex.t.strict.relative_row(momenta[252], day, ss, mapping))
        stability = state.prior.ranks([-abs(corr[b][0].at[day, s] - corr[b][1].at[day, s])
                                      if b is not None else np.nan for s, b in mapping.items()])
        tailrank = state.prior.ranks(tail.loc[day, ss])
        d['core'].append(original_core); d['support'].append(support)
        values_x = (q.vol126.to_numpy(), state.prior.ranks(risk252.loc[day, ss]), rel126, rel252, stability, tailrank)
        for key, value in zip(x, values_x): x[key].append(value)
        values_f = (state.prior.ranks(momenta[252].loc[day, ss]), q.risk294.to_numpy(),
                    state.prior.ranks(efficiency.loc[day, ss]), q.vol126.to_numpy(),
                    q.acceleration.to_numpy(), flow, .5 * rel126 + .5 * rel252, stability, tailrank)
        for key, value in zip(f, values_f): f[key].append(value)
        st = state.state_row(panel['adj'], day, ss, set(m['etfs']))
        a['eligible'][i] = snap['eligible'].to_numpy(bool)
        require(np.array_equal(a['eligible'][i], eligible.loc[day, ss].to_numpy(bool)), 'Eligibility algebra mismatch')
        a['state_weak_narrow'][i] = bool(st['weak_narrow'])
        states.append(st)
    d = {key: np.stack(value) for key, value in d.items()}
    d.update(eligible=a['eligible'], high=(regime.loc[m['dates']] > 0).to_numpy(),
             weak=(~(px.SPY > px.SPY.rolling(200).mean())).loc[m['dates']].to_numpy())
    x = {key: np.stack(value) for key, value in x.items()}
    f = {key: np.stack(value) for key, value in f.items()}
    # Reuse the frozen optimize.score -> local_refine.score arithmetic. The
    # only on-disk reads here are its two immutable configuration JSON files.
    a['baseline'], a['core'], a['core_weight'], _, _, _, frozen_states = state.prior.components(m, d, {}, x, px, f)
    a['state_calm10'] = frozen_states['calm10']
    require(np.array_equal(a['state_calm10'], calm.loc[m['dates']].to_numpy()), 'Calm state mismatch')
    for key in ('baseline', 'core', 'core_weight'): require(np.isfinite(a[key]).all(), 'Nonfinite ' + key)
    return a, states


def residual_feature(px, sectors, m):
    state, _, _ = dependencies()
    raw = state.prior.residual_returns(px, sectors, 126).rolling(210).sum().shift(42)
    rows = []
    for date, ss in zip(m['dates'], m['symbols']):
        row = raw.loc[date, ss].copy()
        row.loc[list(set(ss) & set(m['etfs']))] = np.nan
        rows.append(state.prior.ranks(row))
    return np.stack(rows)


def weekly_inputs(panel, m, weekly_meta=None):
    state, _, _ = dependencies()
    sources = {}
    if weekly_meta is None:
        source = required_source_files()[0]
        seed = json.loads(source.read_text())
        sources[str(source)] = sha(source)
    else: seed = copy.deepcopy(weekly_meta)
    require(len(seed['dates']) == len(seed['ends']) and seed['dates'] == sorted(set(seed['dates'])),
            'Invalid weekly date/end metadata')
    legacy = None
    if any(pd.Timestamp(date) < panel['member'].index[0] for date in seed['dates']):
        source = required_source_files()[1]
        legacy = pd.read_csv(source, index_col=0, parse_dates=True).fillna(False).astype(bool)
        sources[str(source)] = sha(source)
    full = panel['adj']
    feat, mom = state.raw_features(full, panel)
    warm = full.notna().rolling(295).sum().eq(295) & full.gt(0)
    chunks = {key: [] for key in ('X', 'Y', 'adverse', 'good', 'past_score')}
    offsets, symbols, origins, ends = [0], [], [], []
    for date, expected_end in zip(seed['dates'], seed['ends']):
        day = pd.Timestamp(date)
        ss, origin = members_at(panel['member'], day, m['etfs'], legacy)
        st = state.state_row(full, day, ss, set(m['etfs']))
        xx, score, names = state.feature_row(day, ss, feat, mom, full, panel['sectors'], set(m['etfs']), st)
        end, y, adverse, valid = state.forward_labels(full, day, ss, 21)
        require(end == expected_end, 'Weekly calendar label-end drift at ' + date)
        good = (warm.loc[day, ss] & mom[252].loc[day, ss].gt(0)).to_numpy() & valid
        for key, value in zip(chunks, (xx, y, adverse, good, score)): chunks[key].append(value)
        symbols.append(ss); origins.append(origin); ends.append(end); offsets.append(offsets[-1] + len(ss))
    require(bool(symbols), 'No weekly training dates')
    events, monthly_states = [], []
    for date, ss in zip(m['dates'], m['symbols']):
        st = state.state_row(full, date, ss, set(m['etfs']))
        xx, _, event_names = state.feature_row(date, ss, feat, mom, full, panel['sectors'], set(m['etfs']), st)
        require(event_names == names and xx.shape[1] == 84, 'Frozen weekly feature schema drift')
        events.append(xx); monthly_states.append(st)
    data = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    data.update(offsets=np.array(offsets, dtype=int), event=np.stack(events))
    meta = dict(dates=list(seed['dates']), ends=ends, symbols=symbols, pool_origins=origins,
                feature_names=names, feature_interactions=['raw', 'breadth', 'SPY200trend', 'market_highvol'],
                monthly_states=monthly_states)
    if weekly_meta is not None:
        for key in ('symbols', 'pool_origins'):
            if key in weekly_meta: require(meta[key] == weekly_meta[key], 'Supplied weekly identity mismatch: ' + key)
    require(np.isfinite(data['X']).all() and np.isfinite(data['event']).all(), 'Nonfinite ranked weekly features')
    return data, meta, sources


def weekly_forecasts(data, meta, m, eligible):
    state, _, _ = dependencies()
    prediction = np.empty(eligible.shape)
    audit = []
    for year in sorted({pd.Timestamp(end).year for end in m['ends']}):
        ids = [i for i, end in enumerate(m['ends']) if pd.Timestamp(end).year == year]
        first = min(m['dates'][i] for i in ids)
        cutoff = min(first, str((pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=1)).date()))
        coef, intercept, info = state.fit_year(data, meta, cutoff, dict(WEEKLY_CONFIG))
        require(info['latest_label_end'] <= cutoff <= first, 'Unmatured weekly training labels')
        values = np.einsum('ijk,k->ij', data['event'][ids], coef, optimize=False) + intercept
        require(np.isfinite(values).all(), 'Nonfinite weekly predictions')
        for j, i in enumerate(ids): prediction[i] = state.prior.ranks(pd.Series(values[j]).where(eligible[i]))
        audit.append(dict(year=year, cutoff=cutoff, config=dict(WEEKLY_CONFIG), **info,
                          coef=coef.tolist(), intercept=float(intercept)))
    return prediction, audit


def episode_effects(panel, m, eligible):
    _, ep, _ = dependencies()
    symbols = sorted(set(sum(m['symbols'], [])))
    needed = sorted(set(symbols) | {'SPY', 'QQQ'})
    clean = {key: panel[key].reindex(columns=needed).loc[:m['ends'][-1]].copy() for key in FIELDS}
    shape, market, signal, good, vol = ep.features(clean)
    dates = clean['adj'].index.to_numpy(dtype='datetime64[D]')
    opening = clean['adj'] * clean['open'] / clean['close']
    ev = {key: [] for key in ('symbol', 'dates', 'anchor_index', 'signal', 'shape', 'market', 'ends_5', 'utility_5')}
    for symbol in symbols:
        for idx in ep.anchors(signal[symbol].to_numpy(), good[symbol].to_numpy()):
            _, utility = ep.label_path(clean['adj'][symbol].to_numpy(), opening[symbol].to_numpy(),
                                       clean['adj'].SPY.to_numpy(), opening.SPY.to_numpy(), idx, 5)
            values = (symbol, dates[idx], idx, int(signal[symbol].iloc[idx]),
                      [frame[symbol].iloc[idx] for frame in shape], market.iloc[idx].to_numpy(),
                      dates[idx + 5] if idx + 5 < len(dates) else np.datetime64('2262-01-01'), utility)
            for key, value in zip(ev, values): ev[key].append(value)
    events = {key: np.asarray(value, dtype='datetime64[D]' if key in ('dates', 'ends_5') else None)
              for key, value in ev.items()}
    effects = np.zeros(eligible.shape)
    query = dict(shape=np.empty((*eligible.shape, 8)), market=np.empty((len(m['dates']), 3)),
                 signal=np.empty(eligible.shape, int), vol=np.empty(eligible.shape))
    details = []
    for i, (date, ss) in enumerate(zip(m['dates'], m['symbols'])):
        query['shape'][i] = np.column_stack([frame.loc[date, ss] for frame in shape])
        query['market'][i] = market.loc[date]
        query['signal'][i] = signal.loc[date, ss].to_numpy(int)
        query['vol'][i] = vol.loc[date, ss]
        for j, symbol in enumerate(ss):
            q = dict(symbol=symbol, date=np.datetime64(date, 'D'), eligible=bool(eligible[i, j]),
                     shape=query['shape'][i, j], market=query['market'][i],
                     signal=query['signal'][i, j], vol=query['vol'][i, j])
            if not (np.isfinite(q['shape']).all() and np.isfinite(q['market']).all() and np.isfinite(q['vol'])):
                q['eligible'] = False
            effects[i, j], info = ep.estimate(events, q, dict(EPISODE_CONFIG))
            details.append(dict(date=date, symbol=symbol, signal_type=int(q['signal']),
                                effect=float(effects[i, j]), **info))
    require(np.isfinite(effects).all(), 'Nonfinite episode effects')
    return effects, dict(config=dict(EPISODE_CONFIG), anchors=len(events['dates']), details=details), query


def build_base(panel, m, px, op, weekly_meta=None):
    """Return arrays, audit, regenerated weekly data and regenerated identities."""
    validate_panel(panel); validate_meta(m)
    require(op.index.equals(px.index) and op.columns.equals(px.columns), 'Baseline price/open axes differ')
    a, states = base_features(panel, m, px)
    a['resid_126_210_sum'] = residual_feature(px, panel['sectors'], m)
    weekly_data, rebuilt_meta, sources = weekly_inputs(panel, m, weekly_meta)
    a['old_ml4'], fits = weekly_forecasts(weekly_data, rebuilt_meta, m, a['eligible'])
    # score() is original arithmetic, including its simultaneous-budget cap.
    a['ml_4'] = a['old_ml4'].copy()
    a['pre_score'] = dependencies()[0].score(copy.deepcopy(PRIOR), a, len(m['dates']))
    # These are intermediate inputs, not part of the frozen base-array schema.
    del a['ml_4'], a['resid_126_210_sum']
    a['effects'], episode_audit, _ = episode_effects(panel, m, a['eligible'])
    sources.update({str(path): sha(path) for path in required_source_files()[2:]})
    return a, dict(source_hashes=sources, weekly_fits=fits, monthly_states=states,
                   weekly_rows=len(weekly_data['X']), weekly_dates=len(rebuilt_meta['dates']),
                   weekly_features=weekly_data['X'].shape[1], episode=episode_audit,
                   cached_computed_arrays_used=False, old_globals_reassigned=False), weekly_data, rebuilt_meta


class Tests(unittest.TestCase):
    @staticmethod
    def synthetic_panel():
        """Entirely generated data; includes calm and volatile market periods."""
        state, _, _ = dependencies()
        rng = np.random.RandomState(169)
        index = pd.bdate_range('2013-01-01', periods=1000)
        stocks = ['S%02d' % i for i in range(33)]
        etfs = sorted(set(state.ex.t.strict.SECTOR.values()) | {'SPY', 'QQQ', 'DIA', 'IWM', 'EFA', 'EEM'})
        names = sorted(stocks + etfs)
        market = rng.normal(.0008, .004, len(index))
        market[720:] = rng.normal(.0008, .018, len(index) - 720)
        returns = .3 * market[:, None] + rng.normal(.0008, .012, (len(index), len(names)))
        returns[:, names.index('SPY')] = market
        px = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=names)
        opening = px.shift(1).fillna(100) * np.exp(rng.normal(0, .002, px.shape))
        span = rng.uniform(.003, .015, px.shape)
        panel = dict(adj=px, open=opening, close=px.copy(),
                     high=pd.DataFrame(np.maximum(px, opening) * (1 + span), index=index, columns=names),
                     low=pd.DataFrame(np.minimum(px, opening) * (1 - span), index=index, columns=names),
                     volume=pd.DataFrame(rng.uniform(1e5, 1e6, px.shape), index=index, columns=names))
        sectors = list(state.ex.t.strict.SECTOR)
        panel['sectors'] = pd.DataFrame({s: sectors[i % len(sectors)] for i, s in enumerate(stocks)}, index=index)
        panel['member'] = pd.DataFrame([[s != 'S32' for s in stocks]] * 2, index=index[[0, 800]], columns=stocks)
        m = dict(dates=[str(index[i].date()) for i in (650, 880)],
                 ends=[str(index[i + 21].date()) for i in (650, 880)], etfs=etfs,
                 symbols=[sorted(stocks[:32] + etfs) for _ in range(2)])
        weekly = dict(dates=[str(index[i].date()) for i in range(300, 850, 10)],
                      ends=[str(index[i + 21].date()) for i in range(300, 850, 10)])
        return panel, m, weekly

    def test_membership_no_future_or_old_seed(self):
        member = pd.DataFrame([[True, False], [False, True]], index=pd.to_datetime(['2020-01-31', '2020-02-28']), columns=['A', 'B'])
        self.assertEqual(members_at(member, '2020-02-15', ['ETF']), (['A', 'ETF'], '2020-01-31'))
        with self.assertRaises(ValueError): members_at(member, '2020-01-30', ['ETF'])
    def test_legacy_exact_date(self):
        member = pd.DataFrame({'A': [True]}, index=pd.to_datetime(['2020-01-31']))
        legacy = pd.DataFrame({'B': [True]}, index=pd.to_datetime(['2020-01-29']))
        self.assertEqual(members_at(member, '2020-01-29', ['ETF'], legacy)[0], ['B', 'ETF'])
        with self.assertRaises(ValueError): members_at(member, '2020-01-30', ['ETF'], legacy)
    def test_fixed_configuration(self):
        state, ep, _ = dependencies()
        self.assertEqual(state.MODEL_CONFIGS[4], WEEKLY_CONFIG)
        self.assertEqual(ep.models()[2], EPISODE_CONFIG)
    def test_prior_budget_algebra(self):
        state, _, _ = dependencies()
        a = dict(baseline=np.array([[.2, .1], [.2, .1]]), core=np.zeros((2, 2)), core_weight=np.array([[.5], [.52]]),
                 state_calm10=np.array([True, False]), state_weak_narrow=np.array([True, False]),
                 resid_126_210_sum=np.array([[.4, -.1], [.4, -.1]]), ml_4=np.array([[-.2, .3], [-.2, .3]]))
        expected = a['baseline'].copy()
        support = a['baseline'][0] / .5
        expected[0] += .025 * (a['resid_126_210_sum'][0] - support) + .05 * (a['ml_4'][0] - support)
        np.testing.assert_array_equal(state.score(copy.deepcopy(PRIOR), a, 2), expected)
    def test_episode_maturity_and_no_signal(self):
        _, ep, _ = dependencies()
        events = dict(symbol=np.array(['A'] * 4), dates=np.array(['2019-01-01'] * 4, dtype='datetime64[D]'),
                      ends_5=np.array(['2030-01-01'] * 4, dtype='datetime64[D]'), signal=np.ones(4, int), utility_5=np.ones(4))
        query = dict(symbol='A', date=np.datetime64('2020-01-01'), eligible=True, signal=1, vol=.02)
        self.assertEqual(ep.estimate(events, query, EPISODE_CONFIG)[0], 0.)
        query['signal'] = 0
        self.assertEqual(ep.estimate({}, query, EPISODE_CONFIG)[0], 0.)
    def test_weekly_fit_uses_mature_data_only(self):
        state, _, _ = dependencies()
        rng = np.random.RandomState(169)
        data = dict(X=rng.normal(size=(12, 84)), Y=np.linspace(-.2, .2, 12), adverse=np.zeros(12),
                    good=np.ones(12, bool), past_score=np.zeros(12), offsets=np.array([0, 4, 8, 12]))
        meta = dict(dates=['2019-01-02', '2019-02-01', '2019-12-30'], ends=['2019-02-01', '2019-03-01', '2020-02-01'])
        first = state.fit_year(data, meta, '2019-12-31', WEEKLY_CONFIG)
        data['Y'][8:] = 9999.
        second = state.fit_year(data, meta, '2019-12-31', WEEKLY_CONFIG)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[2]['latest_label_end'], '2019-03-01')

    def test_synthetic_whole_base_chain_and_frozen_selection(self):
        panel, m, weekly = self.synthetic_panel()
        original = {key: frame.copy(deep=True) for key, frame in panel.items()}
        a, audit, data, meta = build_base(panel, m, panel['adj'], panel['open'], weekly)
        self.assertEqual(set(a), {'baseline', 'core', 'core_weight', 'state_calm10',
                                  'state_weak_narrow', 'eligible', 'old_ml4', 'pre_score', 'effects'})
        self.assertEqual(a['baseline'].shape, (2, 49))
        self.assertEqual(data['X'].shape, (55 * 49, 84))
        self.assertEqual(data['event'].shape, (2, 49, 84))
        self.assertEqual(a['state_calm10'].tolist(), [True, False])
        self.assertEqual(set(audit['source_hashes']), {str(path) for path in required_source_files()[2:]})
        self.assertFalse(audit['cached_computed_arrays_used'])
        self.assertEqual(meta['dates'], weekly['dates'])
        for key in panel: pd.testing.assert_frame_equal(panel[key], original[key])
        # A separate frozen public entry point checks generated orders and
        # their scores against the rebuild in both market-state branches.
        before_path = list(sys.path)
        try:
            sys.path.insert(0, str(OUTPUTS / 'monthly_stability_monitor'))
            oracle = module('ptr169_synthetic_frozen_selection', OUTPUTS / 'monthly_stability_monitor/strategy.py')
        finally:
            sys.path[:] = before_path
        for i, (date, ss) in enumerate(zip(m['dates'], m['symbols'])):
            expected = oracle.selection(dict(panel, member=panel['member'].reindex(panel['adj'].index, method='ffill')), date, m['etfs'])
            chosen = sorted([s for j, s in enumerate(ss) if a['eligible'][i, j]],
                            key=lambda s: (-a['baseline'][i, ss.index(s)], s))[:3]
            if len(chosen) < 2: chosen = []
            self.assertEqual(chosen, expected['symbols'])
            for s in chosen:
                self.assertAlmostEqual(a['baseline'][i, ss.index(s)], expected['scores'][s], places=14)

    def test_synthetic_membership_rebuilds_training_rows(self):
        panel, m, weekly = self.synthetic_panel()
        old_data, old_meta, _ = weekly_inputs(panel, m, weekly)
        changed = dict(panel, member=panel['member'].copy())
        changed['member'].iloc[1, changed['member'].columns.get_loc('S31')] = False
        changed['member'].iloc[1, changed['member'].columns.get_loc('S32')] = True
        new_m = copy.deepcopy(m)
        new_m['symbols'][1] = members_at(changed['member'], m['dates'][1], m['etfs'])[0]
        new_data, new_meta, _ = weekly_inputs(changed, new_m, weekly)
        self.assertEqual(old_meta['pool_origins'], new_meta['pool_origins'])
        np.testing.assert_array_equal(old_data['event'][0], new_data['event'][0])
        self.assertFalse(np.array_equal(old_data['event'][1], new_data['event'][1]))
        changed_sections = 0
        for i, date in enumerate(weekly['dates']):
            lo, hi = old_data['offsets'][i:i + 2]
            if pd.Timestamp(date) < panel['member'].index[1]:
                np.testing.assert_array_equal(old_data['X'][lo:hi], new_data['X'][lo:hi])
                self.assertEqual(old_meta['symbols'][i], new_meta['symbols'][i])
            else:
                changed_sections += 1
                self.assertIn('S32', new_meta['symbols'][i])
                self.assertNotIn('S31', new_meta['symbols'][i])
                self.assertFalse(np.array_equal(old_data['X'][lo:hi], new_data['X'][lo:hi]))
        self.assertEqual(changed_sections, 5)
        with self.assertRaisesRegex(ValueError, 'Supplied weekly identity mismatch'):
            weekly_inputs(changed, new_m, old_meta)

    def test_dependency_paths_include_fixed_algebra(self):
        paths = set(dependency_files())
        expected = ['monthly_stability_monitor/base_strategy.py', 'monthly_stability_monitor/strategy.py',
                    'monthly_state_rank_research/state_rank.py', 'monthly_daily_episodes/episodes.py',
                    'monthly_quality_research/explore.py', 'monthly_strict_relative/strict_study.py']
        for name in expected: self.assertIn(OUTPUTS / name, paths)
        state, ep, _ = dependencies()
        for function in (state.fit_year, state.ex.ridge.fit_weighted_ridge, state.prior.components,
                         state.prior.local.score, state.prior.old.previous.score,
                         state.ex.t.anchor.snapshot, ep.features, ep.estimate):
            self.assertIn(Path(function.__code__.co_filename).resolve(), paths)
        self.assertTrue(all(path.is_file() for path in paths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['test'])
    parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful(): raise SystemExit(1)
