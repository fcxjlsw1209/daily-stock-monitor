"""Monthly inverse research: skip-two-month momentum, stock/sector relative strength.

No industry absolute-return factor, recent stock-return score, or cached ML model.
Selection labels fit only a retrospective explanatory model; not recovered author code.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import itertools
import sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O / 'monthly_ordered_listwise'))
import experiment as e

e.P = P
read, dump, sha = e.read, e.dump, e.sha
SECTOR = {
    'Basic Materials': 'XLB', 'Communication Services': 'XLC', 'Energy': 'XLE',
    'Financial Services': 'XLF', 'Industrials': 'XLI', 'Technology': 'XLK',
    'Consumer Defensive': 'XLP', 'Real Estate': 'XLRE', 'Utilities': 'XLU',
    'Healthcare': 'XLV', 'Consumer Cyclical': 'XLY',
}
# Explicit inherited research assumptions for names without historical sector data.
FOREIGN = {'ASML': 'XLK', 'ARM': 'XLK', 'TSM': 'XLK', 'BABA': 'XLY',
           'NVO': 'XLV', 'SAP': 'XLK', 'TM': 'XLY', 'SHEL': 'XLE', 'SKHY': 'XLK',
           'BHP': 'XLB', 'HSBC': 'XLF', 'SMFG': 'XLF', 'NVS': 'XLV', 'SNDK': 'XLK'}


def init():
    assert not (P / 'PLAN.json').exists(), 'Do not overwrite an existing experiment'
    dump('PLAN.json', {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'user_correction': 'Industry ETF proxy is accepted. Compare stock with its own sector ETF; do not reward an industry ETF merely for outperforming other sectors.',
        'momentum': 'All stock return/momentum windows skip latest42 sessions or2 calendar monthends, including industry-relative returns and acceleration. Volatility and price-volume shapes may use recent observations.',
        'industry_formula': '(P_stock[t-42]/P_stock[t-n])/(P_sector[t-42]/P_sector[t-n])-1, n126 or252; normalize among stocks only. ETF and unknown classifications get neutral0 after normalization.',
        'excluded': ['sector/industry ETF absolute strength', 'recent industry-relative return', 'recent stock-return penalties', 'cached ML predictions with recent-return inputs', 'daily pullback entry gate'],
        'features': 'Same22 long-core families; volatility126; relative industry126/252; CMF21/63, signed volume63, volume21/126, broad-market-state CMF63; skip42 acceleration.',
        'budgets': 'Core>=50%, volatility<=30%, industry<=20%, combined shapes/state/acceleration<=20%, acceleration<=10%. These caps are research assumptions, not author-disclosed weights.',
        'grid': '22 core families x2 normalizations x3 temperatures(.02,.05,.10) x2 regularization levels(0,.10)=264 fits.',
        'universe': 'Unchanged reconstructed Top32 plus assumed theme17; classification history and cap snapshots remain inherited research proxies. No author-label universe additions.',
        'selection': 'First48 periods ending through2024-12-31 only. Freeze balanced-matching and exact-order leaders before looking at later21. Return is not a selection target. Previously viewed later history is not pristine OOS.',
        'replay': 'Monthend final-close features and same-close zero-cost fill; fixed adjusted total-return units during each month. First start2020-12-31, last mark2026-09-24. Next-open control separately. Same-close is an idealized reproduction convention, not guaranteed executable.',
        'protection': 'No changes to existing observation ledgers, rules, plans, fingerprints, or source inputs.',
    })
    before = read(O / 'monthly_ordered_inverse/PROTECTED_BEFORE.json')
    dump('PROTECTED_BEFORE.json', {f: sha(f) for f in before})
    dump('SECTOR_PROXY_ASSUMPTIONS.json', {'broad': SECTOR, 'missing_foreign': FOREIGN, 'etf_relative_score': 'neutral'})


def benchmark_map(date, symbols, sectors, etfs):
    ans = {}
    for s in symbols:
        if s in etfs:
            ans[s] = None
        elif s in FOREIGN:
            ans[s] = FOREIGN[s]
        elif s in sectors and pd.Timestamp(date) in sectors.index:
            ans[s] = SECTOR.get(sectors.at[pd.Timestamp(date), s])
        else:
            ans[s] = None
    return ans


def relative_row(momentum, date, symbols, benchmarks):
    row = {}
    for s in symbols:
        b = benchmarks.get(s)
        a = momentum.at[pd.Timestamp(date), s]
        z = momentum.at[pd.Timestamp(date), b] if b in momentum else np.nan
        row[s] = (1 + a) / (1 + z) - 1 if np.isfinite(a) and np.isfinite(z) and z > -1 else np.nan
    return pd.Series(row, dtype=float)


def prepare():
    px, raw, members, periods, sources, pools = e.load_prices()
    old = read(O / 'monthly_ordered_listwise/INPUTS.json')
    sf = O / 'index_union/alias_resolved_inputs/sectors.csv.gz'
    sectors = pd.read_csv(sf, index_col=0, parse_dates=True, low_memory=False)
    sources[str(sf)] = sha(sf)
    f, core_names, states = e.previous.panels(px, raw)
    core, side = e.features(px, raw)
    families = old['families']
    core_names = list(dict.fromkeys(k for family in families for k in family))
    clean = {k: (f[k] if k in f else core[k]) for k in core_names}
    for k in ['vol126', 'cmf21', 'cmf63', 'signed_volume63', 'volume_ratio21_126', 'acceleration_skip42']:
        clean[k] = side[k]
    relative = {f'stock_sector_skip42_{n}': px.shift(42) / px.shift(n) - 1 for n in [126, 252]}
    names = list(clean) + list(relative) + ['highvol_cmf63', 'trend_cmf63']
    symbols = [sorted({e.actual(s) for s in pools['32'][p['start']]} | members['theme17'][1]) for p in periods]
    warm = px.notna().rolling(295).sum().eq(295) & px.gt(0)
    eligibility_mom = px.shift(42) / px.shift(252) - 1
    meta = {'periods': periods, 'development': [p['end'] <= '2024-12-31' for p in periods],
            'families': families, 'packs': {}, 'sources_limit': 'Reconstructed cap pool/sector panel; author ETF list unknown'}
    for norm in ['percentile', 'magnitude']:
        X = np.zeros((len(periods), max(map(len, symbols)), len(names)))
        eligible = np.zeros(X.shape[:2], bool)
        labels = np.zeros_like(eligible)
        order = np.full((len(periods), 3), -1, int)
        coverage = []
        for i, (p, ss) in enumerate(zip(periods, symbols)):
            date = p['start']
            benchmarks = benchmark_map(date, ss, sectors, members['theme17'][1])
            tab = pd.DataFrame({k: z.loc[date, ss] for k, z in clean.items()})
            for k, momentum in relative.items():
                tab[k] = relative_row(momentum, date, ss, benchmarks)
            q = e.engine.normalize(tab, norm)
            q['highvol_cmf63'] = q.cmf63 * states.at[date, 'high_market_vol']
            q['trend_cmf63'] = q.cmf63 * states.at[date, 'spy_trend']
            X[i, :len(ss)] = q[names].to_numpy()
            eligible[i, :len(ss)] = (warm.loc[date, ss] & eligibility_mom.loc[date, ss].gt(0)).to_numpy()
            for j, author in enumerate(p['holdings']):
                a = e.actual(author)
                if a in ss:
                    n = ss.index(a)
                    labels[i, n] = True
                    if eligible[i, n]:
                        order[i, j] = n
            coverage.append({'date': date, 'benchmarks': benchmarks,
                             'neutral_etfs': sorted(members['theme17'][1] & set(ss)),
                             'missing_stock_relative': [e.display(s) for s in ss if s not in members['theme17'][1] and pd.isna(tab.at[s, 'stock_sector_skip42_252'])],
                             'author_unavailable': [a for j, a in enumerate(p['holdings']) if order[i, j] < 0]})
        np.savez_compressed(P / (norm + '.npz'), X=X, eligible=eligible, labels=labels, order=order)
        meta['packs'][norm] = {'symbols': symbols, 'names': names, 'norm': norm, 'coverage': coverage}
    # Verify monthend convention against the actual cached trading calendar.
    monthends = px.index.to_series().groupby(px.index.to_period('M')).last()
    for p in periods:
        assert pd.Timestamp(p['start']) == monthends.loc[pd.Timestamp(p['start']).to_period('M')]
    sources[str(O / 'monthly_ordered_listwise/INPUTS.json')] = sha(O / 'monthly_ordered_listwise/INPUTS.json')
    for module in [Path(__file__), Path(e.__file__), Path(e.previous.__file__), Path(e.engine.__file__)]:
        sources[str(module)] = sha(module)
    dump('INPUTS.json', meta)
    dump('SOURCE_HASHES.json', sources)
    print('Prepared', len(periods), 'monthend signals;', len(families), 'families; strict own-sector comparison', flush=True)


def design(d, core, ignored=False):
    spec = [(k, 1., 'core') for k in core]
    spec += [('vol126', sign, 'vol') for sign in [1., -1.]]
    spec += [(f'stock_sector_skip42_{n}', 1., 'industry') for n in [126, 252]]
    for k in ['cmf21', 'cmf63', 'signed_volume63', 'volume_ratio21_126', 'highvol_cmf63', 'trend_cmf63']:
        spec += [(k, sign, 'shape') for sign in [1., -1.]]
    spec += [('acceleration_skip42', 1., 'accel')]
    X = np.stack([d['X'][:, :, d['names'].index(k)] * sign for k, sign, group in spec], axis=-1)
    return X, spec


e.design = design


def train():
    m = read(P / 'INPUTS.json')
    dev = np.array(m['development'])
    assert dev.sum() == 48
    rows = []
    for norm in ['percentile', 'magnitude']:
        d = e.pack(norm, m)
        for fi, core in enumerate(m['families']):
            X, spec = design(d, core)
            for temp, reg in itertools.product([.02, .05, .10], [0., .10]):
                w, loss = e.fit(X, d, dev, spec, temp, reg)
                score = np.einsum('ijk,k->ij', X, w, optimize=False)
                rows.append(dict(id=len(rows), norm=norm, family=fi, core=core, ml=False,
                                 temp=temp, regularization=reg, spec=spec, weights=w.tolist(),
                                 loss=loss, calibration=e.previous.metrics(score, d, dev)))
            if (fi + 1) % 4 == 0:
                print('Strict-relative fits', len(rows), norm, 'families', fi + 1, flush=True)
    assert len(rows) == 264
    balanced = lambda r: (r['calibration']['selection_score'], r['calibration']['order_exact'], r['calibration']['set_exact'], -r['id'])
    ordered = lambda r: (r['calibration']['order_exact'], r['calibration']['set_exact'], r['calibration']['shared_slots'], r['calibration']['pairs_correct'], -r['id'])
    leaders = {'balanced': max(rows, key=balanced), 'ordered': max(rows, key=ordered)}
    dump('TRIALS.json', rows)
    dump('FROZEN.json', {'leaders': leaders, 'trial_count': len(rows), 'created_at': datetime.now(timezone.utc).isoformat(), 'training': 'First48 only; no return-based selection'})
    print('FROZEN', {k: dict(id=v['id'], core=v['core'], calibration=v['calibration']) for k, v in leaders.items()}, flush=True)


def evaluate():
    sources = read(P / 'SOURCE_HASHES.json')
    assert all(sha(f) == h for f, h in sources.items())
    frozen_hash = sha(P / 'FROZEN.json')
    e.evaluate()
    assert sha(P / 'FROZEN.json') == frozen_hash
    # Generic evaluator's ML assertion is inapplicable: this study has no ML.
    validation = read(P / 'VALIDATION.json')
    validation.pop('annual_market_ridge_training_label_cutoffs_verified', None)
    validation.update(source_hashes_verified=True, frozen_hash_unchanged=True,
                      every_signal_is_actual_monthend=True, ml_used=False,
                      industry_comparison='stock versus own sector ETF, matching skip42 endpoints',
                      same_close='Idealized zero-cost descriptive replay, fixed monthly adjusted units')
    dump('VALIDATION.json', validation)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['init', 'prepare', 'train', 'evaluate'])
    globals()[ap.parse_args().phase]()
