"""Same 21-official-session log-excess label on each independent anchor."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

_sp = importlib.util.spec_from_file_location('density70_label_features', Path(__file__).with_name('features.py'))
f = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(f)
HORIZON = 21


def dependency_files():
    return (Path(__file__).resolve(), *f.dependency_files())


def build_labels(adj, meta, X, eligible, calendar=None):
    ds = f.dates(meta['dates'], 'anchors'); ss = f.pools(meta['symbols'], len(ds))
    f.dates(adj.index, 'price dates')
    if not adj.columns.is_unique: raise ValueError('Duplicate price security')
    n = sum(map(len, ss)); X, eligible = np.asarray(X, float), np.asarray(eligible)
    if X.shape != (n, 10) or eligible.shape != (n,) or eligible.dtype.kind != 'b':
        raise ValueError('Labels require aligned original ten features and boolean eligibility')
    offsets = np.r_[0, np.cumsum(list(map(len, ss)))]
    if 'offsets' in meta: np.testing.assert_array_equal(offsets, meta['offsets'])
    calendar = calendar or f.calendar_for(ds)
    y, good = np.full(n, np.nan), np.zeros(n, bool)
    ends = np.empty(n, dtype='datetime64[D]'); records=[]
    for i, (day, names) in enumerate(zip(ds, ss)):
        future = f.sessions(calendar, day, day+pd.Timedelta(days=90))
        if len(future) < HORIZON+1 or future[0] != day:
            raise ValueError('Anchor is not a session or calendar horizon is incomplete')
        path = future[:HORIZON+1]; end = path[-1]; sl = slice(offsets[i], offsets[i+1])
        ends[sl] = np.datetime64(end.date(), 'D')
        known = eligible[sl] & np.isfinite(X[sl]).all(axis=1)
        missing_dates = path[~path.isin(adj.index)]
        record = dict(date=str(day.date()), end=str(end.date()), required_sessions=22,
            missing_sessions=[str(d.date()) for d in missing_dates], valid_names=0,
            status='missing_path', invalid_symbols=[])
        if not len(missing_dates):
            fields = names+([] if 'SPY' in names else ['SPY'])
            # Reindex exposes missing symbols; no filling and no compressed clock.
            values = adj.reindex(index=path, columns=fields).to_numpy(float)
            complete = (np.isfinite(values) & (values > 0)).all(axis=0)
            ok = known & complete[:len(names)] & complete[fields.index('SPY')]
            record['invalid_symbols'] = [s for s, valid in zip(fields, complete) if not valid]
            record['valid_names'] = int(ok.sum())
            record['status'] = 'fewer_than_two' if ok.sum() < 2 else 'complete'
            if ok.sum() >= 2:
                # Log differences avoid overflow of positive finite price ratios.
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    gross = np.log(values[-1])-np.log(values[0])
                    target = np.clip(gross[:len(names)][ok]-gross[fields.index('SPY')], -1., 1.)
                    centered = target-target.mean()
                if not np.isfinite(centered).all(): raise FloatingPointError('Nonfinite centered log label')
                ids = np.arange(offsets[i], offsets[i+1])[ok]
                y[ids] = centered; good[ids] = True
        records.append(record)
    return dict(arrays=dict(density_y=y, density_ends=ends, density_good=good),
                audit=dict(horizon=HORIZON, label='clip(loggross_stock-loggross_SPY,-1,1), then same-anchor center',
                           future_path_only_training_gate=True, records=records))
