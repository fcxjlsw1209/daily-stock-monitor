"""Historical own-volume activity paired with own-sector excess block gains.

Pure source measurement: no source I/O, eligibility, model or portfolio replay.
All required prices and volumes are admitted before any derived signal exists.
"""
from collections import Counter
import math
import numbers

import numpy as np
import pandas as pd
from monthly_nested_trend199_kernel import _frame, _calendar, _queries

PRICE_COUNT, VOLUME_COUNT, SKIP = 253, 315, 42
BLOCK_COUNT, BLOCK_SESSIONS, MEDIAN_LOOKBACK, HIGH_FLAG_COUNT = 12, 21, 63, 252
COUNT_FIELDS = ('valid_price_observations', 'valid_volume_observations',
    'preceding_median_estimates', 'high_volume_comparisons', 'excess_block_observations',
    'activity_block_observations', 'covariance_estimates', 'raw_score_observations',
    'ranked_score_observations')


def empty_measure(reason):
    return dict(available=False, reason=reason, candidate=np.nan, control=np.nan,
        endpoint_indices=[], block_start_indices=[], block_end_indices=[],
        volume_block_start_indices=[], volume_block_end_indices=[], stock_log_gains=[],
        benchmark_log_gains=[], excess_log_returns=[], preceding_medians=[],
        high_volume_flags=[], high_counts=[], activity=[], mean_excess=None,
        mean_activity=None, covariance=None, constant_branch=None, **{k: 0 for k in COUNT_FIELDS})


def measure(stock, benchmark, volume):
    """253 stock/own-sector prices and315 own volumes, all strictly positive."""
    if np.shape(stock) != (PRICE_COUNT,) or np.shape(benchmark) != (PRICE_COUNT,):
        raise ValueError('Exactly253 stock and benchmark prices required')
    if np.shape(volume) != (VOLUME_COUNT,):
        raise ValueError('Exactly315 same-stock volume observations required')
    converted = []
    for name, source in (('stock_price', stock), ('benchmark_price', benchmark), ('volume', volume)):
        raw = np.asarray(source, dtype=object)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
            return empty_measure('nonnumeric_' + name)
        try:
            values = np.asarray(raw, dtype=np.float64)
        except (ValueError, TypeError, OverflowError):
            return empty_measure('nonnumeric_' + name)
        if not (np.isfinite(values) & (values > 0)).all():
            return empty_measure('invalid_' + name)
        converted.append(values)
    stock, benchmark, volume = converted
    medians, flags = [], []
    for j in range(MEDIAN_LOOKBACK, VOLUME_COUNT):
        median = float(sorted(volume[j - MEDIAN_LOOKBACK:j])[31])
        medians.append(median)
        flags.append(int(volume[j] > median))
    high = [sum(flags[a:a + BLOCK_SESSIONS]) for a in range(0, HIGH_FLAG_COUNT, BLOCK_SESSIONS)]
    activity = [count / BLOCK_SESSIONS for count in high]
    endpoints = list(range(0, PRICE_COUNT, BLOCK_SESSIONS))
    starts, ends = endpoints[:-1], endpoints[1:]
    stock_gains, benchmark_gains, excess = [], [], []
    for a, b in zip(starts, ends):
        sg = math.log(float(stock[b])) - math.log(float(stock[a]))
        bg = math.log(float(benchmark[b])) - math.log(float(benchmark[a]))
        stock_gains.append(sg); benchmark_gains.append(bg); excess.append(sg - bg)
    try:
        c = math.fsum(excess) / BLOCK_COUNT
        hbar = math.fsum(activity) / BLOCK_COUNT
        branch = ('all_activity_equal' if all(x == activity[0] for x in activity) else
                  'all_excess_equal' if all(x == excess[0] for x in excess) else None)
        covariance = 0. if branch else math.fsum(
            (h - hbar) * (e - c) for h, e in zip(activity, excess)) / BLOCK_COUNT
        q = c if branch else c + covariance
    except (ValueError, OverflowError) as exc:
        raise FloatingPointError('Nonfinite paired-block arithmetic') from exc
    if not all(math.isfinite(x) for x in stock_gains + benchmark_gains + excess + activity + [c, hbar, covariance, q]):
        raise FloatingPointError('Nonfinite arithmetic from valid sources')
    return dict(available=True, reason='available', candidate=q, control=c,
        endpoint_indices=endpoints, block_start_indices=starts, block_end_indices=ends,
        volume_block_start_indices=list(range(63, 315, 21)),
        volume_block_end_indices=list(range(83, 315, 21)), stock_log_gains=stock_gains,
        benchmark_log_gains=benchmark_gains, excess_log_returns=excess,
        preceding_medians=medians, high_volume_flags=flags, high_counts=high, activity=activity,
        mean_excess=c, mean_activity=hbar, covariance=covariance, constant_branch=branch,
        valid_price_observations=506, valid_volume_observations=315,
        preceding_median_estimates=252, high_volume_comparisons=252, excess_block_observations=12,
        activity_block_observations=12, covariance_estimates=1, raw_score_observations=2,
        ranked_score_observations=2)


def centered_rank(raw, available):
    x, a = np.asarray(raw), np.asarray(available)
    if x.ndim != 1 or x.shape != a.shape or a.dtype != bool or not np.isfinite(x[a]).all():
        raise ValueError('Aligned finite source and boolean mask required')
    result = np.zeros(x.shape, float); ids = np.flatnonzero(a)
    if len(ids) > 1:
        result[ids] = (pd.Series(x[ids]).rank(method='average').to_numpy() - 1) / (len(ids) - 1) - .5
    return result


def build_features(panel, meta, calendar, benchmark_map, limit=None):
    if not isinstance(panel, dict) or set(panel) != {'adj', 'volume', 'sectors'}:
        raise ValueError('Panel requires exactly adj, volume and sectors')
    for frame in panel.values():
        _frame(frame)
    if not callable(benchmark_map):
        raise TypeError('Pinned benchmark mapper must be callable')
    sessions = _calendar(calendar); queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('Invalid query prefix')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = {k: np.full(shape, np.nan) for k in ('candidate', 'control')}
    values.update(available=np.zeros(shape, bool), candidate_rank=np.zeros(shape), control_rank=np.zeros(shape))
    adj, volume, sectors = panel['adj'], panel['volume'], panel['sectors']
    history, reasons, counts, available_by = [], Counter(), Counter(), []
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date()); mapping = benchmark_map(iso, symbols, sectors.loc[:day], meta['etfs'])
        if (not isinstance(mapping, dict) or set(mapping) != set(symbols) or
                any(b is not None and (not isinstance(b, str) or not b or b.strip() != b) for b in mapping.values())):
            raise ValueError('Complete original identity mapping required')
        if any(mapping[s] is not None for s in symbols if s in meta['etfs']):
            raise ValueError('Original candidate ETFs must stay unmapped')
        price_days = sessions[t - 294:t - 41] if t >= 294 else None
        volume_days = sessions[t - 356:t - 41] if t >= 356 else None
        price_dates = price_days.strftime('%Y-%m-%d').tolist() if price_days is not None else []
        volume_dates = volume_days.strftime('%Y-%m-%d').tolist() if volume_days is not None else []
        missing_price = price_days.difference(adj.index).strftime('%Y-%m-%d').tolist() if price_days is not None else []
        missing_volume = volume_days.difference(volume.index).strftime('%Y-%m-%d').tolist() if volume_days is not None else []
        cells = []
        for j, symbol in enumerate(symbols):
            benchmark = mapping[symbol]
            if symbol not in meta['etfs'] and benchmark == symbol:
                raise ValueError('Stock and own-sector benchmark must be distinct')
            if symbol in meta['etfs']: detail = empty_measure('etf_neutral')
            elif benchmark is None: detail = empty_measure('missing_benchmark')
            elif price_days is None or volume_days is None: detail = empty_measure('insufficient_history')
            elif symbol not in adj.columns: detail = empty_measure('missing_symbol')
            elif benchmark not in adj.columns: detail = empty_measure('missing_benchmark_symbol')
            elif symbol not in volume.columns: detail = empty_measure('missing_volume_symbol')
            elif missing_price: detail = empty_measure('missing_official_price_session')
            elif missing_volume: detail = empty_measure('missing_official_volume_session')
            else:
                detail = measure(adj.loc[price_days, symbol].to_numpy(copy=True),
                    adj.loc[price_days, benchmark].to_numpy(copy=True), volume.loc[volume_days, symbol].to_numpy(copy=True))
            for key in ('candidate', 'control', 'available'):
                values[key][i, j] = detail[key]
            cell = dict(detail, symbol=symbol, benchmark=benchmark, date=iso, candidate_rank=0., control_rank=0.)
            for indices, dates in (('endpoint_indices', 'endpoint_dates'), ('block_start_indices', 'block_start_dates'),
                                   ('block_end_indices', 'block_end_dates')):
                cell[dates] = [price_dates[k] for k in detail[indices]]
            for indices, dates in (('volume_block_start_indices', 'volume_block_start_dates'),
                                   ('volume_block_end_indices', 'volume_block_end_dates')):
                cell[dates] = [volume_dates[k] for k in detail[indices]]
            if not detail['available']:
                cell['candidate'] = cell['control'] = None
            cells.append(cell); reasons[detail['reason']] += 1
            for key in COUNT_FIELDS:
                counts[key] += detail[key]
        for raw in ('candidate', 'control'):
            key = raw + '_rank'; values[key][i] = centered_rank(values[raw][i], values['available'][i])
            for j, cell in enumerate(cells): cell[key] = float(values[key][i, j])
        n = int(values['available'][i].sum()); available_by.append(n)
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            price_start=price_dates[0] if price_dates else None, price_end=price_dates[-1] if price_dates else None,
            volume_start=volume_dates[0] if volume_dates else None, volume_end=volume_dates[-1] if volume_dates else None,
            price_dates=price_dates, volume_dates=volume_dates,
            endpoint_dates=price_dates[::21] if price_dates else None, activity_dates=volume_dates[63:],
            median_start_dates=volume_dates[:252], median_end_dates=volume_dates[62:314],
            volume_block_start_dates=volume_dates[63::21], volume_block_end_dates=volume_dates[83::21],
            missing_official_price_sessions=missing_price, missing_official_volume_sessions=missing_volume,
            benchmarks=mapping, cells=cells, available=values['available'][i].tolist(), available_count=n))
    total = sum(available_by)
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=253, volume_count=315, skip_sessions=42, median_lookback=63, high_flag_count=252,
        block_count=12, block_sessions=21, available_cells=total, unavailable_cells=int(values['available'].size)-total,
        available_by_event=available_by, **{k: counts[k] for k in COUNT_FIELDS}, reason_counts=dict(reasons),
        event_history=history, new_model_fits=0, predictive_model_fits=0, common_availability=True,
        ranking_performed=True, ranking_eligibility_filtered=False, trading_eligibility_modified=False,
        portfolio_returns_calculated=False)
