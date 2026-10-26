"""Small auxiliary budget from already matured virtual portfolio differences."""
import numpy as np


def budget_series(dates, ends, log_differences, maximum, window=12):
    dates = np.asarray(dates, dtype='datetime64[D]')
    ends = np.asarray(ends, dtype='datetime64[D]')
    differences = np.asarray(log_differences, float)
    if ends.shape != dates.shape or differences.shape != dates.shape:
        raise ValueError('Mismatched monthly episode arrays')
    if maximum < 0 or window < 2 or not np.isfinite(maximum):
        raise ValueError('Invalid budget/window')
    if np.any(ends <= dates) or np.any(dates[1:] <= dates[:-1]):
        raise ValueError('Monthly episodes must be increasing and forward')
    out = np.zeros(len(dates)); audit = []
    for i, day in enumerate(dates):
        prior = np.flatnonzero((dates < day) & (ends <= day))[-window:]
        row = dict(date=str(day), count=len(prior), latest_end=str(ends[prior].max()) if len(prior) else None)
        if len(prior) == window and np.isfinite(differences[prior]).all():
            d = differences[prior]; mean = float(d.mean()); se = float(d.std(ddof=1)/np.sqrt(window))
            denominator = abs(mean)+se
            out[i] = maximum*max(mean, 0.)/denominator if denominator > 0 else 0.
            row.update(mean=mean, se=se)
        row['budget'] = float(out[i]); audit.append(row)
    return out, audit
