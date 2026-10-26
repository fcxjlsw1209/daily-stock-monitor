"""Pure skip-42 stock-sector recovery race; no file or outcome I/O.

The caller supplies adjusted prices, explicit sector mappings, membership and
an official session calendar. No training labels are consumed. Neither input
eligibility nor the momentum core is read or changed by this module.
"""
import math
import numpy as np
import pandas as pd


LOOKBACK = 252
SKIP = 42
WINDOW = LOOKBACK - SKIP
HORIZON = 21
CENSOR = 22
MIN_JOINT_DOWN = 20
STARTS = WINDOW - HORIZON
SECTOR_ETFS = frozenset(("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"))
INITIAL_PARTIAL = pd.Timestamp("2016-08-17")


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + " must be a valid session date") from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label + " must be a timezone-naive midnight date")
    return day


def _validate_frame(prices):
    if not isinstance(prices, pd.DataFrame) or not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be a DataFrame with a DatetimeIndex")
    index = prices.index
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing")
    if index.tz is not None or not index.equals(index.normalize()):
        raise ValueError("Price dates must be timezone-naive midnight dates")
    if not prices.columns.is_unique:
        raise ValueError("Price security columns must be unique")
    if any(not isinstance(symbol, str) or not symbol for symbol in prices.columns):
        raise ValueError("Price security columns must be nonempty strings")


def _calendar(calendar):
    days = []
    for value in calendar:
        day = _day(value, "Calendar")
        if not isinstance(value, str) or value != str(day.date()):
            raise ValueError("Calendar dates must be ISO YYYY-MM-DD strings")
        if day.weekday() >= 5:
            raise ValueError("Weekend in supplied official session calendar")
        if days and day <= days[-1]:
            raise ValueError("Calendar dates must be unique and increasing")
        days.append(day)
    if not days:
        raise ValueError("Calendar must not be empty")
    return pd.DatetimeIndex(days)


def _events(events, sessions):
    positions = {day: index for index, day in enumerate(sessions)}
    queries = []
    width = None
    for value, members in events:
        day = _day(value, "Event")
        if queries and day <= queries[-1][0]:
            raise ValueError("Event dates must be unique and increasing")
        if isinstance(members, (str, bytes)):
            raise TypeError("Event symbols must be a sequence")
        symbols = list(members)
        if not symbols or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
            raise ValueError("Event symbols must be nonempty strings in a nonempty pool")
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbols in Event")
        if width is not None and len(symbols) != width:
            raise ValueError("All events must have the same number of symbols")
        width = len(symbols)
        if day not in positions:
            raise ValueError("Event is not an official calendar session: " + str(day.date()))
        position = positions[day]
        if day != INITIAL_PARTIAL:
            if position + 1 >= len(sessions):
                raise ValueError("Calendar needs the next session to verify month end: " + str(day.date()))
            if sessions[position + 1].to_period("M") == day.to_period("M"):
                raise ValueError("Noninitial event must be an official month end: " + str(day.date()))
        queries.append((day, symbols, position))
    return queries, (width or 0)


def _mapping_rows(mapping, queries, selected):
    if not isinstance(mapping, dict) or set(mapping) != {'etfs', 'by_date'}:
        raise ValueError('Explicit etfs and by_date sector mapping required')
    etfs, rows = mapping['etfs'], mapping['by_date']
    if not isinstance(etfs, list) or any(not isinstance(x, str) or not x for x in etfs) or len(etfs) != len(set(etfs)):
        raise ValueError('ETF identities must be an explicit unique string list')
    if not isinstance(rows, dict) or set(rows) != {str(day.date()) for day, _, _ in queries}:
        raise ValueError('Sector mapping requires exact event dates')
    output = []
    neutral = set(etfs) | SECTOR_ETFS
    for day, names, _ in selected:
        row = rows[str(day.date())]
        if not isinstance(row, dict) or set(row) != set(names):
            raise ValueError('Sector mapping requires exact original event symbols')
        values = [row[name] for name in names]
        if any(value is not None and (not isinstance(value, str) or value not in SECTOR_ETFS) for value in values):
            raise ValueError('Only frozen allowed sector ETF benchmarks are permitted')
        if any(row[name] is not None for name in names if name in neutral):
            raise ValueError('ETF candidates must have an explicit neutral sector mapping')
        output.append(values)
    return output


def _first_passage(prices, start):
    """First 1..21 session recovery of the pre-down-day level, else 22."""
    if type(start) is not int or not 1 <= start <= STARTS:
        raise ValueError('Recovery start must lie in 1..189')
    for lag in range(1, HORIZON + 1):
        if prices[start + lag] >= prices[start - 1]:
            return lag
    return CENSOR


def _paired_recovery(stock, sector):
    """Both measurements use the same complete paired window and down starts."""
    stock, sector = np.asarray(stock, dtype=float), np.asarray(sector, dtype=float)
    if stock.shape != (WINDOW + 1,) or sector.shape != (WINDOW + 1,):
        raise ValueError('Recovery requires exactly 211 prices for both series')
    if not (np.isfinite(stock) & (stock > 0.)).all():
        return np.nan, np.nan, None, 'invalid_stock_path'
    if not (np.isfinite(sector) & (sector > 0.)).all():
        return np.nan, np.nan, None, 'invalid_sector_path'
    starts = [s for s in range(1, STARTS + 1) if stock[s] < stock[s-1] and sector[s] < sector[s-1]]
    count = len(starts)
    if count < MIN_JOINT_DOWN:
        return np.nan, np.nan, count, 'insufficient_joint_down'
    race, impact = [], []
    for start in starts:
        race.append(_first_passage(sector, start) - _first_passage(stock, start))
        # On strictly down days this difference form is finite, stable and
        # algebraically identical to the specified simple price return.
        rs = (stock[start] - stock[start-1]) / stock[start-1]
        rb = (sector[start] - sector[start-1]) / sector[start-1]
        impact.append((rs - rb) / (abs(rs) + abs(rb)))
    # One final division preserves exact ties between equal discrete means.
    raw_recovery, raw_control = sum(race) / (HORIZON * count), math.fsum(impact) / count
    if not np.isfinite(raw_recovery) or not np.isfinite(raw_control) or abs(raw_recovery) > 1. or abs(raw_control) > 1.:
        raise FloatingPointError('Recovery/impact escaped their finite unit bounds')
    return raw_recovery, raw_control, count, 'available'


def _recovery_values(prices, symbols, benchmarks, etfs, sessions, position):
    raw = np.full((2, len(symbols)), np.nan, dtype=float)
    counts = [None] * len(symbols)
    reasons = ['etf_not_applicable' if name in etfs or name in SECTOR_ETFS else
               'missing_sector_mapping' if benchmark is None else 'insufficient_history'
               for name, benchmark in zip(symbols, benchmarks)]
    if position < LOOKBACK:
        return raw, counts, reasons
    dates = sessions[position - LOOKBACK:position - SKIP + 1]
    needed = list(dict.fromkeys([name for name, benchmark in zip(symbols, benchmarks) if benchmark is not None]
                                + [benchmark for benchmark in benchmarks if benchmark is not None]))
    # All 211 paired prices are consumed; slice before numeric conversion.
    selected = prices.reindex(index=dates, columns=needed).to_numpy(dtype=float)
    columns = {name: selected[:, i] for i, name in enumerate(needed)}
    for j, (name, benchmark) in enumerate(zip(symbols, benchmarks)):
        if benchmark is not None:
            a, b, counts[j], reasons[j] = _paired_recovery(columns[name], columns[benchmark])
            raw[:, j] = a, b
    return raw, counts, reasons


def _json_floats(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def build_features(prices, sector_mapping, events, calendar, limit=None):
    """Return recovery/control deltas on shared direct-stock support.

    At monthend t, consume exactly 211 adjusted prices [t-252,t-42].
    For joint strictly down starts s=1..189, compare each series' first
    recovery of its own pre-down-day level in s+1..s+21, censored at 22.
    Mean (sector_time-stock_time)/21 is the recovery race; mean normalized
    same-day stock-minus-sector simple return is its matched impact control.
    At least 20 starts are required. ETFs, unknown mappings and incomplete
    paired windows are neutral. Both ranks use the same finite stock pool.
    """
    _validate_frame(prices)
    sessions = _calendar(calendar)
    queries, width = _events(events, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    selected = queries if limit is None else queries[:limit]
    maps = _mapping_rows(sector_mapping, queries, selected)
    recovery = np.zeros((len(selected), width), dtype=float)
    control = np.zeros_like(recovery)
    history, support_cells = [], 0
    for row, ((day, names, position), benchmarks) in enumerate(zip(selected, maps)):
        values, counts, reasons = _recovery_values(prices, names, benchmarks, sector_mapping['etfs'], sessions, position)
        available = np.isfinite(values).all(axis=0)
        ranks = np.asarray([pd.Series(v).rank(method='average', ascending=True, pct=True).to_numpy() - .5 for v in values])
        initial = bool(day == INITIAL_PARTIAL)
        support = available & (not initial)
        recovery[row, support] = .04 * ranks[0, support]
        control[row, support] = .04 * ranks[1, support]
        support_cells += int(support.sum())
        history.append(dict(
            date=str(day.date()), symbols=list(names),
            window_start=str(sessions[position-LOOKBACK].date()) if position >= LOOKBACK else None,
            window_end=str(sessions[position-SKIP].date()) if position >= SKIP else None,
            sector_etfs=benchmarks, joint_down_counts=counts, source_status=reasons,
            common_support=support.tolist(), raw_recovery=_json_floats(values[0]), raw_control=_json_floats(values[1]),
            ranks_recovery=_json_floats(ranks[0]), ranks_control=_json_floats(ranks[1]),
            initial_forced_neutral=initial))
    audit = dict(events=len(selected), symbols_per_event=width, window_prices=WINDOW+1,
                 window_returns=WINDOW, skip_sessions=SKIP, start_count=STARTS,
                 recovery_horizon=HORIZON, censor_value=CENSOR, minimum_joint_down=MIN_JOINT_DOWN,
                 common_support_cells=support_cells, neutral_cells=len(selected)*width-support_cells,
                 trading_eligibility_modified=False, outcomes_used=False, event_history=history)
    return {'delta_recovery': recovery, 'delta_control': control}, audit
