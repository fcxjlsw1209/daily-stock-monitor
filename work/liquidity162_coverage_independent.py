"""Pure independent source-coverage census; no files, prices or financial values.

calculate returns the exact row/summary contract; verify compares it with a
producer's outputs. Running this file executes synthetic tests only.
"""
from collections import Counter
from copy import deepcopy
from datetime import date
import json
import unittest
import numpy as np
import pandas as pd

TICKERS = ('MSFT', 'NVDA')
SPLITS = {'early': (0, 53), 'recent': (53, 88), 'train': (0, 88),
          'later': (88, 122), 'full': (0, 122)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def day(value):
    need(type(value) is str and date.fromisoformat(value).isoformat() == value, 'invalid ISO date')
    return date.fromisoformat(value)


def timestamp(value):
    need(type(value) is str, 'timestamp must be string')
    parsed = pd.Timestamp(value)
    need(not pd.isna(parsed) and parsed.tz is not None, 'aware timestamp required')
    return int(parsed.value)


def calculate(records, meta, eligible, closes):
    dates, pools = list(meta['dates']), list(meta['symbols'])
    flags = np.asarray(eligible)
    need(0 < len(dates) <= 122 and dates == sorted(set(dates)), 'ordered unique 1..122 dates required')
    need(len(pools) == len(dates) and flags.dtype == np.dtype(bool), 'pool/boolean mask schema')
    width = len(pools[0])
    need(flags.shape == (len(dates), width), 'eligible shape mismatch')
    for d, pool in zip(dates, pools):
        day(d)
        need(len(pool) == width and all(type(s) is str and s for s in pool), 'pool schema')
        need(len(set(pool)) == len(pool), 'duplicate pool member')
        need(type(closes[d]) is int, 'close must be integer epoch nanoseconds')
    indexed, keys = {t: [] for t in TICKERS}, set()
    for record in records:
        ticker, fy = record['ticker'], record['fiscal_year']
        need(ticker in TICKERS and type(fy) is int, 'source identity')
        sha = record['source_binding']['raw_sha256']
        need(type(sha) is str and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'source SHA256')
        key = f'{ticker}:{fy}:{sha}'
        need(key not in keys, 'duplicate source identity'); keys.add(key)
        ns = timestamp(record['availability']['known_by_at'])
        day(record['fiscal_period_end'])
        need(record['status'] in ('candidate', 'rejected'), 'source status')
        indexed[ticker].append((ns, key, record))
    rows = []
    for i, (d, pool) in enumerate(zip(dates, pools)):
        for ticker in TICKERS:
            positions = [j for j, symbol in enumerate(pool) if symbol == ticker]
            j = positions[0] if positions else None
            row = dict(index=i, date=d, cutoff_ns=closes[d], ticker=ticker, pool_index=j,
                       present=j is not None, eligible=bool(flags[i, j]) if j is not None else False,
                       selected_source=None, known_by_at=None, period_end=None, age_days=None,
                       status=None, source_usable=False, actionable=False, reason='no_known_source')
            available = [event for event in indexed[ticker] if event[0] <= closes[d]]
            if available:
                latest_ns = max(event[0] for event in available)
                latest = [event for event in available if event[0] == latest_ns]
                need(len(latest) == 1, 'ambiguous latest publication')
                _, key, event = latest[0]
                availability = event['availability']
                need(availability['status'] == 'ready' and availability['known_by_is_actual_publication'] is False,
                     'selected publication availability invalid')
                age = (day(d) - day(event['fiscal_period_end'])).days
                reason = ('latest_rejected' if event['status'] != 'candidate' else
                          'fiscal_period_future' if age < 0 else 'stale' if age > 400 else 'usable')
                row.update(selected_source=key, known_by_at=availability['known_by_at'],
                           period_end=event['fiscal_period_end'], age_days=age, status=event['status'],
                           source_usable=reason == 'usable', reason=reason)
                row['actionable'] = i > 0 and row['source_usable'] and row['present'] and row['eligible']
            rows.append(row)
    summary = {}
    for name, (start, end) in SPLITS.items():
        selected = [r for r in rows if start <= r['index'] < end]
        pairs = [selected[k:k + 2] for k in range(0, len(selected), 2)]
        both = [p for p in pairs if all(r['actionable'] for r in p)]
        summary[name] = dict(issuers={}, both_source_usable=sum(all(r['source_usable'] for r in p) for p in pairs),
                             both_pool_usable=sum(all(r['source_usable'] and r['present'] for r in p) for p in pairs),
                             both_actionable=len(both), overlap_reports={},
                             reason_counts=dict(Counter(r['reason'] for r in selected)))
        for ticker in TICKERS:
            issuer = [r for r in selected if r['ticker'] == ticker]
            summary[name]['issuers'][ticker] = dict(source_usable=sum(r['source_usable'] for r in issuer),
                pool_usable=sum(r['source_usable'] and r['present'] for r in issuer),
                actionable=sum(r['actionable'] for r in issuer),
                reports=sorted({r['selected_source'] for r in issuer if r['actionable']}))
            summary[name]['overlap_reports'][ticker] = sorted({r['selected_source'] for p in both for r in p if r['ticker'] == ticker})
    return rows, summary


def verify(records, meta, eligible, closes, rows, summary):
    expected_rows, expected_summary = calculate(records, meta, eligible, closes)
    errors = []
    encode = lambda x: json.dumps(x, sort_keys=True, allow_nan=False, separators=(',', ':'))
    if encode(rows) != encode(expected_rows): errors.append('rows_mismatch')
    if encode(summary) != encode(expected_summary): errors.append('summary_mismatch')
    return dict(passed=not errors, errors=errors, rows_checked=len(expected_rows),
                splits_checked=list(SPLITS), financial_values_read=False, returns_computed=0)


class SyntheticTests(unittest.TestCase):
    def fixture(self, n=122):
        dates = [str(d.date()) for d in pd.date_range('2020-01-01', periods=n)]
        meta = dict(dates=dates, symbols=[list(TICKERS) for _ in dates])
        closes = {d: timestamp(d + 'T21:00:00Z') for d in dates}
        events = [dict(ticker=t, fiscal_year=2019, fiscal_period_end='2019-12-31', status='candidate',
                       source_binding={'raw_sha256': 'a' * 64}, availability=dict(known_by_at='2020-01-01T21:00:00Z',
                       status='ready', known_by_is_actual_publication=False)) for t in TICKERS]
        return events, meta, np.ones((n, 2), dtype=bool), closes

    def test_initial_and_splits(self):
        args = self.fixture(); rows, summary = calculate(*args)
        self.assertEqual(len(rows), 244); self.assertFalse(rows[0]['actionable'])
        self.assertTrue(rows[0]['source_usable'])
        self.assertEqual([summary[k]['both_actionable'] for k in SPLITS], [52, 35, 87, 34, 121])
        self.assertTrue(verify(*args, rows, summary)['passed'])

    def test_latest_reject_no_fallback(self):
        records, meta, mask, closes = self.fixture(3)
        latest = deepcopy(records[0]); latest.update(fiscal_year=2020, status='rejected')
        latest['availability']['known_by_at'] = '2020-01-02T21:00:00Z'; records.append(latest)
        rows, _ = calculate(records, meta, mask, closes)
        self.assertEqual(rows[2]['reason'], 'latest_rejected'); self.assertFalse(rows[4]['source_usable'])

    def test_future_event_and_amount_poison_and_prefix(self):
        records, meta, mask, closes = self.fixture(); rows, _ = calculate(records, meta, mask, closes)
        future = deepcopy(records[0]); future['fiscal_year'] = 2021
        future['availability']['known_by_at'] = '2021-01-01T00:00:00Z'; records.append(future)
        for r in records: r['fields'] = {'liquid_financial_assets': 100, 'short_term_debt': 0}
        self.assertEqual(calculate(records, meta, mask, closes)[0], rows)
        for n in (1, 53, 88, 121):
            prefix = {k: v[:n] for k, v in meta.items()}
            self.assertEqual(calculate(records, prefix, mask[:n], closes)[0], rows[:2*n])
        meta['symbols'][-1] = ['OTHER', 'ANOTHER']; mask[-1] = False
        self.assertEqual(calculate(records, meta, mask, closes)[0][:-2], rows[:-2])

    def test_latest_stale_no_fallback(self):
        records, meta, mask, closes = self.fixture(3)
        latest = deepcopy(records[0]); latest.update(fiscal_year=2018, fiscal_period_end='2018-01-01')
        latest['availability']['known_by_at'] = '2020-01-02T21:00:00Z'; records.append(latest)
        rows, _ = calculate(records, meta, mask, closes)
        self.assertEqual(rows[2]['reason'], 'stale'); self.assertFalse(rows[2]['source_usable'])

    def test_absent_and_ineligible(self):
        records, meta, mask, closes = self.fixture(3); meta['symbols'][1][0] = 'OTHER'; mask[2, 1] = False
        rows, summary = calculate(records, meta, mask, closes)
        self.assertIsNone(rows[2]['pool_index']); self.assertFalse(rows[2]['eligible'])
        self.assertFalse(rows[5]['actionable']); self.assertEqual(summary['full']['both_actionable'], 0)

    def test_age_boundaries(self):
        records, meta, mask, closes = self.fixture(3)
        for period, expected in [('2020-01-02', 'fiscal_period_future'), ('2018-11-27', 'usable'), ('2018-11-26', 'stale')]:
            for r in records: r['fiscal_period_end'] = period
            self.assertEqual(calculate(records, meta, mask, closes)[0][0]['reason'], expected)

    def test_duplicate_and_naive_invalid(self):
        args = self.fixture(2); bad = deepcopy(args); bad[1]['symbols'][0] = ['MSFT', 'MSFT']
        with self.assertRaises(ValueError): calculate(*bad)
        bad = deepcopy(args); duplicate = deepcopy(bad[0][0]); duplicate['fiscal_year'] = 2020; bad[0].append(duplicate)
        with self.assertRaises(ValueError): calculate(*bad)
        bad = deepcopy(args); bad[0][0]['availability']['known_by_at'] = '2020-01-01'
        with self.assertRaises(ValueError): calculate(*bad)
        bad = deepcopy(args); bad[0][0]['availability']['known_by_is_actual_publication'] = True
        with self.assertRaises(ValueError): calculate(*bad)

    def test_before_first_source_and_tampering(self):
        args = self.fixture(2); args[0][0]['availability']['known_by_at'] = '2020-01-02T21:00:00Z'
        rows, summary = calculate(*args); self.assertEqual(rows[0]['reason'], 'no_known_source')
        rows[0]['actionable'] = True; self.assertFalse(verify(*args, rows, summary)['passed'])
        rows, summary = calculate(*args); rows[0]['actionable'] = 0
        self.assertFalse(verify(*args, rows, summary)['passed'])


if __name__ == '__main__':
    unittest.main()
