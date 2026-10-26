"""Synthetic, read-only reporting tests for the four frozen monthly schemas."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from monthly_status import enrich_monthly_status

NOW = datetime(2026, 10, 19, 23, tzinfo=timezone.utc)


class MonthlyStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def account(self, name='regime', *, daily=False):
        folder = self.root / ('monthly_' + name + '_monitor')
        folder.mkdir()
        buffer = name == 'buffer5'
        sid = 'monthly_momentum_buffer5_net_v1' if buffer else 'monthly_momentum_' + name + '_paper_v1'
        config = dict(strategy_id=sid, start='2026-10-08' if buffer else '2026-10-12', initial_capital=100000.)
        if buffer:
            rules = '首个计划调仓日为2026-10-30，信号日为2026-10-29。'
        else:
            config.update(first_signal='2026-10-30', first_execution='2026-11-02')
            rules = '首次2026-10-30完整收盘信号，2026-11-02开盘模拟。'
        (folder / 'config.json').write_text(json.dumps(config))
        (folder / 'RULES.md').write_text(rules)
        self.freeze(folder)
        with sqlite3.connect(folder / 'paper_ledger.sqlite') as db:
            db.executescript('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);'
                            'CREATE TABLE plans(session TEXT PRIMARY KEY,created_utc TEXT NOT NULL,payload TEXT NOT NULL);'
                            'CREATE TABLE daily(session TEXT PRIMARY KEY,payload TEXT NOT NULL);'
                            'CREATE TABLE fills(id TEXT PRIMARY KEY,payload TEXT NOT NULL);')
            if not buffer:
                db.execute('CREATE TABLE closed(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            db.execute('INSERT INTO meta VALUES (?,?)', ('strategy_id', sid))
            if daily:
                row = dict(date='2026-10-16', cash=100000., equity=100000., positions=0 if buffer else {}, fees=0., realized_pnl=0.)
                db.execute('INSERT INTO daily VALUES (?,?)', ('2026-10-16', json.dumps(row)))
        return folder

    def freeze(self, folder):
        (folder / 'FREEZE.json').write_text(json.dumps({
            'outputs/' + folder.name + '/' + name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ['config.json', 'RULES.md']}))

    def report(self):
        return dict(status='WAITING_CURRENT_PANEL', data_as_of='2026-09-24',
                    reason='Current inputs are stale.', error='prior error remains visible',
                    state=dict(cash=100000., equity=100000., positions={}, fees=0., realized_pnl=0.),
                    statistics=None, plan=None, orders=[])

    def run_report(self, folder, report=None, now=NOW):
        return enrich_monthly_status(self.report() if report is None else report, folder, now)

    def test_four_real_schemas_prove_pre_signal_cash_and_keep_stale(self):
        for name in ['buffer5', 'regime', 'industry', 'stability']:
            with self.subTest(name=name):
                folder = self.account(name)
                original = self.report()
                before = deepcopy(original)
                hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
                out = self.run_report(folder, original)
                self.assertEqual(out['execution_phase'], 'WAITING_FIRST_SIGNAL')
                self.assertTrue(out['no_trade_due_today'])
                self.assertEqual(out['status'], 'WAITING_CURRENT_PANEL')
                self.assertEqual(out['data_health'], 'STALE')
                self.assertEqual(out['error'], before['error'])
                self.assertIsNone(out['statistics'])
                self.assertEqual(out['cash'], 100000.)
                self.assertEqual(out['next_signal_date'], '2026-10-29' if name == 'buffer5' else '2026-10-30')
                self.assertEqual(out['next_execution_date'], '2026-10-30' if name == 'buffer5' else '2026-11-02')
                self.assertEqual(original, before)
                self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()})

    def test_existing_data_health_is_not_overwritten(self):
        folder = self.account()
        report = self.report(); report['data_health'] = {'status': 'MISSING', 'issues': ['source unavailable']}
        out = self.run_report(folder, report)
        self.assertEqual(out['data_health'], report['data_health'])
        out['data_health']['issues'].append('copy only')
        self.assertEqual(len(report['data_health']['issues']), 1)

    def test_signal_day_and_later_never_assert_no_trade(self):
        folder = self.account()
        for now in [datetime(2026, 10, 30, 4, tzinfo=timezone.utc), datetime(2026, 11, 2, 15, tzinfo=timezone.utc)]:
            out = self.run_report(folder, now=now)
            self.assertEqual(out['execution_phase'], 'FIRST_SIGNAL_REACHED')
            self.assertIsNone(out['no_trade_due_today'])
            self.assertNotIn('cash', out)
            self.assertNotIn('next_signal_date', out)

    def test_new_york_date_controls_boundary(self):
        folder = self.account()
        out = self.run_report(folder, now=datetime(2026, 10, 30, 3, 59, tzinfo=timezone.utc))
        self.assertTrue(out['no_trade_due_today'])

    def test_unknown_schema_fails_closed(self):
        folder = self.account()
        with sqlite3.connect(folder / 'paper_ledger.sqlite') as db:
            db.execute('CREATE TABLE positions(symbol TEXT, qty REAL)')
        out = self.run_report(folder)
        self.assertEqual(out['execution_phase'], 'UNVERIFIED')
        self.assertIsNone(out['no_trade_due_today'])
        self.assertIn('Unknown monthly ledger tables', out['execution_status_error'])
        self.assertNotIn('cash', out)

    def test_unknown_column_fails_closed(self):
        folder = self.account()
        with sqlite3.connect(folder / 'paper_ledger.sqlite') as db:
            db.execute('ALTER TABLE fills ADD COLUMN newfield TEXT')
        out = self.run_report(folder)
        self.assertIn('columns', out['execution_status_error'])
        self.assertIsNone(out['no_trade_due_today'])

    def test_missing_ledger_is_not_created(self):
        folder = self.account(); (folder / 'paper_ledger.sqlite').unlink()
        out = self.run_report(folder)
        self.assertIn('execution_status_error', out)
        self.assertFalse((folder / 'paper_ledger.sqlite').exists())

    def test_ledger_activity_is_not_treated_as_untouched_cash(self):
        folder = self.account()
        with sqlite3.connect(folder / 'paper_ledger.sqlite') as db:
            db.execute('INSERT INTO plans VALUES (?,?,?)', ('2026-10-20', '2026-10-19', '{}'))
        out = self.run_report(folder)
        self.assertEqual(out['execution_phase'], 'EXISTING_LEDGER_ACTIVITY')
        self.assertIsNone(out['no_trade_due_today'])
        self.assertNotIn('cash', out)

    def test_report_positions_and_cash_conflicts_fail_closed(self):
        folder = self.account()
        for change in [dict(positions={'AAPL': {'qty': 1}}), dict(cash=99000.), dict(fees=1.)]:
            report = self.report(); report['state'].update(change)
            out = self.run_report(folder, report)
            self.assertEqual(out['execution_phase'], 'UNVERIFIED')
            self.assertIsNone(out['no_trade_due_today'])

    def test_recorded_cash_only_daily_rows_are_allowed_for_both_schemas(self):
        for name in ['buffer5', 'regime']:
            out = self.run_report(self.account(name, daily=True))
            self.assertTrue(out['no_trade_due_today'], out)
            self.assertEqual(out['execution_ledger_audit']['table_counts']['daily'], 1)

    def test_unknown_metadata_fails_closed(self):
        folder = self.account()
        with sqlite3.connect(folder / 'paper_ledger.sqlite') as db:
            db.execute('INSERT INTO meta VALUES (?,?)', ('external_deposit', '1000'))
        out = self.run_report(folder)
        self.assertIn('metadata keys', out['execution_status_error'])

    def test_nonempty_prior_result_positions_fail_closed(self):
        folder = self.account(); prior = self.report(); prior['state']['positions'] = {'QQQ': {}}
        with sqlite3.connect(folder / 'paper_ledger.sqlite') as db:
            db.execute('INSERT INTO meta VALUES (?,?)', ('last_result', json.dumps(prior)))
        out = self.run_report(folder)
        self.assertIn('positions', out['execution_status_error'])

    def test_changed_config_fails_closed(self):
        folder = self.account(); config = json.loads((folder / 'config.json').read_text()); config['initial_capital'] = 200000
        (folder / 'config.json').write_text(json.dumps(config))
        out = self.run_report(folder)
        self.assertIn('Frozen monthly input changed', out['execution_status_error'])

    def test_bad_calendar_schedule_fails_closed(self):
        folder = self.account(); config = json.loads((folder / 'config.json').read_text()); config['first_execution'] = '2026-11-03'
        (folder / 'config.json').write_text(json.dumps(config)); (folder / 'RULES.md').write_text('首次2026-10-30，执行2026-11-03。'); self.freeze(folder)
        out = self.run_report(folder)
        self.assertIn('Invalid monthend', out['execution_status_error'])

    def test_naive_now_fails_closed(self):
        out = self.run_report(self.account(), now=datetime(2026, 10, 19))
        self.assertIn('Timezone-aware', out['execution_status_error'])


if __name__ == '__main__':
    unittest.main()
