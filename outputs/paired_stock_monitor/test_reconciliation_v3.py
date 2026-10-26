"""Offline synthetic replay tests. No live ledger or price cache is opened."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent / 'swing_research'))
from reconciliation_v3 import build_decision, reconcile, POLICY_KEY, VERSION, ORIGIN, STATUS


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = load('replay_test_frozen_engine', P.parent / 'daily_stock_monitor' / 'engine.py')
market = load('replay_test_frozen_market', P.parent / 'daily_stock_monitor' / 'market_data.py')


def dumps(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(['2026-10-15', '2026-10-16', '2026-10-19', '2026-10-20'])
        self.calendar = pd.DataFrame({'open': self.dates.tz_localize('UTC') + pd.Timedelta(hours=13, minutes=30),
                                      'close': self.dates.tz_localize('UTC') + pd.Timedelta(hours=20)}, index=self.dates)
        self.cfg = {'symbols': ['BBB', 'AAA'], 'sector_etfs': {'Technology': 'XLK', 'Healthcare': 'XLV'},
                    'sectors': {'BBB': 'Technology', 'AAA': 'Healthcare'}, 'paper_start': '2026-10-16',
                    'admission': 'broad', 'version': 'frozen_synthetic', 'cost_bps': 10,
                    'exit': 'fixed5', 'allocation': .1, 'max_positions': 2, 'sector_cap': 2}
        self.frames = {}
        for i, symbol in enumerate(['BBB', 'AAA', 'SPY', 'QQQ', 'XLK', 'XLV']):
            close = np.array([100., 103., 106.]) + i * 10
            self.frames[symbol] = pd.DataFrame({'Open': close - 1, 'High': close + 2, 'Low': close - 2,
                                               'Close': close, 'Adj Close': close * .9, 'Volume': 1000000.,
                                               'Dividends': 0., 'Stock Splits': 0.}, index=self.dates[:3])
        self.manifest = {'snapshot_id': 'synthetic', 'as_of': '2026-10-19',
                         'retrieved_at_utc': '2026-10-19T21:00:00+00:00', 'failures': []}
        self.now = pd.Timestamp('2026-10-19T22:00:00Z')
        self.authorized = '2026-10-19T21:30:00+00:00'
        self.seen = []
        def features(data, raw):
            self.seen.append((copy.deepcopy(data), copy.deepcopy(raw)))
            close = data.close[self.cfg['symbols']]
            return {'score': close * 0 + 1, 'masks': {'broad': close.notna()},
                    'ma5': close * 100, 'atr': raw['high'] - raw['low']}
        def latest(db):
            row = db.execute('SELECT state FROM sessions ORDER BY date DESC LIMIT 1').fetchone()
            return json.loads(row[0] if row else db.execute("SELECT value FROM meta WHERE key='initial_state'").fetchone()[0])
        self.m = SimpleNamespace(features=features, plan=engine.plan, execute=engine.execute,
                                 session_bar=market.session_bar, latest_state=latest, dumps=dumps,
                                 schedule=lambda now: self.calendar, CACHE=Path('/synthetic/cache'),
                                 fingerprint=lambda: 'frozen-strategy-fingerprint')
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.executescript('''
            CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE plans(session TEXT PRIMARY KEY,created_utc TEXT NOT NULL,payload TEXT NOT NULL,snapshot TEXT NOT NULL,code_hash TEXT NOT NULL);
            CREATE TABLE sessions(date TEXT PRIMARY KEY,payload TEXT NOT NULL,state TEXT NOT NULL,status TEXT NOT NULL,snapshot TEXT NOT NULL);
            CREATE TABLE trades(id INTEGER PRIMARY KEY,data TEXT NOT NULL);
            CREATE TABLE orders(id INTEGER PRIMARY KEY,data TEXT NOT NULL);
            CREATE TABLE corporate_actions(id INTEGER PRIMARY KEY,session TEXT NOT NULL,data TEXT NOT NULL);
        ''')
        self.db.execute('INSERT INTO meta VALUES (?,?)', ('initial_state', dumps(engine.initial_state())))
        self.set_policy({'version': VERSION, 'authorized_at_utc': self.authorized})

    def set_policy(self, policy):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (POLICY_KEY, dumps(policy)))
        self.db.commit()

    def run_replay(self):
        return reconcile(self.m, self.db, self.cfg, self.frames, self.manifest, self.now)

    def decision(self, state=None, day='2026-10-16'):
        return build_decision(self.m, state or engine.initial_state(), day, self.calendar,
                              self.frames, self.manifest, self.cfg)

    def record(self, decision, created='2026-10-16T12:00Z'):
        self.db.execute('INSERT INTO plans VALUES (?,?,?,?,?)',
                        (decision['session'], created, dumps(decision), '/synthetic/old', 'old-code'))
        self.db.commit()

    def narrow_to_first_day(self):
        self.manifest['as_of'] = '2026-10-16'
        self.frames = {s: frame.loc[:'2026-10-16'].copy() for s, frame in self.frames.items()}

    def test_all_inputs_cutoff_and_column_order(self):
        decision = self.decision()
        data, raw = self.seen[-1]
        for matrix in [data.close, data.opening, data.dividends, data.volume, data.sector_prices] + list(raw.values()):
            self.assertEqual(matrix.index.max(), pd.Timestamp('2026-10-15'))
        self.assertEqual(list(raw['adj_close']), ['BBB', 'AAA'])
        self.assertEqual(list(data.close), ['BBB', 'AAA', 'SPY', 'QQQ'])
        self.assertEqual(list(data.sector_prices), ['XLK', 'XLV'])
        self.assertEqual([b['symbol'] for b in decision['buys']], ['BBB', 'AAA'])
        self.assertEqual(decision['decision_as_of'], '2026-10-15')
        self.assertEqual(decision['decision_state_sha256'], hashlib.sha256(dumps(engine.initial_state()).encode()).hexdigest())
        self.assertEqual(decision['parent_snapshot_hashes_sha256'], hashlib.sha256(dumps({}).encode()).hexdigest())
        self.assertNotIn('origin', decision)
        self.assertNotIn('created_utc', decision)

    def test_parent_source_hash_mapping_is_bound_and_type_checked(self):
        self.manifest['hashes'] = {'BBB': 'b' * 64, 'AAA': 'a' * 64}
        decision = self.decision()
        self.assertEqual(decision['parent_snapshot_hashes_sha256'],
                         hashlib.sha256(dumps(self.manifest['hashes']).encode()).hexdigest())
        self.manifest['hashes'] = []
        with self.assertRaisesRegex(RuntimeError, 'verified loader mapping'):
            self.decision()

    def test_future_prices_and_volume_cannot_change_prior_decision(self):
        before = self.decision()
        for frame in self.frames.values():
            frame.loc[frame.index > self.dates[0], ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] *= 777
            frame.loc[frame.index > self.dates[0], 'Dividends'] = 500.
        self.assertEqual(self.decision(), before)

    def test_no_input_mutation(self):
        saved = copy.deepcopy(self.frames)
        state = engine.initial_state()
        state_before = copy.deepcopy(state)
        self.decision(state)
        self.assertEqual(state, state_before)
        for symbol in saved:
            pd.testing.assert_frame_equal(self.frames[symbol], saved[symbol])

    def test_future_dividend_adjustment_is_removed_by_anchor(self):
        before = self.decision()
        for frame in self.frames.values():
            frame.loc[:'2026-10-15', 'Adj Close'] *= .6
            frame.loc['2026-10-16', 'Dividends'] = 3.
        self.assertEqual(self.decision(), before)
        data, raw = self.seen[-1]
        pd.testing.assert_frame_equal(raw['adj_close'], data.close[['BBB', 'AAA']])

    def test_future_split_preserves_decisions_and_dollar_volume(self):
        before = self.decision()
        first_data, first_raw = self.seen[-1]
        for frame in self.frames.values():
            frame.loc[:'2026-10-15', ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Dividends']] /= 2
            frame.loc[:'2026-10-15', 'Volume'] *= 2
            frame.loc['2026-10-16', 'Stock Splits'] = 2.
        after = self.decision()
        self.assertEqual(before['buys'], after['buys'])
        self.assertEqual(before['sells'], after['sells'])
        last_data, last_raw = self.seen[-1]
        for key in ['close', 'opening', 'dividends', 'volume', 'sector_prices']:
            pd.testing.assert_frame_equal(getattr(first_data, key), getattr(last_data, key))
        for key in first_raw:
            pd.testing.assert_frame_equal(first_raw[key], last_raw[key])
        self.assertEqual(set(after['future_split_factors'].values()), {2.})

    def test_frozen_features_are_invariant_to_future_split_and_quotes(self):
        strategy = load('replay_test_frozen_strategy', P.parent / 'daily_stock_monitor' / 'strategy.py')
        self.dates = pd.bdate_range('2025-07-01', periods=304)
        self.calendar = pd.DataFrame(index=self.dates)
        self.cfg['paper_start'] = str(self.dates[-2].date())
        self.manifest['as_of'] = str(self.dates[-1].date())
        t = np.arange(len(self.dates), dtype=float)
        for i, symbol in enumerate(self.frames):
            close = (100. + i * 10) * np.exp(.001 * t + .007 * np.sin(t * (.3 + i * .01)))
            self.frames[symbol] = pd.DataFrame({'Open': close * .999, 'High': close * 1.01,
                'Low': close * .99, 'Close': close, 'Adj Close': close * .9,
                'Volume': 1000000., 'Dividends': 0., 'Stock Splits': 0.}, index=self.dates)
        captured = []
        def frozen_features(data, raw):
            result = strategy.features(data, raw)
            captured.append(result)
            self.assertEqual(data.close.index.max(), self.dates[-3])
            return result
        self.m.features = frozen_features
        before = self.decision(day=self.cfg['paper_start'])
        previous = self.dates[-3]
        for frame in self.frames.values():
            frame.loc[:previous, ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Dividends']] /= 2.
            frame.loc[:previous, 'Volume'] *= 2.
            frame.loc[self.dates[-2], 'Stock Splits'] = 2.
            frame.loc[frame.index > previous, ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] *= 777.
        after = self.decision(day=self.cfg['paper_start'])
        self.assertEqual(before['buys'], after['buys'])
        self.assertEqual(before['sells'], after['sells'])
        self.assertTrue(np.isfinite(captured[0]['score'].iloc[-1]).all())
        self.assertTrue(np.isfinite(captured[0]['atr'].iloc[-1]).all())
        for key in ['score', 'ma5', 'atr', 'cmf']:
            pd.testing.assert_frame_equal(captured[0][key], captured[1][key])
        for key in captured[0]['masks']:
            pd.testing.assert_frame_equal(captured[0]['masks'][key], captured[1]['masks'][key])

    def test_future_plan_uses_prior_state_without_execution_day_bar(self):
        state = engine.initial_state()
        state['last_date'] = '2026-10-19'
        decision = self.decision(state, '2026-10-20')
        self.assertEqual(decision['signal_date'], '2026-10-19')
        self.assertEqual(self.seen[-1][0].close.index.max(), pd.Timestamp('2026-10-19'))

    def test_state_gap_and_nonfirst_initial_state_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'first paper'):
            self.decision(day='2026-10-19')
        state = engine.initial_state()
        state['last_date'] = '2026-10-15'
        with self.assertRaisesRegex(RuntimeError, 'immediately preceding'):
            self.decision(state, '2026-10-19')

    def test_missing_prior_bar_and_future_beyond_manifest_rejected(self):
        self.frames['AAA'] = self.frames['AAA'].iloc[1:]
        with self.assertRaisesRegex(RuntimeError, 'history'):
            self.decision()
        self.frames['AAA'] = self.frames['BBB'].copy()
        self.manifest['as_of'] = '2026-10-16'
        with self.assertRaisesRegex(RuntimeError, 'history'):
            self.decision()

    def test_bad_split_fails_closed(self):
        for value in [float('nan'), float('inf'), -2.]:
            with self.subTest(value=value):
                self.frames['AAA'].loc['2026-10-16', 'Stock Splits'] = value
                with self.assertRaisesRegex(RuntimeError, 'split history'):
                    self.decision()

    def test_standing_policy_is_required(self):
        self.db.execute('DELETE FROM meta WHERE key=?', (POLICY_KEY,))
        self.db.commit()
        with self.assertRaisesRegex(RuntimeError, 'authorization is required'):
            self.run_replay()
        self.assertEqual(self.db.execute('SELECT count(*) FROM sessions').fetchone()[0], 0)

    def test_invalid_and_future_authorization_rejected(self):
        for policy in [[], {}, {'version': 'old', 'authorized_at_utc': self.authorized},
                       {'version': VERSION, 'authorized_at_utc': '2026-10-19'},
                       {'version': VERSION, 'authorized_at_utc': '2026-10-20T00:00Z'}]:
            with self.subTest(policy=policy):
                self.set_policy(policy)
                with self.assertRaisesRegex(RuntimeError, 'Invalid standing'):
                    self.run_replay()

    def test_multiday_next_open_execution_and_idempotence(self):
        self.assertEqual(self.run_replay(), 2)
        orders = [json.loads(row[0]) for row in self.db.execute('SELECT data FROM orders')]
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0]['date'], '2026-10-16')
        self.assertEqual(orders[0]['price'], 102.)
        self.assertNotEqual(orders[0]['price'], 100.)
        self.assertEqual(self.m.latest_state(self.db)['last_date'], '2026-10-19')
        plans = [json.loads(row[0]) for row in self.db.execute('SELECT payload FROM plans ORDER BY session')]
        self.assertEqual(plans[1]['buys'], [])
        for plan in plans:
            self.assertEqual(plan['origin'], ORIGIN)
            self.assertEqual(plan['created_utc'], self.now.isoformat())
            self.assertEqual(plan['policy_authorized_at_utc'], self.authorized)
            self.assertEqual(plan['data_vintage'], 'retrospectively_retrieved')
        self.assertEqual([r[0] for r in self.db.execute('SELECT status FROM sessions')], [STATUS, STATUS])
        counts = [self.db.execute('SELECT count(*) FROM ' + t).fetchone()[0] for t in ['plans', 'sessions', 'orders', 'trades', 'corporate_actions']]
        self.assertEqual(self.run_replay(), 0)
        self.assertEqual(counts, [self.db.execute('SELECT count(*) FROM ' + t).fetchone()[0] for t in ['plans', 'sessions', 'orders', 'trades', 'corporate_actions']])

    def test_later_failure_retains_only_complete_earlier_day(self):
        original = self.m.features
        def fail_on_second(data, raw):
            if data.close.index.max() == pd.Timestamp('2026-10-16'):
                raise RuntimeError('synthetic second-day failure')
            return original(data, raw)
        self.m.features = fail_on_second
        with self.assertRaisesRegex(RuntimeError, 'second-day'):
            self.run_replay()
        self.assertEqual(self.db.execute('SELECT session FROM plans').fetchall(), [('2026-10-16',)])
        self.assertEqual(self.db.execute('SELECT date FROM sessions').fetchall(), [('2026-10-16',)])
        self.m.features = original
        self.assertEqual(self.run_replay(), 1)
        self.assertEqual(self.db.execute('SELECT count(*) FROM orders').fetchone()[0], 2)

    def test_all_daily_writes_roll_back_together(self):
        self.db.executescript("CREATE TRIGGER reject_order BEFORE INSERT ON orders BEGIN SELECT RAISE(ABORT, 'synthetic order failure'); END;")
        with self.assertRaises(sqlite3.IntegrityError):
            self.run_replay()
        for table in ['plans', 'sessions', 'trades', 'orders', 'corporate_actions']:
            self.assertEqual(self.db.execute('SELECT count(*) FROM ' + table).fetchone()[0], 0)
        self.assertFalse(self.db.in_transaction)

    def test_existing_preplan_is_reused_without_features(self):
        self.narrow_to_first_day()
        plan = {'session': '2026-10-16', 'signal_date': '2026-10-15', 'buys': [], 'sells': []}
        self.record(plan)
        self.m.features = lambda *args: self.fail('must not rebuild existing plan')
        self.assertEqual(self.run_replay(), 1)
        self.assertEqual(json.loads(self.db.execute('SELECT payload FROM plans').fetchone()[0]), plan)
        self.assertEqual(self.db.execute('SELECT status FROM sessions').fetchone()[0], 'recorded_plan_executed')

    def test_wrong_state_digest_rejected_before_writes(self):
        self.narrow_to_first_day()
        decision = self.decision()
        decision['decision_state_sha256'] = '0' * 64
        self.record(decision)
        with self.assertRaisesRegex(RuntimeError, 'different account state'):
            self.run_replay()
        self.assertEqual(self.db.execute('SELECT count(*) FROM sessions').fetchone()[0], 0)

    def test_v3_record_cannot_drop_its_state_binding(self):
        self.narrow_to_first_day()
        plan = self.decision()
        del plan['decision_state_sha256']
        self.record(plan)
        with self.assertRaisesRegex(RuntimeError, 'state hash'):
            self.run_replay()

    def test_invalid_future_split_also_blocks_preexisting_plan(self):
        self.record({'session': '2026-10-16', 'signal_date': '2026-10-15', 'buys': [
            {'symbol': 'BBB', 'budget': 10000., 'reference_close': 100., 'atr': 4.}], 'sells': []})
        self.frames['BBB'].loc['2026-10-19', 'Stock Splits'] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'split history'):
            self.run_replay()
        self.assertEqual(self.db.execute('SELECT count(*) FROM sessions').fetchone()[0], 0)

    def test_invalid_prior_ohlc_and_volume_are_not_silently_filtered(self):
        original = self.frames['BBB'].copy()
        for field, value in [('Open', float('nan')), ('High', 50.), ('Low', 200.), ('Volume', 0.)]:
            with self.subTest(field=field):
                self.frames['BBB'] = original.copy()
                self.frames['BBB'].at[self.dates[0], field] = value
                with self.assertRaisesRegex(RuntimeError, 'OHLC/volume'):
                    self.decision()

    def test_separate_open_quote_outside_range_is_preserved(self):
        # Reproduce the verified cache bytes for KO's 2026-10-19 price fields.
        # This tests the frozen signal contract, not exchange-level provenance.
        quote = {'Open': 92.55999755859376, 'High': 93.98999786376952,
                 'Low': 92.62220001220705, 'Close': 93.41999816894533,
                 'Adj Close': 93.41999816894533, 'Volume': 16431046.}
        for field, value in quote.items():
            self.frames['BBB'].at[self.dates[0], field] = value
        decision = self.decision()
        data, raw = self.seen[-1]
        self.assertEqual(data.opening.at[self.dates[0], 'BBB'], quote['Open'])
        self.assertEqual(raw['low'].at[self.dates[0], 'BBB'], quote['Low'])
        self.assertEqual(decision['buys'][0]['reference_close'], quote['Close'])
        # The same contract applies to a separate opening quote above High.
        self.frames['BBB'].at[self.dates[0], 'Open'] = quote['High'] + .1
        self.assertEqual(self.decision()['buys'], decision['buys'])

    def test_late_prospective_rejected(self):
        self.narrow_to_first_day()
        plan = self.decision()
        plan['origin'] = 'prospective'
        self.record(plan, '2026-10-16T13:30Z')
        with self.assertRaisesRegex(RuntimeError, 'origin'):
            self.run_replay()

    def test_legacy_explicit_late_plan_requires_daily_authorization(self):
        self.narrow_to_first_day()
        plan = {'session': '2026-10-16', 'signal_date': '2026-10-15', 'origin': 'user_authorized_retrospective', 'buys': [], 'sells': []}
        self.record(plan, '2026-10-16T14:00Z')
        with self.assertRaisesRegex(RuntimeError, 'daily authorization'):
            self.run_replay()
        self.db.execute('INSERT INTO meta VALUES (?,?)', ('authorized_retrospective_2026-10-16', 'explicit authorization'))
        self.db.commit()
        self.assertEqual(self.run_replay(), 1)
        self.assertEqual(self.db.execute('SELECT status FROM sessions').fetchone()[0], 'user_authorized_retrospective_executed')

    def test_valid_late_v3_plan_reused_and_provenance_required(self):
        self.narrow_to_first_day()
        plan = self.decision()
        plan.update(origin=ORIGIN, execution_policy=VERSION, policy_authorized_at_utc=self.authorized)
        self.record(plan, self.now.isoformat())
        self.assertEqual(self.run_replay(), 1)

    def test_wrong_late_policy_binding_rejected(self):
        self.narrow_to_first_day()
        plan = self.decision()
        plan.update(origin=ORIGIN, execution_policy=VERSION, policy_authorized_at_utc='2026-10-18T00:00Z')
        self.record(plan, self.now.isoformat())
        with self.assertRaisesRegex(RuntimeError, 'authorization/provenance'):
            self.run_replay()

    def test_bad_plan_signal_date_rejected(self):
        self.narrow_to_first_day()
        self.record({'session': '2026-10-16', 'signal_date': '2026-10-16', 'buys': [], 'sells': []})
        with self.assertRaisesRegex(RuntimeError, 'previous session'):
            self.run_replay()

    def test_caller_transaction_and_incomplete_session_rejected(self):
        self.db.execute('INSERT INTO meta VALUES (?,?)', ('uncommitted', 'x'))
        with self.assertRaisesRegex(RuntimeError, 'caller transaction'):
            self.run_replay()
        self.db.rollback()
        self.now = pd.Timestamp('2026-10-19T20:05Z')
        self.set_policy({'version': VERSION, 'authorized_at_utc': '2026-10-19T19:00Z'})
        with self.assertRaisesRegex(RuntimeError, 'not complete'):
            self.run_replay()

    def test_execution_day_split_applies_once_to_new_atr(self):
        self.narrow_to_first_day()
        for frame in self.frames.values():
            frame[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Dividends']] /= 2
            frame.loc[:'2026-10-15', 'Volume'] *= 2
            frame.loc['2026-10-16', 'Stock Splits'] = 2.
        self.assertEqual(self.run_replay(), 1)
        plan = json.loads(self.db.execute('SELECT payload FROM plans').fetchone()[0])
        self.assertEqual(plan['buys'][0]['atr'], 4.)
        position = self.m.latest_state(self.db)['positions']['BBB']
        self.assertEqual(position['atr'], 2.)
        self.assertEqual(position['entry_price'], 51.)
        self.assertAlmostEqual(position['qty'], 10000 / 1.0005 / 51)

    def test_existing_position_split_and_signal_units(self):
        self.narrow_to_first_day()
        self.cfg['exit'] = 'guarded10'
        state = engine.initial_state()
        state['last_date'] = '2026-10-15'
        state['cash'] = 90000.
        state['positions']['BBB'] = {'entry_date': '2026-10-15', 'entry_price': 100., 'qty': 100.,
                                     'cost': 10000., 'entry_fee': 0., 'dividends': 0., 'atr': 4.}
        self.db.execute("UPDATE meta SET value=? WHERE key='initial_state'", (dumps(state),))
        self.db.commit()
        for frame in self.frames.values():
            frame[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Dividends']] /= 2
            frame.loc[:'2026-10-15', 'Volume'] *= 2
            frame.loc['2026-10-16', 'Stock Splits'] = 2.
        self.assertEqual(self.run_replay(), 1)
        position = self.m.latest_state(self.db)['positions']['BBB']
        self.assertEqual(position['qty'], 200.)
        self.assertEqual(position['entry_price'], 50.)
        self.assertEqual(position['atr'], 2.)
        self.assertEqual(self.seen[-1][0].close.at[self.dates[0], 'BBB'], 100.)
        action = json.loads(self.db.execute('SELECT data FROM corporate_actions').fetchone()[0])
        self.assertEqual(action, {'symbol': 'BBB', 'split': 2.})

    def test_legacy_plan_atr_rescaled_without_changing_record(self):
        self.narrow_to_first_day()
        plan = {'session': '2026-10-16', 'signal_date': '2026-10-15', 'buys': [
            {'symbol': 'BBB', 'budget': 10000., 'reference_close': 50., 'atr': 2.}], 'sells': []}
        self.record(plan)
        self.assertEqual(self.run_replay(), 1)
        self.assertEqual(self.m.latest_state(self.db)['positions']['BBB']['atr'], 4.)
        self.assertEqual(json.loads(self.db.execute('SELECT payload FROM plans').fetchone()[0]), plan)


if __name__ == '__main__':
    unittest.main()
