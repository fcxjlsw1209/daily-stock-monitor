import copy
import unittest
from audit_panel import audit


class PanelAuditTests(unittest.TestCase):
    def fixture(self):
        event = dict(event_id='A_1', symbol='A', known_by_at='2024-04-23T20:00:00+00:00')
        row = dict(signal_date='2024-04-30', symbol='A', event_id='A_1', status='verified_event_available',
            signal_asof='2024-04-30T20:00:00+00:00', reactions={'3':dict(status='ready', horizon=3,
            known_by_at=event['known_by_at'], anchor_close_at=event['known_by_at'],
            end_close_at='2024-04-26T20:00:00+00:00', post_known_close_to_close_return=.01)})
        for h in (10, 21):
            row['reactions'][str(h)] = dict(status='waiting_for_closed_window', horizon=h,
                                           known_by_at=event['known_by_at'])
        return row, event

    def test_valid_and_duplicate(self):
        row, event = self.fixture()
        self.assertEqual(audit([row], [row], [event])['future_reaction_values'], 0)
        with self.assertRaisesRegex(ValueError, 'Duplicate monthly'):
            audit([row, row], [row], [event])

    def test_verified_candidate_cannot_silently_drop_horizons(self):
        row, event = self.fixture()
        for keys in ((), ('3',)):
            altered = copy.deepcopy(row)
            altered['reactions'] = {k: row['reactions'][k] for k in keys}
            with self.assertRaisesRegex(ValueError, 'all three reaction'):
                audit([altered], [row], [event])
        row['reactions']['3']['horizon'] = 10
        with self.assertRaisesRegex(ValueError, 'horizon identity'):
            audit([row], [row], [event])

    def test_future_window_is_rejected(self):
        row, event = self.fixture()
        row['reactions']['3']['end_close_at'] = '2024-05-01T20:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'Future'):
            audit([row], [row], [event])

    def test_signal_timestamp_cannot_move_into_future(self):
        row, event = self.fixture()
        row['signal_asof'] = '2030-01-01T20:00:00+00:00'
        row['reactions']['3']['end_close_at'] = '2029-12-31T20:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'canonical XNYS'):
            audit([row], [row], [event])

    def test_missing_is_not_zero_and_latest_event_is_not_replaced(self):
        row, event = self.fixture()
        row['reactions']['3'].update(status='not_yet_complete', post_known_close_to_close_return=0.)
        with self.assertRaisesRegex(ValueError, 'zero-filled'):
            audit([row], [row], [event])
        other = copy.deepcopy(row); other['event_id'] = 'older'
        with self.assertRaisesRegex(ValueError, 'Latest event'):
            audit([other], [row], [event])


if __name__ == '__main__':
    unittest.main()
