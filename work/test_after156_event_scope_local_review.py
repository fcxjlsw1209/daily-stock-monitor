"""Synthetic boundary/adversarial checks only; no actual source reads."""
import copy
from pathlib import Path
import tempfile
import unittest

from after156_event_scope_local_review import Sources, sequence, summarize, counts


DAYS = ['2020-01-%02d' % n for n in (2,3,6,7,8,9,10,13,14,15)]


def event(index, quarter, name=None):
    return dict(event_id=name or 'X_'+DAYS[index],symbol='X',available_session=DAYS[index],
                publication_date=DAYS[index],fiscal_year=2019,fiscal_quarter=quarter)


def row(after, slot='000:00'):
    return dict(slot_id=slot,signal_date='2020-02-01',interval_id='X_interval',after=after)


class IntervalTests(unittest.TestCase):
    def test_internal_gap_repaired_only_by_correct_intermediate_quarter(self):
        base = [event(2,1),event(6,3)]
        self.assertEqual(sequence(base,DAYS,DAYS[2],DAYS[8])['quarter_items'][0]['intermediate_fiscal_labels'],['2019Q2'])
        self.assertEqual(sequence(base+[event(4,2)],DAYS,DAYS[2],DAYS[8])['quarter_items'],[])
        self.assertEqual(len(sequence(base+[event(4,1)],DAYS,DAYS[2],DAYS[8])['quarter_items']),2)

    def test_outside_predecessor_is_context_and_not_consumed(self):
        out = sequence([event(0,1),event(4,3)],DAYS,DAYS[2],DAYS[8])
        self.assertEqual(out['event_ids'],['X_'+DAYS[4]])
        self.assertEqual(out['quarter_items'][0]['scope'],'left_predecessor_crossing_context')

    def test_only_nearest_outside_predecessor_matters(self):
        out = sequence([event(0,4),event(1,1),event(4,2)],DAYS,DAYS[2],DAYS[8])
        self.assertEqual(out['quarter_items'],[])
        self.assertEqual(out['optional_predecessor'],'X_'+DAYS[1])

    def test_inclusive_boundaries_future_events_do_not_affect_result(self):
        base = [event(2,1),event(8,2)]
        out = sequence(base,DAYS,DAYS[2],DAYS[8])
        self.assertEqual(len(out['event_ids']),2)
        self.assertEqual(out,sequence(base+[event(9,4)],DAYS,DAYS[2],DAYS[8]))

    def test_unknown_future_clock_excluded_but_unknown_history_retained(self):
        unknown = event(1,1)
        unknown['available_session'] = None
        future = dict(unknown,event_id='future',publication_date='2021-01-01')
        self.assertEqual(sequence([unknown,future],DAYS,DAYS[2],DAYS[8])['unknown_anchor_ids'],[unknown['event_id']])

    def test_unknown_publication_date_not_discarded(self):
        unknown = dict(event(1,1),available_session=None,publication_date=None)
        self.assertEqual(sequence([unknown],DAYS,DAYS[2],DAYS[8])['unknown_anchor_ids'],[unknown['event_id']])

    def test_duplicate_quarter_events_retained(self):
        out=sequence([event(2,1),event(3,1)],DAYS,DAYS[2],DAYS[8])
        self.assertEqual(len(out['event_ids']),2)
        self.assertEqual(out['quarter_items'][0]['kind'],'duplicate_quarter')

    def test_nonmonotone_quarter_not_treated_as_repair(self):
        self.assertEqual(sequence([event(2,3),event(3,1)],DAYS,DAYS[2],DAYS[8])['quarter_items'][0]['kind'],'nonmonotone_quarter')

    def test_bool_quarter_is_unknown(self):
        self.assertEqual(sequence([event(2,True),event(3,2)],DAYS,DAYS[2],DAYS[8])['quarter_items'][0]['kind'],'unknown_quarter_label')

    def test_year_boundary_continuity(self):
        following=dict(event(4,1),fiscal_year=2020)
        self.assertEqual(sequence([event(2,4),following],DAYS,DAYS[2],DAYS[8])['quarter_items'],[])

    def test_same_pair_counted_once_across_slots_but_scopes_separate(self):
        a=row(sequence([event(2,1),event(6,3)],DAYS,DAYS[2],DAYS[8]))
        b=copy.deepcopy(a);b['slot_id']='001:00'
        c=row(sequence([event(2,1),event(6,3)],DAYS,DAYS[3],DAYS[8]),'002:00')
        self.assertEqual(len(summarize([a,b,c])),2)
        self.assertEqual(counts([a,b,c])['unique_inside_pairs'],1)
        self.assertEqual(counts([a,b,c])['slots_with_any_quarter_flag'],3)

    def test_inputs_unchanged_and_input_order_irrelevant(self):
        items=[event(6,3),event(2,1)];saved=copy.deepcopy(items)
        out=sequence(items,DAYS,DAYS[2],DAYS[8])
        self.assertEqual(items,saved)
        self.assertEqual(out,sequence(list(reversed(items)),DAYS,DAYS[2],DAYS[8]))

    def test_gap_free_empty_window_does_not_invent_events(self):
        out=sequence([],DAYS,DAYS[2],DAYS[8])
        self.assertEqual(out,dict(event_ids=[],quarter_items=[],unknown_anchor_ids=[],optional_predecessor=None))


class SourceTests(unittest.TestCase):
    def test_wrong_hash_size_mutation_and_escape_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'a';p.write_bytes(b'first');src=Sources(temp)
            with self.assertRaises(ValueError):src.read_bytes('a','0'*64)
            with self.assertRaises(ValueError):src.read_bytes('a',size=9)
            src.read_bytes('a');p.write_bytes(b'second')
            with self.assertRaises(ValueError):src.recheck()
            with self.assertRaises(ValueError):src.path('../a')

    def test_symlink_denied_and_nested_reference_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'a';p.write_bytes(b'first');src=Sources(temp)
            src.read_bytes('a');h=src.hashes['a']
            src.refs({'nested':[{'path':'a','sha256':h,'bytes':5}]})
            (Path(temp)/'link').symlink_to(p)
            with self.assertRaises(ValueError):src.read_bytes('link')


if __name__=='__main__':
    unittest.main(verbosity=2)
