"""Synthetic administrative-date/source tests; no market data or lag values."""
from copy import deepcopy
from datetime import date, timedelta, timezone, datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import after157_annual_pair_source_feasibility as target


def weekdays(start='2017-01-02', end='2023-12-29'):
    d=date.fromisoformat(start); stop=date.fromisoformat(end); out=[]
    while d<=stop:
        if d.weekday()<5:out.append(d.isoformat())
        d+=timedelta(days=1)
    return out


def event(number, report='2019-12-31', filed='2020-02-03', form='10-K', **changes):
    value=dict(cik='0000000001',accessionNumber='0000000001-20-%06d'%number,
        form=form,filingDate=filed,reportDate=report,
        raw_acceptanceDateTime=filed+'T12:00:00Z',primaryDocument='annual.htm',
        event_id='event-%s'%number,occurrence=dict(snapshot='a',source='a',row=number))
    value.update(changes);return value


def normal_pair():
    return [event(1,'2018-12-31','2019-02-01'),event(2)]


class PairTests(unittest.TestCase):
    def setUp(self):self.sessions=weekdays()

    def pair(self, events=None, cutoff='2020-06-01'):
        return target.pair_metadata(normal_pair() if events is None else events,cutoff,self.sessions)

    def test_normal_metadata_pair_is_not_ordinary_period_proof(self):
        value=self.pair()
        self.assertTrue(value['tier1']);self.assertEqual(value['reasons'],[])
        self.assertEqual([e['event_id'] for e in value['pair']],['event-2','event-1'])
        self.assertNotIn('tier3',value)

    def test_highest_fiscal_end_not_latest_filing(self):
        events=normal_pair()+[event(3,'2017-12-31','2020-05-01')]
        self.assertEqual([e['event_id'] for e in self.pair(events)['pair']],['event-2','event-1'])

    def test_future_latest_ignored_without_leaking_into_historical_pair(self):
        events=normal_pair()+[event(3,'2020-12-31','2021-02-01')]
        self.assertEqual(self.pair(events),self.pair())

    def test_second_strict_session_friday_to_tuesday(self):
        events=[normal_pair()[0],event(2,filed='2020-02-07')]
        self.assertIn('LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42',self.pair(events,'2020-02-10')['reasons'])
        self.assertEqual(self.pair(events,'2020-02-11')['reasons'],[])

    def test_weekend_uses_same_two_following_sessions(self):
        events=[normal_pair()[0],event(2,filed='2020-02-08')]
        self.assertIn('LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42',self.pair(events,'2020-02-10')['reasons'])
        self.assertEqual(self.pair(events,'2020-02-11')['reasons'],[])

    def test_missing_calendar_tail_blocks(self):
        value=target.pair_metadata(normal_pair(),'2020-02-04',weekdays(end='2020-02-04'))
        self.assertIn('LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42',value['reasons'])

    def test_recent_unavailable_latest_blocks_not_fallback(self):
        events=normal_pair()+[event(3,'2020-12-31','2021-02-01')]
        value=self.pair(events,'2021-02-02')
        self.assertEqual(value['pair'][0]['event_id'],'event-3')
        self.assertIn('LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42',value['reasons'])

    def test_latest_transition_blocks_not_fallback(self):
        value=self.pair(normal_pair()+[event(3,'2020-03-31','2020-05-01','10-KT')])
        self.assertEqual(value['pair'][0]['form'],'10-KT')
        self.assertIn('LATEST_PAIR_CONTAINS_TRANSITION_REPORT',value['reasons'])
        self.assertFalse(value['tier1'])

    def test_duplicate_original_period_blocks(self):
        value=self.pair(normal_pair()+[event(3,filed='2020-03-02')])
        self.assertIn('AMBIGUOUS_ORIGINAL_FOR_PERIOD',value['reasons'])

    def test_bad_latest_report_date_blocks_clean_old_pair(self):
        value=self.pair(normal_pair()+[event(3,report='bad',filed='2020-05-01')])
        self.assertIn('UNLOCATABLE_ORIGINAL_REPORT_DATE',value['reasons'])

    def test_unlocatable_filing_date_blocks(self):
        value=self.pair(normal_pair()+[event(3,filed='bad')])
        self.assertIn('UNLOCATABLE_ANNUAL_FILING_DATE',value['reasons'])

    def test_known_selected_period_amendment_blocks(self):
        value=self.pair(normal_pair()+[event(3,filed='2020-05-01',form='10-K/A')])
        self.assertIn('KNOWN_SELECTED_PERIOD_AMENDMENT_REQUIRES_ADJUDICATION',value['reasons'])

    def test_future_amendment_is_not_known_by_cutoff(self):
        self.assertEqual(self.pair(normal_pair()+[event(3,filed='2020-07-01',form='10-K/A')]),self.pair())

    def test_unselected_old_period_amendment_not_new_original(self):
        self.assertEqual(self.pair(normal_pair()+[event(3,'2017-12-31','2020-05-01','10-K/A')]),self.pair())

    def test_provisional_spacing_bounds_do_not_prove_annual_period(self):
        newer=date(2019,12,31)
        for days,expected in ((329,False),(330,True),(365,True),(400,True),(401,False)):
            with self.subTest(days=days):
                value=self.pair([event(1,(newer-timedelta(days=days)).isoformat(),'2019-03-01'),event(2)])
                self.assertEqual('REPORT_SPACING_OUTSIDE_PROVISIONAL_ANNUAL_SCOPE' not in value['reasons'],expected)

    def test_stale_highest_report_blocks(self):
        self.assertIn('LATEST_ANNUAL_OUTSIDE_PROVISIONAL_550_DAY_SCOPE',self.pair(cutoff='2021-07-05')['reasons'])

    def test_950_day_source_scope_excludes_older_original(self):
        events=[event(1,'2016-12-31','2017-02-01'),event(2)]
        self.assertIn('FEWER_THAN_TWO_ORIGINAL_REPORT_DATES_IN_SCOPE',self.pair(events)['reasons'])

    def test_t_minus_42_is_session_position_not_calendar_subtraction(self):
        sessions=weekdays('2020-01-01','2020-06-30')
        pos,day=target.information_cutoff(sessions,sessions[90])
        self.assertEqual((pos,day),(48,sessions[48]))
        self.assertNotEqual(day,(date.fromisoformat(sessions[90])-timedelta(days=42)).isoformat())

    def test_cutoff_requires_original_signal_and_warmup(self):
        for day in ('2020-01-05','2017-01-03'):
            with self.assertRaises(ValueError):target.information_cutoff(self.sessions,day)


def table(events):
    fields=('accessionNumber','form','filingDate','reportDate','acceptanceDateTime','primaryDocument')
    return {f:[e['raw_acceptanceDateTime' if f=='acceptanceDateTime' else f] for e in events] for f in fields}


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.src=target.Sources(self.root);self.hashes={}
    def tearDown(self):self.tmp.cleanup()

    def write(self,name,value):
        (self.root/name).parent.mkdir(parents=True,exist_ok=True)
        blob=json.dumps(value,sort_keys=True).encode();(self.root/name).write_bytes(blob)
        self.hashes[name]=hashlib.sha256(blob).hexdigest();return self.hashes[name]

    def proof(self,name,value,url):
        h=self.write(name,value);receipt=str(Path(name).with_suffix('.provenance.json'))
        self.write(receipt,dict(sha256=h,source=url,retrieved_at='2023-12-31T00:00:00Z'))
        return dict(path=name,sha256=h,receipt=receipt,source_receipt_verified=True,expected_source_url=url)

    def fixture(self,page_events=None,page_count=None,old_unparsed=False):
        items=normal_pair() if page_events is None else page_events
        desc=dict(name='CIK0000000001-submissions-001.json',filingFrom='2010-01-01',filingTo='2020-12-31',filingCount=len(items) if page_count is None else page_count)
        page='raw/'+desc['name']
        p=self.proof(page,table(items),'https://data.sec.gov/submissions/'+desc['name'])
        page_item=dict(path=page,advertised=desc,status='OUTSIDE_OLD_SCOPE' if old_unparsed else 'PRESENT')
        if not old_unparsed:page_item['source']=p
        rootproof=self.proof('raw/root.json',dict(cik='1',filings=dict(recent=table([items[0]]),files=[desc])),
            'https://data.sec.gov/submissions/CIK0000000001.json')
        return dict(input_hashes=self.hashes,submission_snapshots=[dict(cik='0000000001',source=rootproof,advertised_pages=[page_item])])

    def test_original_receipts_and_exact_duplicates(self):
        snaps,occ=target.load_snapshots(self.src,self.fixture())
        self.assertEqual(len(snaps['raw/root.json']['events']),2)
        self.assertEqual(len(snaps['raw/root.json']['events'][0]['occurrences']),2)
        self.assertEqual(len(occ),3)
        self.assertTrue(snaps['raw/root.json']['pages'][0]['receipt'])

    def test_pre2015_advertised_page_requires_its_own_receipt(self):
        inv=self.fixture(old_unparsed=True);snaps,_=target.load_snapshots(self.src,inv)
        self.assertTrue(snaps['raw/root.json']['pages'][0]['receipt'])
        self.assertIn('raw/CIK0000000001-submissions-001.provenance.json',self.src.hashes)

    def test_missing_receipt_does_not_parse_unbound_old_page(self):
        inv=self.fixture(old_unparsed=True)
        (self.root/'raw/CIK0000000001-submissions-001.provenance.json').unlink()
        snaps,_=target.load_snapshots(self.src,inv)
        page=snaps['raw/root.json']['pages'][0]
        self.assertTrue(page['present']);self.assertFalse(page['receipt'])
        chosen,_=target.choose_snapshot(snaps,['raw/root.json'],'2020-06-01');self.assertIsNone(chosen)

    def test_wrong_receipt_url_rejected(self):
        inv=self.fixture();name='raw/CIK0000000001-submissions-001.provenance.json'
        raw=json.loads((self.root/name).read_text());raw['source']='https://wrong.example/source'
        self.write(name,raw)
        with self.assertRaisesRegex(ValueError,'receipt mismatch'):target.load_snapshots(self.src,inv)

    def test_advertised_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'count mismatch'):target.load_snapshots(self.src,self.fixture(page_count=3))

    def test_source_hash_mismatch_rejected(self):
        inv=self.fixture();(self.root/'raw/root.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Source SHA'):target.load_snapshots(self.src,inv)

    def test_transition_and_nonannual_accession_variants_retained(self):
        original=event(1);other=dict(original,form='8-K');transition=event(3,form='10-KT')
        snaps,occ=target.load_snapshots(self.src,self.fixture([original,other,transition]))
        self.assertEqual({e['form'] for e in snaps['raw/root.json']['events']},{'10-K','10-KT'})
        self.assertEqual({e['form'] for e in occ},{'10-K','8-K','10-KT'})
        cross=target.exceptions.cross_index(occ,timezone(timedelta(hours=-5)))
        self.assertTrue(cross[('0000000001',original['accessionNumber'])]['metadata_conflict'])

    def test_matching_hint_receipt_does_not_create_accession_proof(self):
        p=self.proof('release.json',dict(period_end='2019-12-31'),'https://issuer.example/release')
        hint=dict(source=dict(path=p['path'],sha256=p['sha256'],receipt=p['receipt']),accession_specific_normality=False)
        checked=target.check_hint(self.src,hint)
        self.assertTrue(checked['source_bytes_sha_verified']);self.assertTrue(checked['original_receipt_sha_verified'])
        self.assertFalse(checked['accession_specific_normality'])

    def test_wrong_hint_source_hash_rejected(self):
        self.write('release.json',{})
        with self.assertRaisesRegex(ValueError,'Source SHA'):
            target.check_hint(self.src,dict(source=dict(path='release.json',sha256='0'*64)))


class RoutingAndCounterTests(unittest.TestCase):
    def snapshot(self,path,closed=True):
        return dict(path=path,cik='0000000001',root_receipted=closed,retrieved_date='2023-01-01',pages=[],receipted_records=[])

    def test_source_choice_ignores_better_report_pair(self):
        snaps={p:self.snapshot(p) for p in ('z','a')}
        snaps['a']['events']=[event(1,form='10-KT')];snaps['z']['events']=normal_pair()
        chosen,_=target.choose_snapshot(snaps,['z','a'],'2020-06-01')
        self.assertEqual(chosen,'a')
        self.assertIn('LATEST_PAIR_CONTAINS_TRANSITION_REPORT',target.pair_metadata(snaps[chosen]['events'],'2020-06-01',weekdays())['reasons'])

    def test_source_choice_changes_only_when_closure_fails(self):
        snaps=dict(a=self.snapshot('a',False),z=self.snapshot('z'))
        self.assertEqual(target.choose_snapshot(snaps,['z','a'],'2020-06-01')[0],'z')

    def test_duplicate_cik_slots_do_not_count_as_eight_issuers(self):
        rows=[dict(signal_index=0,is_etf=False,tier1=True,tier2=True,tier3=False,
            cik='one',reasons=[],period_hint_ids=[['a'],['b']]) for _ in range(8)]
        result=target.summaries(rows)['first88']
        self.assertEqual(result['tier2']['slots'],8);self.assertEqual(result['tier2']['distinct_issuers'],1)
        self.assertEqual(result['tier2']['months_at_least8_issuers'],0)

    def test_later34_not_counted_as_first88_and_etf_excluded(self):
        rows=[dict(signal_index=90,is_etf=False,tier1=True,tier2=True,tier3=False,cik='one',reasons=[],period_hint_ids=[[],[]]),
              dict(signal_index=0,is_etf=True,tier1=False,tier2=False,tier3=False,cik=None,reasons=[],period_hint_ids=[])]
        result=target.summaries(rows)
        self.assertEqual(result['first88']['stock_slots'],0)
        self.assertEqual(result['later34']['tier2']['slots'],1)


if __name__=='__main__':unittest.main(verbosity=2)
