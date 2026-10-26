"""Offline evidence tests, no performance or network requests."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec=importlib.util.spec_from_file_location('meta_collect',Path(__file__).with_name('collect.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def article(year=2023,quarter=1,date='April 26, 2023',ticker='META'):
    return ('Meta Reports First Quarter 2023 Results\nMENLO PARK, Calif., '+date+
            ' /PRNewswire/ -- Meta Platforms, Inc. (Nasdaq: '+ticker+') today reported financial results for '
            'the quarter ended March 31, 2023. Conference call at 1:30 p.m. PT / 4:30 p.m. ET.\n')


class EvidenceTests(unittest.TestCase):
    def parse(self,s=None,y=2023,q=1):
        return c.parse_pdf_text(s or article(),y,q,c.PREFIX+'doc_news/test.pdf','outputs/test.pdf',b'%PDF fixture')

    def test_date_only_and_no_call_time_inference(self):
        e=self.parse();self.assertEqual(e['release_date'],'2023-04-26')
        self.assertIsNone(e['publication_timestamp']);self.assertFalse(e['first_public_time_verified'])
        self.assertEqual(e['publication_timezone'],'America/Los_Angeles')
        self.assertEqual(e['evidence_sha256'],hashlib.sha256(b'%PDF fixture').hexdigest())
        self.assertEqual(e['ticker'],'research_equity_FB');self.assertEqual(e['historical_ticker'],'META')

    def test_calendar_announcement_rejected(self):
        with self.assertRaises(ValueError):self.parse(article().replace('Meta Reports','Meta to Announce'))

    def test_wrong_year_or_quarter_rejected(self):
        for y,q in [(2022,1),(2023,2)]:
            with self.assertRaises(ValueError):self.parse(y=y,q=q)

    def test_actual_period_not_vendor_label(self):
        with self.assertRaises(ValueError):self.parse(article().replace('March 31','March 30'))

    def test_release_after_period_required(self):
        with self.assertRaises(ValueError):self.parse(article(date='March 1, 2023'))

    def test_no_dateline_timezone_guess(self):
        with self.assertRaises(ValueError):self.parse(article().replace('MENLO PARK, Calif.,','Date modified:'))

    def test_foreign_or_reused_ticker_rejected(self):
        with self.assertRaises(ValueError):self.parse(article(ticker='FB'))
        with self.assertRaises(ValueError):self.parse(article(ticker='XYZ'))

    def test_historic_fb_keeps_research_identity_and_whitespace(self):
        s='Facebook Reports Third Quarter 2012 Results MENLO PARK , Calif. – Oct. 23 , 2012 – Facebook, Inc. (NASDAQ: FB ) today reported financial results for the third quarter, which ended September 30, 2012.'
        e=self.parse(s,2012,3);self.assertEqual(e['historical_ticker'],'FB');self.assertEqual(e['ticker'],'research_equity_FB')
        self.assertEqual(e['release_date'],'2012-10-23')

    def test_cross_year_q4(self):
        s='Meta Reports Fourth Quarter and Full Year 2022 Results MENLO PARK, Calif., Feb. 1, 2023 /PRNewswire/ -- Meta Platforms, Inc. (Nasdaq: META) today reported financial results for the quarter and full year ended December 31, 2022.'
        e=self.parse(s,2022,4);self.assertEqual(e['fiscal_year'],2022);self.assertEqual(e['release_date'],'2023-02-01')

    def test_expected_horizon_57_not_future(self):
        self.assertEqual(len(c.expected_quarters()),57)
        self.assertEqual(c.expected_quarters()[0],(2012,2));self.assertEqual(c.expected_quarters()[-1],(2026,2))

    def test_cached_checksum_refuses_modified_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td);url=c.PREFIX+'doc_news/test.pdf';key=c.sha(url.encode())[:16]
            (raw/(key+'.pdf')).write_bytes(b'changed')
            (raw/(key+'.meta.json')).write_text(json.dumps({'sha256':c.sha(b'original')}))
            with mock.patch.object(c,'RAW',raw),mock.patch.object(c.urllib.request,'urlopen',side_effect=AssertionError('No network')):
                with self.assertRaises(RuntimeError):c.fetch(url)

    def test_pdf_concatenated_headline_tokens(self):
        e=self.parse(article().replace('2023 Results','2023Results'))
        self.assertEqual(e['fiscal_year'],2023)

    def test_source_stop_honors_retry_after_and_never_retries(self):
        import urllib.error
        import io
        import datetime as dt
        from email.message import Message
        h=Message();h['Retry-After']='172800'
        url=c.PREFIX+'doc_news/test.pdf'
        err=urllib.error.HTTPError(url,429,'limited',h,io.BytesIO(b'limited'))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);raw=p/'raw';raw.mkdir()
            with mock.patch.object(c,'HERE',p),mock.patch.object(c,'RAW',raw),mock.patch.object(c.time,'sleep'),mock.patch.object(c.urllib.request,'urlopen',side_effect=err) as req:
                with self.assertRaises(RuntimeError):c.fetch(url)
                stop=json.loads((p/'SOURCE_STOP.json').read_text())['s21.q4cdn.com']
                self.assertEqual(stop['retry_after'],'172800')
                self.assertGreater(dt.datetime.fromisoformat(stop['stop_until']),dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=47))
                with self.assertRaises(RuntimeError):c.fetch(c.PREFIX+'doc_news/other.pdf')
                self.assertEqual(req.call_count,1)

    def test_wrong_host_rejected_without_network(self):
        with self.assertRaises(ValueError):c.fetch('https://not-meta.example/file.pdf')

if __name__=='__main__':unittest.main()
