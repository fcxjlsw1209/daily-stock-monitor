import copy
import json
from pathlib import Path
import sys
import unittest

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import finalize


class OriginalEvidenceTests(unittest.TestCase):
    def test_failed_lookup_is_not_an_event(self):
        with self.assertRaises(ValueError):
            finalize.tsm(dict(excerpt='quarter ended September 30, 2020',tool_error=True),P/'tsm_observations.json')

    def test_microsoft_title_not_allowed_to_override_real_period(self):
        rows=json.loads((P/'msft_observations.json').read_text())
        row=next(x for x in rows if not x.get('tool_error') and 'FY14 Q2' in x.get('title',''))
        result=finalize.msft(row,P/'msft_observations.json')
        self.assertEqual(result['fiscal_period_end_actual'],'2013-12-31')
        self.assertIsNone(result['publication_timestamp'])
        changed=copy.deepcopy(row);changed['title']='FY14 Q3'
        with self.assertRaises(ValueError):finalize.msft(changed,P/'msft_observations.json')

    def test_berkshire_uses_explicit_prior_publication_not_filing(self):
        obs=json.loads((P/'other_observations.json').read_text())
        row=next(x for x in obs if x.get('ticker')=='BRK.B' and x['fiscal_year']==2010)
        out=finalize.other(row,P/'other_observations.json',obs)
        self.assertEqual(out['release_date'],'2011-02-26')
        self.assertEqual(out['fiscal_period_end_actual'],'2010-12-31')
        row=copy.deepcopy(row);row['excerpt']='Filed February 28, 2011. 2010 Annual Report.'
        with self.assertRaises(ValueError):finalize.other(row,P/'other_observations.json',obs)

    def test_seven_extended_histories_have_continuous_quarters(self):
        sources=['apple','microsoft','nvidia','amazon','amazon/supplement','alphabet','berkshire','tsmc','warmup/verified']
        rows=[]
        for name in sources:
            raw=json.loads((P.parent/name/'events.json').read_text())
            rows.extend(raw.get('events',[]) if isinstance(raw,dict) else raw)
        for ticker in ('AAPL','MSFT','NVDA','AMZN','GOOGL','BRK.B','TSM'):
            values=[r['fiscal_year']*4+r['fiscal_quarter']-1 for r in rows if r['ticker']==ticker]
            self.assertEqual(len(values),len(set(values)),ticker)
            self.assertEqual(set(values),set(range(min(values),max(values)+1)),ticker)
            self.assertEqual(len(values),63,ticker)


if __name__=='__main__':unittest.main()
