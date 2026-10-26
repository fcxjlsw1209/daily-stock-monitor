"""Focused synthetic checks for original HD tables and as-of fiscal boundaries."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
import monthly_hd_original_v2_independent_review as m

P=Path(__file__).resolve().parent.parent/'outputs/monthly_sec_hd_original_admission_v2'
def load(name):
    spec=importlib.util.spec_from_file_location('hd_synthetic_'+name,P/(name+'.py'));x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);return x
p=load('test_parser');old=sys.modules.get('test_parser');sys.modules['test_parser']=p
try:t=load('test_admission')
finally:
    if old is None:del sys.modules['test_parser']
    else:sys.modules['test_parser']=old


class IndependentTests(unittest.TestCase):
    def setUp(self):self.d,self.docs=t.synthetic()
    def read(self,n):return m.strict_json(self.docs[n].decode())
    def change(self,n,v):self.docs[n]=t.b.output_bytes(v);self.d['source_hashes'][n]=m.sha256(self.docs[n]).hexdigest()
    def reader(self,path):return dict(extractor_version='6.10.0',pages=[self.docs[str(path.with_suffix('.txt'))].decode()[:-1]])
    def actual(self):return {n:m.strict_json(v.decode()) for n,v in t.b.produce(self.d,self.docs,t.fake_reconstruct).items()}
    def run_oracle(self,actual=None):return m.reconstruct(Path('.'),self.d,self.docs,actual or self.actual()['EVIDENCE.json'],self.reader)
    def mutate_text(self,n,text):
        self.docs[n+'.txt']=text.encode();self.d['source_hashes'][n+'.txt']=m.sha256(text.encode()).hexdigest()
        r=self.read(n+'.pdf.textmeta');r['text_sha256']=self.d['source_hashes'][n+'.txt'];self.change(n+'.pdf.textmeta',r)
    def test_three_confirmed_split_anchors(self):
        original=m.parse_document(p.fixture())
        for before,after in [('ATLANTA,','AT\nLANTA,'),('ATLANTA,','A\nTLANTA,'),('THE HOME DEPOT, INC.','T\nHE HOME DEPOT, INC.')]:
            with self.subTest(after=after):
                actual=m.parse_document(p.fixture().replace(before,after))
                self.assertEqual(actual['statement']['periods'],original['statement']['periods'])
                self.assertEqual(actual['identity']['release_date'],original['identity']['release_date'])
    def test_similar_wrong_anchors_still_fail(self):
        for before,after in [('ATLANTA,','AT LANTX,'),('ATLANTA,','A TLANTB,'),('THE HOME DEPOT, INC.','T HX HOME DEPOT, INC.')]:
            with self.subTest(after=after),self.assertRaises(ValueError):m.parse_document(p.fixture().replace(before,after))
    def test_original_admissions_preserved_and_only_prior_gaps_added(self):
        old=[dict(record_index_0based=1,amount=5)];coverage=[dict(record_index_0based=1,status='admitted'),dict(record_index_0based=2,status='unverified')]
        self.assertEqual(m.preserved_admissions(old,coverage,old+[dict(record_index_0based=2,amount=7)]),[2])
        for new in ([],[dict(record_index_0based=1,amount=6)],old+[dict(record_index_0based=3,amount=7)],old+old):
            with self.subTest(new=new),self.assertRaises(ValueError):m.preserved_admissions(old,coverage,new)
    def test_complete_output_parity_and_unchanged_inputs(self):
        before=deepcopy((self.d,self.docs));self.assertEqual(self.run_oracle(),self.actual());self.assertEqual(before,(self.d,self.docs))
    def test_all_quarters_column_orders_and_old_headings(self):
        for q in range(1,5):
            for old in (True,False):
                for reverse in (True,False):
                    with self.subTest(q=q,old=old,reverse=reverse):
                        text=p.fixture(q,old,reverse);self.assertEqual(m.parse_document(text),p.p.parse_release(text))
    def test_calendar_only_record_title(self):
        text=p.fixture(4,fy=2015);self.assertIn('Record Fourth',text);self.assertEqual(m.parse_document(text),p.p.parse_release(text))
    def test_53_week_declaration_parser(self):
        text=p.fixture(4,fy=2018);self.assertEqual(m.parse_document(text),p.p.parse_release(text))
    def test_malformed_amounts_identity_and_units(self):
        for a,b in [('NYSE: HD','NYSE: OTHER'),('ATLANTA,','MISSING,'),('in millions','in billions'),('$ 100,000','$ -100,000'),('$ 100,000','$ 100.000'),('July 31, 2016','July 30, 2016')]:
            with self.subTest(a=a),self.assertRaises(ValueError):m.parse_document(p.fixture().replace(a,b))
    def test_legacy_not_fresh_authenticated(self):
        s,_=m.source_context(self.d,self.docs);self.assertFalse(s['raw_facts_provenance_verified']);self.assertFalse(s['http_status_recorded'])
        r=self.read('facts.meta');r['http_status']=200;self.change('facts.meta',r)
        with self.assertRaises(ValueError):m.source_context(self.d,self.docs)
    def test_facts_exact_duplicate(self):
        self.docs['duplicate']+=b' '
        with self.assertRaises(ValueError):m.source_context(self.d,self.docs)
    def test_actual_pdf_companion_disagreement(self):
        with self.assertRaises(ValueError):m.reconstruct(Path('.'),self.d,self.docs,self.actual()['EVIDENCE.json'],lambda path:dict(extractor_version='6.10.0',pages=['other']))
    def test_archive_must_link_exact_source(self):
        self.docs['target.html']=b'<a href="/other.pdf">other</a>';r=self.read('target.html.meta');r.update(sha256=m.sha256(self.docs['target.html']).hexdigest(),bytes=len(self.docs['target.html']));self.change('target.html.meta',r)
        q=self.read('queue');q[0]['archive']['sha256']=r['sha256'];self.change('queue',q);self.d['scope'][0]['queue_entry_sha256']=m.digest(q[0])
        with self.assertRaises(ValueError):m.release_source(Path('.'),self.d['scope'][0],self.docs,q,self.reader)
    def test_future_calendar_cannot_backfill(self):
        self.mutate_text('prior',p.fixture(4,fy=2015).replace('February 23, 2016','September 1, 2016'));self.d['calendar_only'][0]['release_date']='2016-09-01'
        out=self.run_oracle();self.assertEqual(out,self.actual());self.assertEqual(out['COVERAGE.json']['semantic_admitted_count'],0)
    def test_missing_calendar_stays_unverified(self):
        self.mutate_text('prior',p.fixture(4,fy=2015).replace('NYSE: HD','NYSE: OTHER'));out=self.run_oracle();self.assertEqual(out,self.actual());self.assertEqual(out['COVERAGE.json']['unverified_count'],1)
    def test_amount_mismatch_stays_unverified(self):
        self.mutate_text('target',p.fixture().replace('200,000','200,001'));out=self.run_oracle();self.assertEqual(out,self.actual());self.assertEqual(out['COVERAGE.json']['unverified_count'],1)
    def test_53_weeks_require_declaration_and_exact_start(self):
        parsed=m.parse_document(p.fixture(4,fy=2018));pair=dict(current=dict(start='2018-01-29',end='2019-02-03',duration_days=371,val=200000000000),prior=dict(start='2017-01-30',end='2018-01-28',duration_days=364,val=180000000000))
        cal={str(y):[dict(end=end,release_date='2018-02-20',source_path='calendar.pdf',basis='printed_fiscal_year_column',excerpt='original')] for y,end in [(2017,'2018-01-28'),(2016,'2017-01-29')]}
        self.assertEqual(m.select_period(pair,parsed,cal,12),t.a.compare_pair(pair,12,parsed,cal));self.assertEqual(m.select_period(pair,parsed,cal,12)['status'],'exact_match')
        parsed['fiscal_evidence']['week_declarations']=[];self.assertEqual(m.select_period(pair,parsed,cal,12)['reason'],'fiscal_week_declaration_mismatch')
        pair['current']['start']='2018-02-04';pair['current']['duration_days']=365;self.assertEqual(m.select_period(pair,parsed,cal,12)['reason'],'fiscal_start_or_duration_mismatch')
    def test_conflicting_calendar_rejected(self):
        out=self.actual();releases=deepcopy(out['EVIDENCE.json']['releases']);item=next(iter(releases.values()));other=deepcopy(item)
        other['parsed']['fiscal_evidence']['year_ends'][0]['end']='2016-01-24';releases['other']=other
        with self.assertRaises(ValueError):m.fiscal_calendar(releases)
    def test_raw_context_duplicates_and_strict_first_session(self):
        _,filings=m.source_context(self.d,self.docs);r=self.read('records')[0];facts=self.read('facts');facts['facts']['us-gaap']['SalesRevenueNet']['units']['USD'].append(deepcopy(facts['facts']['us-gaap']['SalesRevenueNet']['units']['USD'][0]))
        with self.assertRaises(ValueError):m.compare_record(r,self.d['scope'][0],self.read('sessions'),facts,filings,None,{})
        r['catalog']['selected']['first_usable_date']='2016-08-23'
        with self.assertRaises(ValueError):m.compare_record(r,self.d['scope'][0],self.read('sessions'),self.read('facts'),filings,None,{})


if __name__=='__main__':unittest.main()
