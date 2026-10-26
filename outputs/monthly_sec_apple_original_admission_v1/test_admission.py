"""Synthetic hard gates and output tests; no market or original financial inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock
P=Path(__file__).resolve().parent

def load(n):
 spec=importlib.util.spec_from_file_location('apple_test_'+n,P/(n+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

a=load('admission');b=load('build');t=load('test_parser')

def synthetic():
 flags={k:False for k in a.FLAGS};acc='0000320193-16-000001'
 current=dict(accn=acc,start='2015-09-27',end='2016-09-24',val=44008000000,filed='2016-10-26',form='10-K',fy=2016,fp='FY')
 prior=dict(current,start='2014-09-28',end='2015-09-26',val=40004000000)
 pair=dict(taxonomy='us-gaap',tag='SalesRevenueNet',unit='USD',current=dict(current,row_index=0,row_indices=[0],duration_days=364),prior=dict(prior,row_index=1,row_indices=[1],duration_days=364))
 selected=dict(accession=acc,accessionNumber=acc,form='10-K',reportDate='2016-09-24',report_date='2016-09-24',filingDate='2016-10-26',filing_date='2016-10-26',first_usable_date='2016-10-27',metadata_issues=[],supported_form=True,sources=[dict(table='recent',row=0)])
 record=dict(symbol='AAPL',candidate_cik=a.CIK,signal_date='2016-10-31',**flags,catalog=dict(status='selected',selected=selected,coverage_complete=True),pairing=dict(status='paired',structural_pair_available=True,pairs=[pair],**flags))
 facts=dict(cik=320193,facts={'us-gaap':{'SalesRevenueNet':{'units':{'USD':[current,prior]}}}})
 filings={'recent':{k:[selected[k]] for k in ('accessionNumber','form','filingDate','reportDate')}}
 anchor=dict(accession=acc,report_date=selected['reportDate'],filing_date=selected['filingDate'],form='10-K',first_usable_date=selected['first_usable_date'],release_date='2016-10-25',fiscal_quarter=4,record_indices=[0],record_sha256s={'0':a.digest(record)},evidence_pair_hashes={a.digest(dict(accession=acc,pair=pair)):a.digest(pair)},release_path='fake.pdf',attachment_role='financial_statement')
 docs={'records':b.output_bytes([record]),'facts':b.output_bytes(facts),'sessions':b.output_bytes(['2016-10-26','2016-10-27','2016-10-31'])};hashes={k:a.sha(v) for k,v in docs.items()}
 design=dict(policy_id=a.POLICY,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,allowed_tag_semantics=a.SEMANTICS,source_hashes=hashes,source_records=dict(path='records',sha256=hashes['records']),facts_path='facts',sessions_path='sessions',expected_record_count=1,expected_accessions=1,expected_rows=1,expected_pairs=1,expected_pair_occurrences=1,expected_calendar_only_documents=0,scope=[anchor],calendar_only=[],excluded_indices=[])
 source=dict(source_path='fake.pdf',source_sha256='a'*64,parent_release_path='fake.html',parent_release_sha256='b'*64,publication=a.parser.publication(t.html(),t.URL))
 identity=dict(candidate_cik=a.CIK,anchors=[dict(date='2014-06-04'),dict(date='2019-03-12')],split=dict(release_date='2020-07-30'))
 return design,docs,record,pair,facts,filings,anchor,source,identity

def produce_fixture(change=None):
 d,docs,r,p,f,filings,anchor,source,identity=synthetic()
 if change:change(d,docs,r,p,f,filings,anchor,source,identity)
 status=dict(semantic_source_sufficient=True,raw_facts_provenance_verified=True)
 with mock.patch.object(b,'module',return_value=a),mock.patch.object(a,'filing_source',return_value=filings),mock.patch.object(a,'facts_source',return_value=status),mock.patch.object(a,'source_bundle',return_value=None),mock.patch.object(a,'identity_sources',return_value=identity),mock.patch.object(a,'release_source',return_value=source),mock.patch.object(a,'pdf_payload',return_value=(t.payload(),dict(page_count=3))):
  return {k:a.strict_json(v) for k,v in b.produce(d,docs).items()}

class AdmissionTests(unittest.TestCase):
 def setUp(self):self.d,self.docs,self.r,self.p,self.f,self.filings,self.anchor,self.source,self.identity=synthetic()
 def check_pair(self):return a.validate_pair(self.r,self.p,self.f,['2016-10-26','2016-10-27','2016-10-31'],self.anchor,self.filings)
 def test_exact_raw_context(self):self.assertEqual(self.check_pair(),12)
 def test_full_output_sidecar(self):
  out=produce_fixture();x=out['ADMISSIONS.json'];self.assertEqual(x['semantic_admitted_count'],1);self.assertFalse(x['records'][0]['factor_ready']);self.assertTrue(x['raw_facts_provenance_verified']);self.assertEqual(out['COVERAGE.json']['pair_occurrences'],1)
 def test_raw_duplicate_omission_rejected(self):
  self.f['facts']['us-gaap']['SalesRevenueNet']['units']['USD'].append(copy.deepcopy(self.f['facts']['us-gaap']['SalesRevenueNet']['units']['USD'][0]))
  with self.assertRaises(ValueError):self.check_pair()
 def test_changed_raw_amount_rejected(self):
  self.p['current']['val']+=1
  with self.assertRaises(ValueError):self.check_pair()
 def test_same_day_session_rejected(self):
  self.r['catalog']['selected']['first_usable_date']='2016-10-26'
  with self.assertRaises(ValueError):self.check_pair()
 def test_future_release_rejected(self):
  self.anchor['release_date']='2016-10-28'
  with self.assertRaises(ValueError):self.check_pair()
 def test_wrong_source_row_rejected(self):
  self.r['catalog']['selected']['sources'][0]['row']=1
  with self.assertRaises(ValueError):self.check_pair()
 def test_incomplete_coverage_rejected(self):
  self.r['catalog']['coverage_complete']=False
  with self.assertRaises(ValueError):self.check_pair()
 def test_no_new_identity_for_early_row(self):
  x=a.record_identity(self.identity,'2016-10-25');self.assertEqual(len(x['historical_anchors']),1);self.assertIsNone(x['split']);self.assertFalse(x['continuous_security_identity_verified'])
 def test_future_anchor_unavailable(self):
  with self.assertRaises(ValueError):a.record_identity(self.identity,'2014-06-03')
 def test_later_optional_corroboration(self):
  x=a.record_identity(self.identity,'2020-07-30');self.assertEqual(len(x['historical_anchors']),2);self.assertIsNotNone(x['split'])
 def calendar(self):
  pub=self.source['publication'];parsed=a.parser.financial(t.payload());return a.calendar_index({'x':dict(self.source,parsed=parsed)})
 def test_derived_calendar_exact(self):
  x=a.compare_pair(self.p,12,a.parser.financial(t.payload()),self.calendar(),'2016-10-25');self.assertEqual(x['status'],'exact_match');self.assertEqual(x['selected_period']['current']['duration_weeks'],52)
 def test_future_calendar_not_used(self):
  cal=self.calendar()
  for rows in cal['year_ends'].values():
   for row in rows:row['release_date']='2016-10-26'
  self.assertEqual(a.compare_pair(self.p,12,a.parser.financial(t.payload()),cal,'2016-10-25')['status'],'unverified')
 def test_wrong_expected_start_does_not_choose_boundary(self):
  self.p['current']['start']='2014-09-28';x=a.compare_pair(self.p,12,a.parser.financial(t.payload()),self.calendar(),'2016-10-25');self.assertEqual(x['reason'],'fiscal_start_or_duration_mismatch')
 def test_conflicting_fy_end_aborts(self):
  cal=self.calendar()
  with self.assertRaises(ValueError):a.add_boundary(cal,dict(fiscal_year=2015,end='2015-09-19',basis='explicit',excerpt='x',directly_printed=True),'2015-10-01','y','h')
 def test_amount_mismatch_unverified(self):
  self.p['prior']['val']+=1;self.assertEqual(a.compare_pair(self.p,12,a.parser.financial(t.payload()),self.calendar(),'2016-10-25')['reason'],'reported_amount_mismatch')
 def test_wrong_nominal_period_unverified(self):self.assertEqual(a.compare_pair(self.p,9,a.parser.financial(t.payload()),self.calendar(),'2016-10-25')['status'],'unverified')
 def test_53_week_annual_supported(self):
  x=a.parser.financial(t.payload());x['statement']['current_end']='2016-10-01';self.p['current'].update(end='2016-10-01',duration_days=371);cal=self.calendar();del cal['year_ends']['2016'];self.assertEqual(a.compare_pair(self.p,12,x,cal,'2016-10-25')['status'],'exact_match')
 def test_365_day_not_accepted(self):
  self.p['current']['duration_days']=365;self.assertEqual(a.compare_pair(self.p,12,a.parser.financial(t.payload()),self.calendar(),'2016-10-25')['status'],'unverified')
 def test_false_identity_cannot_admit(self):
  out=produce_fixture();e=list(out['EVIDENCE.json']['evidence'].values())[0];e['release_evidence']['record_identity']['record_level_security_identity_verified']=False
  with self.assertRaises(ValueError):a.admit_record(0,self.r,[e],dict(semantic_source_sufficient=True,raw_facts_provenance_verified=True))
 def test_false_facts_cannot_admit(self):
  out=produce_fixture();e=list(out['EVIDENCE.json']['evidence'].values())[0];self.assertIsNone(a.admit_record(0,self.r,[e],dict(semantic_source_sufficient=True,raw_facts_provenance_verified=False))[0])
 def test_order_binding_rejected(self):
  out=produce_fixture();e=list(out['EVIDENCE.json']['evidence'].values())[0];e['evidence_id']='wrong'
  with self.assertRaises(ValueError):a.admit_record(0,self.r,[e],dict(semantic_source_sufficient=True,raw_facts_provenance_verified=True))
 def test_flags_never_upgraded(self):
  self.r['semantic_verified']=True
  with self.assertRaises(ValueError):a.flags(self.r)
 def test_duplicate_json_rejected(self):
  with self.assertRaises(ValueError):a.strict_json(b'{"x":1,"x":2}')
 def test_original_receipt_exact(self):
  raw=b'<html>';meta=dict(source_url=t.URL,final_url=t.URL,http_status=200,sha256=a.sha(raw),size=len(raw),public_unauthenticated=True,content_type='text/html',retrieved_at='2026-10-16T00:00:00+00:00');a.original_receipt(raw,meta,t.URL)
  meta['http_status']=403
  with self.assertRaises(ValueError):a.original_receipt(raw,meta,t.URL)

class SourceTests(unittest.TestCase):
 def facts_fixture(self):
  raw=b'{"cik":320193}';url='https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json';when='2026-10-15T00:00:00+00:00'
  request=dict(request_index=0,url=url,http_status=200,sha256=a.sha(raw),bytes=len(raw),retrieved_at=when)
  plan=dict(collector_sha256=a.sha(b'collector'),requests=[dict(kind='companyfacts',symbol='AAPL',cik=a.CIK,filename='facts_CIK'+a.CIK+'.json',url=url)])
  pb=b.output_bytes(plan);meta=dict(http_status=200,source=url,sha256=a.sha(raw),bytes=len(raw),kind='companyfacts',retrieved_at=when,plan_sha256=a.sha(pb),collector_sha256=a.sha(b'collector'))
  docs={'original':raw,'fresh':raw,'meta':b.output_bytes(meta),'plan':pb,'collector':b'collector','request':b.output_bytes(request)}
  design=dict(facts_path='original',official_facts_path='fresh',official_facts_receipt_path='meta',acquisition_plan_path='plan',acquisition_collector_path='collector',facts_request_path='request')
  return design,docs,{k:a.sha(v) for k,v in docs.items()}
 def test_http_facts_exact_duplicate(self):
  d,docs,h=self.facts_fixture();self.assertTrue(a.facts_source(d,docs,h)['raw_facts_provenance_verified'])
 def test_http_facts_changed_original(self):
  d,docs,h=self.facts_fixture();docs['original']=b'{"cik":320193,"x":1}';h['original']=a.sha(docs['original'])
  with self.assertRaises(ValueError):a.facts_source(d,docs,h)
 def test_http_facts_request_tamper(self):
  d,docs,h=self.facts_fixture();r=a.strict_json(docs['request']);r['http_status']=403;docs['request']=b.output_bytes(r);h['request']=a.sha(docs['request'])
  with self.assertRaises(ValueError):a.facts_source(d,docs,h)
 def test_http_facts_collector_tamper(self):
  d,docs,h=self.facts_fixture();docs['collector']=b'other';h['collector']=a.sha(docs['collector'])
  with self.assertRaises(ValueError):a.facts_source(d,docs,h)
 def test_legacy_catalog_never_http200(self):
  raw=b'{}';meta=dict(source='https://data.sec.gov/submissions/a',sha256=a.sha(raw),retrieved_at='2026-10-01T00:00:00+00:00');a.legacy_receipt(meta,raw,meta['source']);meta['http_status']=200
  with self.assertRaises(ValueError):a.legacy_receipt(meta,raw,meta['source'])
 def test_identity_transfer_exact_and_tamper(self):
  raw=b'%PDF-test';url='https://www.apple.com/newsroom/pdfs/x.pdf';meta=dict(url=url,final_url=url,status=200,public_unauthenticated=True,request_count=1,content_type='application/pdf',response_sha256=a.sha(raw),response_bytes=len(raw),decoded_sha256=a.sha(raw),decoded_bytes=len(raw),retrieved_at='2026-10-16T00:00:00+00:00',content_encoding=None,decoder='identity')
  a.transfer_receipt(raw,raw,meta,url,'www.apple.com','application/pdf')
  meta['final_url']='https://other.example/x.pdf'
  with self.assertRaises(ValueError):a.transfer_receipt(raw,raw,meta,url,'www.apple.com','application/pdf')
 def test_gzip_transfer_verified(self):
  import gzip
  decoded=b'%PDF-test';raw=gzip.compress(decoded);url='https://www.apple.com/newsroom/pdfs/x.pdf';meta=dict(url=url,final_url=url,status=200,public_unauthenticated=True,request_count=1,content_type='application/pdf',response_sha256=a.sha(raw),response_bytes=len(raw),decoded_sha256=a.sha(decoded),decoded_bytes=len(decoded),retrieved_at='2026-10-16T00:00:00+00:00',content_encoding='gzip',decoder='python:gzip.decompress')
  a.transfer_receipt(raw,decoded,meta,url,'www.apple.com','application/pdf')
  meta['decoder']='gzip'
  with self.assertRaises(ValueError):a.transfer_receipt(raw,decoded,meta,url,'www.apple.com','application/pdf')
 def test_nominal_duration_specific_week_sets(self):
  from datetime import date,timedelta
  fixture=AdmissionTests();fixture.setUp();cal=fixture.calendar();del cal['year_ends']['2016']
  for duration,weeks in [(3,13),(3,14),(6,26),(6,27),(9,39),(9,40),(12,52),(12,53)]:
   with self.subTest(duration=duration,weeks=weeks):
    pair=copy.deepcopy(fixture.p);parsed=a.parser.financial(t.payload());parsed['statement']['duration_months']=[3,duration]
    for role,start in [('current','2015-09-27'),('prior','2014-09-28')]:
     nweeks=weeks if role=='current' else {3:13,6:26,9:39,12:52}[duration];end=(date.fromisoformat(start)+timedelta(days=nweeks*7-1)).isoformat();pair[role].update(end=end,duration_days=nweeks*7);parsed['statement'][role+'_end']=end
    self.assertEqual(a.compare_pair(pair,duration,parsed,cal,'2017-01-01')['status'],'exact_match')
 def test_27_weeks_not_q1(self):
  fixture=AdmissionTests();fixture.setUp();parsed=a.parser.financial(t.payload());parsed['statement']['duration_months']=[3]
  self.assertEqual(a.compare_pair(fixture.p,3,parsed,fixture.calendar(),'2016-10-25')['status'],'unverified')
 def test_missing_statement_is_unverified(self):self.assertEqual(a.compare_pair({},12,None,{},'2016-10-25')['status'],'unverified')

if __name__=='__main__':unittest.main()
