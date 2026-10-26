from copy import deepcopy
from datetime import date,timedelta
from pathlib import Path
import importlib.util,unittest,tempfile
from unittest.mock import patch
from test_parser import payload

def load(n):
 s=importlib.util.spec_from_file_location('oracle_prn_test_'+n,Path(__file__).with_name(n+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=load('admission');b=load('build');ROOT=Path(__file__).resolve().parents[2]
def synthetic(q=3,year=2024,pdf=None,dual=False):
 symbol='ORCL'
 docs={};retrieved='2026-10-16T12:00:00+00:00';cik='0001341439';acc=cik+'-19-000001';cy=year-(q<3);published=f'{cy}-'+{1:'09',2:'12',3:'03',4:'06'}[q]+'-15';filed=published;usable=(date.fromisoformat(filed)+timedelta(days=1))
 while usable.weekday()>=5:usable+=timedelta(days=1)
 usable=usable.isoformat();signal=usable;form='10-K'if q==4 else'10-Q'
 def add(n,x):docs[n]=b.output_bytes(x)
 default=payload(q,year);parsed=a.parser.parse_pdf(default,filed);period=parsed['statement']['periods'][-1];end=period['current']['end'];dates=[];day=date(2009,1,1)
 while day<=date.fromisoformat(signal):
  if day.weekday()<5:dates.append(day.isoformat())
  day+=timedelta(days=1)
 add('sessions',dates);recent=dict(accessionNumber=['start',acc],filingDate=['2010-01-01',filed],reportDate=['2009-12-31',end],form=['8-K',form]);archive=dict(accessionNumber=['archive'],filingDate=['2005-01-01'],reportDate=['2004-12-31'],form=['10-K']);an='CIK'+cik+'-submissions-001.json';root=dict(cik=int(cik),filings=dict(recent=recent,files=[dict(name=an,filingCount=1,filingFrom='2005-01-01',filingTo='2006-01-01')]))
 add('root',root);add('archive',archive)
 for n,u in [('root','CIK'+cik+'.json'),('archive',an)]:add(n+'.meta',dict(source='https://data.sec.gov/submissions/'+u,sha256=a.sha(docs[n]),retrieved_at=retrieved))
 path='outputs/monthly_sec_revenue_data_v3/filings.py';docs[path]=(ROOT/path).read_bytes();ns={};exec(compile(docs[path],path,'exec'),ns);catalog=ns['build_catalog'](root,{an:archive},'2026-10-16');add('catalogs',{symbol:catalog});cat=ns['asof_report'](catalog,signal,dates)
 tag='RevenueFromContractWithCustomerExcludingAssessedTax';pair=dict(taxonomy='us-gaap',tag=tag,unit='USD');raw=[]
 for i,role in enumerate(('current','prior')):
  pp=period[role];item=dict(accn=acc,filed=filed,form=form,start=pp['start'],end=pp['end'],val=pp['revenue_usd']);raw.append(item);pair[role]=dict(**item,duration_days=pp['duration_days'],row_index=i,row_indices=[i])
 tags={tag:dict(units={'USD':raw})};pairs=[pair]
 if dual:
  other=deepcopy(pair);other['tag']='Revenues';pairs.append(other);tags['Revenues']=dict(units={'USD':deepcopy(raw)})
 add('facts',dict(cik=int(cik),facts={'us-gaap':tags}));docs['facts.copy']=docs['facts'];add('facts.meta',dict(source='https://data.sec.gov/api/xbrl/companyfacts/CIK'+cik+'.json',sha256=a.sha(docs['facts']),retrieved_at=retrieved));flags={k:False for k in a.FLAGS};record=dict(symbol=symbol,candidate_cik=cik,signal_date=signal,catalog=cat,pairing=dict(status='paired',structural_pair_available=True,pairs=pairs,**flags),**flags);add('records',[record])
 default=pdf if pdf is not None else default
 pdfpath='fixture/main.pdf';docs[pdfpath]=b'%PDF-'+b.output_bytes(default)
 url='https://s23.q4cdn.com/440135859/files/doc_financials/test.pdf';meta='fixture/main.meta'
 add(meta,dict(url=url,final_url=url,status=200,sha256=a.sha(docs[pdfpath]),bytes=len(docs[pdfpath]),retrieved_at=retrieved,source_representation='original_http_bytes',request_method='GET',request_payload=None,content_type='application/pdf'))
 release=dict(release_path=pdfpath,receipt_path=meta,release_url=url,format='pdf',symbol=symbol,cik=cik,accession=acc,filing_date=filed,first_usable_date=usable,form=form,report_date=end,release_date=published,fiscal_quarter=q,fiscal_year=year,statement_label='Total revenues',record_indices=[0],record_sha256s={'0':a.digest(record)},evidence_pair_hashes={a.digest(dict(accession=acc,pair=v)):a.digest(v)for v in pairs})
 ns_pdf=dict(engine='pdfplumber',version='0.11.9',total_pages=12,pages=[dict(page=i+1,text='',links=[])for i in range(12)],metadata={})
 ns_pdf['pages'][0]['text']='Holger Mueller December 17, 2025 Constellation Research clients'
 ns_pdf['pages'][8]['text']='© 2025 Constellation Research Inc. Oracle Corporation Q4 Fiscal 2025 Financial Results, page 7.'
 ns_pdf['pages'][8]['links']=['http://s23.q4cdn.com/440135859/files/doc_financials/2025/q4/Q425_Form8K_Exhibit99-1_Earnings_Release_Tables_Final.pdf ']*3
 ns_path='fixture/raw/ns.decoded';ns_resp='fixture/raw/ns.response';ns_meta='fixture/raw/ns.meta';docs[ns_path]=b'%PDF-'+b.output_bytes(ns_pdf);docs[ns_resp]=docs[ns_path];ns_url='https://www.oracle.com/il-en/a/ocom/docs/database/oracle-redefines-the-ai-and-data-relationship.pdf'
 add(ns_meta,dict(url=ns_url,final_url=ns_url,status=200,public_unauthenticated=True,decoded_sha256=a.sha(docs[ns_path]),response_sha256=a.sha(docs[ns_resp]),decoded_bytes=len(docs[ns_path]),response_bytes=len(docs[ns_resp]),decoder='identity',content_encoding=None,decoded_path='raw/ns.decoded',response_path='raw/ns.response',retrieved_at=retrieved))
 namespace_source=dict(path=ns_path,response_path=ns_resp,receipt_path=ns_meta,url=ns_url)
 hashes={n:a.sha(v)for n,v in docs.items()};issuer=dict(symbol=symbol,cik=cik,facts_path='facts',legacy_facts_path='facts.copy',legacy_facts_receipt_path='facts.meta',filings_root_path='root',filings_root_receipt_path='root.meta',filings_path='archive',filings_receipt_path='archive.meta',catalogs_path='catalogs',filings_module_path=path)
 design=dict(namespace_source=namespace_source,policy_id=a.POLICY,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,allowed_tag_semantics=a.SEMANTICS,expected_record_count=1,expected_rows=1,expected_accessions=1,expected_pairs=len(pairs),expected_pair_occurrences=len(pairs),source_hashes=hashes,source_records=dict(path='records',sha256=hashes['records']),sessions_path='sessions',issuers={symbol:issuer},scope=[release],source_completions=[],excluded_indices=[],previously_admitted_indices=[])
 return design,docs

def outputs(d,docs):return{n:a.strict_json(x)for n,x in b.produce(d,docs,reader=lambda raw:a.strict_json(raw[5:])).items()}
class AdmissionTests(unittest.TestCase):
 def setUp(self):self.d,self.docs=synthetic()
 def read(self,n):return a.strict_json(self.docs[n])
 def put(self,n,x):self.docs[n]=b.output_bytes(x);self.d['source_hashes'][n]=a.sha(self.docs[n])
 def test_complete_and_read_only(self):
  prior=deepcopy((self.d,self.docs));o=outputs(self.d,self.docs);self.assertEqual(prior,(self.d,self.docs));self.assertEqual(o['COVERAGE.json']['semantic_admitted_count'],1);s=o['ADMISSIONS.json']['records'][0]['approval'][0];self.assertFalse(s['source_status']['raw_facts_provenance_verified']);self.assertTrue(s['source_status']['exact_byte_duplicate_bound']);self.assertFalse(s['release_evidence']['historical_pdf_bytes_available_verified'])
 def test_every_q_and_dual(self):
  for q in (1,2,3,4):
   d,docs=synthetic(q,dual=True);o=outputs(d,docs);self.assertEqual(o['COVERAGE.json']['semantic_admitted_count'],1);self.assertEqual(o['COVERAGE.json']['pair_occurrences'],2)
 def test_original_overlap(self):
  self.d['previously_admitted_indices']=[0]
  with self.assertRaisesRegex(ValueError,'overlap'):outputs(self.d,self.docs)
 def test_http_rejected(self):
  n=self.d['scope'][0]['receipt_path'];x=self.read(n);x['status']=403;self.put(n,x)
  with self.assertRaisesRegex(ValueError,'http'):outputs(self.d,self.docs)
 def test_bad_tenant(self):
  self.d['scope'][0]['release_url']=self.d['scope'][0]['release_url'].replace('440135859','999999999')
  with self.assertRaisesRegex(ValueError,'namespace'):outputs(self.d,self.docs)
 def test_duplicate_facts(self):
  self.docs['facts.copy']+=b' ';self.d['source_hashes']['facts.copy']=a.sha(self.docs['facts.copy'])
  with self.assertRaisesRegex(ValueError,'duplicate'):outputs(self.d,self.docs)
 def test_no_legacy_http_promotion(self):
  x=self.read('facts.meta');x['status']=200;self.put('facts.meta',x)
  with self.assertRaisesRegex(ValueError,'legacy_http'):outputs(self.d,self.docs)
 def test_original_flags(self):
  x=self.read('records');x[0]['semantic_verified']=True;self.put('records',x);self.d['source_records']['sha256']=self.d['source_hashes']['records']
  with self.assertRaisesRegex(ValueError,'flags'):outputs(self.d,self.docs)
 def test_raw_all_duplicate_contexts(self):
  f=self.read('facts');rows=f['facts']['us-gaap']['RevenueFromContractWithCustomerExcludingAssessedTax']['units']['USD'];rows.append(deepcopy(rows[0]));r=self.read('records')[0];sub=dict(self.d['issuers']['ORCL'],scope=self.d['scope'],source_records=self.d['source_records'])
  with self.assertRaisesRegex(ValueError,'context'):a.validate_pair(r,r['pairing']['pairs'][0],f,self.read('sessions'),self.d['scope'][0],a.filing_source(sub,self.docs,self.d['source_hashes']))
 def test_latest_asof(self):
  r=self.read('records');r[0]['catalog']['selected']['accession']='wrong';self.put('records',r);self.d['source_records']['sha256']=self.d['source_hashes']['records'];self.d['scope'][0]['record_sha256s']['0']=a.digest(r[0])
  with self.assertRaisesRegex(ValueError,'asof'):outputs(self.d,self.docs)
 def test_strict_next_session(self):
  r=self.read('records')[0];r['catalog']['selected']['first_usable_date']=self.d['scope'][0]['filing_date'];sub=dict(self.d['issuers']['ORCL'],scope=self.d['scope'],source_records=self.d['source_records'])
  with self.assertRaisesRegex(ValueError,'first_usable'):a.validate_pair(r,r['pairing']['pairs'][0],self.read('facts'),self.read('sessions'),self.d['scope'][0],a.filing_source(sub,self.docs,self.d['source_hashes']))
 def mutated_pdf(self,fn):
  x=payload();fn(x);d,docs=synthetic(pdf=x);return outputs(d,docs)
 def test_bad_amount_gap(self):self.assertEqual(self.mutated_pdf(lambda x:x['pages'][-1].update(text=x['pages'][-1]['text'].replace('1 0,000','1 0,001')))['COVERAGE.json']['unverified_count'],1)
 def test_confidential_gap(self):
  o=self.mutated_pdf(lambda x:x['pages'][0].update(text='Confidential – Oracle Internal '+x['pages'][0]['text']));self.assertEqual(o['COVERAGE.json']['semantic_admitted_count'],0);self.assertIn('confidential_version',o['COVERAGE.json']['rows'][0]['reasons'][0])
 def test_late_version_gap(self):self.assertEqual(self.mutated_pdf(lambda x:x['metadata'].update(ModDate="D:20240316100000-07'00'"))['COVERAGE.json']['unverified_count'],1)
 def test_all_pairs_required(self):
  d,docs=synthetic(dual=True);o=outputs(d,docs);r=a.strict_json(docs['records'])[0];ev=o['EVIDENCE.json']['evidence'];es=[deepcopy(ev[a.digest(dict(accession=d['scope'][0]['accession'],pair=x))])for x in r['pairing']['pairs']];es[1]['status']='unverified';es[1]['reason']='gap';self.assertIsNone(a.admit_record(0,r,es,{'semantic_source_sufficient':True})[0])
 def test_ordered_pairs(self):
  d,docs=synthetic(dual=True);o=outputs(d,docs);r=a.strict_json(docs['records'])[0];ev=o['EVIDENCE.json']['evidence'];es=[ev[a.digest(dict(accession=d['scope'][0]['accession'],pair=x))]for x in r['pairing']['pairs']]
  with self.assertRaisesRegex(ValueError,'order'):a.admit_record(0,r,list(reversed(es)),{'semantic_source_sufficient':True})
 def test_namespace_exact_link(self):
  n=self.d['namespace_source']['path'];x=a.strict_json(self.docs[n][5:]);x['pages'][8]['links'][0]=x['pages'][8]['links'][0].replace('440135859','123');raw=b'%PDF-'+b.output_bytes(x);self.docs[n]=raw;self.docs[self.d['namespace_source']['response_path']]=raw
  for path in(n,self.d['namespace_source']['response_path']):self.d['source_hashes'][path]=a.sha(raw)
  key=self.d['namespace_source']['receipt_path'];m=self.read(key);m.update(decoded_sha256=a.sha(raw),response_sha256=a.sha(raw),decoded_bytes=len(raw),response_bytes=len(raw));self.put(key,m)
  with self.assertRaisesRegex(ValueError,'namespace_links'):outputs(self.d,self.docs)
 def test_create_once(self):
  old=b.P
  try:
   with tempfile.TemporaryDirectory()as tmp,patch.object(b,'module',return_value=a):
    b.P=Path(tmp);x={n:b'one'for n in b.OUTPUTS};self.assertEqual(b.persist_once(x),'created_once');self.assertEqual(b.persist_once(x,True),'verified_existing')
    with self.assertRaisesRegex(ValueError,'mismatch'):b.persist_once({n:b'two'for n in b.OUTPUTS})
  finally:b.P=old
if __name__=='__main__':unittest.main()
