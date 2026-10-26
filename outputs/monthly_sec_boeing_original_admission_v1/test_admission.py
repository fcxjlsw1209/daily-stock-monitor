from copy import deepcopy
from datetime import date,timedelta
from pathlib import Path
import importlib.util,unittest,tempfile
from unittest.mock import patch
from test_parser import html as boeing_html

def load(n):
 s=importlib.util.spec_from_file_location('ba_test_'+n,Path(__file__).with_name(n+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=load('admission');b=load('build');ROOT=Path(__file__).resolve().parents[2]
def synthetic(q=3,year=2019,html=None,external=False,span=1):
 symbol='BA'
 docs={};retrieved='2026-10-16T12:00:00+00:00';cik='0000012927';acc=cik+'-19-000001';published={1:f'{year}-04-24',2:f'{year}-07-24',3:f'{year}-10-23',4:f'{year+1}-01-31'}[q];filed=published;usable=(date.fromisoformat(filed)+timedelta(days=1))
 while usable.weekday()>=5:usable+=timedelta(days=1)
 usable=usable.isoformat();signal=usable;form='10-K'if q==4 else'10-Q'
 def add(n,x):docs[n]=b.output_bytes(x)
 default=boeing_html(q,year,external,span);parsed=a.parser.parse_boeing(default);period=parsed['statement']['periods'][0];end=period['current']['end'];dates=[];day=date(2009,1,1)
 while day<=date.fromisoformat(signal):
  if day.weekday()<5:dates.append(day.isoformat())
  day+=timedelta(days=1)
 add('sessions',dates);recent=dict(accessionNumber=['start',acc],filingDate=['2010-01-01',filed],reportDate=['2009-12-31',end],form=['8-K',form]);archive=dict(accessionNumber=['archive'],filingDate=['2005-01-01'],reportDate=['2004-12-31'],form=['10-K']);an='CIK'+cik+'-submissions-001.json';root=dict(cik=int(cik),filings=dict(recent=recent,files=[dict(name=an,filingCount=1,filingFrom='2005-01-01',filingTo='2006-01-01')]))
 add('root',root);add('archive',archive)
 for n,u in [('root','CIK'+cik+'.json'),('archive',an)]:add(n+'.meta',dict(source='https://data.sec.gov/submissions/'+u,sha256=a.sha(docs[n]),retrieved_at=retrieved))
 path='outputs/monthly_sec_revenue_data_v3/filings.py';docs[path]=(ROOT/path).read_bytes();ns={};exec(compile(docs[path],path,'exec'),ns);catalog=ns['build_catalog'](root,{an:archive},'2026-10-16');add('catalogs',{symbol:catalog});cat=ns['asof_report'](catalog,signal,dates)
 tag=next(iter(a.SEMANTICS[symbol]));pair=dict(taxonomy='us-gaap',tag=tag,unit='USD');raw=[]
 for i,role in enumerate(('current','prior')):
  pp=period[role];item=dict(accn=acc,filed=filed,form=form,start=pp['start'],end=pp['end'],val=pp['revenue_usd']);raw.append(item);pair[role]=dict(**item,duration_days=pp['duration_days'],row_index=i,row_indices=[i])
 add('facts',dict(cik=int(cik),facts={'us-gaap':{tag:dict(units={'USD':raw})}}));docs['facts.copy']=docs['facts'];add('facts.meta',dict(source='https://data.sec.gov/api/xbrl/companyfacts/CIK'+cik+'.json',sha256=a.sha(docs['facts']),retrieved_at=retrieved));flags={k:False for k in a.FLAGS};record=dict(symbol=symbol,candidate_cik=cik,signal_date=signal,catalog=cat,pairing=dict(status='paired',structural_pair_available=True,pairs=[pair],**flags),**flags);add('records',[record])
 def source(n,raw,url,fmt):
  path='fixture/raw/'+n+'.decoded';response='fixture/raw/'+n+'.response';meta='fixture/raw/'+n+'.meta.json';docs[path]=docs[response]=raw
  add(meta,dict(url=url,final_url=url,status=200,retrieved_at=retrieved,public_unauthenticated=True,request_count=1,content_type='application/pdf'if fmt=='pdf'else'text/html',content_encoding=None,decoder='identity',decoded_path='raw/'+n+'.decoded',response_path='raw/'+n+'.response',response_sha256=a.sha(raw),decoded_sha256=a.sha(raw),decoded_bytes=len(raw),response_bytes=len(raw)))
  return dict(release_path=path,response_path=response,receipt_path=meta,release_url=url,format=fmt)
 text=html if html is not None else default
 release=source('main',text.encode(),'https://boeing.mediaroom.com/release','html');release.update(symbol=symbol,cik=cik,accession=acc,filing_date=filed,first_usable_date=usable,form=form,report_date=end,release_date=published,fiscal_quarter=q,statement_label='Total revenues',record_indices=[0],record_sha256s={'0':a.digest(record)},evidence_pair_hashes={a.digest(dict(accession=acc,pair=pair)):a.digest(pair)})
 hashes={n:a.sha(v)for n,v in docs.items()};issuer=dict(symbol=symbol,cik=cik,facts_path='facts',legacy_facts_path='facts.copy',legacy_facts_receipt_path='facts.meta',filings_root_path='root',filings_root_receipt_path='root.meta',filings_path='archive',filings_receipt_path='archive.meta',catalogs_path='catalogs',filings_module_path=path)
 design=dict(policy_id=a.POLICY,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,allowed_tag_semantics=a.SEMANTICS,expected_record_count=1,expected_rows=1,expected_accessions=1,expected_pairs=1,expected_pair_occurrences=1,source_hashes=hashes,source_records=dict(path='records',sha256=hashes['records']),sessions_path='sessions',issuers={symbol:issuer},scope=[release],source_completions=[],excluded_indices=[])
 return design,docs

def outputs(d,docs):return{n:a.strict_json(x)for n,x in b.produce(d,docs).items()}
class AdmissionTests(unittest.TestCase):
 def setUp(self):self.d,self.docs=synthetic()
 def read(self,n):return a.strict_json(self.docs[n])
 def put(self,n,x):self.docs[n]=b.output_bytes(x);self.d['source_hashes'][n]=a.sha(self.docs[n])
 def test_complete(self):
  prior=deepcopy((self.d,self.docs));o=outputs(self.d,self.docs);self.assertEqual(prior,(self.d,self.docs));self.assertEqual(o['COVERAGE.json']['semantic_admitted_count'],1);self.assertFalse(o['ADMISSIONS.json']['records'][0]['approval'][0]['source_status']['raw_facts_provenance_verified']);self.assertTrue(o['ADMISSIONS.json']['records'][0]['approval'][0]['source_status']['exact_byte_duplicate_bound'])
 def test_all_quarters(self):
  for q in(1,2,3,4):
   d,docs=synthetic(q);self.assertEqual(outputs(d,docs)['COVERAGE.json']['semantic_admitted_count'],1)
 def test_external_leap(self):
  for q,y,ext in((1,2018,True),(2,2018,True),(4,2017,False)):
   d,docs=synthetic(q,y,external=ext);self.assertEqual(outputs(d,docs)['COVERAGE.json']['semantic_admitted_count'],1)
 def test_wrong_identity(self):
  d,docs=synthetic(html=boeing_html().replace('NYSE: BA','NYSE: BAD'));self.assertEqual(outputs(d,docs)['COVERAGE.json']['unverified_count'],1)
 def test_wrong_amount(self):
  d,docs=synthetic(html=boeing_html().replace('>100<','>101<').replace('>20<','>21<'));self.assertEqual(outputs(d,docs)['COVERAGE.json']['unverified_count'],1)
 def test_publication_mismatch(self):
  d,docs=synthetic(html=boeing_html().replace('Oct. 23','Oct. 24'));self.assertEqual(outputs(d,docs)['COVERAGE.json']['unverified_count'],1)
 def test_wrong_label(self):
  self.d['scope'][0]['statement_label']='Sales of products';self.assertEqual(outputs(self.d,self.docs)['COVERAGE.json']['unverified_count'],1)
 def test_http_rejected(self):
  n=self.d['scope'][0]['receipt_path'];m=self.read(n);m['status']=403;self.put(n,m)
  with self.assertRaisesRegex(ValueError,'http'):outputs(self.d,self.docs)
 def test_duplicate_facts(self):
  self.docs['facts.copy']+=b' ';self.d['source_hashes']['facts.copy']=a.sha(self.docs['facts.copy'])
  with self.assertRaisesRegex(ValueError,'duplicate'):outputs(self.d,self.docs)
 def test_no_legacy_http_promotion(self):
  m=self.read('facts.meta');m['status']=200;self.put('facts.meta',m)
  with self.assertRaisesRegex(ValueError,'legacy_http'):outputs(self.d,self.docs)
 def test_full_context_duplicate(self):
  f=self.read('facts');f['facts']['us-gaap']['Revenues']['units']['USD'].append(deepcopy(f['facts']['us-gaap']['Revenues']['units']['USD'][0]));r=self.read('records')[0];sub=dict(self.d['issuers']['BA'],scope=self.d['scope'],source_records=self.d['source_records'])
  with self.assertRaisesRegex(ValueError,'context'):a.validate_pair(r,r['pairing']['pairs'][0],f,self.read('sessions'),self.d['scope'][0],a.filing_source(sub,self.docs,self.d['source_hashes']))
 def test_strict_next_session(self):
  r=self.read('records')[0];r['catalog']['selected']['first_usable_date']='2017-07-20';sub=dict(self.d['issuers']['BA'],scope=self.d['scope'],source_records=self.d['source_records'])
  with self.assertRaisesRegex(ValueError,'first_usable'):a.validate_pair(r,r['pairing']['pairs'][0],self.read('facts'),self.read('sessions'),self.d['scope'][0],a.filing_source(sub,self.docs,self.d['source_hashes']))
 def test_selected_catalog_exact(self):
  r=self.read('records');r[0]['catalog']['selected']['accession']='wrong';self.put('records',r);self.d['source_records']['sha256']=self.d['source_hashes']['records'];self.d['scope'][0]['record_sha256s']['0']=a.digest(r[0])
  with self.assertRaisesRegex(ValueError,'asof'):outputs(self.d,self.docs)
 def test_flags(self):
  r=self.read('records');r[0]['semantic_verified']=True;self.put('records',r);self.d['source_records']['sha256']=self.d['source_hashes']['records']
  with self.assertRaisesRegex(ValueError,'flags'):outputs(self.d,self.docs)
 def test_create_once(self):
  old=b.P
  try:
   with tempfile.TemporaryDirectory()as tmp, patch.object(b, 'module', return_value=a):
    b.P=Path(tmp);x={n:b'one'for n in b.OUTPUTS};self.assertEqual(b.persist_once(x),'created_once');self.assertEqual(b.persist_once(x,True),'verified_existing')
    with self.assertRaisesRegex(ValueError,'mismatch'):b.persist_once({n:b'two'for n in b.OUTPUTS})
  finally:b.P=old
if __name__=='__main__':unittest.main()
