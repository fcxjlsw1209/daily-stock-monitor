"""Freeze Apple source metadata and original candidate scope; no financial parser calls."""
import argparse
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parent.parent
POLICY=P.name

def blob(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def digest(x):return sha(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def read(name):return json.loads((ROOT/name).read_bytes())
def once(path,data):
 if path.exists():
  if path.read_bytes()!=data:raise ValueError('existing_registration_changed: '+str(path))
 else:path.write_bytes(data)

def design():
 hashes={}
 def pin(name,expected=None):
  path=Path(name);name=str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
  b=(ROOT/name).read_bytes();h=sha(b)
  if expected is not None and h!=expected:raise ValueError('source_hash_mismatch: '+name)
  hashes[name]=h;return b
 prep=str((P/'PREPARATION.json').relative_to(ROOT));pin(prep)
 for n,h in read(prep)['source_pins'].items():pin(n,h)
 structural='work/monthly_sec_revenue_data_review_v3.json';pin(structural);src='outputs/monthly_sec_revenue_data_v3/'
 for n,h in read(structural)['artifact_hashes'].items():pin(src+n,h)
 records=read(src+'RECORDS.json')
 base='outputs/dynamic_top100/registry_sec/';acq='outputs/monthly_sec_evidence_acquire_20261015_v2/'
 named=dict(facts_path='outputs/dynamic_top100/sec_coverage/0000320193.json',official_facts_path=acq+'facts_CIK0000320193.json',official_facts_receipt_path=acq+'facts_CIK0000320193.provenance.json',facts_request_path=acq+'request_20.json',acquisition_plan_path=acq+'PLAN.json',acquisition_collector_path='work/sec_public_acquire_20261015_v2.py',filings_root_path=base+'0000320193_submissions.json',filings_root_receipt_path=base+'0000320193_submissions.provenance.json',filings_path=base+'filing_indexes/CIK0000320193-submissions-001.json',filings_receipt_path=base+'filing_indexes/CIK0000320193-submissions-001.provenance.json',catalogs_path=src+'CATALOGS.json')
 for n in named.values():pin(n)
 prefix='outputs/monthly_aapl_newsroom_financial_source_v1/'
 collection=dict(plan_path=prefix+'PLAN.json',result_path=prefix+'RESULT.json',completion_path=prefix+'SOURCE_COMPLETION.json',extractions_path=prefix+'EXTRACTIONS.json',client_path=prefix+'source_client.py',collector_path=prefix+'collect.py',extractor_path='work/extract_monthly_aapl_newsroom_sources.py')
 for n in collection.values():pin(n)
 for n,h in read(collection['completion_path'])['source_hashes'].items():pin(n,h)
 plan=read(collection['plan_path']);pin(plan['events_path'],plan['events_sha256'])
 pin('outputs/monthly_sec_apple_ir_crosscheck_v2/parser.py','2f9f4c235ca952159f904d2aebf463d4dce62c2131f8fb0024c42cd6b4b5d59e')
 for n in ['DESIGN.json','PLAN.json','COMPLETION.json','RESULT.json','test_parser.py','test_binding.py']:pin('outputs/monthly_sec_apple_ir_crosscheck_v2/'+n)
 prior=read('outputs/monthly_sec_apple_ir_crosscheck_v2/COMPLETION.json');pin(prior['independent_review']['path'],prior['independent_review']['sha256'])
 for n,h in read('outputs/monthly_sec_apple_ir_crosscheck_v2/DESIGN.json')['inputs'].items():pin(n,h)
 for proof in ['outputs/monthly_aapl_fiscal_boundary_review_v1/SOURCE_COMPLETION.json','outputs/monthly_aapl_nasdaq_identity_source_v1/SOURCE_COMPLETION.json','work/monthly_aapl_calendar_readiness_independent_note.json']:
  pin(proof);value=read(proof)
  for n,h in value['source_hashes'].items():pin(n,h)
  for n,h in value.get('artifact_hashes',{}).items():pin(str(Path(proof).parent/n),h)
 identitybase='outputs/monthly_aapl_nasdaq_identity_source_v1/'
 identity=dict(completion_path=identitybase+'SOURCE_COMPLETION.json',anchors=[])
 for stem,day,id in [('971db4de3b571ebce7f0','2014-06-04','dtn2014-16'),('205bcc65cc2afaae3661','2019-03-12','dtn2019-04')]:
  identity['anchors'].append(dict(path=identitybase+'raw/'+stem+'.decoded',response_path=identitybase+'raw/'+stem+'.response',receipt_path=identitybase+'raw/'+stem+'.meta.json',date=day,url='https://www.nasdaqtrader.com/TraderNews.aspx?id='+id))
 identity['split']=dict(path='outputs/monthly_earnings_repair/apple/raw/42d556a6c493cb8b.html',receipt_path='outputs/monthly_earnings_repair/apple/raw/42d556a6c493cb8b.meta.json',url='https://www.apple.com/newsroom/2020/07/apple-reports-third-quarter-results/')
 results=read(collection['result_path']);extractions=read(collection['extractions_path']);scope=[];calendar=[]
 for index,entry in enumerate(results['entries']):
  t=entry['target'];ids=t['record_indices'];x=[v for v in extractions['reports'] if v['source_path']==entry['path']];assert len(x)==1;x=x[0]
  pin(t['release_path'],t['release_sha256']);pin(t['release_receipt_path'],t['release_receipt_sha256'])
  anchor=dict(report_date=t['report_date'],release_date=t['release_date'],record_indices=ids,attachment_role=t['attachment_role'],release_path=entry['path'],release_url=t['url'],receipt_path=str(Path(entry['path']).with_suffix('.meta.json')),response_path=prefix+entry['receipt']['response_path'],entry_index=index,entry_sha256=digest(entry),extraction=x)
  if ids:
   selected=records[ids[0]]['catalog']['selected'];acc=selected['accession'];fp=records[ids[0]]['pairing']['pairs'][0]['current']['fp'];quarter={'Q1':1,'Q2':2,'Q3':3,'FY':4}[fp]
   assert all(records[i]['symbol']=='AAPL' and records[i]['catalog']['selected']['accession']==acc and len(records[i]['pairing']['pairs'])==1 for i in ids)
   anchor.update(accession=acc,filing_date=selected['filingDate'],form=selected['form'],first_usable_date=selected['first_usable_date'],fiscal_quarter=quarter,record_sha256s={str(i):digest(records[i]) for i in ids},evidence_pair_hashes={digest(dict(accession=acc,pair=p)):digest(p) for i in ids for p in records[i]['pairing']['pairs']})
   scope.append(anchor)
  else:anchor['role']='auxiliary_calendar_only';calendar.append(anchor)
 ids=[i for a in scope for i in a['record_indices']];pairs={k for a in scope for k in a['evidence_pair_hashes']};occ=sum(len(records[i]['pairing']['pairs']) for i in ids)
 assert len(scope)==40 and len(calendar)==7 and len(ids)==len(set(ids))==119 and len(pairs)==40 and occ==119
 assert sorted(ids)==[i for i,r in enumerate(records) if r['symbol']=='AAPL' and r['pairing']['status']=='paired']
 contract=str((P/'CONTRACT_DRAFT.json').relative_to(ROOT));pin(contract)
 inv=blob(dict(schema_version=1,policy_id=POLICY,source_hashes=dict(sorted(hashes.items())),source_count=len(hashes),purpose='Frozen metadata and original candidate scope; no financial parser execution'))
 once(P/'SOURCE_INVENTORY.json',inv)
 d=dict(schema_version=1,policy_id=POLICY,stage='frozen_scope_before_actual_build',source_class='issuer_original_financial_documents',evidence_class='issuer_original_and_http_verified_exact_facts',candidate_cik='0000320193',source_hashes=dict(sorted(hashes.items())),source_inventory_sha256=sha(inv),source_records=dict(path=src+'RECORDS.json',sha256=hashes[src+'RECORDS.json']),sessions_path=src+'SESSIONS.json',structural_review=structural,collection=collection,identity_sources=identity,scope=scope,calendar_only=calendar,excluded_indices=[i for i,r in enumerate(records) if r['symbol']=='AAPL' and i not in ids],expected_record_count=3904,expected_documents=47,expected_accessions=40,expected_rows=119,expected_pairs=40,expected_pair_occurrences=119,expected_calendar_only_documents=7,allowed_tag_semantics={k:'consolidated_net_sales' for k in ('Revenues','SalesRevenueNet','RevenueFromContractWithCustomerExcludingAssessedTax')},contract=read(contract),**named)
 b=blob(d);once(P/'DESIGN.json',b);build=P/'build.py';build.write_text(build.read_text().replace("DESIGN_SHA='PENDING_DESIGN'","DESIGN_SHA='"+sha(b)+"'"));print(json.dumps(dict(design_sha256=sha(b),source_count=len(hashes),scope_rows=len(ids),unique_pairs=len(pairs),pair_occurrences=occ,excluded=len(d['excluded_indices'])),sort_keys=True))

def plan(args):
 d=read(str((P/'DESIGN.json').relative_to(ROOT)))
 for n,h in d['source_hashes'].items():
  if sha((ROOT/n).read_bytes())!=h:raise ValueError('source_changed: '+n)
 impl={str((P/n).relative_to(ROOT)):sha((P/n).read_bytes()) for n in ('parser.py','admission.py','build.py','register.py','test_parser.py','test_admission.py')}
 review={n:sha((ROOT/n).read_bytes()) for n in args.review_file}
 value=dict(schema_version=1,policy_id=POLICY,design_sha256=sha((P/'DESIGN.json').read_bytes()),source_hashes=d['source_hashes'],implementation_hashes=impl,independent_review_hashes=review,synthetic_tests_passed=args.producer_tests,independent_synthetic_tests_passed=args.reviewer_tests,tests_sha256=sha((P/'TESTS.txt').read_bytes()),actual_executed=False,network_requests=0,strategy_trials=0)
 once(P/'PLAN.json',blob(value));print('PLAN',sha(blob(value)))

if __name__=='__main__':
 cli=argparse.ArgumentParser();cli.add_argument('action',choices=('design','plan'));cli.add_argument('--review-file',action='append',default=[]);cli.add_argument('--producer-tests',type=int);cli.add_argument('--reviewer-tests',type=int);args=cli.parse_args()
 if args.action=='design':design()
 else:
  if not args.review_file or not args.producer_tests or not args.reviewer_tests:raise ValueError('completed_tests_required')
  plan(args)
