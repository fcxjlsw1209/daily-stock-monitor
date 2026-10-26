from pathlib import Path
p=Path('outputs/monthly_sec_oracle_later_admission_v1');old=Path('outputs/monthly_sec_oracle_prn_admission_v1/register.py').read_text();tail=old[old.index('def plan(a):'):]
s='''"""Register Oracle later PDF scope and exact dependencies before financial admission."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent.parent;POLICY=P.name
SEMANTICS={'ORCL':{'Revenues':'consolidated_total_revenues','RevenueFromContractWithCustomerExcludingAssessedTax':'consolidated_total_revenues'}}
def blob(v):return(json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def read(n):return json.loads((ROOT/n).read_bytes())
def once(path,b):
 if path.exists():
  if path.read_bytes()!=b:raise ValueError('existing_changed '+str(path))
 else:
  with path.open('xb')as f:f.write(b)
def design():
 hashes={}
 def pin(n,h=None):
  if Path(n).is_absolute()or'..'in Path(n).parts:raise ValueError('unsafe source')
  b=(ROOT/n).read_bytes();actual=sha(b)
  if h is not None and actual!=h:raise ValueError('source_changed '+n)
  hashes[n]=actual;return b
 src='outputs/monthly_sec_revenue_data_v3/';struct='work/monthly_sec_revenue_data_review_v3.json';pin(struct)
 for n,h in read(struct)['artifact_hashes'].items():pin(src+n,h)
 pin(src+'filings.py');pin(src+'test_filings.py');records=read(src+'RECORDS.json')
 old='outputs/monthly_sec_oracle_prn_admission_v1/';pin(old+'DESIGN.json');issuer=read(old+'DESIGN.json')['issuers']['ORCL']
 for k,n in issuer.items():
  if k not in('symbol','cik'):pin(n)
 priorpaths=['outputs/monthly_sec_oracle_visa_original_admission_v1/ADMISSIONS.json',old+'ADMISSIONS.json']
 for n in priorpaths:pin(n);pin(str(Path(n).with_name('COMPLETION.json')))
 previous=sorted(x['record_index_0based']for n in priorpaths for x in read(n)['records']if x['symbol']=='ORCL');assert len(previous)==27
 inventory='work/monthly_oracle_later_source_inventory.json';inv=read(inventory);pin(inventory)
 scope=[]
 for x in inv['documents']:
  ids=x['indices'];selected=records[ids[0]]['catalog']['selected'];acc=selected['accession'];end=selected['reportDate'];q={8:1,11:2,2:3,5:4}[int(end[5:7])];fy=int(end[:4])+(q in(1,2));pin(x['path'],x['sha256']);pin(x['receipt']['path'])
  assert all(all(records[i]['catalog']['selected'][k]==selected[k]for k in('accession','form','filingDate','reportDate','first_usable_date'))for i in ids)
  scope.append(dict(release_path=x['path'],receipt_path=x['receipt']['path'],release_url=x['url'],format='pdf',symbol='ORCL',cik=issuer['cik'],accession=acc,form=selected['form'],filing_date=selected['filingDate'],report_date=end,first_usable_date=selected['first_usable_date'],release_date=x['release_date'],fiscal_quarter=q,fiscal_year=fy,statement_label='Total revenues',record_indices=ids,record_sha256s={str(i):digest(records[i])for i in ids},evidence_pair_hashes={digest(dict(accession=acc,pair=pair)):digest(pair)for i in ids for pair in records[i]['pairing']['pairs']}))
 scope.sort(key=lambda x:x['report_date']);ids=[i for x in scope for i in x['record_indices']]
 assert len(ids)==len(set(ids))==29 and len(scope)==10 and sum(len(records[i]['pairing']['pairs'])for i in ids)==37 and sum(len(x['evidence_pair_hashes'])for x in scope)==13
 assert set(ids).isdisjoint(previous)
 base='outputs/monthly_oracle_later_attribution_v1/';completion=base+'SOURCE_COMPLETION.json';pin(completion)
 for n,h in read(completion)['source_hashes'].items():pin(base+n,h)
 ns=dict(path=base+'raw/7462324aee17c3032356.decoded',response_path=base+'raw/7462324aee17c3032356.response',receipt_path=base+'raw/7462324aee17c3032356.meta.json',url='https://www.oracle.com/il-en/a/ocom/docs/database/oracle-redefines-the-ai-and-data-relationship.pdf')
 excluded=[i for i,r in enumerate(records)if r['symbol']=='ORCL'and i not in ids]
 contract=dict(all_original_pairs_required=True,all_ordered_dual_tags_required=True,previously_admitted_records_not_reissued=True,original_flags_unchanged=True,raw_facts_provenance_verified=False,filing_catalog_http_status_verified=False,exact_byte_duplicate_bound=True,official_archive_outlink_verified=False,historical_pdf_bytes_available_verified=False,source_attribution='Current Oracle-hosted Constellation Research report page9 endnote8 references a specific Oracle financial table on s23.q4cdn.com/440135859. This is third-party authorship and current namespace context, not all-object exact outlinks or historical ownership/publication.',period_start='Independently derive first day of each printed complete calendar-month duration from the exact current/prior month-end column; fiscal year ends May31.',first_usable='First official XNYS session strictly after original selected filing date; reconstruct original catalog and all raw contexts.',financial_evidence_available_date='Original selected filing as conservative reconstruction cutoff. PDF version timestamps must be <= filing; neither version timestamps nor current namespace evidence prove historical blob publication.',confidential_documents='Retain parsed source evidence but refuse admission absent a verified contemporaneous public copy. FY2026Q2 is a known source-status blocker, not a financial-value mismatch.',known_blocked_record_indices=[3639,3671,3703],exact_amounts_no_tolerance=True,main_statement='Only exact named GAAP consolidated statements of operations; require quarterly plus YTD/annual form where applicable; reject duplicates.',amount_columns='Current and prior monetary cells bounded by the printed 100% columns; exclude percent growth and constant-currency cells. Preserve raw row and header locators; allow whitespace inside a single currency cell only.',currency_basis='Printed dollar millions and in US dollars in the same table.',accounting_basis='As-reported current/prior Total revenues for the exact selected filing, no cross-report restatement harmonization.',continuous_security_identity_verified=False,original_inline_context_verified=False,diagnostics_not_financial_parser_input=True,network_requests=0,strategy_trials=0)
 invb=blob(dict(schema_version=1,policy_id=POLICY,source_hashes=dict(sorted(hashes.items())),source_count=len(hashes)));once(P/'SOURCE_INVENTORY.json',invb)
 d=dict(schema_version=1,policy_id=POLICY,stage='frozen_scope_before_actual_build',source_class='issuer_hosted_reference_cdn_original_release',evidence_class='issuer_original_and_legacy_exact_facts',source_hashes=dict(sorted(hashes.items())),source_inventory_sha256=sha(invb),source_records=dict(path=src+'RECORDS.json',sha256=hashes[src+'RECORDS.json']),sessions_path=src+'SESSIONS.json',structural_review=struct,issuers={'ORCL':issuer},namespace_source=ns,source_completions=[completion],scope=scope,excluded_indices=excluded,previously_admitted_indices=previous,previous_admissions_paths=priorpaths,expected_record_count=3904,expected_rows=29,expected_accessions=10,expected_pairs=13,expected_pair_occurrences=37,allowed_tag_semantics=SEMANTICS,contract=contract)
 db=blob(d);once(P/'DESIGN.json',db);code=P/'build.py';code.write_text(code.read_text().replace("DESIGN_SHA='PENDING_DESIGN'","DESIGN_SHA='"+sha(db)+"'"));print('DESIGN',sha(db),'pins',len(hashes))
'''
(p/'register.py').write_text(s+tail)
s=(p/'admission.py').read_text().replace("require(hashes.get(name)==h,'source_completion_dependency_mismatch')","resolved=name if name.startswith(('outputs/','work/'))else str(parent/name)\n            require(hashes.get(resolved)==h,'source_completion_dependency_mismatch')")
(p/'admission.py').write_text(s)
