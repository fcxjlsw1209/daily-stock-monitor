from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,os
R=Path(__file__).resolve().parent.parent;D=R/'outputs/monthly_aapl_identity_recovery_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review=json.loads((D/'REVIEW.json').read_text());gaps=json.loads((R/'work/after142_profitability_admission_gaps.json').read_text())
for path,digest in review['source_hashes'].items():assert sha(R/path)==digest
for path,digest in review['pdf_runtime_hashes'].items():assert sha(Path(path))==digest
for path in ['outputs/monthly_profitability_extension_sources_v1/RAW_FACTS.json','outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json']:
 assert sha(R/path)==gaps['input_bindings_sha256'][path]
pattern=re.compile(r'NASDAQ_AAPL|apple[^"<>]{0,70}annual.report|aapl[^"<>]{0,45}10-k|aapl-20[0-9]{6}',re.I)
scanned={};matches=[]
for relative in ['outputs/monthly_earnings_repair','outputs/dynamic_top100','outputs/index_union']:
 for directory,dirs,files in os.walk(R/relative,followlinks=False):
  dirs[:]=[n for n in dirs if not (Path(directory)/n).is_symlink()]
  for name in files:
   if name.endswith(('meta.json','provenance.json','manifest.json')) or name in ('SOURCE_INVENTORY.json','SOURCES.json'):
    p=Path(directory)/name
    if p.is_symlink():continue
    assert p.stat().st_size<10_000_000
    raw=p.read_bytes();rel=str(p.relative_to(R));scanned[rel]=hashlib.sha256(raw).hexdigest()
    if pattern.search(raw.decode('utf-8',errors='replace')):matches.append(rel)
assert not matches
report=dict(status='COMPLETE_BOUNDED_LOCAL_IDENTITY_SEARCH_NOT_ADMISSION',created_utc=datetime.now(timezone.utc).isoformat(),review_path=str((D/'REVIEW.json').relative_to(R)),review_sha256=sha(D/'REVIEW.json'),missing_registered_paths=16,unique_registered_digests=14,recovered_paths=0,remaining_unresolved_paths=[x['path']for x in review['rows']if x['recovered_object']is None],local_annual_report_metadata_scan=dict(files_scanned=len(scanned),pattern=pattern.pattern,matches=matches,source_hashes=scanned),annual_sources_examined=dict(documents=10,PDF_pages=sum(x['PDF_pages_checked']for x in review['current_annual_original_review']),parent_releases=10,kind='Current-annual earnings releases, not full10-K originals',explicit_dated_AAPL_or_Nasdaq_literals_found=0),decision='No new independent security identity certificate issued. Missing originals remain unresolved.',next_scope='Exact16 originals or their14 byte-identical objects must be migrated under separate authorization, or new official dated identity originals collected under a separate source plan. Existing current SEC ticker arrays and companyfacts are not substituted.',new_identity_admissions=0,network_requests=0,CloudStorage_reads=0,factors_ranks_orders_returns_computed=0,old_frozen_files_modified=0,source_hashes={str(Path(__file__).relative_to(R)):sha(Path(__file__))})
with (D/'FINAL_REVIEW.json').open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(path=str((D/'FINAL_REVIEW.json').relative_to(R)),sha256=sha(D/'FINAL_REVIEW.json'),metadata_files=len(scanned),PDF_pages=report['annual_sources_examined']['PDF_pages'],remaining=16)))
