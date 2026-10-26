"""Inventory retained source bytes; used and unused responses remain distinct."""
import datetime as dt,json
import core as c

def main():
 rows=json.loads((c.HERE/'events.json').read_text());used=set()
 def visit(v):
  if isinstance(v,dict):
   if isinstance(v.get('evidence_path'),str):used.add(v['evidence_path'])
   for x in v.values():visit(x)
  elif isinstance(v,list):
   for x in v:visit(x)
 visit(rows);entries=[];issues=[]
 for p in sorted(c.HERE.glob('*/raw/*')):
  if p.name.endswith(('.meta.json','.textmeta.json')):continue
  mp=p.with_name(p.name+'.meta.json')
  if not mp.exists():
   # Extracted text is tied to its raw PDF by the separate .textmeta.json.
   if p.suffix=='.txt' and any(p.parent.glob(p.stem+'.*.textmeta.json')):continue
   entries.append(dict(path=c.rel(p),used_for_admission=c.rel(p)in used,sha256=c.sha(p.read_bytes()),metadata_missing=True));continue
  m=json.loads(mp.read_text());h=c.sha(p.read_bytes());valid=h==m['sha256'];entries.append(dict(path=c.rel(p),used_for_admission=c.rel(p)in used,source_url=m.get('url'),sha256=h,metadata_path=c.rel(mp),retrieved_at=m.get('retrieved_at'),http_status=m.get('status'),checksum_matches=valid))
  if not valid:issues.append(c.rel(p))
 for p in sorted(c.HERE.glob('*/indexed*.json')):
  if p.name.endswith('.meta.json'):continue
  mp=p.with_name(p.name+'.meta.json');m=json.loads(mp.read_text());h=c.sha(p.read_bytes());entries.append(dict(path=c.rel(p),used_for_admission=c.rel(p)in used,sha256=h,metadata_path=c.rel(mp),retrieved_at=m['retrieved_at'],evidence_class='public_search_index_response_not_original_file',checksum_matches=h==m['sha256']))
  if h!=m['sha256']:issues.append(c.rel(p))
 c.dump(c.HERE/'SOURCE_MANIFEST.json',dict(generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),files=entries,checksum_failures=issues,policy='All failed and unused source bytes retained. Public-index response hashes are not original-issuer file hashes. Extracted PDF text has independent source/text checksum metadata.'))
 c.dump(c.HERE/'NETWORK_FAILURES.json',dict(retrospective_observations=[dict(host='www08.wellsfargomedia.com',source_url=None,failure='TLS certificate expired; no certificate checks disabled',request_time='not individually recorded by initial helper; host-level failure reported in task tool output',raw_response_available=False),dict(source_url='https://news.mcdonalds.com/',failure='DNS name resolution failed',request_time='not individually recorded by initial helper; failure reported in task tool output',raw_response_available=False)],limitation='These failed connections supplied no admitted evidence. Future helper failures record a UTC observation timestamp directly.'))
 tests=(c.HERE/'TESTS.txt').read_text();passed=tests.rstrip().endswith('OK');status=json.loads((c.HERE/'STATUS.json').read_text());status['validation_passed']=passed and not issues;c.dump(c.HERE/'STATUS.json',status)
 c.dump(c.HERE/'VALIDATION.json',dict(generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),events=len(rows),events_sha256=c.sha((c.HERE/'events.json').read_bytes()),unittest_passed=passed,tests_path=c.rel(c.HERE/'TESTS.txt'),test_command="work/venv/bin/python -m unittest discover -s outputs/monthly_earnings_repair/batch_financial_consumer -p 'test_collect.py' -v",raw_cache_checksum_failures=issues,all_admitted_source_facts_quoted_and_hashed=passed,shared_schema_all_ready=passed,known_limitations='Partial coverage; see PENDING.json. Index-only rows explicitly preserve original_source_sha256=null. No claim of immutable contemporaneous capture or first-public time.'))
 print('MANIFEST',len(entries),'FAILURES',len(issues),'TESTS_PASS',passed)
if __name__=='__main__':main()
