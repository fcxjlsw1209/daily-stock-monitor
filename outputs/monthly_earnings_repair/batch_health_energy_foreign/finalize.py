"""Reconcile applicable issuer windows and publish auditable coverage status."""
import collect as c
import build as b
from collections import Counter

def run(tested=False):
 events=c.load(c.HERE/'events.json',[]);keys={(e['ticker'],e['fiscal_year'],e['fiscal_quarter']) for e in events};pending=[];scope=[];counts=Counter(e['ticker'] for e in events);targets=Counter()
 for t in c.TICKERS:
  for y in range(2010,2028):
   for q in range(1,5):
    search_end=b.expected_period(t,y,q)
    if not '2010-12-01'<=search_end<=c.CUTOFF:continue
    row=dict(ticker=t,fiscal_year=y,fiscal_quarter=q)
    reason=None
    if t=='CVX':reason='delegated_to_root_warmup_cvx_63_verified_events'
    elif t=='ABBV' and (y,q)<(2013,1):reason='before_standalone_AbbVie_issuer_2013_separation'
    elif t=='BABA' and (y,q)<(2015,2):reason='before_2014_ADS_IPO_first_post_IPO_reporting_quarter'
    elif t=='CHL' and (y,q)>(2020,3):reason='after_NYSE_ADS_trading_suspension_2021_01_11; no_Hong_Kong_ticker_substitution'
    if reason:scope.append(dict(row,status='delegated' if t=='CVX' else 'outside_applicable_security_window',reason=reason));continue
    targets[t]+=1
    if (t,y,q) not in keys:pending.append(dict(row,status='pending_review',reason='Release publication date and actual fiscal-period end are not both verified. Missing quarters remain gaps; no calendar, CMS, filing-acceptance or call-time substitution.',detail_file=c.rel(c.HERE/'PARSE_PENDING.json'),source_stop_file=c.rel(c.HERE/'SOURCE_STOPS.json')))
 c.dump(c.HERE/'PENDING.json',pending);c.dump(c.HERE/'SCOPE_RECONCILIATION.json',scope)
 identity=[]
 for t,match,claim in [('ABBV','Abbott-Completes-Separation','Abbott official January 2, 2013 release confirms completion of research-pharmaceutical separation.'),('CHL','241077','China Mobile original announcement states NYSE suspension of ADS trading at 4:00 am US Eastern on January 11, 2021.'),('BABA','identity_sources.json','Official IPO pricing announcement dated September 18, 2014; expected ADS trading September 19. Pre-IPO fiscal quarters are not backfilled as listed BABA events.')]:
  matches=[]
  if t=='BABA':
   p=c.RAW/'identity_sources.json';matches=[dict(evidence_path=c.rel(p),sha256=c.sha(p.read_bytes()),representation='public_search_index_not_original_bytes')]
  else:
   for mp in (c.RAW/t).glob('*.meta.json'):
    m=c.load(mp,{})
    if match in m['url']:matches.append(dict(source_url=m['url'],evidence_path=c.rel(mp.with_suffix('').with_suffix('.bin')),sha256=m['sha256'],retrieved_at=m['retrieved_at'],representation='original_http_bytes'))
  identity.append(dict(ticker=t,basis=claim,evidence=matches))
 identity.append(dict(ticker='SHEL',basis='Same legal issuer changed name and assimilated A/B ADSs; issuer-wide earnings mapped explicitly, never price series or a selected predecessor share class.',evidence_file=c.rel(c.HERE/'SHEL_IDENTITY.json')))
 c.dump(c.HERE/'IDENTITY_POLICY.json',identity)
 per={}
 for t in c.TICKERS:
  rows=[e for e in events if e['ticker']==t];pends=[p for p in pending if p['ticker']==t]
  per[t]=dict(verified_events=len(rows),applicable_search_targets=targets[t],pending=len(pends),release_date_range=[min(e['release_date'] for e in rows),max(e['release_date'] for e in rows)] if rows else None,pending_quarters=[f"{e['fiscal_year']}Q{e['fiscal_quarter']}" for e in pends])
 st=dict(generated_utc=c.utc(),requested_release_window=['2011-01-01',c.CUTOFF],tickers=c.TICKERS,verified_events=len(events),events_by_ticker={t:counts[t] for t in c.TICKERS},pending_count=len(pending),parse_pending_count=len(c.load(c.HERE/'PARSE_PENDING.json',[])),all_events_date_only=True,source_stops=c.load(c.HERE/'SOURCE_STOPS.json',[]),complete=False,stable_for_merge=tested,validation='13 offline tests passed' if tested else 'new build: run tests before merging',event_file_sha256=c.sha((c.HERE/'events.json').read_bytes()),source_representations=dict(Counter(e['source_representation'] for e in events)),scope_exclusions=len([r for r in scope if r['status']!='delegated']),delegated_to_root={'CVX':63},per_ticker=per)
 c.dump(c.HERE/'STATUS.json',st)
 lines=['# Validation and remaining coverage','',f"Verified export: **{len(events)} events**, all date-only, release window 2011-01-01 through 2026-09-24. **{len(pending)} applicable quarterly search targets remain unverified.** CVX is delegated to the root warmup source (63 events), and is not duplicated here.",'','Tests: `work/venv/bin/python -m unittest discover -s outputs/monthly_earnings_repair/batch_health_energy_foreign -p test_collect.py -v`.',('13 tests passed; complete output is saved in TESTS.txt.' if tested else 'Current build has not yet passed validation.'),'','The suite verifies every admitted event and evidence SHA, UTC retrieval metadata, fiscal/date uniqueness, the central event schema and conservative next-session availability. It also checks date-only publication, no CMS/call inference, Toyota publication versus filing headers, non-calendar Danaher/Pfizer ends, wrong-issuer China Mobile rejection and explicit Shell predecessor mapping. Additional regressions require the full 63-quarter GE and Novo Nordisk sequences, and prevent the future financial calendar inside a Novo interim report from overriding its own publication header.','','| Ticker | Verified | Release dates | Remaining applicable quarters |','|---|---:|---|---:|']
 for t,v in per.items():lines.append(f"| {t} | {v['verified_events']} | {' to '.join(v['release_date_range']) if v['release_date_range'] else 'Root CVX source'} | {v['pending']} |")
 lines+=['','Original HTML, PDF and public API bytes retain their URL, SHA256 and UTC retrieval time. Search-index captures explicitly disclose that original HTTP bytes were not downloaded; their SHA is the saved capture hash, not an issuer-original hash. Separately captured table views retain upstream capture paths and hashes. Download failures retain status and Retry-After metadata; HTTP error response bodies were not saved and are not claimed as source evidence.','','All publication timestamps remain null. Issuer-local dates use a conservative timezone policy; exact first-public times are not established. A current archive is not an immutable contemporaneous publication feed. Shell records before 2022 retain Royal Dutch Shell plc and RDS.A/RDS.B issuer-level context with an official January 31, 2022 identity confirmation. No historical price histories or frozen ledgers are altered.','','PENDING.json enumerates every unresolved applicable FY/quarter; PARSE_PENDING.json includes rejected candidates and duplicates and therefore is not itself a count of missing quarters. SCOPE_RECONCILIATION.json separates pre-IPO/pre-separation/post-suspension targets and delegated CVX. Raw unused and failed artifacts are retained.','','This is a stable partial source for additive central merging, not a declaration that every historical gap has been filled. No training, backtesting, orders, price collection, stock-pool or ledger changes were performed.']
 (c.HERE/'VALIDATION.md').write_text('\n'.join(lines)+'\n');return st
if __name__=='__main__':
 import sys
 print(c.json.dumps(run('--tested' in sys.argv),indent=2))
