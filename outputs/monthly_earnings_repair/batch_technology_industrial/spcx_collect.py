"""SpaceX issuer identity and first post-IPO regular financial-results event."""
from collect import *
p=HERE/'SPCX/raw/q226_results.bin';m=json.loads(p.with_suffix('.meta.json').read_text());t=clean(extract(p))
period=re.search(r'SpaceX \(Nasdaq: SPCX\) announced the following financial results for the quarter ended ('+DATE+r')',t,re.I);assert period
qm=re.search(r'SpaceX Reports '+ORD+r' Quarter (20\d{2}) Results',t,re.I);assert qm
ip=HERE/'SPCX/public_index_observations/actual_updates_index.json';j=json.loads(ip.read_text());it=clean(j['rendering'])
dm=re.search(r'August 4, 2026 SpaceX Reports Second Quarter 2026 Results',it);assert dm
r=dict(release_date='2026-08-04',fiscal_period_end_actual=iso(period.group(1)),fiscal_year=int(qm.group(2)),fiscal_quarter=QMAP[qm.group(1).lower()],dateline_excerpt=dm.group(),period_excerpt=period.group(),identity_excerpt=qm.group())
e=event('SPCX',p,m,r,'issuer_original_release_pdf_full_bytes_with_public_index_official_publication_archive_date')
proof=dict(evidence_path=rel(ip),sha256=sha(ip.read_bytes()),source_url=j['url'],excerpt=dm.group())
e['release_date_evidence']=[proof];e['additional_evidence']=[proof]
e['release_date_basis']='Official issuer Updates publication archive explicitly lists the actual Reports Second Quarter 2026 Results release on August 4, 2026, separately from the July 20 scheduled-results notice. Undated original PDF supplies actual results and period; no webcast clock or filing timestamp used.'
e['evidence_limitation']='Primary SHA identifies complete original issuer earnings PDF bytes. Publication-date proof is a saved public-index rendering of the issuer actual-release archive; that SHA identifies the observation JSON, not original HTML bytes. Exact first-publication instant unverified.'
e['issuer_identity']='Space Exploration Technologies Corp., 2026 Nasdaq SPCX common stock; not the earlier SPCX fund/ticker holder.'
e['frozen_candidate_months']=['2026-06-30','2026-07-31','2026-08-31']
dump(HERE/'SPCX/events.json',[e])
proofs=[]
for name in ['ipo_pricing','q226_filing']:
 fp=HERE/'SPCX/raw'/f'{name}.bin';fm=json.loads(fp.with_suffix('.meta.json').read_text());proofs.append(dict(evidence_path=rel(fp),sha256=fm['sha256'],source_url=fm['url']))
archive=HERE/'SPCX/public_index_observations/quarterly_archive_index.json';proofs.append(dict(evidence_path=rel(archive),sha256=sha(archive.read_bytes()),source_url='https://ir.spacex.com/investors/'))
dump(HERE/'SPCX/IDENTITY_AND_WINDOW.json',dict(issuer=e['issuer_identity'],ipo_pricing_date='2026-06-11',expected_first_trading_date_in_pricing_original='2026-06-12',actual_ipo_completion_in_financial_report='2026-06-15',first_observed_regular_post_ipo_release='2026-08-04',first_observed_fiscal_period_end='2026-06-30',before_first_observed_regular_release=['2026-06-30','2026-07-31'],interpretation='Official quarterly archive currently contains only the August 4 Q2 release. June and July candidate windows precede it. Do not backfill this event into those months or manufacture earlier release events from IPO prospectus financials. Absence conclusion is limited to verified official archive, not a claim of no prior financial information.',evidence=proofs))
print('SPCX 1; June and July candidate windows precede first archived post-IPO quarterly release')
