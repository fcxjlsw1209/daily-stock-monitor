"""Four-event source-only acquisition proposal; no ingestion or financial arithmetic.

Offline freeze/inspect/verify run only through run_local.sh. The generated shell
collects the two exact CAT PDFs separately; the two HSBC originals reuse cache.
"""
import argparse
from bisect import bisect_right
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import unittest
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after155_event_gap_batch_cat_hsbc.py'
OUT = ROOT / 'outputs/monthly_event_gap156_cat_hsbc'
EVENTS = 'outputs/monthly_earnings_repair/EVENTS.json'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
PDF_PLAN = 'outputs/monthly_cashflow_source_preflight_v1/PLAN.json'
HSBC = 'outputs/monthly_earnings_repair/batch_financial_consumer/HSBC/'
TARGETS = [
 dict(key='CAT_2025Q1', symbol='CAT', issuer='Caterpillar Inc.', publication_date='2025-04-30',
      period_end='2025-03-31', year=2025, quarter=1, period_type='quarterly_results', zone='America/Chicago',
      title='Caterpillar Reports First-Quarter 2025 Results', dateline='IRVING, Texas, April 30, 2025',
      period_marker='Three Months Ended March 31, 2025', actual_marker='announced first-quarter 2025 results',
      url='https://www.caterpillar.com/content/dam/caterpillarDotCom/releases/1q25/1q25-caterpillar-inc-financial-results.pdf',
      discovery_url='https://www.caterpillar.com/en/news/corporate-press-releases/h/1q25-results-caterpillar-inc.html',
      acquisition='one_exact_url_download'),
 dict(key='CAT_2025Q2', symbol='CAT', issuer='Caterpillar Inc.', publication_date='2025-08-05',
      period_end='2025-06-30', year=2025, quarter=2, period_type='quarterly_results', zone='America/Chicago',
      title='Caterpillar Reports Second-Quarter 2025 Results', dateline='IRVING, Texas, Aug. 5, 2025',
      period_marker='Three Months Ended June 30, 2025', actual_marker='announced second-quarter 2025 results',
      url='https://www.caterpillar.com/content/dam/caterpillarDotCom/releases/2q25/2q25-caterpillar-inc-financial-results.pdf',
      discovery_url='https://www.caterpillar.com/en/news/corporate-press-releases/h/2q25-results-caterpillar-inc.html',
      acquisition='one_exact_url_download'),
 dict(key='HSBC_2016H1', symbol='HSBC', issuer='HSBC Holdings plc', publication_date='2016-08-03',
      period_end='2016-06-30', year=2016, quarter=2, period_type='interim_half_year_results', zone='Europe/London',
      title='2016 INTERIM RESULTS', dateline='3 August 2016', period_marker='Half-year to 30 Jun 2016',
      actual_marker='HSBC HOLDINGS PLC',
      url='https://www.hsbc.com/-/files/hsbc/investors/investing-in-hsbc/all-reporting/group/2016/2q-results/hsbc-holdings-plc/interim-results-2016-media-release.pdf',
      discovery_url='https://www.hsbc.com/investors/results-and-announcements/all-reporting/interim-results-2016-quick-read',
      acquisition='reuse_verified_cache', cache_path=HSBC+'raw/dc7a551d806bc6656921.pdf',
      cache_sha256='8ba72be90420844b859009193fe3180280317376d9b4c265dd30e5400e7e1f87'),
 dict(key='HSBC_2016FY', symbol='HSBC', issuer='HSBC Holdings plc', publication_date='2017-02-21',
      period_end='2016-12-31', year=2016, quarter=4, period_type='annual_results', zone='Europe/London',
      title='2016 RESULTS', dateline='21 February 2017', period_marker='Year ended 31 Dec 2016',
      actual_marker='HSBC HOLDINGS PLC',
      url='https://www.hsbc.com/-/files/hsbc/investors/investing-in-hsbc/all-reporting/group/2016/annual-results/hsbc-holdings-plc/170221-annual-results-2016-media-release.pdf',
      discovery_url='https://www.hsbc.com/investors/results-and-announcements/all-reporting/annual-results-2016-quick-read',
      acquisition='reuse_verified_cache', cache_path=HSBC+'raw/a12baac8837504183928.pdf',
      cache_sha256='6990fab8f45d3a0f3518449869ed1f75bf656f2afc32f30c29d38432880d94da'),
]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads(Path(path).read_text())
def rel(path): return str(Path(path).relative_to(ROOT))
def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2); f.write('\n')
def norm(text): return ' '.join(text.split())
def require(ok, why):
    if not ok: raise ValueError(why)


def clock(pub, zone, sessions):
    """Conservative next listed session; close is not a publication timestamp."""
    i = bisect_right(sessions, pub); require(i < len(sessions), 'calendar ends before availability')
    avail = sessions[i]
    boundary = datetime.combine(date.fromisoformat(pub)+timedelta(days=1), time(), ZoneInfo(zone)).astimezone(timezone.utc)
    close = datetime.combine(date.fromisoformat(avail), time(16), ZoneInfo('America/New_York')).astimezone(timezone.utc)
    require(close > boundary, 'next close must be after local date boundary')
    return dict(publication_at=None, publication_date=pub, publication_timezone=zone,
                precision='date_only', source_precision='date_only',
                date_only_upper_bound_at=(boundary-timedelta(microseconds=1)).strftime('%Y-%m-%dT%H:%M:%S')+'.999999999+00:00',
                available_session=avail, proposed_known_by_at=close.isoformat(),
                known_by_is_actual_publication=False, first_public_time_verified=False,
                exact_publication_clock_verified=False,
                close_time_basis='Proposed regular XNYS 16:00 America/New_York close; independent reviewer must confirm no early close for these four sessions.')


def collect_script():
    lines=['#!/bin/bash','set -euo pipefail', 'cd '+shlex.quote(str(ROOT)),
           'test ! -e '+shlex.quote(str(OUT/'ACQUISITION_STARTED.json')),
           'umask 077', 'set -o noclobber',
           'printf \'{"started_utc":"%s","method":"curl_one_request_per_frozen_CAT_URL"}\\n\' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > '+shlex.quote(str(OUT/'ACQUISITION_STARTED.json'))]
    for t in TARGETS:
        if t['acquisition'] != 'one_exact_url_download': continue
        p=str(OUT/t['key']); q=shlex.quote
        lines += ['date -u +%Y-%m-%dT%H:%M:%SZ > '+q(p+'.started_utc.txt'),
            'set +e',
            'curl --silent --show-error --proto =https --connect-timeout 15 --max-time 60 --max-filesize 16777216 --retry 0 --user-agent '+q('Mozilla/5.0 (compatible; historical-source-research/1.0)')+' --dump-header '+q(p+'.headers.txt')+' --output '+q(p+'.pdf')+' --write-out '+q('%{json}')+' '+q(t['url'])+' > '+q(p+'.curl.json')+' 2> '+q(p+'.stderr.txt'),
            'rc=$?', 'set -e', 'printf \'%s\\n\' "$rc" > '+q(p+'.exit.txt'),
            'date -u +%Y-%m-%dT%H:%M:%SZ > '+q(p+'.finished_utc.txt'),
            'status=$(sed -n \'s/^HTTP\/[0-9.]* \\([0-9][0-9][0-9]\\).*/\\1/p\' '+q(p+'.headers.txt')+' | tail -1 | tr -d \'\\r\')',
            'if [ "$status" = 429 ]; then printf \'HTTP429 stopped all remaining acquisition; no retry.\\n\'; exit 0; fi']
    return '\n'.join(lines)+'\n'


def freeze():
    require(not (OUT/'SOURCE_SPEC.json').exists(), 'spec already exists')
    inputs={}
    for p in [EVENTS,CALENDAR,PDF_PLAN,'outputs/monthly_event_overlay_scope156/REMAINING_ITEMS.json',
              'outputs/monthly_event_overlay_scope156/MANIFEST.json',HSBC+'release_queue.json',HSBC+'raw/page_1.html']:
        inputs[p]=sha(ROOT/p)
    for t in TARGETS:
        if t['acquisition']=='reuse_verified_cache':
            p=t['cache_path']; require(sha(ROOT/p)==t['cache_sha256'], 'cached raw mismatch')
            receipt=load(ROOT/(p+'.meta.json'))
            require(receipt['url']==t['url'] and receipt['status']==200 and receipt['sha256']==t['cache_sha256'], 'cached receipt mismatch')
            inputs[p]=sha(ROOT/p); inputs[p+'.meta.json']=sha(ROOT/(p+'.meta.json'))
    old=load(ROOT/EVENTS)
    require(all(not any(e['symbol']==t['symbol'] and (e.get('publication_date')==t['publication_date'] or (e.get('fiscal_year'),e.get('fiscal_quarter'))==(t['year'],t['quarter'])) for e in old) for t in TARGETS), 'old event conflicts')
    runtime=load(ROOT/PDF_PLAN)
    for p,h in runtime['pdf_runtime_hashes'].items(): require(sha(p)==h,'PDF runtime mismatch')
    script=collect_script(); OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'acquire.sh').open('x') as f:f.write(script)
    value=dict(stage='frozen_source_acquisition_targets_only',created_utc=utc(), targets=TARGETS,
         input_hashes=inputs, code_sha256=sha(ROOT/SELF), acquisition_script_sha256=sha(OUT/'acquire.sh'),
         python=sys.version,pdf_runtime=runtime['pdf_runtime'],pdf_runtime_hashes=runtime['pdf_runtime_hashes'],
         policy=dict(one_request_each=True,retries=0,follow_redirects=False,stop_all_on429=True,bypass403=False,
                     financial_values_extracted=False,network_outside_offline_wrapper=True),
         cached_search_scope='Exact issuer CAT and HSBC directories plus existing gap references; HSBC two originals found with HTTP200 receipts; no CAT exact-source cache match.',
         calendar_completeness_certified=False,event_admitted=False,factor_calculations=0,models_fit=0,orders=0,returns=0)
    write(OUT/'SOURCE_SPEC.json',value);print('SOURCE_SPEC_SHA256',sha(OUT/'SOURCE_SPEC.json'))


def spec_checked(expected):
    require(sha(OUT/'SOURCE_SPEC.json')==expected,'spec SHA mismatch')
    s=load(OUT/'SOURCE_SPEC.json'); require(s['code_sha256']==sha(ROOT/SELF),'code changed')
    require(s['acquisition_script_sha256']==sha(OUT/'acquire.sh'),'acquisition script changed')
    for p,h in {**s['input_hashes'],**s['pdf_runtime_hashes']}.items(): require(sha(ROOT/p)==h,'input SHA '+p)
    return s


def metadata_evidence(pages,t):
    markers=[('issuer',t['issuer'].upper(),True),('title',t['title'].upper(),True),
             ('dateline',t['dateline'].upper(),True),('actual_results',t['actual_marker'].upper(),True),
             ('period',t['period_marker'].upper(),False)]
    result=[]
    for kind,marker,first_only in markers:
        found=[]
        for i,p in enumerate(pages[:1] if first_only else pages):
            n=norm(p).upper(); pos=n.find(marker)
            if pos>=0: found.append(dict(kind=kind,page_1based=i+1,normalized_casefold_offset=pos,quote=marker))
        require(found,'metadata marker missing: '+t['key']+'/'+kind)
        result.append(found[0])
    return result


def inspect(expected):
    s=spec_checked(expected); require(not (OUT/'PROPOSAL.json').exists(),'proposal already published')
    sys.path.append(s['pdf_runtime']['bundle_site_packages'])
    from pypdf import PdfReader
    import pypdfium2
    sessions=load(ROOT/CALENDAR)['sessions']; old=load(ROOT/EVENTS)
    proposal=[]; receipts=[]; diagnostics=[]
    for t in s['targets']:
        key=t['key']; receipt=dict(key=key,url=t['url'],acquisition=t['acquisition'])
        if t['acquisition']=='reuse_verified_cache':
            raw=ROOT/t['cache_path']; original=load(ROOT/(t['cache_path']+'.meta.json'))
            receipt.update(status='reused_original_HTTP200_bytes',http_status=original['status'],path=rel(raw),sha256=sha(raw),bytes=raw.stat().st_size,
                           original_receipt_path=t['cache_path']+'.meta.json',original_receipt_sha256=sha(ROOT/(t['cache_path']+'.meta.json')),retrieved_utc=original['retrieved_at'])
        else:
            curl=OUT/(key+'.curl.json')
            if not curl.exists():
                receipts.append(dict(receipt,status='not_attempted_after_acquisition_stop'));continue
            c=load(curl); raw=OUT/(key+'.pdf'); headers=OUT/(key+'.headers.txt')
            code=int((OUT/(key+'.exit.txt')).read_text())
            receipt.update(http_status=c.get('http_code'),curl_exit=code,final_url=c.get('url_effective'),content_type=c.get('content_type'),
                requested_at=(OUT/(key+'.started_utc.txt')).read_text().strip(),retrieved_utc=(OUT/(key+'.finished_utc.txt')).read_text().strip(),
                curl_metadata_sha256=sha(curl),headers_sha256=sha(headers),
                retry_after=[l.partition(':')[2].strip() for l in headers.read_text().splitlines() if l.lower().startswith('retry-after:')])
            if raw.exists(): receipt.update(path=rel(raw),sha256=sha(raw),bytes=raw.stat().st_size)
            if c.get('http_code')!=200 or code!=0:
                receipts.append(dict(receipt,status='unavailable_no_retry'));continue
            receipt['status']='downloaded_HTTP200'
        receipts.append(receipt)
        try:
            blob=raw.read_bytes(); require(blob.startswith(b'%PDF-'),'not PDF bytes')
            reader=PdfReader(io.BytesIO(blob)); require(1<=len(reader.pages)<=100,'unexpected PDF size')
            pages=[p.extract_text() or '' for p in reader.pages]
            evidence=metadata_evidence(pages,t)
            # Rendering is source QA only. No financial numeric cells are parsed.
            doc=pypdfium2.PdfDocument(blob); images=[]
            for pn in sorted({e['page_1based'] for e in evidence}):
                dest=OUT/(key+'_page'+str(pn)+'.png')
                require(not dest.exists(),'render already exists')
                doc[pn-1].render(scale=1.25).to_pil().save(dest)
                images.append(dict(path=rel(dest),sha256=sha(dest),page_1based=pn))
            proposal.append(dict(event_id=t['symbol']+'_'+t['publication_date'],symbol=t['symbol'],issuer=t['issuer'],asset_type='equity',
                fiscal_year=t['year'],fiscal_quarter=t['quarter'],quarter_is_period_endpoint_ordinal=True,
                reporting_period_type=t['period_type'],standalone_quarterly_release_claim=t['period_type']=='quarterly_results',
                official_actual_period_end=t['period_end'],source_url=t['url'],source_kind='issuer_original_earnings_release_pdf',
                evidence_path=rel(raw),evidence_sha256=sha(raw),source_receipt=receipt,evidence=evidence,rendered_pages=images,
                **clock(t['publication_date'],t['zone'],sessions),
                status='producer_proposal_pending_independent_source_review',event_admitted=False,calendar_complete=False))
        except Exception as e: diagnostics.append(dict(key=key,error_type=type(e).__name__,error=str(e)))
    write(OUT/'COLLECTION.json',dict(spec_sha256=expected,created_utc=utc(),receipts=receipts))
    write(OUT/'PROPOSAL.json',dict(stage='producer_source_proposal_not_admission',created_utc=utc(),spec_sha256=expected,
        events=proposal,diagnostics=diagnostics,
        cadence_conclusion='HSBC Q2/Q4 endpoint slots correspond to an interim half-year and annual actual results release, not asserted standalone-quarter releases. Existing Q1/Q3 originals plus these two reports show the four observed 2016 reporting endpoints; no claim of all preliminary/corrective announcements being complete.',
        unresolved=['Independent source/clock review required','Exact first dissemination not certified','No whole-window calendar completeness from four reports','HSBC quarter ordinal is an endpoint normalization, not report duration'],
        financial_values_extracted=0,factor_calculations=0,models_fit=0,orders=0,returns=0,event_admitted=False,calendar_complete=False))
    outputs={rel(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('MANIFEST.json','RUN.log','VERIFY.log')}
    write(OUT/'MANIFEST.json',dict(spec_sha256=expected,code_sha256=sha(ROOT/SELF),output_hashes=outputs,source_input_hashes=s['input_hashes']))
    print(json.dumps(dict(proposed_events=len(proposal),diagnostics=diagnostics,proposal_sha256=sha(OUT/'PROPOSAL.json'))))


def verify(expected):
    spec_checked(expected); m=load(OUT/'MANIFEST.json')
    require(m['spec_sha256']==expected,'manifest spec')
    for p,h in m['output_hashes'].items():require(sha(ROOT/p)==h,'output SHA '+p)
    p=load(OUT/'PROPOSAL.json')
    for event in p['events']:require(sha(ROOT/event['evidence_path'])==event['evidence_sha256'],'raw source SHA')
    print(json.dumps(dict(passed=True,events=len(p['events']),diagnostics=p['diagnostics'],manifest_sha256=sha(OUT/'MANIFEST.json'))))


class Synthetic(unittest.TestCase):
    def test_weekend_next_session(self):self.assertEqual(clock('2025-04-04','America/Chicago',['2025-04-04','2025-04-07'])['available_session'],'2025-04-07')
    def test_london_summer(self):self.assertEqual(clock('2016-08-03','Europe/London',['2016-08-03','2016-08-04'])['proposed_known_by_at'],'2016-08-04T20:00:00+00:00')
    def test_london_winter(self):self.assertEqual(clock('2017-02-21','Europe/London',['2017-02-21','2017-02-22'])['proposed_known_by_at'],'2017-02-22T21:00:00+00:00')
    def test_clock_not_publication(self):self.assertFalse(clock('2017-02-21','Europe/London',['2017-02-22'])['known_by_is_actual_publication'])
    def test_calendar_truncated(self):
        with self.assertRaises(ValueError):clock('2025-04-30','America/Chicago',['2025-04-30'])
    def test_half_year_preserved(self):self.assertEqual(TARGETS[2]['period_type'],'interim_half_year_results')
    def test_annual_preserved(self):self.assertEqual(TARGETS[3]['period_type'],'annual_results')
    def test_missing_marker_rejected(self):
        with self.assertRaises(ValueError):metadata_evidence(['unrelated future conference'],TARGETS[0])
    def test_metadata_all_required(self):
        t=TARGETS[0]; p=' '.join(t[x] for x in ['issuer','title','dateline','period_marker','actual_marker'])
        self.assertEqual(len(metadata_evidence([p],t)),5)
    def test_date_must_be_first_page(self):
        t=TARGETS[0]; p=' '.join(t[x] for x in ['issuer','title','period_marker','actual_marker'])
        with self.assertRaises(ValueError):metadata_evidence([p,t['dateline']],t)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['self-test','freeze','inspect','verify']);ap.add_argument('--spec-sha')
    a=ap.parse_args()
    if a.phase=='self-test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Synthetic));sys.exit(not result.wasSuccessful())
    elif a.phase=='freeze':freeze()
    elif a.phase=='inspect':inspect(a.spec_sha)
    else:verify(a.spec_sha)
