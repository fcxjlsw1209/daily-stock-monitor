"""Independent local source/clock review; no producer import or event ingestion."""
from bisect import bisect_right
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import unittest
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
BASE='outputs/monthly_event_gap156_crm_sap_tsla/'
SELF='work/after155_crm_sap_independent_review.py'
OUTPUT='work/after155_crm_sap_independent_review.json'
PINS={BASE+'SOURCE_PROPOSAL.json':'071d0ae9fff44191859b9ba8d06659bc03382c6add89748f289b3ffbd0916978'}
CAL='outputs/monthly_paired_target143/CALENDAR.json'
CATALOGS=['outputs/monthly_earnings_repair/EVENTS.json','outputs/monthly_event_patch156/ADDED_EVENTS.json','outputs/monthly_event_scope156/EVENTS.json']

def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def load(p):return json.loads((ROOT/p).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
class Visible(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.hidden=0;self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.hidden+=1
    def handle_endtag(self,tag):
        if tag in ('script','style') and self.hidden:self.hidden-=1
    def handle_data(self,text):
        if not self.hidden:self.parts.append(text)
    def text(self):return ' '.join(' '.join(self.parts).split())
def visible(raw):
    p=Visible();p.feed(raw.decode('utf-8'));return p.text()
def evidence(raw,needle):
    target=needle.encode('utf-8');at=raw.find(target);require(at>=0,'raw excerpt missing '+needle)
    require(raw.find(target,at+1)<0,'raw excerpt not unique '+needle)
    return dict(raw_byte_start=at,raw_byte_end=at+len(target),raw_excerpt=needle,line_1based=raw[:at].count(b'\n')+1)
def clock(pub,zone,sessions,calendar):
    at=bisect_right(sessions,pub);require(at<len(sessions),'calendar exhausted')
    session=sessions[at]
    require(calendar.is_session(session),'candidate is not XNYS session')
    close=calendar.session_close(session)
    midnight=datetime.combine(date.fromisoformat(pub)+timedelta(days=1),time(),ZoneInfo(zone)).astimezone(timezone.utc)
    upper=(midnight-timedelta(microseconds=1)).strftime('%Y-%m-%dT%H:%M:%S')+'.999999999+00:00'
    require(close.to_pydatetime()>midnight,'close before complete local publication date')
    return dict(available_session=session,known_by_at=close.isoformat(),date_only_upper_bound_at=upper)
def conflicts(candidate,events):
    found=[]
    for e in events:
        if e.get('symbol')!=candidate['symbol']:continue
        same=e.get('event_id')==candidate['event_id'] or e.get('publication_date')==candidate['publication_date'] or e.get('source_url')==candidate['source_url'] or e.get('official_actual_period_end')==candidate['official_actual_period_end'] or (e.get('fiscal_year'),e.get('fiscal_quarter'))==(candidate['fiscal_year'],candidate['fiscal_quarter'])
        if same:found.append(e.get('event_id'))
    return found

class Checks(unittest.TestCase):
    def test_script_not_visible(self):self.assertEqual(visible(b'<script>datePublished fake</script><p>date</p>'),'date')
    def test_entities_and_tags(self):self.assertEqual(visible(b'<p>SAP <b>SE</b> &amp; results</p>'),'SAP SE & results')
    def test_byte_offsets(self):
        raw='x\n—test'.encode();e=evidence(raw,'—test');self.assertEqual(raw[e['raw_byte_start']:e['raw_byte_end']].decode(),'—test')
    def test_nonunique_reject(self):
        with self.assertRaises(ValueError):evidence(b'a a','a')
    def test_missing_reject(self):
        with self.assertRaises(ValueError):evidence(b'a','b')
    def test_same_period_conflict(self):
        c=dict(symbol='A',event_id='e',publication_date='d',source_url='u',official_actual_period_end='p',fiscal_year=1,fiscal_quarter=2)
        self.assertEqual(conflicts(c,[dict(c,event_id='another')]),['another'])

def audit():
    import exchange_calendars as xcals
    from exchange_calendars import exchange_calendar_xnys
    inputs=dict(PINS)
    for p,h in inputs.items():require(sha(p)==h,'pinned proposal changed')
    proposal=load(BASE+'SOURCE_PROPOSAL.json');spec=load(BASE+'SOURCE_SPEC.json');collection=load(BASE+'COLLECTION.json');notes=load(BASE+'REVIEW_NOTES.json')
    require(sha(BASE+'SOURCE_SPEC.json')==proposal['source_spec_sha256'],'spec binding')
    require(sha(BASE+'COLLECTION.json')==proposal['collection_sha256'],'collection binding')
    require(notes['source_proposal_sha256']==PINS[BASE+'SOURCE_PROPOSAL.json'],'notes binding')
    inputs.update(spec['input_hashes'])
    for p,h in inputs.items():require(sha(p)==h,'source input SHA '+p)
    for p in [SELF,BASE+'SOURCE_SPEC.json',BASE+'COLLECTION.json',BASE+'REVIEW_NOTES.json']:
        inputs[p]=sha(p)
    sessions=load(CAL)['sessions']; require(sessions==sorted(set(sessions)),'calendar order')
    cal=xcals.get_calendar('XNYS',start='2019-01-01',end='2020-12-31')
    specs={
      'CRM':dict(pub='2019-12-03',zone='America/Los_Angeles',end='2019-10-31',year=2020,q=3,kind='actual_quarterly_results',
        visible=['Salesforce Announces Record Third Quarter Fiscal 2020 Results','SAN FRANCISCO, Calif. — Dec. 3, 2019','Salesforce (NYSE: CRM)','today announced results for its fiscal third quarter ended October 31, 2019'],
        raw=['<b>SAN FRANCISCO, Calif. — Dec. 3, 2019 </b>','<p class="wp-block-paragraph"><b>SAN FRANCISCO, Calif. — Dec. 3, 2019 </b>— Salesforce (NYSE: CRM), the global leader in CRM, today announced results for its fiscal third quarter ended October 31, 2019.</p>']),
      'SAP':dict(pub='2020-01-28',zone='Europe/Berlin',end='2019-12-31',year=2019,q=4,kind='actual_preliminary_quarterly_and_annual_results',
        visible=['SAP veröffentlicht vorläufige Ergebnisse für das vierte Quartal und das Gesamtjahr 2019','28. Januar 2020','WALLDORF','Die SAP SE hat heute die vorläufigen Ergebnisse für das vierte Quartal und das Gesamtjahr 2019 (1. Januar – 31. Dezember) veröffentlicht.'],
        raw=['<span class="c-entry-date">28. Januar 2020</span>','<p class="lead"><em><strong>WALLDORF</strong> &#8212; </em>Die <a href="http://www.sap.com/">SAP SE</a> hat heute die vorläufigen Ergebnisse für das vierte Quartal und das Gesamtjahr 2019 (1. Januar – 31. Dezember) veröffentlicht.</p>'])}
    rows=[]
    require([r['symbol'] for r in proposal['results']]==['CRM','SAP','TSLA'],'scope must be three rows')
    for result in proposal['results']:
        symbol=result['symbol'];receipt=result['receipt'];name=symbol.lower()
        path=BASE+name+'.receipt.json';inputs[path]=sha(path)
        require(receipt==load(path),'embedded/standalone receipt mismatch')
        require(receipt==next(x for x in collection['receipts'] if x['name']==name),'collection receipt mismatch')
        if symbol=='TSLA':
            require(result['candidate'] is None and not result['proposal_ready_for_independent_review'],'TSLA must remain blocked')
            require(receipt['http_status']==403,'TSLA403')
            prefix='outputs/monthly_earnings_repair/batch_technology_industrial/TSLA/raw/de691e5b9b29307fe7de'
            meta=load(prefix+'.meta.json');textmeta=load(prefix+'.text.meta.json')
            require(meta['status']==200 and meta['sha256']==sha(prefix+'.bin'),'TSLA original cache')
            require(textmeta['source_sha256']==sha(prefix+'.bin') and textmeta['text_sha256']==sha(prefix+'.txt'),'TSLA text cache')
            text=(ROOT/(prefix+'.txt')).read_text();require('Q3 2019' in text and 'Sep 30, 2019' in text,'TSLA period support')
            rows.append(dict(symbol='TSLA',status='hold_no_verified_release_dateline',single_event_source_admitted=False,
                candidate_absent_verified=True,http_status=403,period_end_supported='2019-09-30',publication_date_verified=False,
                reason='Cached issuer update letter supports period only; webcast schedule and search observation cannot substitute for a verified historical release dateline. No network request or alternate source used by reviewer.'))
            continue
        d=specs[symbol];c=result['candidate'];require(c is not None,'missing candidate')
        require(receipt['http_status']==200 and receipt['status']=='downloaded','original HTTP200 required')
        raw=(ROOT/receipt['path']).read_bytes();inputs[receipt['path']]=sha(receipt['path'])
        require(inputs[receipt['path']]==receipt['sha256']==c['evidence_sha256'],'raw hash')
        require(len(raw)==receipt['bytes'],'raw length')
        text=visible(raw)
        for phrase in d['visible']:require(phrase in text,'visible original evidence missing '+phrase)
        raw_evidence=[evidence(raw,x) for x in d['raw']]
        require((c['publication_date'],c['publication_timezone'],c['official_actual_period_end'],c['fiscal_year'],c['fiscal_quarter'],c['release_kind'])==(d['pub'],d['zone'],d['end'],d['year'],d['q'],d['kind']),'candidate semantics mismatch')
        require(c['source_url']==receipt['requested_url']==receipt['final_url'],'source URL mismatch')
        require(c['event_id']==symbol+'_'+d['pub'],'event identity mismatch')
        expected=clock(d['pub'],d['zone'],sessions,cal)
        for key,value in expected.items():require(c[key]==value,'independent clock mismatch '+key)
        require(c['precision']=='date_only' and c['publication_at'] is None and c['first_public_time_verified'] is False and c['known_by_is_actual_publication'] is False,'unsupported clock certification')
        found={p:conflicts(c,load(p)) for p in CATALOGS};require(not any(found.values()),'catalog conflict')
        rows.append(dict(symbol=symbol,event_id=c['event_id'],status='pass_single_event_for_new_derived_catalog_only',
            single_event_source_admitted=True,admission_is_review_permission_not_catalog_mutation=True,
            verified_candidate=c,source_receipt_sha256=sha(path),raw_evidence=raw_evidence,
            visible_excerpt_checks=d['visible'],clock_independently_reconstructed=expected,catalog_conflicts=found,
            release_kind=d['kind'],period_end=d['end'],period_end_is_context_inference=symbol=='SAP',
            period_basis=('SAP explicitly pairs Q4 with full-year2019 and defines the annual period as 1January–31December; Dec31 is the shared fiscal-year/Q4 endpoint, not a verbatim standalone quarter-ended statement.' if symbol=='SAP' else 'Original lead explicitly states fiscal third quarter ended October31,2019; title identifies fiscal2020.'),
            limitations=['Current archived issuer bytes, not contemporaneous capture','No exact first-public clock certified','CMS datePublished/dateModified excluded from clock','No completeness or absence of other preliminary/final/corrected releases certified']+(['Keep preliminary nature; not final audited annual results; do not automatically deduplicate later final release on quarter key.'] if symbol=='SAP' else [])))
    require(not proposal['calendar_complete'] and not proposal['events_admitted'],'producer cannot claim ingestion')
    return dict(schema_version=1,passed=True,stage='independent_single_event_source_review',reviewed_utc=datetime.now(timezone.utc).isoformat(),
       source_proposal_sha256=PINS[BASE+'SOURCE_PROPOSAL.json'],verified_input_sha256=inputs,input_hashes=inputs,
       source_files={SELF:sha(SELF)},runtime={'python':sys.version,'exchange_calendars':xcals.__version__,'xnys_module_sha256':hashlib.sha256(Path(exchange_calendar_xnys.__file__).read_bytes()).hexdigest()},
       independence='Own HTML visible-text parsing/raw byte locators and calendar arithmetic; no producer imports, no producer source_checks used as evidence.',
       event_reviews=rows,counters=dict(single_events_passed=2,held=1,actual_ingestions=0,network_requests=0,price_values_read=0,financial_values_extracted=0,factor_calculations=0,models_fit=0,orders=0,returns=0),
       calendar_complete=False,whole_window_admitted=False,
       global_limitations=['Two source-only event approvals do not certify whole-window enumeration','Duplicate checks cover only the three frozen supplied catalogs; no absence claim about other archives','Receipt inspection supports recorded acquisition history, not proof that no unrecorded external requests occurred'])

if __name__=='__main__':
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    require(tests.wasSuccessful(),'synthetic checks failed')
    if '--publish' in sys.argv:
        report=audit();report['synthetic_tests_passed']=tests.testsRun
        with (ROOT/OUTPUT).open('x') as f:json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
        print(json.dumps(dict(passed=True,review_sha256=sha(OUTPUT),counters=report['counters'])))
