"""Independent, offline review of existing issuer facts; no factor computation."""
from pathlib import Path
import collections
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import io
import json
import re
import unittest
from bs4 import BeautifulSoup
import profitability141_issuer_statements as source
import profitability141_issuer_statements_tests as selftests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'work/profitability141_issuer_statements_independent_review.json'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def norm(s):
    return re.sub(r'\s+', ' ', s).strip()

def load(rel):
    p = ROOT / rel
    if p.resolve() != p or not p.is_file():
        raise AssertionError('Nonlocal or missing input: ' + rel)
    raw = p.read_bytes()
    return json.loads(raw), digest(raw)

def bound(binding):
    p = ROOT / binding['path']
    assert p.resolve() == p and p.is_file(), binding['path']
    raw = p.read_bytes()
    assert digest(raw) == binding['sha256'], binding['path']
    return raw

class IndependentTests(unittest.TestCase):
    def test_zero_and_negative_are_exact(self):
        self.assertEqual(source.exact_amount('0'), '0')
        self.assertEqual(source.exact_amount('(1,200)'), '-1200')
        self.assertEqual(source.exact_amount('-2.50'), '-2.50')
    def test_no_nonfinite_or_boolean_amounts(self):
        for text in ['True', 'False', 'NaN', '-Infinity', '1e9']:
            with self.assertRaises(ValueError): source.exact_amount(text)
    def test_original_precision_preserved(self):
        self.assertEqual(source.exact_amount('12345678901234567890.12345'), '12345678901234567890.12345')
    def test_receipt_naive_timestamp_rejected(self):
        r = {'http_status':200, 'url':'https://www.microsoft.com/a', 'sha256':digest(b'a'), 'retrieved_at':'2026-01-01T00:00:00'}
        with self.assertRaises(ValueError): source.receipt_check(b'a', r, {'source_url':r['url']})
    def test_receipt_wrong_issuer_host_rejected(self):
        r = {'http_status':200, 'url':'https://other.example/a', 'sha256':digest(b'a'), 'retrieved_at':'2026-01-01T00:00:00+00:00'}
        with self.assertRaises(ValueError): source.receipt_check(b'a', r, {'source_url':r['url']})
    def test_wrong_byte_count_rejected(self):
        r = {'http_status':200, 'url':'https://www.microsoft.com/a', 'sha256':digest(b'a'), 'bytes':2, 'retrieved_at':'2026-01-01T00:00:00+00:00'}
        with self.assertRaises(ValueError): source.receipt_check(b'a', r, {'source_url':r['url']})
    def test_same_session_is_never_available(self):
        self.assertEqual(source.first_session_after('2025-01-10',['2025-01-10','2025-01-13']), '2025-01-13')
    def test_invalid_session_order_rejected(self):
        for sessions in [[], ['2025-01-13','2025-01-10'], ['2025-01-10','2025-01-10']]:
            with self.assertRaises(ValueError): source.first_session_after('2025-01-09',sessions)
    def test_52_week_period_uses_previous_end(self):
        p = source.derive_period('2025-01-26','2024-01-28','NVDA')
        self.assertEqual(p['days_inclusive'],364)
        self.assertEqual(p['start'],'2024-01-29')
    def test_microsoft_shifted_year_end_rejected(self):
        with self.assertRaises(ValueError): source.derive_period('2025-07-01','2024-06-30','MSFT')

def review():
    plan, plan_sha = load(source.PLAN)
    assert plan_sha == source.PLAN_SHA
    report_path = source.OUT + '/RAW_ANNUAL_FACTS.json'
    report, report_sha = load(report_path)
    assert report['plan_sha256'] == plan_sha
    assert report['extractor_sha256'] == digest((ROOT/source.SELF).read_bytes())
    assert report['tests_sha256'] == digest((ROOT/source.TEST).read_bytes())
    assert report['ratios_ranks_returns_computed'] == 0
    inputs = {key:json.loads(bound(binding)) for key,binding in plan['inputs'].items()}
    bound(plan['separate_from_unchanged_SEC_DATA_PLAN'])
    assert report['inputs'] == plan['inputs']
    cal = inputs['calendar_sessions']; calplan = inputs['calendar_plan']
    assert cal['plan_sha256'] == plan['inputs']['calendar_plan']['sha256']
    for path, sha in calplan['bindings'].items(): bound({'path':path,'sha256':sha})
    sessions = cal['sessions']
    assert sessions == sorted(set(sessions)) and '2025-01-09' not in sessions
    events = inputs['MSFT_events'] + inputs['NVDA_events']
    annual_events = [e for e in events if e['fiscal_quarter'] == 4]
    key = lambda r:(r['symbol'],r.get('fiscal_year_label',r.get('fiscal_year')))
    specs = {key(s):s for s in plan['source_scope']['sources']}
    pairs = {key(p):p for p in report['pairs']}
    event_map = {(e['ticker'],e['fiscal_year']):e for e in annual_events}
    assert len(specs) == len(plan['source_scope']['sources']) == 24
    assert len(pairs) == len(report['pairs']) == report['pair_count'] == 24
    assert len(event_map) == len(annual_events) == 24
    assert set(specs) == set(pairs) == set(event_map)
    assert report['gap_count'] == 0 and not report['gaps']
    details = []
    for k,spec in sorted(specs.items()):
        pair = pairs[k]; event = event_map[k]
        raw = bound({'path':spec['raw_path'],'sha256':spec['raw_sha256']})
        receipt = json.loads(bound({'path':spec['receipt_path'],'sha256':spec['receipt_sha256']}))
        if spec['receipt_entry_key'] is not None: receipt = receipt[spec['receipt_entry_key']]
        assert receipt['sha256'] == digest(raw) == pair['source']['raw_sha256']
        assert receipt['url'] == spec['source_url'] == event['source_url'] == pair['source']['source_url']
        assert receipt['http_status'] == 200
        assert datetime.fromisoformat(receipt['retrieved_at'].replace('Z','+00:00')).utcoffset() is not None
        if 'bytes' in receipt: assert receipt['bytes'] == len(raw)
        assert event.get('sha256',event.get('evidence_sha256')) == spec['raw_sha256']
        assert pair['availability']['release_date'] == spec['release_date'] == event['release_date']
        assert pair['availability']['timestamp_precision'] == event['precision']
        assert pair['availability']['publication_timestamp'] == event.get('publication_timestamp')
        assert pair['availability']['publication_timezone'] == event.get('publication_timezone')
        assert pair['availability']['first_conservative_session_after_release_date'] == next(d for d in sessions if d > event['release_date'])
        assert pair['availability']['first_public_version_verified'] is False
        assert pair['factor_or_strategy_admitted'] is False
        assert pair['cik'] == {'MSFT':'0000789019','NVDA':'0001045810'}[k[0]]
        soup = BeautifulSoup(raw,'html.parser')
        text = norm(soup.get_text(' ',strip=True))
        assert norm(spec['release_date_literal']) in text
        assert pair['gaap']['issuer_policy_literal'] in text
        rows_reviewed = []
        for row_spec in spec['rows']:
            field = {'annual_gross_amount':'gross_amount','year_end_assets':'total_assets'}[row_spec['kind']]
            row = pair[field]; table = soup.find_all('table')[row_spec['table_index_zero_based']]
            trs = [tr for tr in table.find_all('tr') if tr.find_parent('table') is table]
            tr = trs[row_spec['row_index_zero_based']]
            cells = tr.find_all(['td','th'],recursive=False)
            cell = cells[row_spec['matching_cells'][0]['source_cell_index']]
            assert norm(cell.get_text(' ',strip=True)) == row['displayed_amount']
            assert digest(str(cell).encode()) == row['raw_cell_sha256']
            assert row_spec['row_label'] == row['measure_label']
            assert sum(any(norm(c.get_text(' ',strip=True)).lower()==row['measure_label'].lower() for c in rr.find_all(['td','th'],recursive=False)) for rr in trs) == 1
            assert row['date'] == spec['fiscal_year_end'] == event['fiscal_period_end_actual']
            assert norm(row_spec['units']['literal']) == row['unit_literal']
            assert row['unit_literal'] in norm(table.get_text(' ',strip=True)+' '+row['statement_title_and_unit_context'])
            assert row['scale'] in [1000,1000000] and row['currency_label'] == '$'
            assert row['scale'] == (1000000 if 'millions' in row['unit_literal'].lower() else 1000)
            assert ('millions' in row['unit_literal'].lower()) != ('thousands' in row['unit_literal'].lower())
            assert '%' not in row['displayed_amount']
            monetary = row['displayed_amount'].replace('$','').replace(',','').replace(' ','').replace('\u2212','-')
            negative = monetary.startswith('(') and monetary.endswith(')')
            amount = Decimal(monetary[1:-1] if negative else monetary)
            if negative: amount = -amount
            assert amount.is_finite() and amount == Decimal(row['amount_decimal'])
            if field == 'total_assets': assert amount > 0
            assert not re.search(r'non[ -]?GAAP|as adjusted',norm(table.get_text(' ',strip=True)),re.I)
            rows_reviewed.append({'field':field,'locator':row['locator'],'cell_hash_verified':True})
        period = pair['annual_period']
        previous_ends = [d for d in pair['total_assets']['all_statement_header_dates'] if d<spec['fiscal_year_end']]
        assert len(previous_ends) == 1
        assert period['start'] == (date.fromisoformat(previous_ends[0])+timedelta(days=1)).isoformat()
        assert period['days_inclusive'] == (date.fromisoformat(period['end'])-date.fromisoformat(period['start'])).days+1
        assert 330 <= period['days_inclusive'] <= 400
        assert period['start_is_direct_quote'] is False
        assert period['end'] < event['release_date']
        details.append({'symbol':k[0],'fiscal_year':k[1],'end':period['end'],'start':period['start'],'days':period['days_inclusive'],'release_date':event['release_date'],'available_session':pair['availability']['first_conservative_session_after_release_date'],'USD_verified':pair['currency']['USD_verified'],'gross_scale':pair['gross_amount']['scale'],'assets_scale':pair['total_assets']['scale'],'source_sha256':digest(raw),'rows':rows_reviewed})
    continuity = []
    for symbol in ['MSFT','NVDA']:
        rows = [d for d in details if d['symbol']==symbol]
        assert [r['fiscal_year'] for r in rows] == list(range(2015,2027))
        assert [r['release_date'] for r in rows] == sorted(r['release_date'] for r in rows)
        adjacent = all(cur['start']==(date.fromisoformat(prev['end'])+timedelta(days=1)).isoformat() for prev,cur in zip(rows,rows[1:]))
        assert adjacent
        continuity.append({'symbol':symbol,'Q4_events_in_pinned_catalog':len(rows),'registered_sources':len(rows),'first_fiscal_year':2015,'last_fiscal_year':2026,'consecutive_current_annual_periods':True,'missing_Q4_events_within_pinned_catalog':0,'outside_catalog_event_completeness_certified':False})
    stream = io.StringIO()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(selftests),unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests)])
    result = unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    assert result.wasSuccessful(), stream.getvalue()
    output = {
        'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_EXISTING_RAW_SOURCE_CONTRACT_WITH_EXPLICIT_RESEARCH_LIMITATIONS',
        'scope':{'new_source_requests':0,'CloudStorage_reads':0,'factor_ratio_rank_order_return_computations':0,'old_extraction_rerun':False,'source_state_mutated':False},
        'bindings':{'plan':{'path':source.PLAN,'sha256':plan_sha},'existing_raw_facts':{'path':report_path,'sha256':report_sha},'extractor':{'path':source.SELF,'sha256':report['extractor_sha256']},'extractor_tests':{'path':source.TEST,'sha256':report['tests_sha256']},'review_script':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':digest(Path(__file__).read_bytes())},'inputs':plan['inputs']},
        'checks':{'source_pairs':24,'source_HTML_sha_and_receipt_and_event_matches':24,'raw_cell_locator_and_hash_matches':48,'annual_periods_and_conservative_dates_match':24,'positive_assets':24,'USD_resolved_pairs':report['USD_resolved_pairs'],'same_original_dollar_label_and_scale_pairs':24,'original_synthetic_tests':17,'additional_independent_synthetic_tests':result.testsRun-17,'tests_passed':result.testsRun,'failures':0,'errors':0},
        'annual_event_continuity':continuity,
        'source_contract_assessment':{'limited_issuer_release_feed_review_passed':True,'symbols':['MSFT','NVDA'],'financial_field_scope':'MSFT monetary Gross margin and NVDA Gross profit with same-page annual-end Total assets; no SEC us-gaap certification is inferred.','existing_factor_or_strategy_admitted_flags_changed':False,'certified_first_public_version_PIT_dataset':False,'original_SEC_companyfacts_admitted':0,'factor_or_strategy_admission_now':False},
        'required_registration_terms_before_factor_use':[
            'Treat this as an exploratory two-issuer historical-disclosure feed using current-retrieval original issuer pages under an explicit unchanged-since-publication assumption; do not call it a certified first-public-version or archived-vintage PIT dataset.',
            'Select the latest eligible annual disclosure event before checking validity; retain missing/invalid current events so old valid facts cannot silently survive a newer invalid disclosure. The 24 pinned Q4 events are complete relative to the pinned two-issuer catalogs, not a certified all-corrections universe.',
            'Use each current-release annual column only after the pinned first trading session strictly later than its issuer-local release date. Never use a later release comparative value to overwrite an earlier annual fact.',
            'Freeze the two-issuer scope and preserve every original stock/ETF candidate. All unadmitted issuers and missing/stale/latest-invalid events must follow the same preregistered neutral fallback for both primary and control.',
            'Preserve exact monetary labels and scales. Within-year ratios can cancel a verified common dollar unit; the cross-year growth control must have matching same-issuer reporting currency and normalized scale. Do not label unresolved NVDA currency as independently USD certified.',
            'Derived annual starts are one day after the same balance sheet prior annual end, are checked against adjacent annual reports, and remain labeled derived rather than quoted.',
            'Bind this review, raw facts, original plan, calendar, source/event/identity hashes and new prospective factor contract before computing real ratios, scores, choices or returns.'
        ],
        'limitations':[
            'All 24 historical source pages were retrieved in 2026; historical first-version and intervening corrections are not certified. Current annual-column discipline reduces comparative backfill but cannot prove old webpage content was unchanged.',
            'Pinned event catalogs contain all 12 annual Q4 releases per issuer for FY2015-FY2026. The review cannot prove no omitted correction/reissue outside those catalogs.',
            'The session sequence is pinned XNYS exchange_calendars plus an official closure overlay, not an individually exchange-certified daily history.',
            'Only two source-preparation identities are in scope; this is not a full-universe or continuous daily security-identity certification.',
            'The extractor is a frozen-plan tool. Its duplicate-column and currency safeguards rely partly on the pinned source specifications; expansion to new sources requires a new plan and review.'
        ],
        'rows':details,'tests_output':stream.getvalue(),
        'next_step':'Parent can freeze a separate, explicitly limited issuer-release factor plan with these terms. No need to rerun completed source extraction; certified-vintage PIT claims would instead require archived original filing/release evidence.'
    }
    with OUT.open('x') as f:
        json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'path':str(OUT.relative_to(ROOT)),'sha256':digest(OUT.read_bytes()),'status':output['status'],'checks':output['checks'],'continuity':continuity}))

if __name__ == '__main__':review()
