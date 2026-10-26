"""Source-only, offline MSFT annual revenue preparation; never strategy admission.

Freeze an extraction scope before reading report content, then extract exact
reported levels and cross-report equality evidence. No growth features, prices,
models, selections, or returns are calculated here.
"""
import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unittest

from bs4 import BeautifulSoup
import profitability141_issuer_statements as layout


ROOT = Path(__file__).resolve().parent.parent
SELF = 'work/after155_revenue_source_prepare.py'
OUT = 'outputs/monthly_revenue_source156_preparation'
SPEC = OUT + '/EXTRACTION_SPEC.json'
YEARS = tuple(range(2015, 2027))
RAW_BASE = 'outputs/monthly_earnings_repair/microsoft/raw'
INPUTS = {
    'prior_review': 'work/after155_revenue_persistence_review.json',
    'events': 'outputs/monthly_earnings_repair/microsoft/events.json',
    'calendar_plan': 'outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json',
    'sessions': 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json',
    'extractor': SELF,
    'layout_helpers': 'work/profitability141_issuer_statements.py',
    'launcher': 'run_local.sh',
    'launcher_code': 'local_run.py',
    'sandbox': 'local.sandbox.sb',
}
RESTATEMENT = re.compile(
    r'restat\w*|recast\w*|retrospect\w*|reclassif\w*|'
    r'revenue recognition|accounting (?:standard|principle|change)|'
    r'(?:Topic|ASC)\s*606|adopt(?:ed|ion|ing)', re.I)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def path(rel):
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Nonlocal relative path: ' + rel)
    p = ROOT / p
    if p.resolve() != p:
        raise ValueError('Symlink/path redirection: ' + rel)
    return p


def bind(rel):
    p = path(rel)
    if not p.is_file():
        return {'path': rel, 'exists': False, 'sha256': None}
    raw = p.read_bytes()
    return {'path': rel, 'exists': True, 'sha256': sha(raw), 'bytes': len(raw)}


def pinned(binding):
    if not binding['exists']:
        raise ValueError('Missing at scope freeze: ' + binding['path'])
    raw = path(binding['path']).read_bytes()
    if len(raw) != binding['bytes'] or sha(raw) != binding['sha256']:
        raise ValueError('Pinned source changed: ' + binding['path'])
    return raw


def write_new(rel, data):
    p = path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with p.open('xb') as f:
        f.write(raw)
    return {'path': rel, 'sha256': sha(raw)}


def freeze():
    """Hash report bytes without parsing them; fix every later extraction rule."""
    sources = [{
        'symbol': 'MSFT', 'fiscal_year': y,
        'raw': bind(f'{RAW_BASE}/FY{y}Q4.html'),
        'receipt': bind(f'{RAW_BASE}/FY{y}Q4.meta.json'),
        'expected_url': f'https://www.microsoft.com/en-us/Investor/earnings/FY-{y}-Q4/press-release-webcast',
    } for y in YEARS]
    spec = {
        'schema_version': 1, 'status': 'FROZEN_SOURCE_ONLY_EXTRACTION_SCOPE',
        'created_utc': now(), 'symbol': 'MSFT', 'cik': '0000789019',
        'annual_fiscal_years': list(YEARS), 'priority_fiscal_years': [2015, 2016, 2017, 2018],
        'scope_basis': 'All existing FY2015-FY2026 Q4 report paths of the same issuer, chosen by calendar/file identity before numerical parsing; no holdings or performance selection.',
        'inputs': {k: bind(v) for k, v in INPUTS.items()}, 'sources': sources,
        'contract': {
            'measure': 'Exact Revenue/Total revenue/Revenues row in an unadjusted formal INCOME STATEMENTS table, with cost-of-revenue context; no revenue-minus-cost reconstruction, non-GAAP or constant-currency revenue.',
            'columns': 'Annual Twelve Months Ended June30 current-year and previous-year columns in the same release; retain both with distinct period and vintage roles. Exclude Three Months columns.',
            'period': 'Derived start is July1 of the preceding year, supported by the explicit June30 end plus Twelve Months Ended header; never label the derived start as a direct quote. Preserve 365/366 calendar duration.',
            'amounts': 'Decimal strings in the displayed unit with the explicit table-local multiplier separately recorded. No financial ratios or growth values.',
            'currency': 'Preserve $ literal. Normalize to USD only when the report explicitly states conversion into United States dollars. Otherwise record unresolved currency and a gap.',
            'publication': 'Verify existing event date literal against raw report text and receipt URL/hash/status. Preserve date-only precision. Conservative usability is the next pinned official session strictly after the issuer-local release date; do not fabricate a clock time.',
            'vintage': 'Every historical page was retrieved later. Original historical first-public bytes are not certified. Same-report prior-year figures retain current-report availability; never backdate them or replace the prior report current-year amount.',
            'comparison': 'Compare exact scaled monetary equality only between previous release current-year and next release prior-year. Mismatch or scope/adoption notes are evidence for separate review, not permission to recast the earlier record.',
            'notes': 'Collect bounded context for restatement/recast/retrospective/reclassification/revenue-recognition/accounting-adoption terms. A keyword alone is not a confirmed restatement; absence is not proof of comparability.',
            'ambiguity': 'Missing, duplicate or incompatible source/table/header/cell/date/currency remains explicit. Do not pick an arbitrary candidate or silently skip it.',
            'admission': 'SOURCE_ADMISSION_PROPOSAL only; all revenue/factor/strategy admission flags remain false pending independent source and vintage review.',
            'forbidden': ['network', 'OneDrive', 'companyfacts substitution', 'future comparative backfill', 'price reads', 'financial feature calculation', 'fit', 'selection', 'strategy plan', 'returns', 'observer changes'],
        },
    }
    print(json.dumps(write_new(SPEC, spec)))


def statement_candidates(soup, fiscal_year):
    """Find uniquely supported annual revenue cells, retaining exact locators."""
    candidates, rejections = [], []
    wanted = [f'{fiscal_year - 1}-06-30', f'{fiscal_year}-06-30']
    for ti, table in enumerate(soup.find_all('table')):
        context = layout.statement_context(table)
        if not re.search(r'INCOME STATEMENTS|CONSOLIDATED STATEMENTS OF (?:INCOME|OPERATIONS)', context, re.I):
            continue
        try:
            text = layout.norm(table.get_text(' ', strip=True))
            layout.validate_statement(context, text, 'annual_gross_amount', 'MSFT')
            rows, grid, physical = layout.table_grid(table)
            stop = layout.header_stop(rows)
            matches = [ri for ri, row in enumerate(rows) if any(
                re.fullmatch(r'(?:total )?revenues?:?', layout.norm(c.get_text(' ', strip=True)), re.I)
                for c in row.find_all(['td', 'th'], recursive=False))]
            if len(matches) != 1:
                raise ValueError('Not exactly one revenue row')
            ri = matches[0]
            unit_context = layout.norm(context + ' ' + ' '.join(
                r.get_text(' ', strip=True) for r in rows[:stop]))
            scale = layout.scale_from(unit_context)
            if '$' not in text:
                raise ValueError('No statement dollar symbol')
            found = {d: [] for d in wanted}
            for (r, pi), cell in physical.items():
                if r != ri:
                    continue
                try:
                    amount = layout.exact_amount(cell['text'])
                except ValueError:
                    continue
                headers = layout.headers_for(grid, ri, cell['columns'][0], stop)
                dates = layout.header_dates(headers)
                if 'Twelve Months Ended' not in ' '.join(headers) or dates not in [[d] for d in wanted]:
                    continue
                year_end = dates[0]
                cell_tag = rows[ri].find_all(['td', 'th'], recursive=False)[pi]
                raw_cell = str(cell_tag)
                found[year_end].append({
                    'displayed_amount': cell['text'], 'amount_decimal': amount,
                    'scale': scale, 'currency_literal': '$', 'period_end': year_end,
                    'period_start_derived': f'{int(year_end[:4]) - 1}-07-01',
                    'period_start_is_direct_quote': False,
                    'duration_literal': 'Twelve Months Ended',
                    'days_inclusive': (date.fromisoformat(year_end) - date(int(year_end[:4]) - 1, 7, 1)).days + 1,
                    'header_literals': headers,
                    'locator': {'table_index_zero_based': ti, 'row_index_zero_based': ri,
                                'physical_cell_index_zero_based': pi, 'logical_columns': cell['columns']},
                    'cell_html': raw_cell, 'cell_html_sha256': sha(raw_cell.encode()),
                })
            if any(len(v) != 1 for v in found.values()):
                raise ValueError('Annual current/prior revenue cells absent or ambiguous: ' + str({d: len(v) for d, v in found.items()}))
            candidates.append({
                'current': found[wanted[1]][0], 'comparative': found[wanted[0]][0],
                'statement_context': context, 'unit_context': unit_context,
                'revenue_row_text': layout.norm(rows[ri].get_text(' ', strip=True)),
                'revenue_row_html': str(rows[ri]), 'table_html': str(table),
                'table_html_sha256': sha(str(table).encode()),
            })
        except (ValueError, KeyError, IndexError) as exc:
            rejections.append({'table_index_zero_based': ti, 'reason': str(exc)})
    if len(candidates) != 1:
        raise ValueError('Expected one annual statement candidate; got ' + str(len(candidates)) + '; ' + json.dumps(rejections))
    return candidates[0], rejections


def contexts(text):
    items = []
    for match in RESTATEMENT.finditer(text):
        if items and match.start() <= items[-1]['match_start_normalized_text'] + 100:
            continue
        items.append({'keyword': match.group(0), 'match_start_normalized_text': match.start(),
                      'excerpt': text[max(0, match.start()-150):min(len(text), match.end()+300)]})
        if len(items) >= 30:
            break
    return items


def extract(source, events, sessions, calendar_start):
    raw, receipt_raw = pinned(source['raw']), pinned(source['receipt'])
    receipt = json.loads(receipt_raw)
    layout.receipt_check(raw, receipt, {'source_url': source['expected_url']})
    year = source['fiscal_year']
    hits = [e for e in events if (e['ticker'], e['fiscal_year'], e['fiscal_quarter']) == ('MSFT', year, 4)]
    if len(hits) != 1:
        raise ValueError('Missing or ambiguous annual event')
    event = hits[0]
    if event.get('sha256', event.get('evidence_sha256')) != source['raw']['sha256'] or event['source_url'] != source['expected_url']:
        raise ValueError('Event raw hash or URL mismatch')
    if event['fiscal_period_end_actual'] != f'{year}-06-30':
        raise ValueError('Annual event end mismatch')
    soup = BeautifulSoup(raw, 'html.parser')
    text = layout.norm(soup.get_text(' ', strip=True))
    literal = layout.norm(event['date_literal'])
    if literal not in text:
        raise ValueError('Publication-date literal missing from raw page')
    publication_index = text.index(literal)
    statement, rejected = statement_candidates(soup, year)
    policy = re.search(r'(?:To supplement.{0,350}?presented in accordance with GAAP|financial results.{0,100}?reported in accordance with.{0,100}?GAAP|financial results in accordance with.{0,100}?GAAP|measures of financial performance prepared in accordance with GAAP)', text, re.I)
    if policy is None:
        raise ValueError('Original-report GAAP policy absent')
    usd = re.search(r'.{0,70}converted into United States dollars.{0,100}', text, re.I)
    usable = layout.first_session_after(event['release_date'], sessions, calendar_start)
    if usable is None:
        raise ValueError('Next official session outside pinned calendar')
    if event['release_date'] <= f'{year}-06-30':
        raise ValueError('Release not after annual period')
    notes = contexts(text)
    for role in ['current', 'comparative']:
        statement[role]['source_release_date'] = event['release_date']
        statement[role]['first_conservative_session'] = usable
        statement[role]['vintage_role'] = 'current_release_current_year' if role == 'current' else 'current_release_prior_year_never_backdated'
    table_html = statement.pop('table_html')
    result = {
        'symbol': 'MSFT', 'cik': '0000789019', 'fiscal_year': year,
        'status': 'EXTRACTED_FOR_INDEPENDENT_SOURCE_REVIEW',
        'source': source, 'revenue': statement,
        'gaap_policy_excerpt': policy.group(0),
        'currency': {'iso_currency': 'USD' if usd else None, 'USD_context_found': bool(usd),
                     'literal': '$', 'evidence': usd.group(0) if usd else None},
        'availability': {
            'release_date': event['release_date'], 'publication_timestamp': event.get('publication_timestamp'),
            'precision': event['precision'], 'timezone': event.get('publication_timezone'),
            'date_literal': literal, 'normalized_text_index': publication_index,
            'dateline_excerpt': text[max(0, publication_index-90):publication_index+len(literal)+120],
            'first_conservative_session_after_release': usable,
            'retrieved_at': receipt['retrieved_at'], 'historical_first_public_bytes_verified': False,
        },
        'accounting_change_keyword_context': notes,
        'keyword_context_is_confirmed_restatement': False,
        'rejected_statement_candidates': rejected,
        'review_flags': ['original_webpage_vintage_not_certified', 'derived_annual_start',
                         'accounting_scope_and_comparability_need_independent_review'] + ([] if usd else ['currency_unresolved']),
        'revenue_source_admitted': False, 'factor_admitted': False, 'strategy_admitted': False,
    }
    return result, table_html


def compare_releases(records):
    by_year = {r['fiscal_year']: r for r in records}
    checks = []
    for year in YEARS[1:]:
        if year not in by_year or year-1 not in by_year:
            checks.append({'current_release_fy': year, 'status': 'MISSING_ONE_OR_BOTH_REPORTS'})
            continue
        previous, current = by_year[year-1], by_year[year]
        a = previous['revenue']['current']
        b = current['revenue']['comparative']
        # Equality validation is source reconciliation, not a growth feature.
        equal = Decimal(a['amount_decimal']) * a['scale'] == Decimal(b['amount_decimal']) * b['scale']
        checks.append({
            'current_release_fy': year, 'compared_fiscal_end': a['period_end'],
            'previous_release_current_amount': a['amount_decimal'],
            'next_release_comparative_amount': b['amount_decimal'],
            'previous_scale': a['scale'], 'next_scale': b['scale'],
            'same_period': a['period_end'] == b['period_end'],
            'exact_scaled_amount_equal': equal,
            'status': 'AMOUNTS_EQUAL_NOT_PROOF_OF_SCOPE_CONTINUITY' if equal else 'COMPARATIVE_CHANGED_DO_NOT_BACKFILL',
            'later_comparative_available_only_from': current['availability']['first_conservative_session_after_release'],
            'earlier_original_preserved': True,
            'restatement_cause_confirmed': False,
        })
    return checks


def prepare(spec_sha):
    raw = path(SPEC).read_bytes()
    if sha(raw) != spec_sha:
        raise ValueError('Extraction spec digest mismatch')
    spec = json.loads(raw)
    if spec['annual_fiscal_years'] != list(YEARS):
        raise ValueError('Unexpected frozen scope')
    inputs = {k: pinned(b) for k, b in spec['inputs'].items()}
    calendar_plan, calendar = json.loads(inputs['calendar_plan']), json.loads(inputs['sessions'])
    if calendar['plan_sha256'] != spec['inputs']['calendar_plan']['sha256']:
        raise ValueError('Calendar-plan mismatch')
    for rel, expected in calendar_plan['bindings'].items():
        if sha(path(rel).read_bytes()) != expected:
            raise ValueError('Calendar dependency mismatch: ' + rel)
    events = json.loads(inputs['events'])
    records, gaps, evidence_bindings = [], [], []
    for source in spec['sources']:
        try:
            record, table_html = extract(source, events, calendar['sessions'], calendar_plan['start'])
            rel = OUT + f'/evidence/FY{source["fiscal_year"]}Q4_income_table.html'
            p = path(rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as f:
                f.write(table_html.encode())
            record['revenue']['derived_table_evidence'] = bind(rel)
            evidence_bindings.append(bind(rel))
            records.append(record)
        except (ValueError, KeyError, IndexError, OSError) as exc:
            gaps.append({'fiscal_year': source['fiscal_year'], 'source': source, 'reason': str(exc)})
    report = {
        'schema_version': 1, 'created_utc': now(),
        'status': 'SOURCE_ONLY_PROPOSAL_NOT_ADMITTED', 'spec': {'path': SPEC, 'sha256': spec_sha},
        'inputs': spec['inputs'], 'source_count': len(spec['sources']),
        'extracted_count': len(records), 'extraction_gap_count': len(gaps),
        'records': records, 'extraction_gaps': gaps,
        'cross_release_comparative_checks': compare_releases(records),
        'evidence_bindings': evidence_bindings,
        'revenue_source_admitted': False, 'factor_admitted': False, 'strategy_admitted': False,
        'activity': {'new_requests': 0, 'price_reads': 0, 'growth_features': 0, 'fits': 0, 'orders': 0, 'returns': 0},
        'required_independent_review': [
            'Verify each raw-byte hash, rendered/logical table locator, exact annual revenue amount and USD context.',
            'Verify issuer identity, original event date, conservative next-session rule and original historical webpage vintage limitation.',
            'Inspect accounting notes and every changed same-year comparative; decide whether any chain is semantically continuous before a factor plan is proposed.',
            'Do not insert comparative values into an earlier information set or replace original current-year figures.',
            'Original-publication bytes are not authenticated by a later retrieved URL; explicit policy/evidence review is still required.',
            'Scope is MSFT only. No claim of full-pool coverage or strategy readiness.',
        ],
    }
    binding = write_new(OUT + '/SOURCE_ADMISSION_PROPOSAL.json', report)
    print(json.dumps(dict(binding, extracted_count=len(records), extraction_gap_count=len(gaps),
                          comparative_mismatches=sum(c.get('exact_scaled_amount_equal') is False for c in report['cross_release_comparative_checks']))))


class SyntheticTests(unittest.TestCase):
    def sample(self, current='12,300', prior='11,000', duration='Twelve Months Ended', extra=''):
        return BeautifulSoup('<h2>INCOME STATEMENTS</h2><p>(In millions)</p><div class="c-table"><table>'
            '<tr><th></th><th colspan="2">' + duration + '</th></tr>'
            '<tr><th></th><th>June 30, 2015</th><th>June 30, 2016</th></tr>'
            '<tr><td>Revenue</td><td>$ ' + prior + '</td><td>$ ' + current + '</td></tr>'
            '<tr><td>Cost of revenue</td><td>1</td><td>2</td></tr>' + extra + '</table></div>', 'html.parser')
    def test_annual_exact(self):
        item, _ = statement_candidates(self.sample(), 2016)
        self.assertEqual(item['current']['amount_decimal'], '12300')
        self.assertEqual(item['comparative']['period_end'], '2015-06-30')
        self.assertEqual(item['current']['scale'], 1000000)
    def test_duration(self):
        with self.assertRaises(ValueError): statement_candidates(self.sample(duration='Three Months Ended'), 2016)
    def test_percent(self):
        with self.assertRaises(ValueError): statement_candidates(self.sample(current='12%'), 2016)
    def test_duplicate_revenue(self):
        with self.assertRaises(ValueError): statement_candidates(self.sample(extra='<tr><td>Revenue</td><td>2</td><td>3</td></tr>'), 2016)
    def test_non_gaap(self):
        with self.assertRaises(ValueError): statement_candidates(self.sample(extra='<tr><td>Non-GAAP</td></tr>'), 2016)
    def test_wrong_year(self):
        with self.assertRaises(ValueError): statement_candidates(self.sample(), 2017)
    def test_leap_duration(self):
        item, _ = statement_candidates(self.sample(), 2016)
        self.assertEqual(item['current']['days_inclusive'], 366)
        self.assertEqual(item['comparative']['days_inclusive'], 365)
    def test_zero_preserved(self):
        item, _ = statement_candidates(self.sample(current='0'), 2016)
        self.assertEqual(item['current']['amount_decimal'], '0')
    def test_two_statements(self):
        soup = BeautifulSoup(str(self.sample()) * 2, 'html.parser')
        with self.assertRaises(ValueError): statement_candidates(soup, 2016)
    def test_notes_are_context_not_conclusions(self):
        self.assertEqual(contexts('We adopted an accounting standard.')[0]['keyword'], 'adopted')
    def test_paths(self):
        for rel in ['/tmp/x', '../x']:
            with self.assertRaises(ValueError): path(rel)
    def test_no_same_day_use(self):
        self.assertEqual(layout.first_session_after('2015-07-21', ['2015-07-21', '2015-07-22']), '2015-07-22')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    parser.add_argument('--spec-sha')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.freeze:
        freeze()
    elif not args.spec_sha:
        parser.error('--prepare requires --spec-sha')
    else:
        prepare(args.spec_sha)
