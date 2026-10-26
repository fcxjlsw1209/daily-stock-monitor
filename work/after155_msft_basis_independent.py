"""Read-only original-page checks of MSFT revenue boundary evidence and dates.

Revalidates existing cells by exact equality; never computes growth or factors.
"""
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
BASE = 'outputs/monthly_msft_revenue_basis156/'
FINAL = 'work/after155_msft_revenue_basis_review.json'
PROPOSAL = 'outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json'
PINS = {
    FINAL: 'ade332bf2cab7c8f8a06ce832fc3389dd0a08902ac4c05515ba22c487e055b40',
    BASE + 'BOUNDARY_DECISIONS.json': '77c137cc871bf0973091b65c36a85091e584ccc85130f449e5c087ba7d94a293',
    BASE + 'CONTEXT_EVIDENCE.json': 'ae9cd4ab971c3b2472c2d21f6b6b5ea9d8fc58981011ee071488055974fe99e4',
    BASE + 'NOTE_EVIDENCE.json': 'e329aee0bd1809742ff4483d258631d46fa4891e5fb3758c37b9a5c2015d50f0',
    BASE + 'SPEC.json': '27b5bf62baef91502ecc64ccee8726e39575c15b527d589d55ef727efe27e268',
    'work/after155_revenue_chain_policy_review.json': '8564d0d052e4a7cfb89bd73c553b46c6ea831afacd384a5d69fe63c52732ce36',
    'work/after155_revenue_extended_review.json': 'f18d530b9cea67d0fce5d82f8910b8a7a20264813e058e296058c8e783a687bd',
    PROPOSAL: '0f5aa563ac7eac950b9594740d14111dfa8aac1608f5e73558d39baee6066aef',
}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read(rel):
    return json.loads((ROOT / rel).read_text())


def norm(text):
    return ' '.join(text.split())


def signature(element):
    return norm(element.get_text(' ', strip=True))


def main():
    for path, expected in PINS.items():
        assert digest((ROOT / path).read_bytes()) == expected, path
    review = read(FINAL)
    decisions = read(BASE + 'BOUNDARY_DECISIONS.json')
    spec = read(BASE + 'SPEC.json')
    context = read(BASE + 'CONTEXT_EVIDENCE.json')
    prior = {r['fiscal_year']: r for r in read('work/after155_revenue_extended_review.json')['reports']}
    proposed = {r['fiscal_year']: r for r in read(PROPOSAL)['records']}
    assert [s['fiscal_year'] for s in review['sources']] == list(range(2015, 2027))
    assert len(review['boundaries']) == len(decisions['boundaries']) == 11
    assert review['decisions_sha256'] == PINS[BASE + 'BOUNDARY_DECISIONS.json']
    assert review['note_evidence_sha256'] == PINS[BASE + 'NOTE_EVIDENCE.json']
    assert review['spec_sha256'] == PINS[BASE + 'SPEC.json']
    assert review['policy']['sha256'] == PINS['work/after155_revenue_chain_policy_review.json']
    sys.path.insert(0, str(ROOT / 'outputs/monthly_earnings_repair'))
    from event_schema import normalize_event
    from research_calendar import market_calendar
    calendar = market_calendar(start='2015-01-01', end='2027-01-01')
    sessions = read('outputs/monthly_paired_target143/CALENDAR.json')['sessions']
    input_paths = set(PINS) | {str(Path(__file__).relative_to(ROOT)),
        'outputs/monthly_paired_target143/CALENDAR.json',
        'outputs/monthly_earnings_repair/event_schema.py', 'outputs/monthly_earnings_repair/research_calendar.py',
        'outputs/index_union/CALENDAR_EXCEPTIONS.json', 'outputs/index_union/sources/closure_20250109.html',
        'work/venv/lib/python3.9/site-packages/exchange_calendars/exchange_calendar_xnys.py',
        'run_local.sh', 'local_run.py', 'local.sandbox.sb'}
    reports = {}
    soups = {}
    note_sets = {}
    independently_found_context = []
    section_names = {'INCOME STATEMENTS', 'CASH FLOWS STATEMENTS', 'STATEMENTS OF CASH FLOWS',
        'SEGMENT RESULTS', 'SEGMENT REVENUE AND OPERATING INCOME', 'Segment Revenue Constant Currency Reconciliation'}
    for source in review['sources']:
        year = source['fiscal_year']
        path = source['path']
        blob = (ROOT / path).read_bytes()
        raw_sha = digest(blob)
        assert raw_sha == source['sha256'] == prior[year]['sha256']
        receipt_path = source['receipt']['path']
        receipt = read(receipt_path)
        assert digest((ROOT / receipt_path).read_bytes()) == source['receipt']['sha256']
        assert receipt['sha256'] == raw_sha and receipt['http_status'] == 200
        input_paths |= {path, receipt_path}
        soup = BeautifulSoup(blob.decode('utf-8-sig'), 'html.parser')
        for hidden in soup(['script', 'style', 'noscript']):
            hidden.decompose()
        soups[year] = soup
        ps = soup.find_all('p')
        context_nodes = soup.find_all(['p','h1','h2','h3','h4','h5','h6'])
        elements = soup.find_all(['p','li','h1','h2','h3','h4','h5','h6'])
        note_sets[year] = {signature(p) for p in elements}
        date_elements = [e for e in elements if re.search(r'REDMOND,\s*Wash', signature(e), re.I)
                         and re.search(r'July\s+\d{1,2},?\s+' + str(year), signature(e))]
        assert date_elements, ('missing original dateline', year)
        dateline_element = date_elements[0]
        date_match = re.search(r'July\s+(\d{1,2}),?\s+(' + str(year) + ')', signature(dateline_element))
        release_date = f'{year}-07-{int(date_match.group(1)):02d}'
        assert release_date == source['release_date'] == prior[year]['release_date']
        event = normalize_event(dict(symbol='MSFT', official=dict(actual_period_end=f'{year}-06-30',
            publication_date=release_date, publication_timezone='America/Los_Angeles', source_url=receipt['url'],
            verified=True, publication_kind='issuer_release')), calendar=calendar)
        assert event['status'] == 'ready'
        assert event['available_session'] == sessions[bisect_right(sessions, release_date)] == source['conservative_session']
        # Locate the consolidated income statement without the producer's row offsets.
        candidates = []
        tables = soup.find_all('table')
        for ti, table in enumerate(tables):
            table_text = signature(table).lower()
            if not ('cost of revenue' in table_text and 'diluted' in table_text and 'income' in table_text):
                continue
            for tr in table.find_all('tr'):
                if tr.find_parent('table') is not table:
                    continue
                cells = tr.find_all(['td','th'], recursive=False)
                if not cells or signature(cells[0]).casefold() not in ('revenue','total revenue'):
                    continue
                numbers = re.findall(r'\d[\d,]*', ' '.join(signature(c) for c in cells[1:]))
                if len(numbers) == 4:
                    candidates.append((ti+1, table, tr, [int(n.replace(',','')) for n in numbers]))
        assert len(candidates) == 1, (year, [(r[0],r[3]) for r in candidates])
        table_no, table, tr, amounts = candidates[0]
        annual = amounts[2:]
        assert annual == [prior[year]['annual_current_usd_millions'], prior[year]['annual_prior_usd_millions']]
        pr = proposed[year]
        assert pr['source']['raw']['sha256'] == raw_sha
        assert pr['availability']['release_date'] == release_date
        assert pr['availability']['first_conservative_session_after_release'] == event['available_session']
        assert pr['currency']['iso_currency'] == 'USD'
        for role, amount, period_year in [('current', annual[0], year), ('comparative', annual[1], year - 1)]:
            cell = pr['revenue'][role]
            assert int(cell['amount_decimal']) == amount and cell['scale'] == 1000000
            assert cell['period_end'] == f'{period_year}-06-30'
            assert cell['source_release_date'] == release_date
            assert cell['first_conservative_session'] == event['available_session']
        assert table_no == prior[year]['table_ordinal_1based']
        table_text = signature(table)
        assert re.search(r'(?:Twelve Months|Year) Ended', table_text, re.I), year
        assert 'June 30' in table_text and str(year) in table_text and str(year-1) in table_text
        units = [signature(e) for e in elements if 'In millions' in signature(e) and len(signature(e)) < 170]
        assert units, year
        recast_elements = [p for p in ps if signature(p).startswith('We have recast certain prior period amounts')]
        for p in recast_elements:
            pindex = next(i for i, pp in enumerate(ps) if pp is p)
            before = [(i,pp) for i,pp in enumerate(ps[:pindex]) if signature(pp) in section_names]
            after = [(i+pindex+1,pp) for i,pp in enumerate(ps[pindex+1:]) if signature(pp) in section_names]
            ancestors = list(p.find_parents('table'))
            previous_table = p.find_previous('table')
            tables_here = []
            for role, t in [('ancestor',a) for a in ancestors] + [('previous', previous_table)]:
                if t is not None:
                    idx = next(i+1 for i, tt in enumerate(tables) if tt is t)
                    tables_here.append(dict(role=role, table_ordinal_1based=idx, html_line=t.sourceline,
                        first_two_paragraphs=[signature(q) for q in t.find_all('p')[:2]]))
            independently_found_context.append(dict(fiscal_year=year, source_path=path, source_sha256=raw_sha,
                html_line=p.sourceline, paragraph_ordinal_1based=pindex+1,
                paragraph_and_heading_node_ordinal_1based=next(i+1 for i,node in enumerate(context_nodes) if node is p), text=signature(p),
                preceding_section=signature(before[-1][1]) if before else None,
                following_section=signature(after[0][1]) if after else None, tables=tables_here))
        notice = [e for e in elements if 'current as of the earnings date, and not restated' in signature(e)]
        basis_flag_texts = sorted({signature(e) for e in elements
            if re.search(r'recast|restat|reclass|revenue recognition|new accounting standards|useful li|continuing operations|discontinued operations|gross versus net', signature(e), re.I)
            and not e.find_parent('table')})
        reports[year] = dict(fiscal_year=year, source_path=path, source_sha256=raw_sha,
            receipt_path=receipt_path, receipt_sha256=source['receipt']['sha256'], receipt_retrieved_at=receipt['retrieved_at'],
            source_url=receipt['url'], dateline_html_line=dateline_element.sourceline,
            original_dateline_date=release_date, known_by_at=event['known_by_at'],
            available_session=event['available_session'], date_only_upper_bound_at=event['date_only_upper_bound_at'],
            statement_table_ordinal_1based=table_no, statement_row_html_line=tr.sourceline,
            annual_current=annual[0], annual_comparative=annual[1], units=units[0],
            row_labels_verified=True, original_existing_cell_pair_verified=True,
            proposal_cells_periods_units_and_availability_verified=True,
            independent_non_table_basis_note_scan=[dict(text=text, text_sha256=digest(text.encode())) for text in basis_flag_texts],
            archival_notice_present=bool(notice), archival_notice_line=notice[0].sourceline if notice else None)
    assert len(independently_found_context) == len(context['records']) == 5
    for found, supplied in zip(independently_found_context, context['records']):
        assert found['fiscal_year'] == supplied['fiscal_year'] and found['text'] == supplied['text']
        assert found['paragraph_and_heading_node_ordinal_1based'] == supplied['paragraph_ordinal_1based']
        assert found['html_line'] == supplied['line']
        assert found['preceding_section'] == (supplied['nearest_preceding_section'] or {}).get('text')
        assert found['following_section'] == (supplied['nearest_following_section'] or {}).get('text')
    boundary_checks = []
    for boundary, decision in zip(review['boundaries'], decisions['boundaries']):
        a,b = boundary['earlier_fiscal_year'],boundary['later_fiscal_year']
        assert b == a + 1
        for key, value in decision.items():
            assert boundary[key] == value, ('decision disagreement', a, key)
        assert boundary['earlier_report_available_at'] == reports[a]['known_by_at']
        assert boundary['later_report_available_at'] == reports[b]['known_by_at']
        assert boundary['known_by'] == boundary['available_from'] == reports[b]['known_by_at']
        assert boundary['known_no_earlier_than'] == reports[b]['known_by_at']
        evidence_checked = []
        for ev in boundary['evidence']:
            year = int(re.match(r'FY(\d{4}):', ev['evidence_id']).group(1))
            assert year in (a,b), ('future source', a, ev['evidence_id'])
            assert ev['source_sha256'] == reports[year]['source_sha256']
            assert ev['text'] in note_sets[year], ('paragraph absent', ev['evidence_id'])
            assert digest(ev['text'].encode()) == ev['text_sha256']
            assert ev['source_available_at'] == reports[year]['known_by_at'] <= boundary['known_by']
            evidence_checked.append(ev['evidence_id'])
        if 'context_evidence_ref' in boundary:
            cref = boundary['context_evidence_ref']
            assert cref['sha256'] == PINS[BASE + 'CONTEXT_EVIDENCE.json']
            assert all(context['records'][i]['fiscal_year'] in (a,b) for i in cref['records_index_0based'])
        same_amount = reports[a]['annual_current'] == reports[b]['annual_comparative']
        assert same_amount == boundary['preverified_comparative_witness']['exact_scaled_amount_equal']
        assert same_amount == (b != 2018)
        boundary_checks.append(dict(earlier_fiscal_year=a,later_fiscal_year=b,
            producer_status=boundary['status'],earliest_permitted_cutoff=boundary['known_by'],
            same_original_and_later_comparative_amount=same_amount,evidence_paragraphs_checked=evidence_checked,
            evidence_time_bound_verified=True, no_future_report_used=True))
    output = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
        reviewed_artifact=FINAL, reviewed_artifact_sha256=PINS[FINAL],
        input_hashes={p:digest((ROOT/p).read_bytes()) for p in sorted(input_paths)},
        method='Independent BeautifulSoup traversal of all 12 original pages; no producer extraction functions called. Existing annual pairs revalidated by equality only. Original datelines and all selected boundary paragraphs verified. Frozen-session lookup and pinned close policy independently recomputed.',
        reports=list(reports.values()), context_occurrences=independently_found_context,
        boundaries=boundary_checks, source_reports=12, existing_cells_revalidated=24,
        boundary_count=11, context_occurrence_count=5, exact_amount_matches=10,
        growth_calculations=0,four_level_chains_computed=0,factor_calculations=0,returns_computed=0,
        network_requests=0,old_artifacts_modified=False)
    print(json.dumps(output,ensure_ascii=False,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
