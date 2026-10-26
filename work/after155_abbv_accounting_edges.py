"""Source-only ABBV annual accounting-edge review; no prices or strategies."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import re
import sys
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
EXTRACTION = 'outputs/monthly_abbv_revenue156/EXTRACTION_V2.json'
POLICY = 'work/after155_multi_issuer_chain_policy.json'
REVIEW = 'work/after155_abbv_revenue_independent.json'
OUT = 'work/after155_abbv_accounting_edges.json'
PINS = {
    EXTRACTION: '5c81e1e23eefda6118739b938cd06074119ad9083f827e499cb5567ffe8f93fd',
    POLICY: '25820e3cdb6273a9bedadfc8253a415bc2f6515b3d61e9d59ab7a465cdc0acc0',
    REVIEW: '0fc8b40786cfb3df532d94700ac46a7d78ae416d1f39a7a3bd0c3faee08a8f9d',
}
SEEN = {}

def sha(raw): return hashlib.sha256(raw).hexdigest()

def read(path, expected=None):
    p = ROOT / path
    assert p.resolve() == p and ROOT in p.parents, path
    raw = p.read_bytes()
    h = sha(raw)
    assert expected is None or h == expected, path
    assert path not in SEEN or SEEN[path] == h, path
    SEEN[path] = h
    return raw

def load(path): return json.loads(read(path, PINS.get(path)))

def norm(node): return ' '.join(node.get_text(' ', strip=True).split())

def originals():
    data = load(EXTRACTION)
    review = load(REVIEW)
    assert review['passed'] is True
    load(POLICY)
    records = []
    for r in data['reports']:
        b = r['source']['raw']
        raw = read(b['path'], b['sha256'])
        assert len(raw) == b['bytes']
        soup = BeautifulSoup(raw, 'html.parser')
        whole = norm(soup)
        assert sha(whole.encode()) == r['context']['normalized_document_text_sha256']
        body = soup.select('.wd_body')
        assert len(body) == 1
        records.append((r, soup, whole, body[0]))
    return records

def inspect(years):
    for r, soup, whole, body in originals():
        if years and r['fiscal_year'] not in years: continue
        print('YEAR', r['fiscal_year'])
        paras = []
        for p in body.find_all(['p', 'li']):
            if p.find(['p', 'li']) is not None: continue
            t = norm(p)
            if re.search(r'(?i)reclass|restate|recast|revenue recognition|accounting polic|accounting standard|adopt|divest|consolidat|royalt|milestone revenue|Pharmacyclics|Allergan|ImmunoGen', t):
                if t and t not in paras: paras.append(t)
        for t in paras:
            if len(t) > 2200: continue
            print(t)

# These anchors were selected after reading original release paragraphs. They
# locate evidence; their presence alone never assigns an accounting status.
ANCHORS = {
    2013: ['On Jan. 1, 2013, Abbott Laboratories distributed'],
    2014: ['Other revenue principally includes royalty income from prior periods'],
    2015: ['Reflects Imbruvica revenue from the May 26 close date',
           'Other revenue is associated with a milestone payment received'],
    2016: ['Acquisition related costs primarily include the amortization',
           'prior period royalty revenue related to a patent lawsuit settlement'],
    2017: ['Financial results for 2017 reflect provisional amounts related',
           'Acquisition related costs primarily includes the amortization'],
    2018: ['Certain reclassifications were made to conform the prior period financial results'],
    2019: ['AbbVie and Allergan announced that Allergan has entered into definitive agreements to divest',
           'The company will issue 2020 proforma guidance following the close'],
    2020: ['Net revenues include Allergan product revenues from the date of the acquisition, May 8, 2020, through December 31, 2020.',
           'All historically reported Allergan revenues have been recast',
           '"Comparable Operational" comparisons include full-period'],
    2021: ['All historically reported Allergan revenues have been recast',
           '"Comparable Operational" comparisons include full-period',
           'Acquisition and integration costs reflect Allergan integration costs'],
    2022: ['During the three months ended March 31, 2022, AbbVie changed its classification'],
    2023: ['This guidance includes a $0.32 per share dilutive impact related',
           'Intangible asset impairment primarily reflects partial impairment charges'],
    2024: ['Reflects partial year Elahere revenue based on the February 12, 2024 close date',
           'Acquisition and integration costs primarily reflect costs related to the ImmunoGen'],
    2025: ['Acquisition and integration costs primarily reflect costs related to the Capstan'],
}

COMMENT = {
    2014: 'Reported total includes royalty income for prior periods recognized in Q4 2014 after licensing settlement. Keep recognition-year GAAP total; do not redistribute to earlier periods or substitute adjusted sales. The January 2013 Abbott separation is disclosed in the earlier original.',
    2015: 'Net sales label becomes net revenues. Later original explicitly includes Imbruvica only from the May26 Pharmacyclics acquisition close, and notes a collaboration milestone payment. Exact FY2014 reported comparative remains19960. Treat both labels as the issuer consolidated GAAP top line, not like-for-like or organic revenue.',
    2016: 'Reported revenue retains milestone and prior-period patent-settlement royalty recognition; Pharmacyclics inventory step-up and Stemcentrx/Boehringer acquisition costs are identified in earnings adjustments. Preserve nominal consolidated total; no removal of unusual revenue or acquisition effects.',
    2017: 'The source explicitly identifies provisional Tax Cuts and Jobs Act estimates and potential subsequent tax updates. These statements concern tax estimates, not a replacement revenue top line. Earlier acquisition/royalty context remains visible. This does not certify all later amendments or organic comparability.',
    2018: 'Unresolved: source explicitly states prior-period reclassifications but does not identify their accounts or establish revenue-policy impact. Matching FY2017 total28216 and generic GAAP policy do not resolve this identified gap. Do not clear this edge without a timed original explanatory note.',
    2019: 'FY2019 release treats Allergan acquisition and related divestitures as pending and its2020 proforma guidance as future. Reported FY2018 comparative32753 matches the original. Earlier unresolved2017→2018 edge is not cleared by this later comparison. Reata divestiture is separately visible in GAAP-to-adjusted earnings reconciliation; no organic-growth claim.',
    2020: 'Source explicitly says reported net revenues include Allergan only from May8 through December31. Separate comparable-operational tables instead pretend January1 2019 acquisition and recast acquired Allergan revenue for rebates/discounts and Zenpep/Viokace divestitures. Those proforma/constant-currency columns are not this input. The formal GAAP comparative remains original ABBV2019 total33266.',
    2021: 'Comparable-operational product tables repeat full-period hypothetical Allergan acquisition and recast acquired revenue. Formal consolidated GAAP statement uses actual FY2020 comparative45804 unchanged. Full2021 versus partial2020 consolidation and Soliton acquisition effects remain in reported total; not homogeneous growth.',
    2022: 'Affirmative footnote scopes prior-period revision to development milestone expense reclassified from R&D to acquired IPR&D and milestones. It states no impact on total operating costs/expenses and earnings/equity. Top-line total56197 also matches FY2021 original. Pylera divestiture is separately shown in earnings reconciliation; not revenue normalisation.',
    2023: 'Source distinguishes proposed ImmunoGen/Cerevel acquisitions and2024 guidance from actual2023 revenue; Imbruvica/CoolSculpting impairment and prior-product-liability gain are earnings matters. Preserve actual nominal total and FY2022 comparative58054. Acquisition/divestiture economic effects are not removed.',
    2024: 'Source explicitly states Elahere revenue is partial-year from February12 2024 ImmunoGen close; acquisition costs and Cerevel intangible impairment are separately described. FY2023 comparative54318 is unchanged. Actual consolidation scope is retained and not proforma-restated.',
    2025: 'Source separately describes Capstan acquisition/integration costs and prior ImmunoGen/Cerevel costs/impairment. Current and prior consolidated reported totals use the explicit GAAP all-revenue policy; FY2024 comparative56334 is unchanged. No claim of fixed acquisition perimeter or organic growth.',
}

def known(r):
    value = r['publication_clock']['canonical_event_known_by_at']
    dt = datetime.fromisoformat(value)
    assert dt.utcoffset() is not None and dt.microsecond == 0
    return int(dt.timestamp()) * 1000000000

def evidence(r, whole, text, kind):
    start = whole.find(text)
    assert start >= 0 and text, (r['fiscal_year'], text)
    return {'kind': kind, 'report_id': r['source']['metadata']['source_id'],
            'source_path': r['source']['raw']['path'],
            'source_sha256': r['source']['raw']['sha256'],
            'source_url': r['source']['metadata']['source_url'],
            'original_publication_date': r['publication_clock']['release_date'],
            'known_ns': known(r), 'known_by_at': r['publication_clock']['canonical_event_known_by_at'],
            'excerpt': text, 'start_normalized_text': start,
            'end_normalized_text': start + len(text), 'excerpt_sha256': sha(text.encode())}

def notes(r, whole, body):
    year = r['fiscal_year']
    paragraphs = list(dict.fromkeys(norm(p) for p in body.find_all(['p', 'li'])
                                    if p.find(['p', 'li']) is None and norm(p)))
    out = []
    for anchor in ANCHORS[year]:
        hits = [p for p in paragraphs if anchor in p]
        if hits:
            text = min(hits, key=len)
        else:
            start = whole.find(anchor)
            assert start >= 0, (year, anchor)
            text = whole[max(0, start - 160): min(len(whole), start + 900)]
        out.append(evidence(r, whole, text, 'scoped_original_note'))
    # Divestiture labels are retained with surrounding reconciliation columns;
    # they are not converted into revenue adjustments or amendment evidence.
    for label in ('Reata divestiture', 'Pylera divestiture'):
        relevant = [x['excerpt'] for x in r['context']['accounting_boundary_contexts']
                    if label in x['excerpt'] and 'Other operating' in x['excerpt']]
        if relevant:
            out.append(evidence(r, whole, max(relevant, key=len), 'earnings_reconciliation_scope'))
    return out

def review():
    assert not (ROOT / OUT).exists(), 'Write-once review'
    source_review = load(REVIEW)
    for path, digest in source_review['input_hashes'].items():
        read(path, digest)
    records = originals()
    assert [r[0]['fiscal_year'] for r in records] == list(range(2013, 2026))
    ev = {}
    source_checks = []
    for r, soup, whole, body in records:
        year = r['fiscal_year']
        gaap = r['context']['reported_gaap_policy']
        assert whole[gaap['start_normalized_text']:gaap['end_normalized_text']] == gaap['excerpt']
        assert re.fullmatch(r'Reported results were prepared in accordance with (?:GAAP|generally accepted accounting principles in the United States \(GAAP\)) and include all revenue and expenses recognized during the period\.', gaap['excerpt'])
        ev[year] = [evidence(r, whole, gaap['excerpt'], 'affirmative_reported_total_policy')]
        ev[year] += notes(r, whole, body)
        table = soup.find_all('table')[r['statement_profile']['table_index']]
        assert sha(str(table).encode()) == r['statement_profile']['table_html_sha256']
        for role in ('current', 'comparative'):
            f = r['revenue'][role]
            assert f['source_raw_sha256'] == r['source']['raw']['sha256']
            assert f['table_html_sha256'] == r['statement_profile']['table_html_sha256']
            assert f['currency_literal'] == '$' and f['currency_iso'] is None
            assert f['scale'] == 1000000
            assert Decimal(f['amount_decimal']).is_finite() and Decimal(f['amount_decimal']) > 0
            assert f['period_start'] == str(f['fact_year']) + '-01-01'
            assert f['period_end'] == str(f['fact_year']) + '-12-31'
        for c in r['context']['accounting_boundary_contexts']:
            assert whole[c['excerpt_start_normalized_text']:c['excerpt_end_normalized_text']] == c['excerpt']
        source_checks.append({'fiscal_year': year, 'report_id': r['source']['metadata']['source_id'],
                              'known_ns': known(r), 'normalized_text_sha256': sha(whole.encode()),
                              'statement_html_sha256': r['statement_profile']['table_html_sha256'],
                              'source_sha256': r['source']['raw']['sha256'],
                              'full_context_excerpts_rechecked': len(r['context']['accounting_boundary_contexts']),
                              'selected_note_evidence': ev[year]})
    edges = []
    for early, late in zip(records, records[1:]):
        a, b = early[0], late[0]
        year = b['fiscal_year']
        ac, bc = a['revenue']['current'], b['revenue']['comparative']
        equal = Decimal(ac['amount_decimal']) == Decimal(bc['amount_decimal'])
        compatible = all(ac[k] == bc[k] for k in ('period_start', 'period_end', 'scale', 'currency_literal', 'measure'))
        assert equal and compatible
        pending = year == 2018
        edge_evidence = ev[year - 1] + ev[year]
        edges.append({
            'issuer': 'ABBV', 'edge_id': f'ABBV_original_total_{year-1}_{year}_release_scope_v1',
            'earlier_fiscal_year': year - 1, 'later_fiscal_year': year,
            'earlier_report_id': a['source']['metadata']['source_id'],
            'later_report_id': b['source']['metadata']['source_id'],
            'known_ns': known(b), 'known_by_at': b['publication_clock']['canonical_event_known_by_at'],
            'evidence_known_ns': sorted(set(x['known_ns'] for x in edge_evidence)),
            'status': 'pending_identified_reclassification_note' if pending else 'checked_reported_total_edge',
            'source_review_sha256': None,
            'source_review_binding_instruction': 'Caller must verify this review and bind the SHA256 of this complete JSON; null is intentionally not directly admissible.',
            'original_amount': ac['amount_decimal'], 'later_comparative_amount': bc['amount_decimal'],
            'exact_decimal_equal': equal, 'same_period_scale_measure': compatible,
            'earlier_label': ac['measure_label'], 'later_comparative_label': bc['measure_label'],
            'period_start': ac['period_start'], 'period_end': ac['period_end'], 'scale': '1000000',
            'currency': 'reported_dollar', 'currency_basis': 'declared_within_issuer_reported_dollar_v1',
            'currency_iso': None, 'currency_iso_certified': False,
            'currency_policy_sha256': PINS[POLICY],
            'currency_assumption': 'Within-issuer same reported dollars is an explicit research assumption, not independently certified ISO currency.',
            'conclusion': COMMENT[year], 'evidence': edge_evidence,
            'economic_homogeneity_certified': False, 'organic_growth_certified': False,
            'exhaustive_amendment_history_verified': False,
            'timing_limitation': 'Current-archive original-dated releases; next-session-close clock; not proven first-public-version bytes.',
            'missing_original_note': ({
                'document': 'Original ABBV FY2018 Form10-K or contemporaneous explanatory accounting note',
                'required_sections': ['scope of prior-period reclassifications', 'revenue recognition/accounting-policy changes and transition treatment'],
                'required_evidence': 'Exact primary bytes, issuer identity, first availability date and a scoped statement proving whether the identified reclassification changes the annual reported-revenue basis.',
                'later_availability_rule': 'A note released after January25 2019 cannot retroactively clear earlier decisions; edge availability must advance to its verified next-session close.',
                'no_known_revenue_break_asserted': True,
            } if pending else None),
        })
    assert len(edges) == 12 and sum(x['status'] == 'checked_reported_total_edge' for x in edges) == 11
    read('work/after155_abbv_accounting_edges.py')
    output = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'ABBV original annual releases FY2013-FY2025; adjacent reported-total accounting review',
              'passed_source_checks': True, 'input_hashes': dict(sorted(SEEN.items())),
              'source_checks': source_checks, 'edges': edges,
              'summary': {'original_reports': 13, 'adjacent_edges': 12, 'exact_equal_comparatives': 12,
                          'checked_limited_reported_total_edges': 11, 'pending_identified_note': 1,
                          'confirmed_revenue_basis_breaks': 0, 'new_chain_rows': 0,
                          'new_factors': 0, 'new_models': 0, 'new_returns': 0, 'new_trials': 0},
              'review_limit': 'Limited consolidated nominal GAAP total continuity under the declared original-release scope. Generic GAAP text and equality are not sufficient where an identified unresolved reclassification exists. Full audited accounting-policy histories and economic homogeneity are not certified.',
              'caller_requirements': ['Independently verify this review, all raw hashes and scoped interpretations.',
                                      'Bind output digest to edge source_review_sha256 and map the matching report IDs explicitly.',
                                      'Keep FY2017→FY2018 pending until additional timed original note evidence is reviewed.',
                                      'Apply four-original chain membership and cutoff checks; this review does not admit any financial factor or strategy.'],
              'observer_writes': 0}
    raw = (json.dumps(output, indent=2, ensure_ascii=False) + '\n').encode()
    with (ROOT / OUT).open('xb') as f: f.write(raw)
    print(json.dumps({'output': OUT, 'sha256': sha(raw), 'summary': output['summary']}))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'inspect':
        inspect([int(x) for x in sys.argv[2:]])
    elif len(sys.argv) > 1 and sys.argv[1] == 'review':
        review()
    else:
        raise SystemExit('Use inspect [years] or review.')
