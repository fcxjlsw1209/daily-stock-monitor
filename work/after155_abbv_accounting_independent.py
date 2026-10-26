"""Independent ABBV original-release accounting audit; no producer imports."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlparse
import hashlib
import json
import re
import sys
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = 'work/after155_abbv_accounting_independent.json'
EDGES = 'work/after155_abbv_accounting_edges.json'
EXTRACT = 'outputs/monthly_abbv_revenue156/EXTRACTION_V2.json'
CELLS = 'work/after155_abbv_revenue_independent.json'
POLICY = 'work/after155_multi_issuer_chain_policy.json'
EXPECTED = {
    EDGES: '115a6e1d48502a87de34ee0376dfdedea7936eac5320dda353f6dd7226ad18ec',
    EXTRACT: '5c81e1e23eefda6118739b938cd06074119ad9083f827e499cb5567ffe8f93fd',
    CELLS: '0fc8b40786cfb3df532d94700ac46a7d78ae416d1f39a7a3bd0c3faee08a8f9d',
    POLICY: '25820e3cdb6273a9bedadfc8253a415bc2f6515b3d61e9d59ab7a465cdc0acc0',
    'work/after155_abbv_accounting_edges.py': 'ebef6aecfadcb832238ef78ec45ae71db37fbb093aca1bcd8012ad31da26780a',
    'work/after155_abbv_revenue_independent.py': 'f3c5844c484f17c777f5835ac10d142402727dc177e4b1714bd7ec2fd19540e9',
    'work/after155_multi_issuer_chain.py': 'f106f40624afd6fc233263301df06cf85ed8961bca8b822007c9161f65b80bba',
}
SEEN = {}

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, expected=None, size=None):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    require(p.resolve() == p and ROOT in p.parents, 'Nonlocal/redirected input: ' + str(p))
    key = str(p.relative_to(ROOT))
    raw = p.read_bytes()
    digest = sha(raw)
    require(expected is None or expected == digest, 'Input digest mismatch: ' + key)
    require(size is None or size == len(raw), 'Input size mismatch: ' + key)
    require(key not in SEEN or SEEN[key] == digest, 'Input changed during audit: ' + key)
    SEEN[key] = digest
    return raw

def load(path):
    return json.loads(read(path, EXPECTED.get(path)))

def norm(node):
    return ' '.join(node.get_text(' ', strip=True).split())

def sources():
    data = load(EXTRACT)
    require([r['fiscal_year'] for r in data['reports']] == list(range(2013, 2026)), 'Exact annual source universe')
    result = {}
    for r in data['reports']:
        b = r['source']['raw']
        raw = read(b['path'], b['sha256'], b['bytes'])
        soup = BeautifulSoup(raw, 'html.parser')
        whole = norm(soup)
        require(sha(whole.encode()) == r['context']['normalized_document_text_sha256'], 'Normalized source text digest')
        body = soup.select('.wd_body')
        require(len(body) == 1, 'Unique original release body')
        paras = list(dict.fromkeys(norm(p) for p in body[0].find_all(['p', 'li'])
                                  if not p.find(['p', 'li']) and norm(p)))
        result[r['fiscal_year']] = (r, soup, whole, paras)
    return result

def inspect(years, currency=False):
    for year, (r, soup, whole, paras) in sources().items():
        if years and year not in years:
            continue
        print('YEAR', year)
        for text in paras:
            pattern = r'currency|dollar|\beuro\b|\byen\b|sterling' if currency else r'reclass|recast|revenue recognition|royalt|milestone revenue|Net revenues include|partial year|separation|Imbruvica revenue|Capstan|Tax Cuts|proforma guidance'
            if len(text) < 2600 and re.search(pattern, text, re.I):
                print(text)


# Independent dispositions, selected after reading the original releases. These
# are accounting interpretations, not conclusions produced by a keyword test.
# Every admitted edge also needs both original GAAP statements and scoped notes.
REASONS = {
    2014: 'Both originals present AbbVie consolidated annual net sales. The 2013 separation occurred on January 1 and is explicitly described; the 2014 royalty settlement is revenue recognized in 2014, even though it relates to prior periods. The annual GAAP basis retains that recognition timing. The matching 2013 comparative corroborates the original, without reallocating the settlement or assuming a constant business perimeter.',
    2015: 'The original GAAP statement changes its top-line label from Net sales to Net revenues but retains the prior annual consolidated total as its comparative. The release explicitly includes Imbruvica from the Pharmacyclics closing date and collaboration milestone revenue. These are actual recognized revenues within the stated all-revenue GAAP policy, not a pro-forma reconstruction. This supports continuity of the nominal reported-total measure only.',
    2016: 'The original annual statement continues consolidated net revenues. Its notes explicitly retain milestone revenue and prior-period royalty revenue recognized through a patent settlement; acquisition inventory step-up and transaction costs are separately described as earnings adjustments. The source therefore supports actual recognized annual total revenue, including these uneven items, and the prior original is corroborated by the later GAAP comparative.',
    2017: 'The 2017 source explicitly scopes provisional amounts to the Tax Cuts and Jobs Act, and describes the mandatory tax and tax-balance-sheet effects. Acquisition inventory costs are disclosed separately. These notes and both annual consolidated statements support the same nominal reported-revenue measure; neither provisional taxes nor matching revenue is treated as proof that all accounting policies are unchanged.',
    2018: 'Pending. The 2018 original affirmatively says prior-period financial results were reclassified to its current presentation but does not identify the affected accounts or revenue transition. Its unchanged 2017 comparative and generic GAAP statement cannot resolve that identified uncertainty. An original explanatory note with verified availability and revenue-policy scope is still required.',
    2019: 'The 2018 current-year original already uses the post-reclassification presentation; the unresolved earlier transition does not itself disqualify this subsequent pair. The 2019 release describes the Allergan acquisition and related divestitures as pending and pro-forma guidance as future. Both formal annual statements preserve actual consolidated GAAP revenue and the 2018 comparative corroborates its own original. The separately identified Reata divestiture affects reported economics and is retained, not normalized away.',
    2020: 'The 2020 original expressly includes acquired Allergan product revenue only from May 8 through December 31. Its separate comparable-operational tables assume a January 2019 acquisition, use constant currency and recast historical Allergan revenue. The selected formal GAAP annual statement instead preserves actual consolidation and the original AbbVie 2019 comparative. The scope distinction affirmatively supports a reported-total edge while forbidding organic, same-perimeter or pro-forma interpretation.',
    2021: 'The 2021 source again identifies historical Allergan recasts and full-period hypothetical acquisition comparisons as comparable-operational presentations, distinct from the selected consolidated GAAP annual statement. The 2020 original explicitly describes its partial-year acquisition inclusion. The unchanged formal comparative preserves that actual 2020 basis; 2021 full-year inclusion and Soliton acquisition effects remain part of reported economics.',
    2022: 'The 2022 original precisely identifies its comparative revision as development milestone expense moving from research and development to acquired IPR&D and milestones. It states no impact on total operating costs or earnings and equity. This is affirmative scope evidence, not a guess from an unchanged revenue figure. Both annual GAAP revenue statements remain the same nominal consolidated measure; divestiture economics are not removed.',
    2023: 'The prior-year expense reclassification is explicitly scoped by the 2022 original. The 2023 release describes ImmunoGen/Cerevel as proposed acquisitions affecting 2024 guidance, and the cited impairments and development-liability gain as earnings items. Its actual consolidated annual revenue and prior GAAP comparative support reported-total continuity while preserving acquisition, product and disposal economics.',
    2024: 'The 2024 original explicitly limits Elahere revenue to the period following the February 12 ImmunoGen close, and separately identifies Cerevel impairment and acquisition costs. The prior release treats these acquisitions as proposed future transactions. Both formal GAAP annual statements therefore measure actual recognized consolidated revenues with changing scope, and the later comparative corroborates the 2023 original without pro-forma substitution.',
    2025: 'The 2025 original continues consolidated annual net revenues under its explicit United States GAAP all-revenue policy. Capstan acquisition and integration costs and intangible impairments are separately identified; the 2024 source specifies its partial-year Elahere inclusion. The formal 2024 comparative preserves that actual reported basis. This admits the nominal consolidated total under the declared source scope, not an assertion of identical businesses, accounting-policy histories or organic growth.',
}

ANCHORS = {
    2013: ['On Jan. 1, 2013, Abbott Laboratories distributed'],
    2014: ['Other revenue principally includes royalty income from prior periods'],
    2015: ['Reflects Imbruvica revenue from the May 26 close date', 'Other revenue is associated with a milestone payment received'],
    2016: ['prior period royalty revenue related to a patent lawsuit settlement'],
    2017: ['Financial results for 2017 reflect provisional amounts', 'The net charge included the one-time impact'],
    2018: ['Certain reclassifications were made to conform the prior period financial results'],
    2019: ['AbbVie and Allergan announced that Allergan has entered into definitive agreements to divest', 'The company will issue 2020 proforma guidance'],
    2020: ['Net revenues include Allergan product revenues from the date of the acquisition', 'All historically reported Allergan revenues have been recast', '"Comparable Operational" comparisons include full-period'],
    2021: ['All historically reported Allergan revenues have been recast', '"Comparable Operational" comparisons include full-period', 'Acquisition and integration costs reflect Allergan integration costs'],
    2022: ['During the three months ended March 31, 2022, AbbVie changed its classification'],
    2023: ['This guidance includes a $0.32 per share dilutive impact', 'Intangible asset impairment primarily reflects partial impairment charges'],
    2024: ['Reflects partial year Elahere revenue based on the February 12, 2024 close date', 'Acquisition and integration costs primarily reflect costs related to the ImmunoGen'],
    2025: ['Acquisition and integration costs primarily reflect costs related to the Capstan'],
}

def known(r):
    value = r['publication_clock']['canonical_event_known_by_at']
    instant = datetime.fromisoformat(value)
    require(instant.utcoffset() is not None and instant.microsecond == 0, 'Invalid source clock')
    return int(instant.timestamp()) * 1000000000

def excerpt(r, whole, text, kind):
    start = whole.find(text)
    require(start >= 0 and bool(text), 'Missing independent source excerpt')
    return dict(kind=kind, report_id=r['source']['metadata']['source_id'],
                source_path=r['source']['raw']['path'], source_sha256=r['source']['raw']['sha256'],
                source_url=r['source']['metadata']['source_url'], known_ns=known(r),
                original_publication_date=r['publication_clock']['release_date'],
                start_normalized_text=start, end_normalized_text=start + len(text),
                excerpt=text, excerpt_sha256=sha(text.encode()))

def independent_notes(r, whole, paragraphs):
    out = []
    for anchor in ANCHORS[r['fiscal_year']]:
        hits = [p for p in paragraphs if anchor in p]
        require(bool(hits), 'Missing independently selected accounting note: ' + anchor)
        out.append(excerpt(r, whole, min(hits, key=len), 'independently_selected_scoped_note'))
    return out

def verify_quote(e, original):
    r, soup, whole, paras = original
    require(e['report_id'] == r['source']['metadata']['source_id'], 'Evidence report identity')
    require(e['source_path'] == r['source']['raw']['path'] and e['source_sha256'] == r['source']['raw']['sha256'], 'Evidence raw binding')
    require(e['source_url'] == r['source']['metadata']['source_url'], 'Evidence URL binding')
    require(e['original_publication_date'] == r['publication_clock']['release_date'], 'Evidence original date')
    require(e['known_ns'] == known(r), 'Evidence next-session clock')
    require(e.get('known_by_at', r['publication_clock']['canonical_event_known_by_at']) == r['publication_clock']['canonical_event_known_by_at'], 'Evidence close timestamp')
    require(whole[e['start_normalized_text']:e['end_normalized_text']] == e['excerpt'], 'Evidence normalized offsets')
    require(sha(e['excerpt'].encode()) == e['excerpt_sha256'], 'Evidence excerpt digest')

def verified_source(original, cell_check):
    r, soup, whole, paragraphs = original
    year = r['fiscal_year']
    source = r['source']
    meta = source['metadata']
    receipt_b = source['receipt']
    receipt = json.loads(read(receipt_b['path'], receipt_b['sha256'], receipt_b['bytes']))
    url = urlparse(meta['source_url'])
    require(url.scheme == 'https' and url.hostname == 'news.abbvie.com', 'Original issuer domain')
    require(meta['symbol'] == receipt['ticker'] == 'ABBV', 'Issuer identity')
    require(meta['source_kind'] == 'issuer_original_release' and meta['fiscal_year'] == year, 'Original annual identity')
    require(receipt['url'] == receipt['final_url'] == meta['source_url'] and receipt['status'] == 200, 'Receipt URL/status')
    require(receipt['sha256'] == source['raw']['sha256'] and receipt['bytes'] == source['raw']['bytes'], 'Receipt source binding')
    require(receipt['source_representation'] == 'original_http_bytes', 'Receipt original representation')
    dates = soup.select('.wd_date')
    require(len(dates) == 1, 'Unique original publication date')
    release = datetime.strptime(norm(dates[0]), '%B %d, %Y').date().isoformat()
    require(release == meta['publication_date'] == r['publication_clock']['release_date'] == cell_check['release_date'], 'Original publication date agreement')
    require(url.path.startswith('/' + release + '-'), 'Date in issuer release URL')
    require(sha(str(dates[0]).encode()) == r['publication_clock']['date_element_html_sha256'], 'Publication date HTML digest')
    require(cell_check['source_sha256'] == source['raw']['sha256'] and cell_check['source_path'] == source['raw']['path'], 'Independent cell review source identity')
    require(cell_check['independent_known_by_close'] == r['publication_clock']['canonical_event_known_by_at'], 'Independent known-by clock agreement')
    require(cell_check['independent_available_session'] == meta['available_session'], 'Independent availability agreement')
    require(cell_check['receipt_verified'] and cell_check['reported_gaap_policy_verified'], 'Bound source review checks')
    require(r['publication_clock']['precision'] == 'date_only' and r['publication_clock']['publication_timestamp'] is None, 'No exact publication time inferred')
    policy = r['context']['reported_gaap_policy']
    require(whole[policy['start_normalized_text']:policy['end_normalized_text']] == policy['excerpt'], 'Original GAAP policy offsets')
    require(re.fullmatch(r'Reported results were prepared in accordance with (?:GAAP|generally accepted accounting principles in the United States \(GAAP\)) and include all revenue and expenses recognized during the period\.', policy['excerpt']) is not None, 'Affirmative reported-total GAAP policy')
    table = soup.find_all('table')[r['statement_profile']['table_index']]
    require(sha(str(table).encode()) == r['statement_profile']['table_html_sha256'], 'Original formal statement digest')
    rows = [x for x in table.find_all('tr') if x.find_parent('table') is table]
    ri = r['statement_profile']['row_index']
    head = ' '.join(norm(x) for x in rows[:ri])
    require('AbbVie Inc.' in head and 'Consolidated Statements of Earnings' in head, 'Formal consolidated statement identity')
    require(not re.search(r'non[ -]?GAAP|adjusted|pro.forma|comparable operational', head, re.I), 'Selected formal header excludes alternative measures')
    require('(in millions, except per share data)' in head.lower(), 'Selected formal statement units')
    row = rows[ri]
    physical = row.find_all(['td', 'th'], recursive=False)
    require(norm(physical[0]) == ('Net sales' if year <= 2014 else 'Net revenues'), 'Original top-line label')
    checked_cells = {}
    for role, target in (('current', year), ('comparative', year - 1)):
        fact = r['revenue'][role]
        loc = fact['locator']
        cc = cell_check['cells'][role]
        require(cc['exact_amount_verified'] and cc['own_annual_column_dollar_evidence_verified'] and cc['period_verified'], 'Independent annual-column checks')
        require(loc['row_index_zero_based'] == cc['row_index'] == ri and loc['table_index_zero_based'] == cc['table_index'] == r['statement_profile']['table_index'], 'Statement geometry binding')
        ci = loc['physical_cell_index_zero_based']
        require(ci == cc['physical_cell_index'], 'Independent cell geometry binding')
        cell = physical[ci]
        require(sha(str(cell).encode()) == fact['cell_html_sha256'] == cc['cell_sha256'], 'Original monetary cell digest')
        require(sha(str(row).encode()) == fact['row_html_sha256'], 'Original annual row digest')
        require(norm(cell) == fact['displayed_amount'], 'Original displayed amount')
        literal = norm(cell).replace('$', '').replace(',', '').strip()
        require(re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', literal) is not None and Decimal(literal) == Decimal(fact['amount_decimal']), 'Exact monetary string')
        require(Decimal(literal) > 0, 'Positive annual total')
        markers = fact['currency_marker_physical_cells']
        require('$' in norm(cell) or (len(markers) == 1 and norm(physical[markers[0]]) == '$'), 'Own annual amount dollar marker')
        require(fact['currency_literal'] == '$' and fact['currency_iso'] is None and fact['scale'] == cc['scale'] == 1000000, 'Dollar literal and millions, no ISO assertion')
        require(fact['fact_year'] == cc['fiscal_year'] == target, 'Original annual vintage')
        require(fact['period_start'] == str(target) + '-01-01' and fact['period_end'] == str(target) + '-12-31', 'Actual annual endpoints')
        require(fact['measure'] == 'consolidated_total_net_revenue', 'Nominal annual reported measure')
        checked_cells[role] = dict(fiscal_year=target, amount_decimal=fact['amount_decimal'], cell_sha256=fact['cell_html_sha256'], scale=1000000, currency_literal='$', own_column_dollar_verified=True)
    for context in r['context']['accounting_boundary_contexts']:
        require(whole[context['excerpt_start_normalized_text']:context['excerpt_end_normalized_text']] == context['excerpt'], 'Accounting context offsets')
        require(whole[context['match_start_normalized_text']:context['match_end_normalized_text']] == context['keyword'], 'Accounting context match offsets')
    currency_notes = [excerpt(r, whole, p, 'currency_context') for p in paragraphs if len(p) < 2600 and re.search(r'currency|dollar|\beuro\b|\byen\b|sterling', p, re.I)]
    notes = [excerpt(r, whole, policy['excerpt'], 'affirmative_reported_total_policy')] + independent_notes(r, whole, paragraphs)
    return dict(fiscal_year=year, report_id=meta['source_id'], source_path=source['raw']['path'], source_sha256=source['raw']['sha256'], source_url=meta['source_url'], release_date=release, known_ns=known(r), available_session=cell_check['independent_available_session'], known_by_at=cell_check['independent_known_by_close'], receipt_verified=True, statement_identity_verified=True, statement_html_sha256=r['statement_profile']['table_html_sha256'], source_normalized_text_sha256=sha(whole.encode()), cells=checked_cells, accounting_contexts_rechecked=len(r['context']['accounting_boundary_contexts']), independent_evidence=notes, currency_contexts=currency_notes)

def audit():
    for p, h in EXPECTED.items():
        read(p, h)
    producer = load(EDGES)
    cell_review = load(CELLS)
    policy = load(POLICY)
    require(producer['passed_source_checks'] is True and cell_review['passed'] is True, 'Required source reviews passed')
    require(policy['new_factor_or_strategy_admission'] is False, 'Source-only policy')
    for evidence_doc in (producer, cell_review):
        for p, h in evidence_doc['input_hashes'].items():
            read(p, h)
    originals = sources()
    cell_checks = {x['fiscal_year']: x for x in cell_review['reports']}
    reports = {y: verified_source(original, cell_checks[y]) for y, original in originals.items()}
    source_checks = {x['fiscal_year']: x for x in producer['source_checks']}
    require(set(source_checks) == set(originals), 'Producer exact source universe')
    all_ids = {r[0]['source']['metadata']['source_id']: y for y, r in originals.items()}
    quotes_checked = 0
    for year, checks in source_checks.items():
        r = originals[year][0]
        require(checks['report_id'] == reports[year]['report_id'] and checks['known_ns'] == known(r), 'Producer source identity/clock')
        require(checks['source_sha256'] == reports[year]['source_sha256'] and checks['normalized_text_sha256'] == reports[year]['source_normalized_text_sha256'] and checks['statement_html_sha256'] == reports[year]['statement_html_sha256'], 'Producer source content digests')
        for e in checks['selected_note_evidence']:
            verify_quote(e, originals[year])
            quotes_checked += 1
    require(len(producer['edges']) == 12 and {e['later_fiscal_year'] for e in producer['edges']} == set(REASONS), 'Exact producer adjacent universe')
    admitted = []
    for e in producer['edges']:
        later = e['later_fiscal_year']
        earlier = later - 1
        a, b = originals[earlier][0], originals[later][0]
        require(e['issuer'] == 'ABBV' and e['earlier_fiscal_year'] == earlier, 'Adjacent issuer/year identity')
        require(e['earlier_report_id'] == reports[earlier]['report_id'] and e['later_report_id'] == reports[later]['report_id'], 'Adjacent original report identities')
        require(e['edge_id'] == 'ABBV_original_total_%s_%s_release_scope_v1' % (earlier, later), 'Exact edge identity')
        require(e['known_ns'] == known(b) and e['known_by_at'] == b['publication_clock']['canonical_event_known_by_at'], 'Edge actual available clock')
        require(e['evidence_known_ns'] == sorted({known(a), known(b)}), 'Edge evidence clocks cover both originals')
        require(e['source_review_sha256'] is None, 'Producer remains intentionally non-admissible until bound')
        require(e['currency_policy_sha256'] == EXPECTED[POLICY] and e['currency'] == 'reported_dollar' and e['currency_basis'] == 'declared_within_issuer_reported_dollar_v1' and e['currency_iso'] is None and e['currency_iso_certified'] is False, 'Edge declared currency basis')
        expected_evidence = source_checks[earlier]['selected_note_evidence'] + source_checks[later]['selected_note_evidence']
        require(e['evidence'] == expected_evidence, 'Full source evidence included in each edge')
        for ev in e['evidence']:
            require(ev['report_id'] in all_ids and all_ids[ev['report_id']] in (earlier, later), 'No foreign edge evidence')
            verify_quote(ev, originals[all_ids[ev['report_id']]])
            require(ev['known_ns'] <= e['known_ns'], 'No backdated supporting note')
            quotes_checked += 1
        original, witness = a['revenue']['current'], b['revenue']['comparative']
        require(Decimal(original['amount_decimal']) == Decimal(witness['amount_decimal']), 'Exact corroborating comparative')
        require(e['original_amount'] == original['amount_decimal'] and e['later_comparative_amount'] == witness['amount_decimal'], 'Producer original/witness values')
        for field in ('period_start', 'period_end', 'measure', 'scale', 'currency_literal'):
            require(original[field] == witness[field], 'Same annual reported-total unit and period: ' + field)
        require(e['period_start'] == original['period_start'] and e['period_end'] == original['period_end'] and e['scale'] == '1000000', 'Producer units/period')
        require(e['earlier_label'] == original['measure_label'] and e['later_comparative_label'] == witness['measure_label'], 'Producer label transition')
        require(e['exact_decimal_equal'] is True and e['same_period_scale_measure'] is True, 'Producer numerical check results')
        require(not e['economic_homogeneity_certified'] and not e['organic_growth_certified'] and not e['exhaustive_amendment_history_verified'], 'Scope restrictions preserved')
        status = 'pending_identified_reclassification_note' if later == 2018 else 'checked_reported_total_edge'
        require(e['status'] == status, 'Independent disposition disagrees with producer')
        if later == 2018:
            require(e['missing_original_note'] is not None and e['missing_original_note']['no_known_revenue_break_asserted'], 'Unresolved is not a confirmed basis break')
        else:
            require(e['missing_original_note'] is None, 'Unexpected pending original note')
        admitted.append(dict(edge_id=e['edge_id'], issuer='ABBV', earlier_fiscal_year=earlier, later_fiscal_year=later,
                             earlier_report_id=e['earlier_report_id'], later_report_id=e['later_report_id'],
                             known_ns=e['known_ns'], known_by_at=e['known_by_at'], evidence_known_ns=e['evidence_known_ns'],
                             status=status, producer_status=e['status'], admitted=later != 2018,
                             source_review_sha256=EXPECTED[EDGES], independent_review_binding_required=True,
                             currency_policy_sha256=EXPECTED[POLICY], exact_comparative_correspondence_verified=True,
                             reason=REASONS[later], source_evidence=reports[earlier]['independent_evidence'] + reports[later]['independent_evidence'],
                             organic_growth_certified=False, economic_homogeneity_certified=False,
                             pending_requirements=e['missing_original_note']))
    read(str(Path(__file__)))
    for p, h in list(SEEN.items()):
        read(p, h)
    currency_review = dict(passed=True, issuer='ABBV', policy_sha256=EXPECTED[POLICY], currency='reported_dollar',
                           currency_basis='declared_within_issuer_reported_dollar_v1', currency_iso=None,
                           currency_iso_certified=False, sources_checked=13, monetary_cells_checked=26,
                           actual_source_preconditions_verified=True, within_issuer_assumption_admitted=True,
                           scope_fiscal_years=list(range(2013, 2026)),
                           evidence='Every selected annual original and comparative has its own verified dollar marker in the same formal consolidated statement with millions units. Original currency passages concern Shire transaction positions or explicitly operational constant-currency comparisons; those are not the selected nominal GAAP cells. No contradictory reporting-currency evidence was identified within the reviewed original releases.',
                           limitations=['Same-issuer reported dollars is the explicitly declared research assumption, not an independently certified ISO currency.', 'No cross-issuer aggregation, currency conversion, constant-currency replacement, or exhaustive historical currency-policy certification.', 'A later contradictory timed original note must block or revise admission prospectively.'])
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                stage='INDEPENDENT_BOUNDED_ORIGINAL_REPORTED_TOTAL_ACCOUNTING_EDGE_REVIEW',
                reviewed_edge_review_sha256=EXPECTED[EDGES], input_hashes=dict(sorted(SEEN.items())),
                actual_checks=dict(source_reports=13, monetary_cells=26, accounting_contexts_rechecked=sum(r['accounting_contexts_rechecked'] for r in reports.values()), producer_evidence_occurrences_rechecked=quotes_checked, source_and_receipt_hashes=True, original_issuer_identities_and_dates=True, edge_clocks_match_independent_source_availability=True, scoped_original_note_interpretations=True, producer_imports=0, evaluator_imports=0),
                currency_policy_review=currency_review, reports=list(reports.values()), edges=admitted,
                summary=dict(adjacent_edges=12, independently_admitted_reported_total_edges=11, pending_identified_reclassification_edges=1, confirmed_revenue_basis_breaks=0),
                source_scope=dict(original_release_bytes=True, current_archive_assumption=True, first_public_version_bytes_certified=False, exhaustive_amendment_history_verified=False, prepared_2013_clock_gap_retained=True, original_2013_clock_available_only_via_bound_independent_review=True),
                caller_requirements=['Verify this artifact and all its input digests, and bind this independent JSON digest before assigning source_review_sha256 in the evaluator.', 'Use the explicitly admitted status for exactly the matching adjacent original report IDs; FY2017 to FY2018 remains pending.', 'Independently enforce four-original membership, decision cutoffs and currency-policy digest; no source-chain rows are admitted by this audit alone.', 'Map the FY2013 clock only from the bound independent source review without changing the frozen null extraction clock.'],
                limits=['Affirmative annual GAAP statement scope plus individually reviewed accounting notes supports nominal reported-total continuity; matching values or keyword absence alone never admits an edge.', 'Acquisitions, divestitures, settlements, milestone recognition, and uneven consolidation durations remain in the original totals. No organic or economically homogeneous growth is certified.', 'The unresolved 2018 note requires timed primary evidence describing the prior-period reclassification and revenue-accounting transition. Any later availability must advance the edge clock.'],
                activity_counts=dict(new_source_chain_rows=0, new_revenue_ratios=0, new_factors=0, new_returns=0, new_models=0, new_trials=0, observer_writes=0))

def main(mode):
    require(mode in ('audit', 'verify'), 'Use audit, verify, inspect or inspect_currency')
    if mode == 'audit':
        require(not (ROOT / OUT).exists(), 'Independent accounting audit is write-once')
    result = audit()
    if mode == 'audit':
        raw = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
        with (ROOT / OUT).open('xb') as f:
            f.write(raw)
    else:
        raw = (ROOT / OUT).read_bytes()
        saved = json.loads(raw)
        result.pop('created_utc')
        saved.pop('created_utc')
        require(saved == result, 'Independent audit replay differs from saved artifact')
    print(json.dumps(dict(path=OUT, sha256=sha(raw), passed=True, mode=mode, summary=result['summary'], currency_policy_review_passed=result['currency_policy_review']['passed'])))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('inspect', 'inspect_currency'):
        inspect([int(x) for x in sys.argv[2:]], sys.argv[1] == 'inspect_currency')
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else 'audit')
