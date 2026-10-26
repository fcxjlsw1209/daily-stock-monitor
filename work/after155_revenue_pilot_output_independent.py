"""Independent source-grid audit. Never imports the production driver/evaluator."""
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = 'outputs/monthly_revenue_pilot_source156/'
OUT = 'work/after155_revenue_pilot_output_independent.json'
TEST = 'work/test_after155_revenue_pilot_output_independent.py'
PLAN = 'work/after155_revenue_pilot_source_plan.json'
POLICY = 'work/after155_multi_issuer_chain_policy.json'
META = 'outputs/monthly_local_linear70/META.json'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
EXTRACT = 'outputs/monthly_abbv_revenue156/EXTRACTION_V2.json'
PROPOSAL = 'work/after155_abbv_accounting_edges.json'
MSFT = 'outputs/monthly_msft_revenue_chain156/'
USABLE = 'checked_reported_total_chain_with_declared_source_scope'
BASIS = 'declared_within_issuer_reported_dollar_v1'
VINTAGE = 'original_dated_releases_retrieved_later_not_certified_first_public_bytes'
PINS = {
    PLAN: '5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10',
    POLICY: '25820e3cdb6273a9bedadfc8253a415bc2f6515b3d61e9d59ab7a465cdc0acc0',
    META: '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    CALENDAR: '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    EXTRACT: '5c81e1e23eefda6118739b938cd06074119ad9083f827e499cb5567ffe8f93fd',
    PROPOSAL: '115a6e1d48502a87de34ee0376dfdedea7936eac5320dda353f6dd7226ad18ec',
}
REVIEW_PINS = {
    'msft': ('work/after155_msft_chain_output_independent.json', '8dd79daa2891cd093fb16dd01015be92e59d38c9545d4ff04a3e45826ae4dadc'),
    'abbv_source': ('work/after155_abbv_revenue_independent.json', '0fc8b40786cfb3df532d94700ac46a7d78ae416d1f39a7a3bd0c3faee08a8f9d'),
    'abbv_accounting': ('work/after155_abbv_accounting_independent.json', '2cadcda44b07096ac02ad6c60ac81ef881c2a33ee840b4d6aa5492964efb22be'),
    'chain': ('work/after155_multi_issuer_chain_review.json', '1aabe00960b8c09c3794ad2d0bc9de2d03159fadaddb7acb211a07ea32232e8c'),
    'gate': ('work/after155_revenue_coverage_independent.json', 'c260e8ba0cbee0e637131c1c4cf04ab02f08b11ab8d539e2e02249f6a1da719c'),
    'driver': ('work/after155_revenue_pilot_driver_independent.json', '9d8e4c4f8adc7e70132a591d1c8c6b2088b390e99f0ec59edba64441cd914d90'),
}
HASHES = {}

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, expected=None, size=None):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    require(p.resolve() == p and ROOT in p.parents, 'Nonlocal/redirected input: ' + str(path))
    key = str(p.relative_to(ROOT))
    raw = p.read_bytes()
    h = sha(raw)
    require(expected is None or h == expected, 'Hash mismatch: ' + key)
    require(size is None or size == len(raw), 'Size mismatch: ' + key)
    require(key not in HASHES or HASHES[key] == h, 'Input changed: ' + key)
    HASHES[key] = h
    return raw

def load(path, expected=None):
    return json.loads(read(path, expected if expected is not None else PINS.get(path)))

def utc_ns(value):
    t = datetime.fromisoformat(value)
    require(t.utcoffset() is not None and t.microsecond == 0, 'Timezone/whole-second clock')
    return int(t.timestamp()) * 10**9

def catalog_by(items, key):
    require(isinstance(items, list), 'Catalog is not a list')
    result = {x[key]: x for x in items}
    require(len(result) == len(items), 'Duplicate catalog key: ' + key)
    return result

def own_oracle(records, boundaries, cutoff, decision_day):
    """Small independent state machine on source-derived records, not output rows.

    Values/periods/source bindings are checked separately against original cells.
    This oracle decides latest-first membership, timeliness and boundary status.
    It deliberately preserves partial member/edge provenance on a hold.
    """
    answer = dict(status='unresolved_source_or_comparability', reasons=[], latest_report_id=None,
                  report_ids=[], edge_ids=[])
    def finish(status, reason=None):
        answer['status'] = status
        answer['reasons'] = [] if reason is None else [reason]
        return answer
    visible = sorted((r for r in records if r['known'] <= cutoff), key=lambda r: r['known'], reverse=True)
    if not visible:
        return finish('missing_required_source', 'no_known_original_annual_report')
    if len(visible) > 1 and visible[0]['known'] == visible[1]['known']:
        return finish('unresolved_source_or_comparability', 'conflicting_latest_release')
    newest = visible[0]
    answer['latest_report_id'] = newest['id']
    if not newest['checked']:
        return finish('unresolved_source_or_comparability', 'latest_source_not_checked')
    if max(r['year'] for r in visible) > newest['year']:
        return finish('unresolved_source_or_comparability', 'later_release_for_older_year_requires_review')
    age = (date.fromisoformat(decision_day) - date.fromisoformat(newest['end'])).days
    if age < 0:
        return finish('unresolved_source_or_comparability', 'future_latest_period')
    if age > 550:
        return finish('stale_latest_annual', 'latest_annual_period_older_than_550_days')
    chosen = []
    for year in range(newest['year'] - 3, newest['year'] + 1):
        matches = [r for r in visible if r['year'] == year]
        if not matches:
            return finish('missing_required_source', 'missing_own_year_' + str(year))
        if len(matches) > 1:
            return finish('unresolved_source_or_comparability', 'conflicting_own_year_' + str(year))
        r = matches[0]
        if not r['checked']:
            return finish('unresolved_source_or_comparability', 'invalid_or_unreviewed_member_' + str(year))
        chosen.append(r)
        answer['report_ids'].append(r['id'])
    for position in range(1, 4):
        old, new = chosen[position - 1], chosen[position]
        matches = [e for e in boundaries if e['years'] == (old['year'], new['year']) and e['known'] <= cutoff]
        if len(matches) != 1:
            return finish('unresolved_source_or_comparability', 'missing_or_conflicting_edge_' + str(new['year']))
        edge = matches[0]
        answer['edge_ids'].append(edge['id'])
        if (edge['reports'] != (old['id'], new['id']) or edge['known'] < max(old['known'], new['known'])
                or not edge['evidence'] or any(t > edge['known'] for t in edge['evidence'])):
            return finish('unresolved_source_or_comparability', 'unbound_or_mistimed_edge_' + str(new['year']))
        if edge['status'] == 'blocked_known_basis_break':
            return finish('blocked_known_basis_break', edge['id'])
        if edge['status'] != 'checked_reported_total_edge':
            return finish('unresolved_source_or_comparability', 'unreviewed_edge_' + str(new['year']))
        if Decimal(new['witness']) != Decimal(old['amount']):
            return finish('blocked_known_basis_break', 'comparative_mismatch_' + str(new['year']))
    return finish(USABLE)

def independent_gate(meta, rows, plan):
    """Recompute coverage by per-month sets and issuer counters."""
    issuers = plan['fixed_issuer_scope']
    limits = plan['gate_before_strategy_registration']
    by_month = {i: set() for i in range(122)}
    buckets = {issuer: [] for issuer in issuers}
    for row in rows:
        if row['symbol'] in buckets:
            buckets[row['symbol']].append(row)
        if row['status'] == USABLE:
            require(row['symbol'] in issuers and row['symbol'] not in meta['etfs'], 'Usable nonpilot/ETF row')
            by_month[row['signal_index']].add(row['symbol'])
    segments = {}
    for name, indexes in [('early53', range(53)), ('recent35', range(53, 88)),
                          ('first88', range(88)), ('later34', range(88, 122)), ('all122', range(122))]:
        segment = {}
        for issuer in issuers:
            selected = [r for r in buckets[issuer] if r['signal_index'] in indexes]
            usable = [r for r in selected if r['status'] == USABLE]
            segment[issuer] = dict(candidate_rows=len(selected), usable_rows=len(usable),
                distinct_latest_endpoints=len(set(r['latest_report_id'] for r in usable)),
                status_counts=dict(Counter(r['status'] for r in selected)))
        segments[name] = dict(issuers=segment, cutoffs_with_minimum_issuers=[meta['dates'][i] for i in indexes
            if len(by_month[i]) >= limits['minimum_issuers']])
    qualifying = []
    for issuer in issuers:
        c = segments['first88']['issuers'][issuer]
        if (c['usable_rows'] >= limits['minimum_usable_candidate_rows_per_qualifying_issuer_in_first88']
                and c['distinct_latest_endpoints'] >= limits['minimum_distinct_latest_original_annual_endpoints_per_qualifying_issuer_in_first88']):
            qualifying.append(issuer)
    overlap = [meta['dates'][i] for i in range(88)
               if len(by_month[i].intersection(qualifying)) >= limits['minimum_issuers']]
    return dict(source_coverage_floor_passed=len(qualifying) >= limits['minimum_issuers'] and
                len(overlap) >= limits['minimum_first88_cutoffs_with_at_least_two_usable_issuers'],
                qualifying_issuers=qualifying, qualifying_joint_cutoffs=overlap, segments=segments,
                rows_verified=len(rows), source_reviews_verified_by_this_function=False,
                financial_values_or_returns_computed=False, strategy_registration_authorized_by_this_function=False)

def build_calendar():
    frozen = load(CALENDAR)
    path = 'outputs/monthly_earnings_repair/research_calendar.py'
    require(path in HASHES, 'Unbound calendar code')
    spec = importlib.util.spec_from_file_location('independent_source_output_calendar', ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exchange = module.market_calendar(frozen['start'], frozen['end'])
    days = [str(d.date()) for d in exchange.sessions]
    require(days == frozen['sessions'], 'Independent frozen calendar differs')
    return days, {day: exchange.session_close(day) for day in days}

def audit_catalog(outputs, reviews, days, closes):
    extracted = load(EXTRACT)['reports']
    source_review = catalog_by(reviews['abbv_source']['reports'], 'fiscal_year')
    report_map = catalog_by(outputs['REPORTS.json']['ABBV'], 'fiscal_year')
    require(sorted(report_map) == list(range(2013, 2026)), 'Exact ABBV original catalog required')
    records = []
    for original in extracted:
        year = original['fiscal_year']
        observed = report_map[year]
        source = original['source']
        clock = original['publication_clock']
        check = source_review[year]
        c, w = original['revenue']['current'], original['revenue']['comparative']
        release = clock['release_date']
        available = days[bisect_right(days, release)]
        close = closes[available]
        require(check['independent_available_session'] == available and check['independent_known_by_close'] == close.isoformat(), 'Source date/calendar review disagreement')
        require(check['source_path'] == source['raw']['path'] and check['source_sha256'] == source['raw']['sha256'], 'Independent source identity')
        require(check['receipt_verified'] and check['reported_gaap_policy_verified'], 'Source review permission')
        for key in ('raw', 'receipt'):
            bound = source[key]
            read(bound['path'], bound['sha256'], bound['bytes'])
        for role, fact, fact_year in [('current', c, year), ('comparative', w, year - 1)]:
            cc = check['cells'][role]
            require(cc['cell_sha256'] == fact['cell_html_sha256'] and cc['fiscal_year'] == fact_year and
                    cc['exact_amount_verified'] and cc['period_verified'] and cc['own_annual_column_dollar_evidence_verified'], 'Unverified source monetary cell')
            require(fact['period_start'] == '%s-01-01' % fact_year and fact['period_end'] == '%s-12-31' % fact_year,
                    'Original annual calendar period')
            require(fact['scale'] == 1000000 and fact['currency_literal'] == '$' and fact['currency_iso'] is None, 'Original unit basis')
            require(Decimal(fact['amount_decimal']).is_finite() and Decimal(fact['amount_decimal']) > 0, 'Original finite positive amount')
        expected = dict(issuer='ABBV', fiscal_year=year, report_id=source['metadata']['source_id'],
            source_checked=True, currency_basis_checked=True, source_sha256=source['raw']['sha256'],
            source_review_sha256=REVIEW_PINS['abbv_source'][1], currency_policy_sha256=PINS[POLICY],
            currency_review_sha256=REVIEW_PINS['abbv_accounting'][1], current=c['amount_decimal'],
            scale=str(c['scale']), period_start=c['period_start'], period_end=c['period_end'],
            measure='consolidated_gaap_total_revenue', comparative=w['amount_decimal'],
            comparative_scale=str(w['scale']), comparative_period_start=w['period_start'],
            comparative_period_end=w['period_end'], comparative_measure='consolidated_gaap_total_revenue',
            comparative_source_report_id=source['metadata']['source_id'], release_date=release,
            known_by=close.isoformat(), known_ns=utc_ns(close.isoformat()), original_publication_clock=clock,
            source=source, original_holds_preserved=original['holds'], archival_vintage_assumption=VINTAGE,
            historical_first_public_bytes_verified=False, economic_homogeneity_certified=False,
            retrieved_at=clock['retrieved_at'], current_cell_html_sha256=c['cell_html_sha256'],
            comparative_cell_html_sha256=w['cell_html_sha256'], current_locator=c['locator'],
            comparative_locator=w['locator'], period_derivation=c['period_derivation'],
            measure_label=c['measure_label'], source_currency_status=c['currency_status'])
        for prefix in ('', 'comparative_'):
            expected.update({prefix+'currency': 'reported_dollar', prefix+'currency_basis': BASIS,
                prefix+'currency_iso': None, prefix+'currency_iso_certified': False})
        require(observed == expected, 'Normalized ABBV report differs from independent source reconstruction: ' + str(year))
        records.append(dict(year=year, id=source['metadata']['source_id'], known=utc_ns(close.isoformat()),
            end=c['period_end'], checked=True, amount=c['amount_decimal'], witness=w['amount_decimal']))
    require(len({r['id'] for r in records}) == 13, 'Unique original report identities')
    require(report_map[2013]['original_publication_clock']['available_session'] is None, 'Original early clock was overwritten')
    proposal = catalog_by(load(PROPOSAL)['edges'], 'edge_id')
    decisions = catalog_by(reviews['abbv_accounting']['edges'], 'edge_id')
    observed_edges = catalog_by(outputs['EDGES.json']['ABBV'], 'edge_id')
    require(set(proposal) == set(decisions) == set(observed_edges) and len(proposal) == 12, 'Exact accounting-edge coverage')
    by_id = {r['id']: r for r in records}
    boundaries = []
    for edge_id, source in proposal.items():
        verdict, observed = decisions[edge_id], observed_edges[edge_id]
        earlier, later = source['earlier_fiscal_year'], source['later_fiscal_year']
        require(later == earlier + 1, 'Adjacent accounting years')
        expected_status = 'pending_identified_reclassification_note' if later == 2018 else 'checked_reported_total_edge'
        require(verdict['status'] == expected_status and verdict['admitted'] is (later != 2018), 'Independent accounting permission')
        expected = dict(source, status=expected_status, source_review_sha256=REVIEW_PINS['abbv_accounting'][1],
            producer_review_sha256=PINS[PROPOSAL], independent_disposition=verdict)
        require(observed == expected, 'Normalized edge differs from independently admitted disposition')
        a, b = by_id[source['earlier_report_id']], by_id[source['later_report_id']]
        require((a['year'], b['year']) == (earlier, later) and source['known_ns'] >= max(a['known'], b['known']), 'Edge identity/availability')
        for evidence in source['evidence']:
            associated = by_id[evidence['report_id']]
            require(associated['id'] in (a['id'], b['id']) and evidence['known_ns'] == associated['known'] <= source['known_ns'], 'Unrelated/future supporting evidence')
            read(evidence['source_path'], evidence['source_sha256'])
        require(Decimal(a['amount']) == Decimal(b['witness']), 'Independent exact comparative continuity')
        boundaries.append(dict(id=edge_id, years=(earlier, later), reports=(a['id'], b['id']),
            known=source['known_ns'], evidence=source['evidence_known_ns'], status=expected_status))
    return records, boundaries, report_map, observed_edges

def audit_rows(meta, rows, old_rows, records, boundaries, report_map, edge_map, closes, policy_id):
    require(len(meta['dates']) == len(meta['symbols']) == 122 and meta['dates'] == sorted(set(meta['dates'])), 'Original month axis')
    require(len(rows) == len(old_rows) == 122 * 49, 'All 5978 original slots required')
    require(all(len(names) == len(set(names)) == 49 for names in meta['symbols']), 'Original 49-candidate axis')
    old = {(r['signal_index'], r['candidate_index']): r for r in old_rows}
    require(len(old) == 122 * 49, 'Unique original MSFT grid')
    record_ids = {r['report_id']: r for r in report_map.values()}
    seen = set()
    decisions = []
    for position, row in enumerate(rows):
        i, j = divmod(position, 49)
        day, symbol = meta['dates'][i], meta['symbols'][i][j]
        require((row['signal_index'], row['candidate_index'], row['decision_date'], row['symbol']) == (i, j, day, symbol), 'Changed/reordered candidate slot')
        require((i, j) not in seen, 'Duplicate slot')
        seen.add((i, j))
        require(day in closes, 'Decision date outside frozen calendar')
        cutoff = utc_ns(closes[day].isoformat())
        require(row['decision_cutoff_ns'] == cutoff and row['decision_cutoff'] == closes[day].isoformat(), 'Decision cutoff mismatch')
        require(all(row[k] is False for k in ('factor_computed', 'factor_admitted', 'strategy_admitted')), 'Financial admission leaked into source-only grid')
        if symbol == 'MSFT':
            require(all(row.get(k) == v for k, v in old[i, j].items()), 'Existing MSFT contract/decision changed')
            extension = {'reused_source_row_file': MSFT+'ROWS.json',
                         'reused_source_row_file_sha256': HASHES[MSFT+'ROWS.json'],
                         'source_review_sha256s': {'msft': REVIEW_PINS['msft'][1]}}
            require(set(row) == set(old[i, j]) | set(extension) and all(row[k] == v for k, v in extension.items()), 'Unexpected MSFT extension')
            continue
        require(row['policy_id'] == policy_id, 'Wrong pilot policy')
        require(row['slot_id'] == '%03d:%02d' % (i, j), 'Slot identity')
        if symbol == 'ABBV':
            independent = own_oracle(records, boundaries, cutoff, day)
            require(all(row[k] == v for k, v in independent.items()), 'Independent ABBV decision differs: ' + day)
            selected = [r for r in report_map.values() if r['report_id'] in independent['report_ids']]
            edges = [e for e in edge_map.values() if e['edge_id'] in independent['edge_ids']]
            require(row['source_refs'] == [r['source']['raw'] for r in selected], 'Selected original source refs mismatch')
            require(row['per_report_known_by'] == {r['report_id']: r['known_by'] for r in selected}, 'Selected report clock provenance')
            require(row['per_edge_known_by'] == {e['edge_id']: e['known_by_at'] for e in edges}, 'Selected edge clock provenance')
            require(row['source_review_sha256s'] == {name: REVIEW_PINS[name][1] for name in ('abbv_source', 'abbv_accounting', 'chain')}, 'Per-row independent review binding')
            require(row['currency'] == 'reported_dollar' and row['currency_basis'] == BASIS and row['currency_iso'] is None and row['currency_iso_certified'] is False, 'Per-row assumed currency basis')
            require(row['archival_vintage_assumption'] == VINTAGE and row['economic_homogeneity_certified'] is False, 'Per-row source scope')
            if independent['status'] == USABLE:
                require(len(selected) == 4 and len(edges) == 3, 'Four originals and three boundaries required')
                require(all(r['known_ns'] <= cutoff for r in selected) and all(e['known_ns'] <= cutoff for e in edges), 'Future evidence selected')
            decisions.append(dict(decision_date=day, signal_index=i, **independent))
        else:
            require(row['report_ids'] == row['edge_ids'] == [] and row['latest_report_id'] is None, 'Nonadmitted issuer has source-chain provenance')
            if symbol == 'AAPL':
                require(row['status'] == 'unresolved_source_or_comparability' and row['reasons'] == ['AAPL_accounting_chain_not_independently_admitted'], 'AAPL pending accounting status changed')
            elif symbol in meta['etfs']:
                require(row['status'] == 'not_applicable_etf' and row['reasons'] == ['Original ETF candidate retained'], 'ETF not preserved')
            else:
                require(row['status'] == 'outside_fixed_revenue_pilot_scope' and row['reasons'] == ['Original candidate retained'], 'Out-of-scope original candidate not preserved')
    return decisions

def main(mode):
    require(mode in ('audit', 'verify'), 'Use audit or verify')
    if mode == 'audit':
        require(not (ROOT / OUT).exists(), 'Independent output audit is write-once')
    read(TEST)
    spec = importlib.util.spec_from_file_location('independent_output_oracle_tests', ROOT / TEST)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    log = io.StringIO()
    test_result = unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_module))
    require(test_result.wasSuccessful(), log.getvalue())
    manifest = load(BASE+'MANIFEST.json')
    require(manifest['source_only'] is True and manifest['strategy_registration'] is False, 'Manifest source-only scope')
    expected_files = {'ROWS.json', 'REPORTS.json', 'EDGES.json', 'COVERAGE_GATE.json', 'SUMMARY.json'}
    require(set(manifest['outputs']) == expected_files, 'Unexpected manifest output set')
    require({p.name for p in (ROOT / BASE).iterdir()} == expected_files | {'MANIFEST.json'}, 'Unexpected source output directory member')
    for path, h in manifest['input_hashes'].items():
        require(re.fullmatch(r'[0-9a-f]{64}', h) is not None, 'Malformed input digest')
        read(path, h)
    for path, h in PINS.items():
        read(path, h)
        require(manifest['input_hashes'].get(path) == h, 'Manifest omits fixed input')
    reviews = {}
    for name, (path, h) in REVIEW_PINS.items():
        require(manifest['review_paths'][name] == path and manifest['review_sha256s'][name] == h, 'Manifest review identity')
        reviews[name] = load(path, h)
        require(reviews[name]['passed'] is True, 'Independent source permission failed')
        for p, digest in reviews[name]['input_hashes'].items():
            require(manifest['input_hashes'].get(p) == digest, 'Manifest input closure omits review dependency')
            read(p, digest)
    outputs = {name: json.loads(read(BASE+name, spec['sha256'], spec['bytes'])) for name, spec in manifest['outputs'].items()}
    for name in ('REPORTS.json', 'EDGES.json'):
        require(set(outputs[name]) == {'MSFT', 'ABBV', 'AAPL'}, 'Pilot issuer catalog coverage')
        require(outputs[name]['MSFT'] == load(MSFT+name), 'MSFT existing source catalog changed')
        require(outputs[name]['AAPL'] == [], 'AAPL source chain admitted prematurely')
    plan, meta, policy = load(PLAN), load(META), load(POLICY)
    require(plan['fixed_issuer_scope'] == ['MSFT', 'AAPL', 'ABBV'], 'Fixed pilot scope')
    days, closes = build_calendar()
    records, boundaries, report_map, edge_map = audit_catalog(outputs, reviews, days, closes)
    decisions = audit_rows(meta, outputs['ROWS.json'], load(MSFT+'ROWS.json'), records, boundaries, report_map, edge_map, closes, policy['policy_id'])
    expected_gate = independent_gate(meta, outputs['ROWS.json'], plan)
    expected_gate.update(source_reviews_verified_by_preparation_driver=True,
        source_review_sha256s=manifest['review_sha256s'], strategy_registration_pending=True)
    require(outputs['COVERAGE_GATE.json'] == expected_gate, 'Independent common-coverage gate differs')
    summary = outputs['SUMMARY.json']
    require(summary['source_coverage_floor_passed'] == expected_gate['source_coverage_floor_passed'] and summary['rows'] == 5978, 'Summary gate/grid mismatch')
    require(summary['issuer_scope'] == plan['fixed_issuer_scope'] and summary['msft_contract_reused_unchanged'] is True and summary['aapl_accounting_chain_admitted'] is False, 'Summary source scope')
    require(summary['abbv_currency_policy_review'] == reviews['abbv_accounting']['currency_policy_review'], 'Summary actual-source currency permission')
    require(summary['parent_completed_round'] == plan['parent_completed_round'] == 155 and summary['parent_total_trials'] == plan['parent_total_trials'] == 978, 'Original research counters')
    require(all(summary[k] == 0 for k in ('new_strategy_trials', 'financial_factors_computed', 'returns_computed', 'models_fit', 'observer_writes')), 'Unexpected financial activity')
    require(summary['financial_factor_admitted'] is False and summary['strategy_registration_pending'] is True and summary['gate_pass_is_strategy_registration'] is False, 'Gate promoted into financial/strategy admission')
    read(TEST)
    read(str(Path(__file__)))
    for p, h in list(HASHES.items()):
        read(p, h)
    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        actual_grid_verified=True, input_hashes=dict(sorted(HASHES.items())),
        production_driver_or_evaluator_imports=0, production_gate_imports=0,
        checks=dict(candidate_rows=5978, decision_dates=122, candidates_per_date=49,
            independent_oracle_tests_passed=test_result.testsRun,
            exact_original_axis=True, msft_existing_contract_preserved=True, aapl_explicit_pending=True,
            abbv_original_reports=13, abbv_accounting_boundaries=12, held_2017_2018_preserved=True,
            original_source_catalog_reconstructed=True, current_values_never_replaced_by_later_comparatives=True,
            independent_calendar_closes=True, latest_known_original_before_validity=True,
            original_four_year_membership=True, all_supporting_evidence_available_by_cutoff=True,
            common_coverage_gate_recomputed=True, complete_manifest_input_and_output_closure=True),
        abbv_decisions=decisions, abbv_status_counts=dict(Counter(x['status'] for x in decisions)),
        gate_result=dict(source_coverage_floor_passed=expected_gate['source_coverage_floor_passed'],
            qualifying_issuers=expected_gate['qualifying_issuers'], qualifying_joint_cutoff_count=len(expected_gate['qualifying_joint_cutoffs']),
            segments=expected_gate['segments']),
        limitations=['Approval covers only the existing bounded original-source catalogs and declared within-issuer reported-dollar assumption.',
            'The unresolved FY2017 to FY2018 accounting edge remains unavailable; no organic-growth claim.',
            'Coverage feasibility is not a numerical factor, strategy registration or performance result.'],
        activity_counts=dict(new_source_grids=0, financial_factors=0, rankings=0, models=0, returns=0, strategies_registered=0, observer_writes=0))
    if mode == 'audit':
        raw = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
        with (ROOT / OUT).open('xb') as f:
            f.write(raw)
    else:
        raw = (ROOT / OUT).read_bytes()
        saved = json.loads(raw)
        saved.pop('created_utc')
        result.pop('created_utc')
        require(saved == result, 'Independent output audit replay differs')
    print(json.dumps(dict(path=OUT, sha256=sha(raw), passed=True, actual_grid_verified=True,
        input_files=len(HASHES), abbv_status_counts=result['abbv_status_counts'],
        qualifying_joint_cutoff_count=result['gate_result']['qualifying_joint_cutoff_count'])))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'audit')
