"""Package reviewed original annual source facts; no financial feature calculation.

Run only via run_local.sh. This builds no monthly query grid and imports no
strategy, model, accounting-return or price module. All output is write-once.
"""
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after219_roa_stability_source_admission_v1.py'
OUT = 'outputs/monthly_roa_stability_source220_v1'
PREFLIGHT = 'work/after219_roa_stability_source_preflight_v2.json'
PREFLIGHT_SHA = '547ff0f39f2fb037bc72266520ae33159dade14c90794b2f90ca3c84b0a3f206'
CONTEXT = 'work/after219_roa_stability_context_v2.json'
CONTEXT_SHA = 'a713c6ce62c8a6ec9608b495d8cbfbe19b474f4d9d9f40a3d3dd7acd27cf5e55'
CALENDAR = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
REVIEW_SPECS = {
    'work/after219_roa_stability_independent_source_review_v1.json': '108ed6009f26f5a7a99c74aa683eeb3aefbfa1f069d0d4eb70349456466a9d61',
    'work/after219_roa_stability_catalog_review_v1.json': '11d126e12aed82eda65c86fd8e7d6696c6d252334b7040b200f0dd78d9c559cb',
}
FIELDS = ('annual_net_income', 'opening_total_assets', 'closing_total_assets')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def version(kind, value):
    return kind + ':' + hashlib.sha256(canonical(value)).hexdigest()


def day(value):
    require(type(value) is str, 'Explicit source date required')
    d = date.fromisoformat(value)
    require(d.isoformat() == value, 'Canonical source date required')
    return d


class Reader:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        p = Path(path)
        p = p if p.is_absolute() else ROOT / p
        require(not p.is_symlink() and 'CloudStorage' not in p.parts, 'Forbidden source location')
        p = p.resolve()
        key = str(p.relative_to(ROOT))
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(expected is None or digest == expected, 'Source digest mismatch: ' + key)
        require(key not in self.hashes or self.hashes[key] == digest, 'Source changed during run: ' + key)
        self.hashes[key] = digest
        return data

    def load(self, path, expected=None):
        return json.loads(self.read(path, expected))


def amount(cell):
    """Literal monetary equality / sign checking only; never a ratio."""
    value, scale = cell['value_decimal'], cell['scale']
    require(type(value) is str and type(scale) is str, 'Explicit decimal strings required')
    require(Decimal(value).is_finite() and Decimal(scale).is_finite() and Decimal(scale) > 0,
            'Finite monetary literal and positive scale required')
    return Fraction(value) * Fraction(scale)


def field_cells(event, batch):
    f = event['fields']
    if batch == 0:
        return {
            'annual_net_income': ('fields.current.net_income', f['current']['net_income']),
            'opening_total_assets': ('fields.comparison.total_assets', f['comparison']['total_assets']),
            'closing_total_assets': ('fields.current.total_assets', f['current']['total_assets']),
        }
    return {
        'annual_net_income': ('fields.annual_net_income', f['annual_net_income']),
        'opening_total_assets': ('fields.prior_total_assets', f['prior_total_assets']),
        'closing_total_assets': ('fields.current_total_assets', f['current_total_assets']),
    }


def clock_record(row, sessions):
    released = row.get('release_date', row.get('publication_date'))
    day(released)
    supplied = row.get('available_from', row.get('available_session'))
    if supplied is not None:
        day(supplied)
    pos = bisect_right(sessions, released)
    covered = 0 < pos < len(sessions)
    expected = sessions[pos] if covered else None
    if covered:
        require(supplied == expected, 'Catalog next-session mismatch ' + row['event_id'])
    return dict(
        release_date=released, source_supplied_available_from=supplied,
        assumed_available_from=expected, first_pinned_next_session=expected,
        next_session_rule_verified=covered,
        status='verified_calendar_rule_only' if covered else 'outside_pinned_calendar',
        inherited_clock_review_present='clock_review' in row,
        inherited_clock_review=deepcopy(row.get('clock_review')),
        calendar_ref=CALENDAR, calendar_start=sessions[0], calendar_end=sessions[-1],
        assumption='Original document date plus first pinned following trading session; not verified first-publication time.',
        first_public_time_verified=False, first_public_version_verified=False,
        complete_correction_history_verified=False,
    )


def source_evidence(report):
    return dict(event_id=report['event_id'], report_version_id=report['report_version_id'],
                source=deepcopy(report['source']),
                assumed_available_from=report['available_from'],
                known_at_is_assumption=True)


def main():
    reader = Reader()
    require(len(REVIEW_SPECS) == 2, 'Two final independent review artifacts must be pinned before execution')
    preflight = reader.load(PREFLIGHT, PREFLIGHT_SHA)
    reader.load(CONTEXT, CONTEXT_SHA)
    reader.read(SELF)
    for path, digest in preflight['input_sha256'].items():
        reader.read(path, digest)
    reviews = {path: reader.load(path, digest) for path, digest in REVIEW_SPECS.items()}
    for review in reviews.values():
        for map_key in ('input_sha256', 'verified_preflight_input_sha256', 'additional_hash_bound_prior_visual_reviews'):
            for path, ref in review.get(map_key, {}).items():
                reader.read(path, ref if isinstance(ref, str) else ref['sha256'])
    source_review = reviews['work/after219_roa_stability_independent_source_review_v1.json']
    require(source_review['source_preflight']['sha256'] == PREFLIGHT_SHA,
            'Independent source review must bind final preflight')
    require(reviews['work/after219_roa_stability_catalog_review_v1.json']['passed'] is True,
            'Independent catalog review must pass its source-only scope')
    sessions = reader.load(CALENDAR)['sessions']
    require(sessions == sorted(set(sessions)), 'Unique ascending pinned sessions required')
    for s in sessions:
        day(s)
    require(preflight['counts']['reports'] == 91 and preflight['counts']['factor_admitted_chains'] == 0,
            'Unexpected preflight scope')
    reviewed_at = datetime.now(timezone.utc).isoformat()
    chain_ids = sorted(c['chain_id'] for c in preflight['candidate_chains']
                       if c['preflight_class'] == 'B_reported_fields_match_conditional_scope_only')
    require(len(chain_ids) == 40, 'Expected exact reviewed forty B chains')
    selected_chains = [c for c in preflight['candidate_chains'] if c['chain_id'] in chain_ids]
    member_ids = {x for c in selected_chains for x in c['event_ids']}
    admitted_edge_ids = {x for c in selected_chains for x in c['edge_ids']}
    require(len(member_ids) == 62 and len(admitted_edge_ids) == 51, 'Reviewed report/edge membership mismatch')
    require(chain_ids == sorted(c['chain_id'] for c in source_review['chains']), 'Exact independent chain list mismatch')
    require(member_ids == {r['event_id'] for r in source_review['reports']}, 'Exact independent report list mismatch')
    require(admitted_edge_ids == {e['edge_id'] for e in source_review['edges']}, 'Exact independent edge list mismatch')

    # Preserve all processed and unprocessed records; no hash-based event deduplication.
    event_map, batch_map, package_rows = {}, {}, []
    for batch, package in enumerate(preflight['source_catalogs']):
        path = package['source_dataset']
        data = reader.load(path, preflight['input_sha256'][path])
        native = {e['event_id']: e for e in data['events']}
        require(len(native) == len(data['events']), 'Duplicate native event ID')
        for eid, event in native.items():
            require(eid not in event_map, 'Cross-package event ID duplicate')
            event_map[eid] = event
            batch_map[eid] = batch
        rows = data['events'] if batch == 0 else data['latest_event_catalog']
        package_rows.extend((batch, path, deepcopy(row)) for row in rows)
    require(len(package_rows) == 191, 'Complete supplied catalog must contain191 events')
    identity_registry = deepcopy(preflight['issuers'])
    allowed_identities = {(i['issuer_id'], i['security_id']) for i in identity_registry}
    require(len(allowed_identities) == 15, 'Fifteen exact declared source identities required')
    catalog, catalog_by_id, clock_receipts = [], {}, []
    for batch, package_path, original in package_rows:
        eid = original['event_id']
        require(eid not in catalog_by_id, 'Catalog event duplicate')
        symbol = original['symbol']
        fy = original['fiscal_year']
        require(type(fy) is int and type(symbol) is str and symbol.strip() == symbol and symbol,
                'Explicit fiscal year and source symbol required')
        native = event_map.get(eid)
        gate_key = 'cash_quality_admission' if batch == 0 else 'ni_assets_admission'
        gate = native.get(gate_key) if native is not None else None
        state = 'unprocessed' if native is None else 'admitted' if gate and gate.get('admitted') is True else 'held'
        period_end = (original.get('periods', {}).get('current', {}).get('period_end') if batch == 0
                      else original.get('official_actual_period_end'))
        if period_end is not None:
            day(period_end)
        anchor_status = 'explicit_fiscal_year_and_period_end' if period_end else 'unknown_period_end_blocks_selection'
        identity = deepcopy(native.get('identity')) if native is not None else None
        issuer_id = identity.get('issuer_id') if isinstance(identity, dict) else None
        security_id = identity.get('security_id') if isinstance(identity, dict) else None
        exact_identity = (issuer_id, security_id) in allowed_identities and security_id == symbol
        cr = clock_record(original, sessions)
        clock_receipts.append(dict(event_id=eid, **cr))
        payload = dict(event_id=eid, symbol=symbol, fiscal_year=fy, fiscal_period_end=period_end,
                       original_catalog=original, source_dataset=package_path)
        period_start = (native.get('periods', {}).get('current', {}).get('period_start') if native is not None else None)
        if period_start is not None:
            day(period_start)
        item = dict(payload, catalog_version_id=version('catalog', payload),
                    canonical_symbol=symbol, period_start=period_start, period_end=period_end,
                    publication_date=cr['release_date'], assumed_available_session=cr['assumed_available_from'],
                    clock_status=cr['status'], report_version_id=None, kind='original_annual_event',
                    corrects_report_id=None,
                    native_event_present=native is not None, inherited_field_admission=deepcopy(gate),
                    original_field_state=state, identity=identity,
                    issuer_id=issuer_id if exact_identity else None,
                    security_id=security_id if exact_identity else None,
                    issuer_identity_verified_for_join=exact_identity,
                    identity_status='explicit_existing_source_identity' if exact_identity else 'unknown_identity_blocker_only',
                    conservative_blocker_symbol=symbol, fiscal_anchor_status=anchor_status,
                    clock=cr, available_from=cr['assumed_available_from'],
                    source_only_admission_does_not_clear_unknown_catalog=True,
                    corrects_event_id=None, affected_fiscal_year=fy, affected_fiscal_period_end=period_end,
                    correction_links_verified=False, archived_at_utc=reviewed_at)
        catalog.append(item)
        catalog_by_id[eid] = item
    states = dict(Counter(c['original_field_state'] for c in catalog))
    require(states == {'admitted': 91, 'held': 38, 'unprocessed': 62}, 'Catalog original state counts mismatch')
    calendar_counts = dict(Counter(c['status'] for c in clock_receipts))
    require(calendar_counts == {'verified_calendar_rule_only': 161, 'outside_pinned_calendar': 30},
            'Calendar coverage counts mismatch')
    absent_clock = [c for c in clock_receipts if not c['inherited_clock_review_present']]
    require(len(absent_clock) == 44 and all(c['next_session_rule_verified'] for c in absent_clock),
            'Original44 clockless schema must be independently checked')

    reports = []
    reports_by_id = {}
    for old in preflight['reports']:
        eid = old['event_id']
        event = event_map[eid]
        cat = catalog_by_id[eid]
        require(cat['original_field_state'] == 'admitted' and cat['issuer_identity_verified_for_join'],
                'Admitted original report needs explicit source identity')
        require(cat['clock']['next_session_rule_verified'] and cat['available_from'] == old['assumed_known_session'],
                'Original report clock mismatch')
        require(cat['fiscal_period_end'] == old['period']['period_end'], 'Original report anchor mismatch')
        period = deepcopy(old['period'])
        start, end = day(period['period_start']), day(period['period_end'])
        require((end-start).days + 1 == period['days_inclusive'], 'Annual period length mismatch')
        require(end < day(cat['clock']['release_date']) < day(cat['available_from']), 'Report period/release clock mismatch')
        source = deepcopy(old['source'])
        reader.read(source['path'], source['sha256'])
        fields = {}
        for name, (field_path, cell) in field_cells(event, batch_map[eid]).items():
            require(cell is not None and cell['source']['path'] == source['path'] and
                    cell['source']['sha256'] == source['sha256'], 'Own-report field-source mismatch')
            require(cell.get('locator') is not None and cell['available_from'] == cat['available_from'],
                    'Own-report locator/field clock missing')
            require(cell['currency_literal'] == '$', 'Literal currency mismatch')
            numeric = amount(cell)
            if name != 'annual_net_income':
                require(numeric > 0, 'Positive source asset endpoint required')
            expected_date = (start-timedelta(days=1)).isoformat() if name == 'opening_total_assets' else end.isoformat()
            actual_date = cell.get('as_of') or (cell.get('period') or {}).get('period_end') or cell.get('locator', {}).get('period_end')
            require(actual_date == expected_date, 'Source field fiscal endpoint mismatch')
            fields[name] = dict(original_field_path=field_path, as_of=actual_date,
                                source_cell=deepcopy(cell), ni_attribution='reported_consolidated_total_NI_before_NCI_deduction' if name == 'annual_net_income' else None,
                                inherited_field_scope_review_ref=old['field_review_source'])
        require(len({x['source_cell'].get('currency_iso') for x in fields.values()}) == 1,
                'Inconsistent source ISO currency declaration')
        payload = dict(event_id=eid, issuer_id=old['issuer_id'], security_id=old['security_id'],
                       fiscal_year=old['fiscal_year'], period=period, source=source, fields=fields)
        rep = dict(payload, report_version_id=version('report', payload),
                   symbol=old['symbol'], catalog_version_id=cat['catalog_version_id'],
                   available_from=cat['available_from'], known_at=cat['available_from'], evidence_known_from=cat['available_from'],
                   known_at_is_original_date_next_session_assumption=True,
                   inherited_single_report_field_admission=deepcopy(old['declared_field_admission']),
                   inherited_single_report_fields_admitted=True,
                   member_of_reviewed_three_report_chain=eid in member_ids,
                   three_report_policy_is_separate_from_original_field_admission=True,
                   source_identity=deepcopy(old['identity_evidence']),
                   identity_source_use_window_not_permanent_issuer_life=True,
                   accounting_scope='as_reported_total_NI_and_same_original_report_asset_endpoints',
                   combined_and_consolidated_allowed=eid == 'PYPL_2016-01-27',
                   same_report_review=deepcopy(old['same_report_accounting_review']),
                   source_review_refs=[old['field_review_source']],
                   cross_report_review_refs=list(REVIEW_SPECS) if eid in member_ids else [],
                   source_field_reviews_preserved=True, reviewed_at_utc=reviewed_at,
                   first_public_version_verified=False, complete_correction_history_verified=False,
                   corrected_original_values_substituted=False,
                   supersedes_version_id=None, immutable=True)
        reports.append(rep)
        reports_by_id[eid] = rep
        cat['report_version_id'] = rep['report_version_id']
    require(len(reports) == 91, 'All91 original report admissions must be preserved')

    edges = []
    edge_by_id = {}
    for old in preflight['edges']:
        earlier = reports_by_id[old['earlier_event_id']]
        later = reports_by_id[old['later_event_id']]
        pair = (earlier, later)
        approved = old['edge_id'] in admitted_edge_ids
        require((earlier['issuer_id'], earlier['security_id']) == (later['issuer_id'], later['security_id']),
                'Cross-issuer edge')
        require(later['fiscal_year'] == earlier['fiscal_year'] + 1 and
                day(later['period']['period_start']) == day(earlier['period']['period_end']) + timedelta(days=1),
                'Consecutive annual edge required')
        if approved:
            require(old['preflight_class'] == 'B_reported_fields_match_conditional_scope_only', 'Only reviewed B edges')
            witness = old['comparative_ni_witness']
            require(witness is not None and witness['source']['sha256'] == later['source']['sha256'],
                    'Later comparative NI evidence required')
            require(amount(earlier['fields']['annual_net_income']['source_cell']) == amount(witness),
                    'Original/comparative NI mismatch')
            require(amount(earlier['fields']['closing_total_assets']['source_cell']) ==
                    amount(later['fields']['opening_total_assets']['source_cell']), 'Original/comparative assets mismatch')
        evidence = [source_evidence(r) for r in pair]
        evidence_known_from = max(e['assumed_available_from'] for e in evidence)
        payload = dict(edge_id=old['edge_id'], issuer_id=earlier['issuer_id'], security_id=earlier['security_id'],
                       earlier_report_version_id=earlier['report_version_id'], later_report_version_id=later['report_version_id'],
                       evidence_refs=evidence, comparative_ni_witness=deepcopy(old['comparative_ni_witness']),
                       conditional_admitted=approved, policy_scope='as_reported_original_NI_assets_three_annual_reports_v1')
        edge = dict(payload, edge_version_id=version('edge', payload),
                    earlier_event_id=earlier['event_id'], later_event_id=later['event_id'],
                    earlier_fiscal_year=earlier['fiscal_year'], later_fiscal_year=later['fiscal_year'],
                    evidence_known_from=evidence_known_from, available_from=evidence_known_from,
                    known_at=evidence_known_from, conditional_allowed=approved,
                    known_at_is_assumption=True, reviewed_at_utc=reviewed_at,
                    independent_review_refs=list(REVIEW_SPECS) if approved else [],
                    decision='conditional_as_reported_source_use' if approved else 'not_admitted',
                    preflight_class=old['preflight_class'], original_preflight_status=old['source_chain_status'],
                    original_preflight_reasons=deepcopy(old['reasons']),
                    NI_original_comparison=deepcopy(old['ni_original_vs_later_comparison']),
                    asset_original_comparison=deepcopy(old['assets_original_vs_later_comparison']),
                    broad_scope_alerts=deepcopy(old['broad_scope_alerts']),
                    manual_review_assumption='Comparative-equal original reported total NI/total-assets chain is accepted for observed as-reported stability, including disclosed real business-perimeter changes; not invariant accounting-policy counterfactual.',
                    constant_economic_perimeter_verified=False, first_public_version_verified=False,
                    complete_correction_history_verified=False,
                    supersedes_version_id=None, immutable=True,
                    later_revision_policy='Append a new evidence version with its own evidence_known_from; never replace this record or backdate future notes.')
        edges.append(edge)
        edge_by_id[edge['edge_id']] = edge
    require(sum(e['conditional_admitted'] for e in edges) == 51, 'Reviewed51 conditional edges required')

    chains = []
    for old in preflight['candidate_chains']:
        members = [reports_by_id[e] for e in old['event_ids']]
        used_edges = [edge_by_id[e] for e in old['edge_ids']]
        approved = old['chain_id'] in chain_ids
        require(not approved or all(e['conditional_admitted'] for e in used_edges), 'Unreviewed chain edge')
        nine = [dict(report_version_id=r['report_version_id'], event_id=r['event_id'], field=name,
                     source_sha256=r['source']['sha256'], field_path='fields.'+name,
                     original_field_path=r['fields'][name]['original_field_path'])
                for r in members for name in FIELDS]
        require(len(nine) == 9, 'Exactly nine own-report field references')
        payload = dict(chain_id=old['chain_id'], issuer_id=old['issuer_id'], security_id=old['security_id'],
                       fiscal_years=old['fiscal_years'], report_version_ids=[r['report_version_id'] for r in members],
                       edge_version_ids=[e['edge_version_id'] for e in used_edges], nine_required_fields=nine,
                       conditional_admitted=approved)
        chain_known_at=max([r['available_from'] for r in members] + [e['evidence_known_from'] for e in used_edges])
        chains.append(dict(payload, chain_version_id=version('chain', payload),
                           conditional_allowed=approved, known_at=chain_known_at,
                           event_ids=old['event_ids'], edge_ids=old['edge_ids'],
                           latest_fiscal_period_end=members[-1]['period']['period_end'],
                           available_from=chain_known_at,
                           known_at_is_assumption=True, reviewed_at_utc=reviewed_at,
                           original_preflight_class=old['preflight_class'], original_preflight_status=old['status'],
                           status='conditional_as_reported_source_use' if approved else 'not_admitted',
                           independent_review_refs=list(REVIEW_SPECS) if approved else [],
                           broader_scope_alert_event_ids=old['broader_scope_alert_event_ids'],
                           monthly_query_eligibility_computed=False, first_public_version_verified=False,
                           complete_correction_history_verified=False, immutable=True, supersedes_version_id=None))
    require(sum(c['conditional_admitted'] for c in chains) == 40, 'Exactly40 conditional chains required')

    policy = dict(
        schema_version=1, method='as_reported_original_NI_assets_three_annual_reports_v1',
        created_utc=reviewed_at, status='CONDITIONAL_SOURCE_POLICY_NOT_STRATEGY_REGISTRATION',
        definition='Three consecutive own-year original-report total GAAP NI values, each paired with that same report opening and closing total assets. No future comparative values substitute for original observations.',
        scope='Only supplied191-event catalog for15 declared local source issuer/security pairs. This is not an exhaustive global original-release or correction history.',
        approval='40 exact B chains independently reviewed;91 original single-report field admissions preserved separately; all other chains remain unavailable.',
        accepts_disclosed_real_business_perimeter_changes=True,
        accepts_combined_and_consolidated_original_PYPL_FY2015=True,
        constant_business_perimeter_claim=False, unchanged_global_accounting_policy_claim=False,
        assumptions=['Original source publication date followed by first pinned trading session is assumed availability, not certified historical dissemination.',
                     'Source bytes were collected retrospectively; first-publication vintage and complete correction history are unverified.',
                     'Observed as-reported stability intentionally retains disclosed economic acquisitions/deconsolidations/carve-out effects; no constant-perimeter or counterfactual accounting normalization.',
                     'Exact existing issuer identities are source-scoped, not a certified permanent historical security master; unknown identities block rather than join by ticker.'],
        selection_protocol=dict(
            step1='From complete supplied catalog, identify known events for the requested verified issuer/security. Unknown identity rows associated with its symbol remain conservative blockers; never use them to join issuer values.',
            step2='Order annual anchors by fiscal_year and fiscal_period_end, not publication date. Unknown/inconsistent fiscal anchors block selection, rather than silently dropping a row.',
            step3='For the latest known fiscal anchor, held/unprocessed/ambiguous/unknown-clock/unknown-identity rows block fallback. Later publications about older fiscal years cannot replace this anchor.',
            step4='An older-period correction/review version only affects its own report/edge from the maximum availability of all its required evidence. Do not replace original numeric observations with later restated amounts.',
            step5='Require the exact latest anchor and its two preceding consecutive own-year reports plus two admitted evidence versions. No skipped year or convenient older chain.',
            step6='At query time require all9 fields and edges known, latest fiscal-period age <=550days, old reports checked at their own availability, and exact issuer/security mapping. This packaging step performs none of those monthly queries.',
            unknown_clock_knownness='For conservative blocking only, a source publication date before the query can establish that an unverified catalog row must not be silently ignored. It never admits values.',
            same_fiscal_anchor_ties='Ambiguous competing original/correction records require explicit version/corrects links; no arbitrary max-publication or lexicographic selection.'),
        negative_original_NI_permitted=True, positive_original_asset_endpoints_required=True,
        first_public_version_verified=False, complete_correction_history_verified=False,
        certified_daily_security_master=False, no_strategy_registration=True,
    )
    review_evidence = dict(schema_version=1, saved_at_utc=reviewed_at,
                          preflight=dict(path=PREFLIGHT, sha256=PREFLIGHT_SHA),
                          context=dict(path=CONTEXT, sha256=CONTEXT_SHA),
                          independent_reviews=[dict(path=p,sha256=REVIEW_SPECS[p],content=reviews[p]) for p in REVIEW_SPECS],
                          approved_chain_ids=chain_ids,
                          approved_edge_ids=sorted(admitted_edge_ids), reviewed_report_event_ids=sorted(member_ids),
                          numeric_checks_scope='Original monetary signs/units/endpoint identity and exact original-to-comparative equality only; no ratios.',
                          manual_interpretation_scope=policy['assumptions'],
                          review_timestamp_is_not_historical_publication_time=True)
    dataset = dict(schema_version=1, method=policy['method'], created_utc=reviewed_at,
                   conditional_research_only=True, first_public_version_verified=False,
                   complete_correction_history_verified=False, certified_daily_security_master=False,
                   identity_registry=identity_registry, catalog=catalog, reports=reports, edges=edges, chains=chains,
                   missing_chain_members=deepcopy(preflight['missing_chain_members']),
                   report_fields=FIELDS, source_pins=dict(sorted(reader.hashes.items())),
                   policy_ref=OUT+'/POLICY.json', review_ref=OUT+'/REVIEW_EVIDENCE.json')
    clock_review = dict(schema_version=1, created_utc=reviewed_at, calendar_ref=CALENDAR,
                        calendar_sha256=reader.hashes[CALENDAR], counts=calendar_counts,
                        inherited_clock_review_absent_count=len(absent_clock),
                        absent_clock_independently_verified_count=sum(c['next_session_rule_verified'] for c in absent_clock),
                        records=clock_receipts, first_public_time_verified=False,
                        no_outside_calendar_clock_invented=True, actual_signal_queries=0)
    validation = dict(schema_version=1, created_utc=reviewed_at, passed=True,
                      source_only=True, implementation_sha256=reader.hashes[SELF],
                      counts=dict(catalog_events=len(catalog), original_field_states=states,
                                  reports=len(reports), own_report_fields=len(reports)*3,
                                  conditionally_reviewed_member_reports=len(member_ids),
                                  edges=len(edges), conditional_edges=len(admitted_edge_ids),
                                  chains=len(chains), conditional_chains=len(chain_ids),
                                  chain_required_field_refs=len(chains)*9,
                                  calendar=calendar_counts, verified_input_files=len(reader.hashes)),
                      checks=dict(raw_digest_bindings=True,own_report_three_field_bindings=True,
                                  original_field_admissions_preserved=True,exact_identity_no_ticker_join=True,
                                  reported_NI_and_assets_checked_independently=True,
                                  admitted_edge_exact_scaled_source_equality=True,
                                  event_versions_not_deduplicated_by_raw_sha=True,
                                  complete_supplied_catalog_preserved=True,
                                  no_first_public_PIT_certification=True,
                                  ROA_calculations=0,means=0,standard_deviations=0,factors=0,ranks=0,
                                  monthly_support_matches=0,orders=0,returns=0,network_requests=0,
                                  source_admission_updates_outside_new_directory=0,observer_changes=0))
    products={'POLICY.json':policy,'DATASET.json':dataset,'REVIEW_EVIDENCE.json':review_evidence,
              'CLOCK_REVIEW.json':clock_review,'VALIDATION.json':validation}
    encoded={name:(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode() for name,data in products.items()}
    manifest=dict(schema_version=1,created_utc=reviewed_at,source_only=True,
                  files={name:hashlib.sha256(raw).hexdigest() for name,raw in encoded.items()},
                  input_sha256=dict(sorted(reader.hashes.items())),implementation_path=SELF,
                  implementation_sha256=reader.hashes[SELF],write_once=True)
    encoded['MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    target=ROOT/OUT
    require(not target.exists(),'Write-once source package already exists')
    target.mkdir()
    for name,raw in encoded.items():
        with (target/name).open('xb') as stream:
            stream.write(raw)
    print(json.dumps(dict(output=OUT,sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in encoded.items()},
                          counts=validation['counts']),ensure_ascii=False))


if __name__=='__main__':
    main()
