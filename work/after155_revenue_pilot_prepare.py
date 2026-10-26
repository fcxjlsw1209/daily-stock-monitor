"""Source-only pilot grid. Preparation requires exact independent-review bindings.

No financial factor, ranking, portfolio, model or outcome is computed here.
MSFT rows retain their reviewed contract; ABBV uses the separate reported-dollar
assumption; AAPL remains an explicit accounting-chain hold.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_revenue_pilot_source156'
PLAN = 'work/after155_revenue_pilot_source_plan.json'
META = 'outputs/monthly_local_linear70/META.json'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
POLICY = 'work/after155_multi_issuer_chain_policy.json'
EXTRACT = 'outputs/monthly_abbv_revenue156/EXTRACTION_V2.json'
EDGE_PROPOSAL = 'work/after155_abbv_accounting_edges.json'
MSFT_BASE = 'outputs/monthly_msft_revenue_chain156/'
SELF = 'work/after155_revenue_pilot_prepare.py'
TEST = 'work/test_after155_revenue_pilot_prepare.py'
REVIEW_PATHS = {
    'msft': 'work/after155_msft_chain_output_independent.json',
    'abbv_source': 'work/after155_abbv_revenue_independent.json',
    'abbv_accounting': 'work/after155_abbv_accounting_independent.json',
    'chain': 'work/after155_multi_issuer_chain_review.json',
    'gate': 'work/after155_revenue_coverage_independent.json',
    'driver': 'work/after155_revenue_pilot_driver_independent.json',
}
PINS = {
    PLAN: '5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10',
    META: '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    CALENDAR: '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    POLICY: '25820e3cdb6273a9bedadfc8253a415bc2f6515b3d61e9d59ab7a465cdc0acc0',
    EXTRACT: '5c81e1e23eefda6118739b938cd06074119ad9083f827e499cb5567ffe8f93fd',
}
USABLE = 'checked_reported_total_chain_with_declared_source_scope'
DOLLAR_BASIS = 'declared_within_issuer_reported_dollar_v1'
VINTAGE = 'original_dated_releases_retrieved_later_not_certified_first_public_bytes'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


class Inputs:
    """Exact local byte closure, validated again before any output is written."""
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.hashes = {}

    def local(self, path):
        p = Path(path)
        p = p if p.is_absolute() else self.root / p
        require(p.is_relative_to(self.root) and p.resolve() == p,
                'Nonlocal, escaping or symlink input: ' + str(path))
        return p

    def read(self, path, expected=None):
        p = self.local(path)
        key = str(p.relative_to(self.root))
        raw = p.read_bytes()
        h = digest(raw)
        require(expected is None or (valid_digest(expected) and h == expected),
                'Input SHA-256 mismatch: ' + key)
        require(key not in self.hashes or self.hashes[key] == h,
                'Input changed during preparation: ' + key)
        self.hashes[key] = h
        return raw

    def load(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def closure(self, hashes):
        require(isinstance(hashes, dict) and hashes, 'Missing independent input closure')
        for path, h in hashes.items():
            require(valid_digest(h), 'Invalid review input digest: ' + path)
            self.read(path, h)

    def bound(self, review, *paths):
        for path in paths:
            key = str(self.local(path).relative_to(self.root))
            require(review.get('input_hashes', {}).get(key) == self.hashes.get(key)
                    and key in self.hashes, 'Review does not bind required input: ' + key)

    def unchanged(self):
        for path, h in list(self.hashes.items()):
            self.read(path, h)


def review_inputs(inputs, review_shas, review_paths=None, require_driver=True):
    paths = dict(REVIEW_PATHS)
    paths.update(review_paths or {})
    for path, h in PINS.items():
        inputs.read(path, h)
    reviews = {}
    names = list(REVIEW_PATHS) if require_driver else [n for n in REVIEW_PATHS if n != 'driver']
    for name in names:
        require(valid_digest(review_shas.get(name)), 'Required independent review SHA missing: ' + name)
        review = inputs.load(paths[name], review_shas[name])
        require(review.get('passed') is True, 'Independent review not passed: ' + name)
        inputs.closure(review.get('input_hashes'))
        reviews[name] = review
    required = {
        'msft': [META, CALENDAR, MSFT_BASE+'ROWS.json', MSFT_BASE+'REPORTS.json', MSFT_BASE+'EDGES.json'],
        'abbv_source': [EXTRACT, CALENDAR],
        'abbv_accounting': [EDGE_PROPOSAL, EXTRACT, POLICY, paths['abbv_source']],
        'chain': ['work/after155_multi_issuer_chain.py', POLICY, PLAN],
        'gate': ['work/after155_revenue_coverage_gate.py', PLAN],
        'driver': [SELF, TEST, POLICY, PLAN],
    }
    for name in names:
        inputs.bound(reviews[name], *required[name])
    source_review = reviews['abbv_source']
    require(source_review.get('admission', {}).get('source_cells_verified') is True,
            'ABBV source review did not verify cells')
    for path, h in inputs.load(PLAN)['reference_files'].items():
        inputs.read(path, h)
    return reviews, paths


def calendar(inputs):
    """Regenerate exchange closes using the reviewed frozen session axis."""
    implementation = 'outputs/monthly_earnings_repair/research_calendar.py'
    require(implementation in inputs.hashes, 'Calendar implementation lacks independent binding')
    spec = importlib.util.spec_from_file_location('pilot_source_calendar156', inputs.local(implementation))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = inputs.load(CALENDAR)
    cal = module.market_calendar(frozen['start'], frozen['end'])
    days = [str(day.date()) for day in cal.schedule.index]
    require(days == frozen['sessions'], 'Frozen 143 session axis mismatch')
    return days, dict(zip(days, cal.schedule['close']))


def currency_review(review, policy_sha):
    c = review.get('currency_policy_review', review.get('currency', {}))
    require(c.get('policy_sha256') == policy_sha, 'Actual-source currency policy not bound')
    require(c.get('actual_source_preconditions_verified') is True
            and c.get('within_issuer_assumption_admitted') is True,
            'Actual-source reported-dollar assumption not reviewed/admitted')
    require(c.get('currency') == 'reported_dollar' and c.get('currency_basis') == DOLLAR_BASIS
            and c.get('currency_iso') is None and c.get('currency_iso_certified') is False,
            'Reported dollars cannot be represented as certified ISO currency')
    require(c.get('scope_fiscal_years') == list(range(2013, 2026)),
            'Actual-source dollar review does not cover bounded ABBV catalog')
    return deepcopy(c)


def unique_by(records, key, name):
    require(isinstance(records, list) and all(isinstance(r, dict) for r in records), 'Invalid '+name)
    result = {r[key]: r for r in records}
    require(len(result) == len(records), 'Duplicate '+name)
    return result


def normalize_abbv(extraction, source_review, accounting_review, source_sha,
                   accounting_sha, policy_sha, days, closes):
    """Use reviewed exact source cells; never replace an original with a witness."""
    currency_review(accounting_review, policy_sha)
    raw = unique_by(extraction['reports'], 'fiscal_year', 'ABBV original year')
    checked = unique_by(source_review['reports'], 'fiscal_year', 'ABBV reviewed year')
    years = list(range(2013, 2026))
    require(sorted(raw) == sorted(checked) == years, 'Incomplete bounded ABBV release catalog')
    reports = []
    for year in years:
        r, check = raw[year], checked[year]
        src, clock = r['source'], r['publication_clock']
        c, w = r['revenue']['current'], r['revenue']['comparative']
        require(r['symbol'] == src['metadata']['symbol'] == 'ABBV', 'Wrong ABBV source issuer')
        require(check['source_path'] == src['raw']['path'] and check['source_sha256'] == src['raw']['sha256'],
                'ABBV independent source identity mismatch')
        require(check.get('receipt_verified') is True and check.get('reported_gaap_policy_verified') is True,
                'ABBV receipt or reported GAAP policy not verified')
        release = clock['release_date']
        session = next((d for d in days if d > release), None)
        require(session is not None and session == check['independent_available_session'],
                'ABBV independent next-session mismatch')
        close = closes[session]
        require(close.isoformat() == check['independent_known_by_close'] and release == check['release_date'],
                'ABBV independent close mismatch')
        require(clock['precision'] == 'date_only', 'Unexpected ABBV source clock precision')
        for role, cell in [('current', c), ('comparative', w)]:
            verified = check['cells'][role]
            require(all(verified.get(k) is True for k in ['exact_amount_verified', 'period_verified',
                    'own_annual_column_dollar_evidence_verified']), 'Unreviewed ABBV '+role+' cell')
            require(verified['cell_sha256'] == cell['cell_html_sha256']
                    and verified['scale'] == cell['scale']
                    and verified['fiscal_year'] == cell['fact_year'], 'ABBV cell review binding mismatch')
            require(cell['currency_iso'] is None and cell['currency_literal'] == '$',
                    'Unexpected ABBV literal currency')
            require(cell['source_raw_sha256'] == src['raw']['sha256'], 'ABBV monetary source mismatch')
            require(cell['measure'] == 'consolidated_total_net_revenue', 'Unexpected source revenue measure')
        require(c['fact_year'] == year and w['fact_year'] == year-1,
                'Original/current comparative fiscal-year identity mismatch')
        report_id = src['metadata']['source_id']
        report = dict(issuer='ABBV', fiscal_year=year, report_id=report_id,
            source_checked=True, currency_basis_checked=True,
            source_sha256=src['raw']['sha256'], source_review_sha256=source_sha,
            currency_policy_sha256=policy_sha, currency_review_sha256=accounting_sha,
            current=c['amount_decimal'], scale=str(c['scale']),
            period_start=c['period_start'], period_end=c['period_end'],
            measure='consolidated_gaap_total_revenue',
            comparative=w['amount_decimal'], comparative_scale=str(w['scale']),
            comparative_period_start=w['period_start'], comparative_period_end=w['period_end'],
            comparative_measure='consolidated_gaap_total_revenue', comparative_source_report_id=report_id,
            release_date=release, known_by=close.isoformat(), known_ns=int(close.value),
            original_publication_clock=deepcopy(clock), source=deepcopy(src),
            original_holds_preserved=deepcopy(r['holds']),
            archival_vintage_assumption=VINTAGE, historical_first_public_bytes_verified=False,
            economic_homogeneity_certified=False, retrieved_at=clock['retrieved_at'],
            current_cell_html_sha256=c['cell_html_sha256'], comparative_cell_html_sha256=w['cell_html_sha256'],
            current_locator=deepcopy(c['locator']), comparative_locator=deepcopy(w['locator']),
            period_derivation=c['period_derivation'], measure_label=c['measure_label'],
            source_currency_status=c['currency_status'])
        for prefix in ('', 'comparative_'):
            report.update({prefix+'currency': 'reported_dollar', prefix+'currency_basis': DOLLAR_BASIS,
                           prefix+'currency_iso': None, prefix+'currency_iso_certified': False})
        reports.append(report)
    require(len({r['report_id'] for r in reports}) == 13, 'Duplicate ABBV report identity')
    return reports


def normalize_edges(proposal, review, reports, independent_sha, producer_sha, policy_sha):
    source = unique_by(proposal['edges'], 'edge_id', 'ABBV producer edge')
    checked = unique_by(review['edges'], 'edge_id', 'ABBV independent edge')
    require(set(source) == set(checked) and len(source) == 12, 'Incomplete ABBV accounting-edge review')
    report_map = unique_by(reports, 'report_id', 'ABBV normalized report')
    edges = []
    pairs = set()
    for edge_id, p in source.items():
        check = checked[edge_id]
        for key in ['earlier_fiscal_year', 'later_fiscal_year', 'earlier_report_id', 'later_report_id', 'known_ns']:
            require(check[key] == p[key], 'Independent ABBV edge identity/clock mismatch: '+edge_id)
        earlier, later = (report_map[p[key]] for key in ['earlier_report_id', 'later_report_id'])
        pair = (p['earlier_fiscal_year'], p['later_fiscal_year'])
        require(pair == (earlier['fiscal_year'], later['fiscal_year']) and pair[1] == pair[0]+1,
                'Invalid ABBV accounting pair')
        pairs.add(pair)
        require(p['known_ns'] >= max(earlier['known_ns'], later['known_ns']), 'Premature ABBV edge')
        require(p['currency_policy_sha256'] == policy_sha, 'ABBV edge currency policy mismatch')
        require(check.get('source_review_sha256') == producer_sha, 'Independent edge producer hash mismatch')
        require(check.get('admitted') is True or check.get('admitted') is False,
                'Missing explicit per-edge disposition')
        if check['admitted'] is True:
            require(check['status'] == p['status'] == 'checked_reported_total_edge',
                    'Unreviewed positive ABBV accounting status')
        else:
            require(check['status'] != 'checked_reported_total_edge', 'Rejected edge has positive status')
        e = deepcopy(p)
        e.update(status=check['status'], source_review_sha256=independent_sha,
                 producer_review_sha256=producer_sha, independent_disposition=deepcopy(check))
        for evidence in e['evidence']:
            r = report_map[evidence['report_id']]
            require(evidence['source_path'] == r['source']['raw']['path']
                    and evidence['source_sha256'] == r['source_sha256']
                    and evidence['known_ns'] == r['known_ns']
                    and evidence['known_ns'] <= e['known_ns'], 'Unbound ABBV note source/clock')
        require(e['evidence_known_ns'] == sorted({v['known_ns'] for v in e['evidence']})
                and check['evidence_known_ns'] == e['evidence_known_ns'], 'ABBV evidence clock closure mismatch')
        edges.append(e)
    require(pairs == {(y, y+1) for y in range(2013, 2025)}, 'Missing consecutive ABBV edge')
    return edges


def verify_source_objects(inputs, reports, edges):
    for report in reports:
        for role in ['raw', 'receipt']:
            src = report['source'][role]
            raw = inputs.read(src['path'], src['sha256'])
            require(len(raw) == src['bytes'], 'Source size mismatch')
    for edge in edges:
        for e in edge.get('evidence', []):
            inputs.read(e['source_path'], e['source_sha256'])


def attach_provenance(row, reports, edges, review_shas):
    selected = [r for r in reports if r['report_id'] in row.get('report_ids', [])]
    selected_edges = [e for e in edges if e['edge_id'] in row.get('edge_ids', [])]
    row['source_refs'] = [deepcopy(r['source']['raw']) for r in selected]
    row['per_report_known_by'] = {r['report_id']: r['known_by'] for r in selected}
    row['per_edge_known_by'] = {e['edge_id']: e.get('known_by', e.get('known_by_at')) for e in selected_edges}
    row['source_review_sha256s'] = deepcopy(review_shas)
    row['archival_vintage_assumption'] = VINTAGE
    row['economic_homogeneity_certified'] = False
    if row['status'] == USABLE:
        require(len(selected) == 4 and len(selected_edges) == 3, 'Usable row lacks four originals/three edges')
        require(all(r['known_ns'] <= row['decision_cutoff_ns'] for r in selected)
                and all(e['known_ns'] <= row['decision_cutoff_ns'] for e in selected_edges),
                'Usable source is after decision cutoff')


def grid(meta, prior_rows, reports, edges, closes, policy_id, review_shas, evaluate):
    dates, symbols = meta['dates'], meta['symbols']
    require(len(dates) == len(symbols) == 122 and dates == sorted(set(dates)), 'Invalid original 122-month axis')
    old = {}
    for row in prior_rows:
        key = (row['signal_index'], row['candidate_index'])
        require(key not in old, 'Duplicate previous grid slot')
        old[key] = row
    require(len(old) == 122*49, 'Incomplete reviewed MSFT source grid')
    rows = []
    for i, (day, names) in enumerate(zip(dates, symbols)):
        require(len(names) == len(set(names)) == 49 and day in closes, 'Invalid original candidate/calendar axis')
        close = closes[day]
        require(str(close.tz_convert('America/New_York').date()) == day, 'Decision date/close mismatch')
        for j, name in enumerate(names):
            previous = old[(i, j)]
            require((previous['decision_date'], previous['symbol'], previous['decision_cutoff_ns'])
                    == (day, name, int(close.value)), 'Reviewed MSFT grid axis/clock mismatch')
            if name == 'MSFT':
                # Byte-origin row contract preserved, including its policy/status.
                row = deepcopy(previous)
                row['reused_source_row_file'] = MSFT_BASE+'ROWS.json'
                row['reused_source_row_file_sha256'] = review_shas['msft_rows']
                row['source_review_sha256s'] = {'msft': review_shas['msft']}
            else:
                row = dict(slot_id=f'{i:03d}:{j:02d}', signal_index=i, candidate_index=j,
                    symbol=name, decision_date=day, decision_cutoff=close.isoformat(),
                    decision_cutoff_ns=int(close.value), policy_id=policy_id,
                    factor_admitted=False, factor_computed=False, strategy_admitted=False,
                    report_ids=[], edge_ids=[], latest_report_id=None)
                if name == 'ABBV':
                    row.update(evaluate('ABBV', reports, edges, int(close.value), day))
                    attach_provenance(row, reports, edges,
                        {k: review_shas[k] for k in ['abbv_source', 'abbv_accounting', 'chain']})
                    row.update(currency='reported_dollar', currency_basis=DOLLAR_BASIS,
                               currency_iso=None, currency_iso_certified=False)
                elif name == 'AAPL':
                    row.update(status='unresolved_source_or_comparability',
                        reasons=['AAPL_accounting_chain_not_independently_admitted'],
                        catalog_scope='Original annual financial statement attachments FY2016-FY2025; no chain admitted')
                elif name in meta['etfs']:
                    row.update(status='not_applicable_etf', reasons=['Original ETF candidate retained'])
                else:
                    row.update(status='outside_fixed_revenue_pilot_scope', reasons=['Original candidate retained'])
            require(row.get('factor_computed') is False and row.get('factor_admitted') is False
                    and row.get('strategy_admitted') is False, 'Unexpected financial/strategy admission')
            rows.append(row)
    return rows


def prepare(inputs, reviews, paths, review_shas):
    from after155_multi_issuer_chain import evaluate
    from after155_revenue_coverage_gate import coverage_gate
    plan, meta, policy = (inputs.load(p) for p in [PLAN, META, POLICY])
    require(plan['fixed_issuer_scope'] == ['MSFT', 'AAPL', 'ABBV'], 'Pilot issuer scope changed')
    days, closes = calendar(inputs)
    reports = normalize_abbv(inputs.load(EXTRACT), reviews['abbv_source'], reviews['abbv_accounting'],
        review_shas['abbv_source'], review_shas['abbv_accounting'], inputs.hashes[POLICY], days, closes)
    edges = normalize_edges(inputs.load(EDGE_PROPOSAL), reviews['abbv_accounting'], reports,
        review_shas['abbv_accounting'], inputs.hashes[EDGE_PROPOSAL], inputs.hashes[POLICY])
    verify_source_objects(inputs, reports, edges)
    old_rows = inputs.load(MSFT_BASE+'ROWS.json')
    msft_reports, msft_edges = (inputs.load(MSFT_BASE+p) for p in ['REPORTS.json', 'EDGES.json'])
    verify_source_objects(inputs, msft_reports, msft_edges)
    row_shas = dict(review_shas, msft_rows=inputs.hashes[MSFT_BASE+'ROWS.json'])
    rows = grid(meta, old_rows, reports, edges, closes, policy['policy_id'], row_shas, evaluate)
    for row in rows:
        if row['symbol'] == 'MSFT':
            if row['status'] == USABLE:
                require(len(row['source_refs']) == len(row['report_ids']) == 4 and len(row['edge_ids']) == 3,
                        'Reviewed usable MSFT row lacks provenance')
                require(all(inputs.hashes[r['path']] == r['sha256'] for r in row['source_refs']),
                        'Reviewed usable MSFT row source mismatch')
    # Source admission and all independent reviews have been verified before gate evaluation.
    gate = coverage_gate(meta, rows, plan)
    gate.update(source_reviews_verified_by_preparation_driver=True,
                source_review_sha256s=deepcopy(review_shas), strategy_registration_pending=True)
    currency = currency_review(reviews['abbv_accounting'], inputs.hashes[POLICY])
    summary = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        stage='source_only_pilot_preparation', source_coverage_floor_passed=gate['source_coverage_floor_passed'],
        rows=len(rows), issuer_scope=plan['fixed_issuer_scope'],
        original_axis='122 original decision dates x 49 original candidates',
        msft_contract_reused_unchanged=True, aapl_accounting_chain_admitted=False,
        abbv_currency_policy_review=currency, archival_vintage_assumption=VINTAGE,
        source_closure_verified_before_and_after=True,
        parent_completed_round=155, parent_total_trials=978, new_strategy_trials=0,
        financial_factors_computed=0, returns_computed=0, models_fit=0, observer_writes=0,
        strategy_registration_pending=True, financial_factor_admitted=False,
        gate_pass_is_strategy_registration=False,
        limitation='Bounded original-release catalog; no exhaustive amendment history or first-public-byte certification')
    outputs = {'ROWS.json': rows, 'REPORTS.json': {'MSFT': msft_reports, 'ABBV': reports, 'AAPL': []},
               'EDGES.json': {'MSFT': msft_edges, 'ABBV': edges, 'AAPL': []},
               'COVERAGE_GATE.json': gate, 'SUMMARY.json': summary}
    inputs.unchanged()
    return outputs


def write_once(inputs, outputs, review_shas, review_paths):
    out = inputs.local(OUT)
    require(not out.exists(), 'Write-once output directory already exists: '+OUT)
    # Finish serialization and input verification before creating the directory.
    encoded = {name: (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False)+'\n').encode() for name, value in outputs.items()}
    inputs.unchanged()
    manifest = dict(schema_version=1, input_hashes=dict(sorted(inputs.hashes.items())),
        review_sha256s=review_shas, review_paths=review_paths,
        outputs={name: dict(sha256=digest(raw), bytes=len(raw)) for name, raw in encoded.items()},
        source_only=True, strategy_registration=False)
    encoded['MANIFEST.json'] = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    out.mkdir()
    for name, raw in encoded.items():
        with (out/name).open('xb') as stream:
            stream.write(raw)
    inputs.unchanged()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['check-inputs', 'prepare'])
    for name, path in REVIEW_PATHS.items():
        flag = name.replace('_', '-')
        parser.add_argument('--'+flag+'-review-sha', required=name != 'driver')
        parser.add_argument('--'+flag+'-review-path', default=path)
    args = parser.parse_args()
    shas = {name: getattr(args, name+'_review_sha') for name in REVIEW_PATHS}
    paths = {name: getattr(args, name+'_review_path') for name in REVIEW_PATHS}
    inputs = Inputs()
    inputs.read(SELF)
    inputs.read(TEST)
    reviews, paths = review_inputs(inputs, shas, paths, require_driver=args.phase == 'prepare')
    if args.phase == 'check-inputs':
        currency_review(reviews['abbv_accounting'], inputs.hashes[POLICY])
        inputs.unchanged()
        print(json.dumps(dict(phase=args.phase, passed=True, verified_input_files=len(inputs.hashes),
                              actual_grid_evaluated=False)))
        return
    outputs = prepare(inputs, reviews, paths, shas)
    write_once(inputs, outputs, shas, paths)
    print(json.dumps(dict(phase=args.phase, output_directory=OUT,
        rows=outputs['SUMMARY.json']['rows'],
        source_coverage_floor_passed=outputs['COVERAGE_GATE.json']['source_coverage_floor_passed'],
        strategy_registration=False)))


if __name__ == '__main__':
    main()
