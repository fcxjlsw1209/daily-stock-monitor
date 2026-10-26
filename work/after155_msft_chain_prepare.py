"""Frozen source-only MSFT annual-chain preparation; no factors or returns."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_msft_revenue_chain156'
PINS = {
    'outputs/monthly_local_linear70/META.json': '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    'outputs/monthly_paired_target143/CALENDAR.json': '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    'outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json': '0f5aa563ac7eac950b9594740d14111dfa8aac1608f5e73558d39baee6066aef',
    'work/after155_revenue_extended_review.json': 'f18d530b9cea67d0fce5d82f8910b8a7a20264813e058e296058c8e783a687bd',
    'work/after155_revenue_chain_policy_review.json': '8564d0d052e4a7cfb89bd73c553b46c6ea831afacd384a5d69fe63c52732ce36',
    'work/after155_msft_revenue_basis_review.json': 'ade332bf2cab7c8f8a06ce832fc3389dd0a08902ac4c05515ba22c487e055b40',
    'work/after155_revenue_chain.py': 'b7921dc1d20b6ccf602a1a8e14e3544d2b8e41bba86d4843b3dc0f445ed2b83c',
    'work/test_after155_revenue_chain.py': '3490f5c0c90b165f63ebb10fa38967754934de0cde6613dae895055eeffce970',
}
POLICY = 'msft_original_annual_revenue_four_level_chain_v1'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def local(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    check(p.is_relative_to(ROOT) and p.resolve() == p, 'Nonlocal/symlink source')
    return p


def sha(path):
    return hashlib.sha256(local(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(local(path).read_text())


def write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def validate(pins):
    for path, h in pins.items():
        check(sha(path) == h, 'Source hash mismatch: ' + path)
    check(all(pins.get(p) == h for p, h in PINS.items()), 'Mandatory pin mismatch')


def reviewed_inputs(pins):
    """A checksum identifies a review; it does not establish that it passed."""
    requirements = {
        'work/after155_msft_basis_independent.json': [
            'work/after155_msft_revenue_basis_review.json',
            'work/after155_revenue_chain_policy_review.json',
            'outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json'],
        'work/after155_revenue_chain_implementation_review_v2.json': [
            'work/after155_revenue_chain.py', 'work/test_after155_revenue_chain.py',
            'work/after155_revenue_chain_policy_review.json'],
    }
    for path, required in requirements.items():
        review = read(path)
        check(review.get('passed') is True, 'Independent review not passed: '+path)
        for p in required:
            check(review['input_hashes'].get(p)==pins[p], 'Review input not bound: '+p)
        for p,h in review['input_hashes'].items():
            check(pins.get(str(local(p).relative_to(ROOT)))==h, 'Review closure missing: '+p)


def calendar():
    spec = importlib.util.spec_from_file_location('chain_source_calendar156',
        ROOT / 'outputs/monthly_earnings_repair/research_calendar.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = read('outputs/monthly_paired_target143/CALENDAR.json')
    cal = module.market_calendar(frozen['start'], frozen['end'])
    days = [str(d.date()) for d in cal.schedule.index]
    check(days == frozen['sessions'], 'Session axis changed')
    return days, {d: t for d, t in zip(days, cal.schedule['close'])}


def catalog(pd, days, closes):
    proposal = read('outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json')
    basis = read('work/after155_msft_revenue_basis_review.json')
    reports = []
    for r in proposal['records']:
        y = r['fiscal_year']; a = r['availability']; c = r['revenue']['current']; w = r['revenue']['comparative']
        check(r['symbol'] == 'MSFT' and a['timezone'] == 'America/Los_Angeles'
              and a['precision'] == 'date_only', 'Unexpected source identity/clock')
        session = next(d for d in days if d > a['release_date'])
        check(session == a['first_conservative_session_after_release'], 'Original clock mismatch')
        check(c['duration_literal'] == w['duration_literal'] == 'Twelve Months Ended', 'Nonannual value')
        check(r['currency']['iso_currency'] == 'USD', 'NonUSD source')
        check(c['first_conservative_session'] == w['first_conservative_session'] == session,
              'Cell time differs from report')
        receipt = read(r['source']['receipt']['path'])
        check(receipt['http_status']==200 and receipt['url']==r['source']['expected_url']
              and receipt['sha256']==r['source']['raw']['sha256'], 'Original receipt mismatch')
        rid = 'MSFT_FY' + str(y) + '_original_Q4'
        reports.append(dict(report_id=rid, fiscal_year=y, issuer='MSFT',
            source_checked=True, source=r['source'], currency='USD',
            measure='consolidated_gaap_total_revenue', known_ns=closes[session].value,
            known_by=closes[session].isoformat(), release_date=a['release_date'],
            period_start=c['period_start_derived'], period_end=c['period_end'],
            current=c['amount_decimal'], scale=str(c['scale']),
            comparative=w['amount_decimal'], comparative_scale=str(w['scale']),
            comparative_currency='USD', comparative_measure='consolidated_gaap_total_revenue',
            comparative_period_start=w['period_start_derived'], comparative_period_end=w['period_end'],
            comparative_source_report_id=rid,
            proposal_record_ref=dict(path='outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json',
                fiscal_year=y),
            current_locator=c['locator'], comparative_locator=w['locator'],
            current_cell_html_sha256=c['cell_html_sha256'], comparative_cell_html_sha256=w['cell_html_sha256'],
            row_html_sha256=hashlib.sha256(r['revenue']['revenue_row_html'].encode()).hexdigest(),
            derived_income_table=r['revenue']['derived_table_evidence'],
            archival_vintage='currently_retrieved_original_dated_issuer_release',
            historical_first_public_bytes_verified=False))
    check(sorted(r['fiscal_year'] for r in reports) == list(range(2015,2027)), 'Catalog coverage differs')
    by_year = {r['fiscal_year']: r for r in reports}
    edges = []
    for b in basis['boundaries']:
        before, after = b['earlier_fiscal_year'], b['later_fiscal_year']
        check(after == before+1, 'Nonadjacent source review')
        check(b['available_from'] == b['known_by'] == b['known_no_earlier_than'], 'Ambiguous note availability')
        check(pd.Timestamp(b['known_by']).value == by_year[after]['known_ns'], 'Note/report timing mismatch')
        edges.append(dict(edge_id=f'MSFT_FY{before}_to_FY{after}', earlier_fiscal_year=before,
            later_fiscal_year=after, earlier_report_id=by_year[before]['report_id'],
            later_report_id=by_year[after]['report_id'], known_ns=pd.Timestamp(b['known_by']).value,
            known_by=b['known_by'], evidence_known_ns=[pd.Timestamp(e['source_available_at']).value for e in b['evidence']],
            status={'pass_with_declared_scope':'checked_reported_total_edge',
                    'blocked_known_basis_break':'blocked_known_basis_break'}[b['status']],
            evidence_refs=b['evidence_refs'], evidence_sources=[dict(path=e['source_path'], sha256=e['source_sha256']) for e in b['evidence']],
            context_evidence_ref=b.get('context_evidence_ref'),
            reason=b['reason'], economic_homogeneity_certified=False))
    check(len(edges) == 11, 'Boundary count differs')
    return reports, edges


def prepare(pd):
    from after155_revenue_chain import evaluate
    from test_after155_revenue_chain import SourceChainTest
    log = StringIO()
    tests = unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourceChainTest))
    check(tests.wasSuccessful() and tests.testsRun == 26, 'Synthetic tests failed: ' + log.getvalue())
    days, closes = calendar()
    reports, edges = catalog(pd, days, closes)
    meta = read('outputs/monthly_local_linear70/META.json')
    check(len(meta['dates'])==len(meta['symbols'])==122 and meta['dates']==sorted(set(meta['dates'])),
          'Month axis mismatch')
    rows = []
    for i, (day, symbols) in enumerate(zip(meta['dates'], meta['symbols'])):
        check(day in closes and len(symbols) == 49 and len(set(symbols)) == 49, 'Candidate axis mismatch')
        cutoff = closes[day]
        # The caller, rather than arbitrary per-row input, binds date to its actual exchange close.
        check(str(cutoff.tz_convert('America/New_York').date()) == day, 'Decision date/cutoff mismatch')
        for j, symbol in enumerate(symbols):
            base = dict(policy_id=POLICY, slot_id=f'{i:03d}:{j:02d}', signal_index=i,
                candidate_index=j, symbol=symbol, decision_date=day,
                decision_cutoff=cutoff.isoformat(), decision_cutoff_ns=cutoff.value,
                factor_admitted=False, strategy_admitted=False)
            if symbol == 'MSFT':
                base.update(evaluate(reports, edges, cutoff.value, day))
                selected = [r for r in reports if r['report_id'] in base['report_ids']]
                selected_edges = [e for e in edges if e['edge_id'] in base['edge_ids']]
                base['source_refs'] = [r['source']['raw'] for r in selected]
                base['per_report_known_by'] = {r['report_id']: r['known_by'] for r in selected}
                base['per_edge_known_by'] = {e['edge_id']: e['known_by'] for e in selected_edges}
                base['archival_vintage_assumption'] = 'original_dated_releases_retrieved_later_not_certified_first_public_bytes'
            else:
                base.update(status='not_applicable_etf' if symbol in meta['etfs'] else 'outside_msft_source_contract',
                    reasons=['No chain evaluated; original candidate remains on the research axis.'],
                    factor_computed=False)
            rows.append(base)
    check(len(rows) == 5978 and sum(r['symbol']=='MSFT' for r in rows)==122, 'Full candidate grid not preserved')
    check(sum(r['symbol'] not in meta['etfs'] for r in rows)==3904, 'Stock axis changed')
    summary = dict(stage='SOURCE_CHAIN_PROPOSAL_REQUIRES_INDEPENDENT_REVIEW', policy_id=POLICY,
        rows=len(rows), stock_rows=sum(r['symbol'] not in meta['etfs'] for r in rows),
        etf_rows=sum(r['symbol'] in meta['etfs'] for r in rows),
        msft_status_counts=dict(Counter(r['status'] for r in rows if r['symbol']=='MSFT')),
        full_status_counts=dict(Counter(r['status'] for r in rows)), synthetic_tests=tests.testsRun,
        independent_actual_chain_review_complete=False, factor_admitted=False, new_trials=0,
        growth_ratios_computed=0, labels_models_ranks_returns_computed=0,
        selection_scope='Unchanged 122 x 49 candidate axis; non-MSFT rows explicitly not evaluated by this issuer-specific source contract.',
        financial_scope='Reported nominal consolidated GAAP USD totals, not organic revenue or accounting homogeneity.')
    return {'REPORTS.json':reports, 'EDGES.json':edges, 'ROWS.json':rows, 'SUMMARY.json':summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze','prepare','verify'])
    parser.add_argument('--spec-sha')
    parser.add_argument('--basis-review-sha')
    parser.add_argument('--implementation-review-sha')
    args = parser.parse_args()
    if args.phase == 'freeze':
        check(args.basis_review_sha and args.implementation_review_sha, 'Both independent review hashes required')
        pins = dict(PINS)
        def add(path, h):
            p=str(local(path).relative_to(ROOT))
            check(p not in pins or pins[p]==h, 'Conflicting source hash: '+p)
            check(sha(p)==h, 'Input changed: '+p)
            pins[p]=h
        add('work/after155_msft_basis_independent.json', args.basis_review_sha)
        add('work/after155_revenue_chain_implementation_review_v2.json', args.implementation_review_sha)
        for p in ('work/after155_msft_basis_independent.json',
                  'work/after155_revenue_chain_implementation_review_v2.json'):
            for path,h in read(p)['input_hashes'].items():
                add(path,h)
        # Bind the note extraction and table-context artifacts, not only a prose verdict.
        base='outputs/monthly_msft_revenue_basis156/'
        bm=read(base+'MANIFEST.json')
        add(base+'MANIFEST.json',sha(base+'MANIFEST.json'))
        add(base+'SPEC.json',bm['spec_sha256'])
        for name,h in bm['outputs'].items():
            add(name if name.startswith('work/') else base+name,h)
        for name in ('CONTEXT_EVIDENCE.json','inspect_note_context.py'):
            add(base+name,sha(base+name))
        def add_refs(obj):
            if isinstance(obj,dict):
                if 'path' in obj and 'sha256' in obj:
                    add(obj['path'],obj['sha256'])
                for value in obj.values():
                    add_refs(value)
            elif isinstance(obj,list):
                for value in obj:
                    add_refs(value)
        add_refs(read(base+'SPEC.json'))
        add_refs(read('outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json'))
        add_refs(read('work/after155_msft_revenue_basis_review.json'))
        cell_review=read('work/after155_revenue_extended_review.json')
        check(cell_review['status']=='pass_24_source_cells_and_dates_comparability_remains_bounded',
              'Source cells not independently verified')
        add(cell_review['scope']['prior_review_path'],cell_review['scope']['prior_review_sha256'])
        add('work/after155_msft_chain_prepare.py', sha(__file__))
        for p in ('run_local.sh','local_run.py','local.sandbox.sb'):
            add(p, sha(p))
        for p,h in read('outputs/monthly_paired_target143/CALENDAR.json')['provenance']['source_hashes'].items():
            add(p, h)
        for r in read('outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json')['records']:
            for value in (r['source']['raw'], r['source']['receipt'], r['revenue']['derived_table_evidence']):
                add(value['path'], value['sha256'])
        for b in read('work/after155_msft_revenue_basis_review.json')['boundaries']:
            for e in b['evidence']:
                add(e['source_path'], e['source_sha256'])
        validate(pins)
        reviewed_inputs(pins)
        write('SPEC.json', dict(stage='FROZEN_SOURCE_CHAIN_PREPARATION', policy_id=POLICY,
            created_utc=datetime.now(timezone.utc).isoformat(), hashes=pins, runtime=sys.version,
            scope='One issuer source eligibility on unchanged 5978 candidate grid, no growth/factor/strategy.',
            independent_reviews_required=True))
        print(json.dumps(dict(phase='freeze', spec_sha256=sha(OUT/'SPEC.json'))))
        return
    check(args.spec_sha and sha(OUT/'SPEC.json')==args.spec_sha, 'Exact spec hash required')
    spec = read(OUT/'SPEC.json'); validate(spec['hashes'])
    reviewed_inputs(spec['hashes'])
    check(spec['runtime']==sys.version, 'Runtime mismatch')
    import pandas as pd
    payload = prepare(pd)
    validate(spec['hashes'])
    if args.phase == 'prepare':
        for name,value in payload.items():
            write(name,value)
        write('MANIFEST.json',dict(spec_sha256=args.spec_sha,
            outputs={name:sha(OUT/name) for name in payload}, independent_actual_chain_review_complete=False))
    else:
        manifest=read(OUT/'MANIFEST.json')
        check(manifest['spec_sha256']==args.spec_sha, 'Manifest bound to wrong spec')
        for name,value in payload.items():
            check(sha(OUT/name)==manifest['outputs'][name] and read(OUT/name)==value,
                  'Output mismatch: '+name)
    print(json.dumps(dict(phase=args.phase,passed=True,summary=payload['SUMMARY.json'])))


if __name__=='__main__':
    main()
