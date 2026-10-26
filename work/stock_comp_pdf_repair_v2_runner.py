"""Write-once three-event SBC patch; freeze is metadata-only, then build/verify.

Existing v1 evidence and all admission decisions stay immutable. New source
values remain candidates until separately reviewed. No ratios or strategies.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import stock_comp_pdf_repair_v2 as parser

base, safe, require = parser.base, parser.base.safe, parser.base.require
ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_stock_comp_pdf_repair_v2'
V1 = 'outputs/monthly_stock_comp_source_v1'
V1_PLAN_SHA = 'ef563d83e864c3e592c75f232dccc1327aed9b1fc95b5a2115c703ad51df45ba'
V1_MANIFEST_SHA = '5e25899f13b31bc631d1ef5bebc3d5f76e0a14bbb5866bd41ff104b288736f31'
TARGETS = ('AAPL:FY2021:2021-10-28', 'AAPL:FY2022:2022-10-27', 'AAPL:FY2023:2023-11-02')
TESTS = 'work/stock_comp_pdf_repair_v2_TESTS.json'
DIAGNOSTIC = 'work/stock_comp_pdf_repair_v2_diagnostic.json'
DIAGNOSTIC_CODE = 'work/stock_comp_pdf_repair_v2_diagnose.py'
ZERO_COUNTS = ('fact_admissions', 'model_admissions', 'strategy_admissions', 'financial_calculations',
               'ratios_computed', 'ranks_computed', 'model_fits', 'orders_computed', 'returns_computed')


def policy():
    return dict(scope='Exactly three previously-null current Apple SBC fields; no other v1 event changes.',
                event_ids=list(TARGETS), maximum_new_cells=3, field=parser.old.FIELD,
                pypdf='Two annual columns; unique contiguous token partition by unchanged exact monetary syntax; no expected-value, whitespace-width or plausibility selection.',
                maximum_pypdf_tokens=parser.MAX_TOKENS,
                pdfium='Frozen cashflow_pdf_repair_v4 physical rows, source indices, numeric kerning, exact row uniqueness, date/year geometry and per-column literal agreement unchanged.',
                metadata='Frozen v1 inventory, original documents, receipts, period, scale, currency, release and availability remain unchanged.',
                inherited_holds='All original 13 holds remain; patch cannot alter admission.',
                currency_iso_certified=False, first_public_version_verified=False,
                complete_correction_history_verified=False, certified_daily_security_master=False,
                financial_calculations_permitted=False, strategy_registration_permitted=False,
                source_only=True)


def validate_targets(plan_events, raw_events):
    require(type(plan_events) is list and type(raw_events) is list, 'Original event arrays required')
    pmap = {e['event_id']: e for e in plan_events}
    rmap = {e['event_id']: e for e in raw_events}
    require(len(plan_events) == len(pmap) == len(raw_events) == len(rmap) == 44 and set(pmap) == set(rmap),
            'Complete unique original44 population required')
    require(sum(not e['cash_quality_admission']['admitted'] for e in plan_events) == 13,
            'Original13 holds required')
    specs = []
    for event_id in TARGETS:
        spec, event = pmap[event_id], rmap[event_id]
        require(spec['symbol'] == 'AAPL' and spec['cash_quality_admission']['admitted'] is True,
                'Exact previously eligible Apple target')
        for key in ('event_id', 'symbol', 'fiscal_year', 'release_date', 'available_from',
                    'identity', 'periods', 'source', 'inherited_rejection_codes', 'cash_quality_admission'):
            require(event[key] == spec[key], 'Original target metadata changed: '+key)
        result = event['stock_compensation']
        require(result['value'] is None and result['gaps'] == [dict(
                    reason='invalid_or_ambiguous_current_monetary_cell', error_type='ValueError')],
                'Patch may target only the declared v1 null failures')
        require(all(event[k] is False and result[k] is False for k in
                    ('fact_admitted', 'model_admitted', 'strategy_admitted')),
                'No inherited new-field admission')
        inv = spec['locator_inventory']
        require(inv['format'] == 'pdf' and not inv['gaps'] and len(inv['candidates']) == 1,
                'Unchanged unique PDF inventory required')
        loc = inv['candidates'][0]
        require(loc['row_label'] in parser.old.PDF_LABELS and loc['financial_column_index_0based'] == 0
                and len(loc['header_dates']) == 2, 'Declared first annual column')
        specs.append(deepcopy(spec))
    return specs


def verified_inputs(root):
    plan = base.read(root, V1+'/PLAN.json', V1_PLAN_SHA)
    manifest = base.read(root, V1+'/MANIFEST.json', V1_MANIFEST_SHA)
    require(plan['stage'] == 'frozen_stock_compensation_source_census' and plan['runtime'] == sys.version,
            'Original frozen source plan/runtime required')
    require(manifest['plan_sha256'] == V1_PLAN_SHA and
            set(manifest['output_hashes']) == {'RAW_FACTS.json', 'COVERAGE.json', 'AUDIT.json'},
            'Original complete source manifest required')
    hashes = {V1+'/PLAN.json': V1_PLAN_SHA, V1+'/MANIFEST.json': V1_MANIFEST_SHA,
              **{V1+'/'+name: digest for name, digest in manifest['output_hashes'].items()}}
    for group in (plan['source_hashes'], plan['source_code_hashes']):
        for path, digest in group.items():
            require(path not in hashes or hashes[path] == digest, 'Conflicting inherited source binding')
            hashes[path] = digest
    safe.verify_hashes(root, hashes)
    base.configure_pdf(plan)
    data = base.read(root, V1+'/RAW_FACTS.json')
    specs = validate_targets(plan['events'], data['events'])
    return plan, specs, hashes


def verify_tests(root):
    tests = base.read(root, TESTS)
    require(tests.get('passed') is True and tests.get('synthetic_only') is True and
            tests.get('source_code_hashes') == parser.code_hashes(root), 'Current synthetic receipt required')
    require(type(tests.get('tests_run')) is int and tests['tests_run'] > 0 and
            all(type(tests.get(k)) is int and tests[k] == 0 for k in
                ('errors', 'failures', 'skipped', 'real_source_cells_extracted')), 'Clean synthetic counters required')
    ref = tests['log']
    require(base.sha(root, ref['path']) == ref['sha256'], 'Synthetic log changed')
    log = safe.local(root, ref['path']).read_text()
    require(re.search(r'Ran '+str(tests['tests_run'])+r' tests? in ', log) and log.rstrip().endswith('OK'),
            'Synthetic log does not confirm completed passing run')
    return {TESTS: base.sha(root, TESTS), ref['path']: ref['sha256']}


def freeze(root, review_path):
    require(not safe.local(root, OUT+'/PLAN.json').exists(), 'Patch PLAN already frozen')
    original, specs, hashes = verified_inputs(root)
    hashes.update(verify_tests(root))
    diagnostic = base.read(root, DIAGNOSTIC)
    require(diagnostic['plan_sha256'] == V1_PLAN_SHA and diagnostic['manifest_sha256'] == V1_MANIFEST_SHA
            and diagnostic['new_values_published'] == 0 and diagnostic['new_parser_executed'] is False,
            'Original masked diagnostic required')
    require([x['event_id'] for x in diagnostic['findings']] == list(TARGETS) and
            all(x['frozen_extractor_error'] == dict(type='ValueError', message='exact_pypdf_column_count_required')
                for x in diagnostic['findings']), 'Wrong diagnosed failure mechanism')
    require(diagnostic['diagnostic_code_sha256'] == base.sha(root, DIAGNOSTIC_CODE), 'Diagnostic code changed')
    hashes.update({DIAGNOSTIC: base.sha(root, DIAGNOSTIC), DIAGNOSTIC_CODE: base.sha(root, DIAGNOSTIC_CODE)})
    review = base.read(root, review_path)
    require(review.get('passed') is True and review.get('source_plan_sha256') == V1_PLAN_SHA
            and review.get('source_code_hashes') == parser.code_hashes(root)
            and review.get('real_source_cells_extracted') == 0, 'Independent pre-build code review required')
    hashes[review_path] = base.sha(root, review_path)
    for spec in specs:
        raw = safe.local(root, spec['source']['path']).read_bytes()
        require(safe.digest(raw) == spec['source']['sha256'], 'Original PDF changed')
        require(parser.old.inventory(raw, spec['periods'], original) == spec['locator_inventory'],
                'Metadata inventory changed before freeze')
    safe.verify_hashes(root, hashes)
    plan = dict(schema_version=1, stage='frozen_stock_comp_pdf_source_repair',
                created_utc=datetime.now(timezone.utc).isoformat(), runtime=sys.version,
                source_plan_sha256=V1_PLAN_SHA, source_manifest_sha256=V1_MANIFEST_SHA,
                source_hashes=hashes, source_code_hashes=parser.code_hashes(root),
                pdf_runtime=original['pdf_runtime'], pdf_runtime_hashes=original['pdf_runtime_hashes'],
                policy=policy(), events=specs, real_source_cells_extracted=0)
    base.publish(root, OUT+'/PLAN.json', plan)
    return dict(plan_sha256=base.sha(root, OUT+'/PLAN.json'), events=3, real_source_cells_extracted=0)


def load_plan(root, plan_sha):
    base.expect_sha(plan_sha)
    plan = base.read(root, OUT+'/PLAN.json', plan_sha)
    require(plan['stage'] == 'frozen_stock_comp_pdf_source_repair' and plan['runtime'] == sys.version
            and plan['policy'] == policy(), 'Frozen patch plan/runtime/policy required')
    require(plan['source_code_hashes'] == parser.code_hashes(root), 'Frozen patch code changed')
    require(plan['source_plan_sha256'] == V1_PLAN_SHA and plan['source_manifest_sha256'] == V1_MANIFEST_SHA,
            'Wrong original source baseline')
    safe.verify_hashes(root, plan['source_hashes'])
    safe.verify_hashes(root, plan['source_code_hashes'])
    base.configure_pdf(plan)
    original, specs, hashes = verified_inputs(root)
    require(plan['events'] == specs and all(plan['source_hashes'].get(p) == digest for p, digest in hashes.items()),
            'Frozen target metadata or inherited binding changed')
    require(plan['pdf_runtime'] == original['pdf_runtime'] and plan['pdf_runtime_hashes'] == original['pdf_runtime_hashes'],
            'Original PDF runtime binding changed')
    return plan


def products(root, plan, plan_sha):
    events = []
    for spec in plan['events']:
        raw = safe.local(root, spec['source']['path']).read_bytes()
        require(safe.digest(raw) == spec['source']['sha256'], 'Original PDF changed during patch')
        result = parser.extract_current(raw, spec['periods'], spec['locator_inventory'], plan)
        if result['value'] is not None:
            value = result['value']
            require(value['source_sha256'] == spec['source']['sha256'] and value['period'] == spec['periods']['current'],
                    'Patch current-cell provenance mismatch')
            value['source'] = deepcopy(spec['source'])
            value['available_from'] = spec['available_from']
        require(all(result[k] is False for k in ('fact_admitted', 'model_admitted', 'strategy_admitted')),
                'Source patch cannot admit a field')
        events.append(dict(event_id=spec['event_id'], result=result))
    require([e['event_id'] for e in events] == list(TARGETS), 'Exact three ordered patch events required')
    count = sum(e['result']['value'] is not None for e in events)
    counts = dict(events=3, requested_new_cells=3, extracted_new_cells=count, gap_cells=3-count,
                  inherited_population=44, inherited_holds=13, **{key: 0 for key in ZERO_COUNTS})
    return dict(schema_version=1, stage='stock_compensation_source_patch_not_admitted',
                plan_sha256=plan_sha, source_plan_sha256=V1_PLAN_SHA,
                source_manifest_sha256=V1_MANIFEST_SHA, counts=counts, events=events)


def audit_for(patch, patch_sha):
    return dict(passed=True, stage='source_patch_process_audit_not_admission',
                plan_sha256=patch['plan_sha256'], source_plan_sha256=V1_PLAN_SHA,
                source_manifest_sha256=V1_MANIFEST_SHA, patch_sha256=patch_sha, counts=patch['counts'],
                patch_targets_previously_null_only=True, original_v1_files_unchanged=True,
                inherited_event_metadata_preserved=True, inherited_admission_unchanged=True,
                independent_field_and_visual_review_pending=True)


def build(root, plan_sha, verify=False):
    plan = load_plan(root, plan_sha)
    names = ('PATCH.json', 'AUDIT.json', 'MANIFEST.json')
    if not verify:
        require(not any(safe.local(root, OUT+'/'+name).exists() for name in names), 'Source patch outputs are write-once')
    patch = products(root, plan, plan_sha)
    patch_sha = safe.digest(safe.encoded(patch))
    audit = audit_for(patch, patch_sha)
    objects = {'PATCH.json': patch, 'AUDIT.json': audit}
    hashes = {name: safe.digest(safe.encoded(obj)) for name, obj in objects.items()}
    manifest = dict(plan_sha256=plan_sha, output_hashes=hashes)
    safe.verify_hashes(root, plan['source_hashes']); safe.verify_hashes(root, plan['source_code_hashes'])
    if verify:
        require(base.read(root, OUT+'/MANIFEST.json') == manifest, 'Patch manifest reproduction mismatch')
        for name, obj in objects.items():
            require(base.read(root, OUT+'/'+name, hashes[name]) == obj, 'Patch reproduction mismatch: '+name)
    else:
        for name, obj in objects.items(): base.publish(root, OUT+'/'+name, obj)
        base.publish(root, OUT+'/MANIFEST.json', manifest)
    return dict(passed=True, counts=patch['counts'], source_only=True,
                manifest_sha256=base.sha(root, OUT+'/MANIFEST.json'))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('freeze', 'build', 'verify'))
    cli.add_argument('--review'); cli.add_argument('--plan-sha')
    args = cli.parse_args()
    if args.phase == 'freeze':
        require(args.review is not None, 'Independent review path required')
        result = freeze(ROOT, args.review)
    else:
        result = build(ROOT, args.plan_sha, verify=args.phase == 'verify')
    print(safe.encoded(result).decode())


if __name__ == '__main__':
    main()
