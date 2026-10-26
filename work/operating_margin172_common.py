"""Prospective source-only registry for the operating-margin research proposal.

No market panels, financial ratios, rankings or returns are loaded here.
This registry does not admit a factor or alter the central strategy record.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_operating_margin172_sources'
PDF_PLAN = ROOT / 'outputs/monthly_aapl_revenue156/v4/PLAN.json'
PDF_PLAN_SHA = '3c841c5d80659e5f1911035e027518e51a788f7b7e5afba529d15a0324e3be6b'
PDF_BUNDLE = Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
MANIFESTS = {
    'html': ROOT / 'outputs/monthly_revenue_pilot_source156/REPORTS.json',
    'aapl': ROOT / 'outputs/monthly_aapl_revenue156/v4/DATASET.json',
}
CODE_NAMES = ('operating_margin172_common.py', 'operating_margin172_html_sources.py',
              'test_operating_margin172_html_sources.py', 'operating_margin172_aapl_sources.py',
              'test_operating_margin172_aapl_sources.py', 'operating_margin172_independent.py',
              'test_operating_margin172_independent.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    require(path.is_relative_to(ROOT) and 'CloudStorage' not in path.parts,
            'Source path escapes offline research root')
    require(path.is_file(), 'Missing source: ' + str(path))
    return path


def verify(pins):
    require(isinstance(pins, dict) and pins, 'Source hashes required')
    for name, digest in pins.items():
        require(sha(source_path(name)) == digest, 'Source hash mismatch: ' + name)


def source_refs(value, pins):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and re.fullmatch(r'[a-f0-9]{64}', str(value.get('sha256', ''))):
            p = source_path(value['path']); key = str(p); digest = value['sha256']
            require(key not in pins or pins[key] == digest, 'Conflicting source bindings: ' + key)
            require(sha(p) == digest, 'Existing source binding changed: ' + key)
            pins[key] = digest
        for item in value.values():
            source_refs(item, pins)
    elif isinstance(value, list):
        for item in value:
            source_refs(item, pins)


def runtime():
    return dict(version=platform.python_version(), full=sys.version)


def verify_pdf_runtime(pins=None):
    require(sha(PDF_PLAN) == PDF_PLAN_SHA, 'Original PDF runtime registry changed')
    expected = read(PDF_PLAN)['pdf_runtime_hashes']
    require(expected and (pins is None or pins == expected), 'PDF runtime binding differs')
    for name, digest in expected.items():
        path = Path(name)
        require(path.resolve() == path and path.is_relative_to(PDF_BUNDLE), 'Unapproved PDF runtime path')
        require(path.is_file() and sha(path) == digest, 'PDF runtime changed: ' + name)
    return expected


def contract():
    return dict(
        phase='source_preparation_only',
        scope={'MSFT': list(range(2015, 2027)), 'ABBV': list(range(2013, 2026)),
               'AAPL': list(range(2016, 2026))},
        quantities=['annual_consolidated_revenue_current', 'annual_consolidated_revenue_comparative',
                    'annual_consolidated_operating_income_current', 'annual_consolidated_operating_income_comparative'],
        proposal='Long-term momentum may receive a small operating-margin confirmation; no factor formula or trial admitted by this source registry.',
        units='Preserve literal units and currency evidence. Do not infer USD from issuer nationality. Income may be negative; no absolute-value replacement.',
        accounting='Extract current and comparative amounts from the SAME consolidated annual statement in each original release. Never substitute quarterly, non-GAAP, segment, other-operating-income or later-release amounts.',
        time='Both current and comparative cells share the current release information time. Retain original date-only next-session clock and historical first-public-vintage limitations. No backdating comparatives.',
        exclusions='Every scoped report must be represented. Ambiguity, missing cells, unknown periods or scope cause an explicit hold; never omit a difficult report or synthesize data.',
        verification='Independent raw HTML/PDF cell, header, units, period and source-hash review before any proposed admission. Existing revenue review does not certify operating income.',
        no_new_network=True, no_market_prices=True, no_financial_ratios=True,
        no_features_orders_returns=True, no_observer_writes=True, no_central_trial_increment=True,
        no_certified_pristine_pit_claim=True)


def register():
    require(not OUT.exists(), 'Source registry is write-once')
    html, aapl = read(MANIFESTS['html']), read(MANIFESTS['aapl'])
    rows = {'MSFT': html['MSFT'], 'ABBV': html['ABBV'], 'AAPL': aapl['rows']}
    for name, years in contract()['scope'].items():
        require(sorted(r['fiscal_year'] for r in rows[name]) == years, 'Source scope changed: ' + name)
    pins = {str(p): sha(p) for p in MANIFESTS.values()}
    pins[str(PDF_PLAN)] = PDF_PLAN_SHA
    pdf_pins = verify_pdf_runtime()
    require(sha(CALENDAR) == CALENDAR_SHA, 'Historical calendar registry changed')
    pins[str(CALENDAR)] = CALENDAR_SHA
    for name, digest in read(CALENDAR)['provenance']['source_hashes'].items():
        path = source_path(name)
        require(sha(path) == digest, 'Historical calendar dependency changed')
        pins[str(path)] = digest
    for group in rows.values():
        source_refs(group, pins)
    code = {str(ROOT / 'work' / name): sha(ROOT / 'work' / name) for name in CODE_NAMES}
    central = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    state = read(central)
    require(state['current'] == 'monthly_price_distribution171' and not state['registered_queue'],
            'Round171 must be complete before source registration')
    require(state['current_status'] == 'completed_independently_verified', 'Round171 not complete')
    verify(pins); verify(code)
    OUT.mkdir()
    dump(OUT / 'PLAN.json', dict(schema='operating_margin172_source_v1',
        registered_utc=datetime.now(timezone.utc).isoformat(), contract=contract(), runtime=runtime(),
        source_manifests={k: str(v) for k, v in MANIFESTS.items()}, input_sha256=pins, code_sha256=code,
        pdf_runtime_hashes=pdf_pins, calendar=str(CALENDAR),
        prior_central_sha256=sha(central), extraction_complete=False, source_fields_admitted=False))
    print(json.dumps(dict(status='SOURCE_SCOPE_REGISTERED', reports=sum(map(len, rows.values())),
        source_files=len(pins), plan_sha256=sha(OUT / 'PLAN.json'))))


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit source PLAN SHA required')
    plan = read(OUT / 'PLAN.json')
    require(plan['contract'] == contract() and plan['runtime'] == runtime(), 'Source contract/runtime changed')
    verify(plan['input_sha256']); verify(plan['code_sha256']); verify_pdf_runtime(plan['pdf_runtime_hashes'])
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register', 'verify'])
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.phase == 'register':
        register()
    else:
        gate(args.plan_sha256)
        print('SOURCE_REGISTRY_VERIFIED')
