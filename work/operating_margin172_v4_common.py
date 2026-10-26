"""Versioned correction of independent annual-column ownership checks."""
from datetime import datetime, timezone
import argparse
import json
import operating_margin172_v3_common as prior

ROOT = prior.ROOT
OUT = prior.OUT.parent / 'v4'
PDF_BUNDLE = prior.PDF_BUNDLE
require, read, dump, sha, source_path = prior.require, prior.read, prior.dump, prior.sha, prior.source_path
verify_pdf_runtime = prior.verify_pdf_runtime
V3_SHA = 'd9034467d0116008fe27091257bc70abb0244476d0cdd4b3731f6c269895367e'
PREFLIGHT = ROOT / 'work/operating_margin172_preflight_v4_20261021T212624692053Z.json'
PREFLIGHT_SHA = 'bfea9963c2f10ada77a37d28ccb376d0fa40699f32edd2d5c8035159c94a9a8f'
NAMES = ('operating_margin172_v4_common.py', 'operating_margin172_independent_v4.py',
         'test_operating_margin172_independent_v4.py')


def original():
    return prior.gate(V3_SHA)


def contract():
    return dict(reason='ABBV FY2013-FY2021 mixed quarter-and-annual table titles caused v3 to classify quarterly cells as annual despite the local quarterly header.',
                scope='Use column-owned single-duration annual headers; mixed-period table titles cannot confer annual eligibility and local quarterly headers exclude a column. Preserve unique formal statement, numeric value, annual period, unit, source and publication-clock checks.',
                no_new_source_extraction=True, no_ratios_features_returns=True,
                original_failures_preserved=True, source_fields_admitted=False)


def register():
    p = original()
    require(sha(PREFLIGHT) == PREFLIGHT_SHA, 'Source-only preflight changed')
    require(not OUT.exists(), 'V4 source audit registry is write-once')
    code = {str(ROOT / 'work' / n): sha(ROOT / 'work' / n) for n in NAMES}
    prior.prior.old.verify(code)
    OUT.mkdir()
    p = dict(p, schema='operating_margin172_source_audit_v4', source_v3_plan_sha256=V3_SHA,
             source_only_preflight=dict(path=str(PREFLIGHT), sha256=PREFLIGHT_SHA),
             v3_code_sha256=p['code_sha256'], code_sha256=code,
             annual_column_correction_contract=contract(), registered_utc=datetime.now(timezone.utc).isoformat())
    dump(OUT / 'PLAN.json', p)
    dump(OUT / 'V3_FAILURE.json', dict(phase='independent_raw_source_verification', exit_code=1,
         error='ValueError: Unique annual monetary cell required',
         source_v3_plan_sha256=V3_SHA, source_fields_admitted=False, new_strategy_trials=0))
    print(json.dumps(dict(status='INDEPENDENT_ANNUAL_COLUMN_CORRECTION_REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))))


def gate(expected):
    original()
    require(sha(PREFLIGHT) == PREFLIGHT_SHA, 'Source-only preflight changed')
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit V4 PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['source_v3_plan_sha256'] == V3_SHA and p['annual_column_correction_contract'] == contract(), 'Annual-column correction changed')
    prior.prior.old.verify(p['code_sha256'])
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register', 'verify'])
    parser.add_argument('--plan-sha256')
    a = parser.parse_args()
    if a.phase == 'register': register()
    else:
        gate(a.plan_sha256)
        print('SOURCE_ANNUAL_COLUMN_CORRECTION_VERIFIED')
