"""Independent DOM-context correction, preserving both original source runs."""
from datetime import datetime, timezone
import argparse
import json
import operating_margin172_v2_common as prior

ROOT = prior.ROOT
OUT = prior.OUT.parent / 'v3'
PDF_BUNDLE = prior.PDF_BUNDLE
require, read, dump, sha, source_path = prior.require, prior.read, prior.dump, prior.sha, prior.source_path
verify_pdf_runtime = prior.verify_pdf_runtime
V2_SHA = '9fa4d3b896b0fdeb93a8e79e1e081956759f0e110607a85a630a7d586e32ac87'
HTML_SHA = '791f1a6994e6a2193125b788fcb841d855bc5d4aa6ce5487013f49718629df34'
NAMES = ('operating_margin172_v3_common.py', 'operating_margin172_independent_v3.py',
         'test_operating_margin172_independent_v3.py')


def original():
    p = prior.gate(V2_SHA)
    require(sha(prior.OUT / 'html/EXTRACTION.json') == HTML_SHA, 'V2 HTML extraction changed')
    return p


def contract():
    return dict(reason='Independent v2 scan failed to associate the Microsoft formal statement heading with its table, although original source contains the heading.',
                scope='Repair independent DOM heading association only. Preserve amount, annual column, unit, period, raw-source and publication-clock checks.',
                no_new_source_extraction=True, no_ratios_features_returns=True,
                original_failures_preserved=True, source_fields_admitted=False)


def register():
    p = original()
    require(not OUT.exists(), 'V3 source audit registry is write-once')
    code = {str(ROOT / 'work' / n): sha(ROOT / 'work' / n) for n in NAMES}
    prior.old.verify(code)
    OUT.mkdir()
    p = dict(p, schema='operating_margin172_source_audit_v3', source_v2_plan_sha256=V2_SHA,
             source_v2_html_sha256=HTML_SHA, previous_code_sha256=p['code_sha256'], code_sha256=code,
             audit_correction_contract=contract(), registered_utc=datetime.now(timezone.utc).isoformat())
    dump(OUT / 'PLAN.json', p)
    dump(OUT / 'V2_FAILURE.json', dict(phase='independent_raw_source_verification', exit_code=1,
         error='ValueError: Consolidated Microsoft statement title absent',
         source_v2_plan_sha256=V2_SHA, source_fields_admitted=False, new_strategy_trials=0))
    print(json.dumps(dict(status='INDEPENDENT_DOM_CORRECTION_REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))))


def gate(expected):
    original()
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit V3 PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['source_v2_plan_sha256'] == V2_SHA and p['audit_correction_contract'] == contract(), 'Audit correction changed')
    prior.old.verify(p['code_sha256'])
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register', 'verify'])
    parser.add_argument('--plan-sha256')
    a = parser.parse_args()
    if a.phase == 'register': register()
    else:
        gate(a.plan_sha256)
        print('SOURCE_AUDIT_CORRECTION_VERIFIED')
