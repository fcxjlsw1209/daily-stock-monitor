"""Versioned source-parser correction; no strategy inputs or outcomes computed."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import operating_margin172_common as old

ROOT = old.ROOT
OUT = old.OUT / 'v2'
ORIGINAL_SHA = '7c2f3179d4d5a7c763746a5cb8072dc4407fdbc7ff8dc30361b8555cca83ff58'
HTML_SHA = 'ed6b5848bb65e9b0fee60ad07a02dc5399e2554def32076c4779f7ee501b88f7'
AAPL_SHA = 'be0c7bb99a996bd37a8c0b383d134b35fdb686280eb232a14a505128c73f6c2f'
PDF_BUNDLE = old.PDF_BUNDLE
require, read, dump, sha, source_path = old.require, old.read, old.dump, old.sha, old.source_path
verify_pdf_runtime = old.verify_pdf_runtime
NAMES = ('operating_margin172_v2_common.py', 'operating_margin172_html_v2.py',
         'test_operating_margin172_html_v2.py', 'operating_margin172_independent_v2.py',
         'test_operating_margin172_independent_v2.py')


def contract():
    return dict(original=old.contract(), correction='Preserve absent literal currency as null, never invent a dollar glyph or ISO currency. All numerical, consolidated annual table, unit, period, source and clock checks remain.',
                cause='11 Microsoft reports held because the selected total revenue row has no dollar glyph; original source extraction and hold records remain unchanged.',
                no_new_financial_ratios=True, no_factors_or_returns=True,
                reuses_original_aapl_output=True, source_fields_admitted=False)


def validate_original():
    plan = old.gate(ORIGINAL_SHA)
    require(sha(old.OUT / 'html/EXTRACTION.json') == HTML_SHA, 'Original HTML output changed')
    require(sha(old.OUT / 'aapl/REPORTS.json') == AAPL_SHA, 'Original Apple output changed')
    return plan


def register():
    p = validate_original()
    require(not OUT.exists(), 'V2 source registry is write-once')
    code = {str(ROOT / 'work' / name): sha(ROOT / 'work' / name) for name in NAMES}
    old.verify(code)
    OUT.mkdir()
    p = dict(p, schema='operating_margin172_source_v2',
             original_plan_sha256=ORIGINAL_SHA, original_aapl_sha256=AAPL_SHA,
             original_html_sha256=HTML_SHA, original_code_sha256=p['code_sha256'],
             code_sha256=code, registered_utc=datetime.now(timezone.utc).isoformat(),
             correction_contract=contract())
    dump(OUT / 'PLAN.json', p)
    print(json.dumps(dict(status='VERSIONED_SOURCE_CORRECTION_REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))))


def gate(expected):
    validate_original()
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit V2 PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['original_plan_sha256'] == ORIGINAL_SHA and p['correction_contract'] == contract(), 'V2 correction contract changed')
    require(p['runtime'] == old.runtime() and p['contract'] == old.contract(), 'Original source contract/runtime changed')
    old.verify(p['input_sha256']); old.verify(p['code_sha256'])
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register', 'verify'])
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.phase == 'register': register()
    else:
        gate(args.plan_sha256)
        print('SOURCE_CORRECTION_VERIFIED')
