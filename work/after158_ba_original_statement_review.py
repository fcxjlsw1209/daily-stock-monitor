"""Offline PDF evidence extraction for the frozen five-document BA source plan.

Produces page-level evidence, not a financial factor, admission or strategy.
Original selection and accounting adjudication are recorded separately after
visual inspection. No source file is modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import unittest

import after158_ba_source_plan as contract

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after158_ba_original_statement_review.py'
PLAN = 'work/after158_ba_source_plan.json'
OUT = 'work/after158_ba_original_statement_review'
PDFIUM_SITE = Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
PATTERNS = {
    'formal_statement_heading': r'Consolidated Statements of Operations',
    'adoption_or_restatement': r'\badopt(?:ing|ed|ion)?\b|\brestat(?:ed|ement)\b|\bretrospective',
    'scope_or_classification': r'\breclass|\bdiscontinued|\bconsolidation|\bprogram accounting|\bchange in accounting',
}


def classify(text, number):
    result = ['dateline_page'] if number == 1 else []
    result += [name for name, pattern in PATTERNS.items() if re.search(pattern, text, re.I)]
    return result


def write_once(path, raw):
    with path.open('xb') as handle:
        handle.write(raw)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def verify(plan):
    for path, sha in {**plan['source_hashes'], **plan['context_hashes']}.items():
        contract.read(ROOT, path, sha)


def extract(plan_sha, script_sha):
    plan = json.loads(contract.read(ROOT, PLAN, plan_sha))
    contract.read(ROOT, SELF, script_sha)
    contract.require(plan['source_objects'] == 22 and len(plan['targets']) == 5, 'Unexpected plan scope')
    verify(plan)
    sys.path.append(str(PDFIUM_SITE))
    from pypdf import PdfReader
    import pypdfium2 as pdfium
    import pypdf
    from PIL import Image
    directory = contract.local(ROOT, OUT)
    directory.mkdir(exist_ok=True)
    contract.require(not any(directory.iterdir()), 'Evidence directory is not empty; never overwrite')
    runtime = {}
    for module in (pypdf, pdfium, Image):
        location = Path(module.__file__).resolve()
        runtime[str(location)] = contract.digest(location.read_bytes())
    records = []
    for target in plan['targets']:
        year = target['fiscal_year']
        raw_path = target['paths']['raw']
        raw = contract.read(ROOT, raw_path, plan['source_hashes'][raw_path])
        reader = PdfReader(io.BytesIO(raw))
        document = pdfium.PdfDocument(raw)
        contract.require(len(reader.pages) == len(document), 'PDF parser page-count mismatch')
        pages = []
        for idx, page in enumerate(reader.pages):
            number = idx + 1
            text = page.extract_text() or ''
            tags = classify(text, number)
            pdfium_page = document[idx]
            textpage = pdfium_page.get_textpage()
            other_text = textpage.get_text_range()
            textpage.close()
            item = dict(page_1based=number, text=text, text_sha256=contract.digest(text.encode()),
                        pdfium_text=other_text, pdfium_text_sha256=contract.digest(other_text.encode()),
                        tags=tags)
            if tags:
                relative = OUT + '/FY%d_page%02d.png' % (year, number)
                path = contract.local(ROOT, relative)
                image = pdfium_page.render(scale=1.4).to_pil()
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                encoded = buffer.getvalue()
                write_once(path, encoded)
                item['rendered_image'] = dict(path=relative, sha256=contract.digest(encoded))
            pages.append(item)
            pdfium_page.close()
        document.close()
        record = dict(fiscal_year=year, role=target['role'], raw_path=raw_path,
                      raw_sha256=plan['source_hashes'][raw_path], page_count=len(pages), pages=pages)
        relative = OUT + '/FY%d_pages.json' % year
        encoded = json_bytes(record)
        write_once(contract.local(ROOT, relative), encoded)
        records.append(dict(fiscal_year=year, path=relative, sha256=contract.digest(encoded),
                            page_count=len(pages), rendered_pages=[p['page_1based'] for p in pages if 'rendered_image' in p]))
    verify(plan)
    contract.read(ROOT, PLAN, plan_sha)
    contract.read(ROOT, SELF, script_sha)
    manifest = dict(schema='after158_ba_pdf_evidence_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    input_hashes={PLAN: plan_sha, SELF: script_sha, **plan['source_hashes'], **plan['context_hashes']},
                    runtime_module_entrypoint_hashes=runtime, records=records,
                    rendering='PDFium 1.4x; selected complete pages; all pages independently extracted with pypdf and PDFium',
                    source_admission_changed=False, new_strategy_trials=0, financial_factors_computed=0,
                    returns_computed=0, visual_review_completed=False,
                    limitation='Candidate tags locate pages only. No keyword absence or parser output establishes accounting continuity.')
    encoded = json_bytes(manifest)
    path = OUT + '/MANIFEST.json'
    write_once(contract.local(ROOT, path), encoded)
    print(json.dumps(dict(path=path, sha256=contract.digest(encoded), records=records), indent=2))


class Tests(unittest.TestCase):
    def test_formal_and_supplement_both_remain_visible(self):
        text = 'Consolidated Statements of Operations\nRestated following adoption of ASC 606'
        self.assertEqual(classify(text, 3), ['formal_statement_heading', 'adoption_or_restatement'])

    def test_negative_keyword_search_is_not_admission(self):
        self.assertEqual(classify('Ordinary operating discussion', 5), [])

    def test_first_page_always_rendered(self):
        self.assertEqual(classify('Boeing January 31', 1), ['dateline_page'])

    def test_valid_date_does_not_create_a_timing_model(self):
        self.assertEqual(classify('Published January 31, 2018', 2), [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--script-sha256')
    args = parser.parse_args()
    if args.test_only:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    contract.require(args.extract and args.plan_sha256 and args.script_sha256, 'Pinned extraction arguments required')
    extract(args.plan_sha256, args.script_sha256)


if __name__ == '__main__':
    main()
