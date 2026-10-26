"""Independent, bounded source inspection; no financial feature or admission.

Render the first (income statement) page of the ten pinned Apple PDFs and
inspect owned HTML statement tables for seven preselected structural examples.
This deliberately does not import the producer or extract all 44 share pairs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after158_basic_shares_independent.py'
OUT = 'work/after158_basic_shares_independent'
PLAN = 'work/after158_basic_shares_source_plan.json'
PLAN_SHA = '3ff000c9a1b9c17d04d2a2edf2f79f21b18811f4e32263b38b768ace805ca60b'
INVENTORY = 'outputs/after158_basic_shares_inventory_v1/INVENTORY.json'
ADDENDUM = 'outputs/after158_basic_shares_inventory_v1/PDF_TEXT_ADDENDUM.json'
SAMPLE = {
    'AMD:FY2016:2017-01-31', 'AMD:FY2023:2024-01-30',
    'MSFT:FY2015:2015-07-21', 'MSFT:FY2022:2022-07-26',
    'NVDA:FY2015:2015-02-11', 'NVDA:FY2022:2022-02-16', 'NVDA:FY2025:2025-02-26',
}
PDF_SITE = '/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(relative, expected=None):
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
        raise ValueError('Nonlocal input')
    raw = path.read_bytes()
    if expected is not None and sha(raw) != expected:
        raise ValueError('Hash mismatch: ' + relative)
    return raw


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


def owned_rows(table):
    return [r for r in table.find_all('tr') if r.find_parent('table') is table]


def html_evidence(raw):
    soup = BeautifulSoup(raw, 'html.parser')
    tables = []
    for index, table in enumerate(soup.find_all('table')):
        rows = owned_rows(table)
        texts = [norm(r.get_text(' ', strip=True)) for r in rows]
        indices = [i for i, text in enumerate(texts) if re.search(
            r'(?:weighted[\s-]*average.*shares|shares.*(?:computation|calculation|computing))', text, re.I)]
        if not indices:
            continue
        # Text around the table is evidence only, never used as its GAAP label.
        prefix = []
        for node in table.previous_siblings:
            text = norm(node.get_text(' ', strip=True) if hasattr(node, 'get_text') else str(node))
            if text:
                prefix.append(text)
            if len(prefix) >= 5:
                break
        selected = set(range(min(8, len(rows))))
        for i in indices:
            selected.update(range(max(0, i-1), min(len(rows), i+4)))
        evidence = []
        for i in sorted(selected):
            cells = [c for c in rows[i].find_all(['td', 'th']) if c.find_parent('tr') is rows[i]]
            evidence.append({'row_index': i, 'text': texts[i], 'cells': [
                {'physical_index': j, 'text': norm(c.get_text(' ', strip=True)),
                 'colspan': str(c.get('colspan', 1)), 'rowspan': str(c.get('rowspan', 1))}
                for j, c in enumerate(cells)]})
        tables.append({'table_index': index, 'preceding_siblings_nearest_first': prefix,
                       'owned_row_count': len(rows), 'rows': evidence})
    # Sentence-like block evidence includes split disclosures without relabeling
    # any table on the basis of non-GAAP words elsewhere in the document.
    notes = []
    for node in soup.find_all(['p', 'div', 'td']):
        if node.find(['p', 'div', 'td']):
            continue
        text = norm(node.get_text(' ', strip=True))
        if len(text) < 2500 and re.search(r'\bsplit\b|retroactiv|retrospectiv|recast', text, re.I):
            if text not in notes:
                notes.append(text)
    return {'tables': tables, 'split_or_restatement_blocks': notes}


def write(path, raw):
    with (ROOT / path).open('xb') as handle:
        handle.write(raw)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def inspect():
    raw_plan = read(PLAN, PLAN_SHA)
    plan = json.loads(raw_plan)
    pins = {PLAN: PLAN_SHA, SELF: sha(read(SELF)), INVENTORY: sha(read(INVENTORY)),
            ADDENDUM: sha(read(ADDENDUM))}
    events = plan['events']
    inventory = json.loads(read(INVENTORY))
    if len(events) != 44 or [e['event_id'] for e in events] != [e['event_id'] for e in inventory['events']]:
        raise ValueError('Original 44-event denominator mismatch')
    # Verify raw and receipt identities for the complete denominator, without
    # parsing contents outside the predeclared independent inspection sample.
    for event in events:
        source = event['source']
        for ref in (source, source['receipt']):
            read(ref['path'], ref['sha256'])
            if ref['path'] in pins and pins[ref['path']] != ref['sha256']:
                raise ValueError('Conflicting pin')
            pins[ref['path']] = ref['sha256']
    out = ROOT / OUT
    out.mkdir(exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Independent inspection directory must be empty')
    sys.path.append(PDF_SITE)
    from pypdf import PdfReader
    import pypdfium2 as pdfium
    import pypdf
    from PIL import Image
    runtime = {str(Path(m.__file__).resolve()): sha(Path(m.__file__).read_bytes())
               for m in (pypdf, pdfium, Image)}
    results = []
    for event in events:
        if event['symbol'] != 'AAPL' and event['event_id'] not in SAMPLE:
            continue
        source = event['source']
        raw = read(source['path'], source['sha256'])
        result = {'event_id': event['event_id'], 'source': source, 'periods': event['periods']}
        if event['symbol'] == 'AAPL':
            doc = pdfium.PdfDocument(raw)
            page = doc[0]
            textpage = page.get_textpage()
            image = page.render(scale=1.6).to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_path = OUT + '/AAPL_FY%s_page1.png' % event['fiscal_year']
            write(image_path, buffer.getvalue())
            result.update(page_1based=1, page_count=len(doc),
                          full_page_image={'path': image_path, 'sha256': sha(buffer.getvalue())},
                          layout_text=PdfReader(io.BytesIO(raw)).pages[0].extract_text(extraction_mode='layout'),
                          pdfium_text=textpage.get_text_range(),
                          geometry_note='Full original page image supplied for human column alignment; no automatic certification.')
            textpage.close()
            page.close()
            doc.close()
        else:
            result['html_evidence'] = html_evidence(raw)
        results.append(result)
    for path, digest in pins.items():
        read(path, digest)
    report = {'schema':'after158_basic_shares_independent_inspection_v1',
              'created_utc':datetime.now(timezone.utc).isoformat(), 'input_hashes':pins,
              'runtime_entrypoint_hashes':runtime, 'original_event_count':44,
              'raw_and_receipt_pins_verified':len(set(e['source']['path'] for e in events)) +
                    len(set(e['source']['receipt']['path'] for e in events)),
              'inspected_sample':results, 'selection_rule':'All 10 Apple page1 layouts; seven predefined HTML structural/split examples, independent of outcomes.',
              'source_admission_changed':False, 'financial_features_computed':0,
              'returns_computed':0, 'orders_computed':0, 'new_strategy_trials':0,
              'visual_review_completed':False}
    target = OUT + '/INSPECTION.json'
    data = encoded(report)
    write(target, data)
    print(json.dumps({'path':target, 'sha256':sha(data), 'events':len(results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true', required=True)
    parser.parse_args()
    inspect()
