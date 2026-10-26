"""Diagnose only the three declared failed Apple SBC cells; emit no amounts."""
from pathlib import Path
import json
import re
import sys
import stock_comp_source_v1 as old

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'outputs/monthly_stock_comp_source_v1/PLAN.json'
PLAN_SHA = 'ef563d83e864c3e592c75f232dccc1327aed9b1fc95b5a2115c703ad51df45ba'
MANIFEST = 'outputs/monthly_stock_comp_source_v1/MANIFEST.json'
MANIFEST_SHA = '5e25899f13b31bc631d1ef5bebc3d5f76e0a14bbb5866bd41ff104b288736f31'
EVENT_IDS = ('AAPL:FY2021:2021-10-28', 'AAPL:FY2022:2022-10-27', 'AAPL:FY2023:2023-11-02')


def main():
    base, pdf = old.base, old.pdf_repair
    plan = base.read(ROOT, PLAN, PLAN_SHA)
    base.read(ROOT, MANIFEST, MANIFEST_SHA)
    base.require(plan['runtime'] == sys.version, 'Frozen runtime')
    base.safe.verify_hashes(ROOT, plan['source_code_hashes'])
    _, pdfium = base.configure_pdf(plan)
    findings = []
    for event_id in EVENT_IDS:
        specs = [e for e in plan['events'] if e['event_id'] == event_id]
        base.require(len(specs) == 1, 'Exact declared event')
        spec = specs[0]
        source = spec['source']
        base.safe.verify_hashes(ROOT, {source['path']: source['sha256'], source['receipt']['path']: source['receipt']['sha256']})
        locs = spec['locator_inventory']['candidates']
        base.require(len(locs) == 1 and not spec['locator_inventory']['gaps'], 'Unique frozen locator')
        loc = locs[0]
        raw = base.safe.local(ROOT, source['path']).read_bytes()
        texts = base.pdf_texts(raw, plan)
        finding = dict(event_id=event_id, source=source, locator=loc)
        try:
            pdf.extract_cell(raw, texts, loc, plan)
        except (ValueError, IndexError, KeyError) as exc:
            finding['frozen_extractor_error'] = dict(type=type(exc).__name__, message=str(exc))
        else:
            raise ValueError('Expected frozen extractor failure; diagnostic cannot publish successful values')
        line = texts[loc['page_1based']-1].splitlines()[loc['line_1based']-1]
        finding['pypdf_row_masked'] = re.sub(r'\d', '#', line)
        doc = pdfium.PdfDocument(raw); page = doc[loc['page_1based']-1]; tp = page.get_textpage()
        try:
            rows = pdf.physical_rows(tp)
            diagnostic_rows = []
            label = pdf.compact(loc['row_label'])
            for row in rows:
                full = pdf.row_text(row)
                if label not in pdf.compact(full):
                    continue
                groups = []
                tail = row[len(label):]
                for group in pdf.numeric_groups(tail):
                    gaps = []
                    for left, right in zip(group, group[1:]):
                        gaps.append(dict(indices=[left['index'], right['index']],
                                         gap=right['bbox'][0]-left['bbox'][2]))
                    groups.append(dict(masked_text=re.sub(r'\d', '#', pdf.row_text(group)),
                                       bbox=pdf.bbox(group), indices=[c['index'] for c in group], gaps=gaps))
                try:
                    pdf.numeric_tail(tail)
                    tail_error = None
                except (ValueError, IndexError, KeyError) as exc:
                    tail_error = dict(type=type(exc).__name__, message=str(exc))
                diagnostic_rows.append(dict(masked_text=re.sub(r'\d', '#', full),
                                            bbox=pdf.bbox(row), startswith_exact_label=full.lower().startswith(label),
                                            groups=groups, numeric_tail_error=tail_error))
            finding['pdfium_rows'] = diagnostic_rows
            finding['header_match_counts'] = [sum(pdf.compact(pdf.row_text(r)) == pdf.compact(h) for r in rows) for h in loc['header_literals']]
        finally:
            tp.close(); page.close(); doc.close()
        findings.append(finding)
    result = dict(stage='masked_frozen_extractor_diagnostic_only', plan_sha256=PLAN_SHA,
                  manifest_sha256=MANIFEST_SHA, diagnostic_code_sha256=base.sha(ROOT, 'work/stock_comp_pdf_repair_v2_diagnose.py'),
                  new_values_published=0, new_parser_executed=False, findings=findings)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
