"""Offline extraction of the registered historical-price source probes."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_ptr_pool_source169/missing_quote'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, data):
    with p.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    assert sha(OUT/'PLAN.json') == 'b1d5ac84ebfc94db46a47671b24c5d10c614cceeee4e51c90ba8dda34b08cc32'
    plan = json.loads((OUT/'PLAN.json').read_text())
    pins = {'PLAN.json': sha(OUT/'PLAN.json')}
    receipts = {}
    for name, url in plan['sources'].items():
        receipt = OUT/(name+'.receipt.json')
        record = json.loads(receipt.read_text())
        assert record['url'] == url and record['status'] == 200
        assert record['body_sha256'] == sha(OUT/name)
        assert record['plan_sha256'] == sha(OUT/'PLAN.json')
        pins[name] = sha(OUT/name)
        pins[receipt.name] = sha(receipt)
        receipts[name] = record
    runtime = ROOT/'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'
    assert sha(runtime) == 'cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50'
    cfg = json.loads(runtime.read_text())
    for p, digest in cfg['pdf_runtime_hashes'].items():
        assert sha(Path(p)) == digest
    sys.path.append(cfg['pdf_runtime']['bundle_site_packages'])
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument((OUT/'cjsc_20190430.pdf').read_bytes())
    pages = []
    for idx in range(len(doc)):
        page = doc[idx]
        text = page.get_textpage().get_text_range()
        txt = OUT/('cjsc_page%02d.txt' % (idx+1))
        assert not txt.exists()
        txt.write_text(text)
        pins[txt.name] = sha(txt)
        if idx == 0 or '601857' in text or 'A股（人民币）' in text or 'A股(人民币)' in text:
            image = txt.with_suffix('.png')
            assert not image.exists()
            page.render(scale=1.75).to_pil().save(image)
            pins[image.name] = sha(image)
            pages.append(dict(pdf_page=idx+1, text=txt.name, image=image.name))
    doc.close()
    sina = receipts['sina_2019q2.html']
    write(OUT/'EXTRACTION.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), code_sha256=sha(Path(__file__)),
        runtime_sha256=sha(runtime), source_and_output_sha256=pins, pages_for_visual_review=pages,
        sina_rejected=dict(reason='Redirected from historical request to current quote page; HTTP200 does not establish historical table coverage.', final_url=sina['final_url']),
        quote_admitted=False, visual_review_pending=True,
        scope='Source inspection only. No price fill, market cap, pool or return calculation.'))
    print(json.dumps(dict(pages=pages, sina_final_url=sina['final_url']), ensure_ascii=False))

if __name__ == '__main__':
    main()
