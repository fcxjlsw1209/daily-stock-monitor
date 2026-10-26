"""Offline extraction/rendering of the acquired pre-start annual source."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_ptr_pool_source169'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source = OUT / 'petrochina_2015_annual.pdf'
    receipt = json.loads((OUT / 'petrochina_2015_annual.pdf.receipt.json').read_text())
    assert receipt['status'] == 200 and sha(source) == receipt['body_sha256']
    runtime = ROOT / 'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'
    assert sha(runtime) == 'cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50'
    config = json.loads(runtime.read_text())
    for p, h in config['pdf_runtime_hashes'].items():
        assert sha(Path(p)) == h
    sys.path.append(config['pdf_runtime']['bundle_site_packages'])
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(source.read_bytes())
    derived = OUT / 'derived'
    derived.mkdir(exist_ok=True)
    records = []
    for i in [7, 12, 20, 159, 222, 235]:
        page = doc[i]
        text = page.get_textpage().get_text_range()
        txt = derived / ('page%03d.txt' % (i + 1))
        png = derived / ('page%03d.png' % (i + 1))
        assert not txt.exists() and not png.exists()
        txt.write_text(text)
        page.render(scale=1.5).to_pil().save(png)
        records.append(dict(pdf_page=i+1, text=str(txt.relative_to(ROOT)),
                            text_sha256=sha(txt), image=str(png.relative_to(ROOT)), image_sha256=sha(png)))
    doc.close()
    with (derived / 'INDEX.json').open('x') as f:
        json.dump(dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
            runtime_plan_sha256=sha(runtime), script_sha256=sha(Path(__file__)), pages=records), f, indent=2)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
