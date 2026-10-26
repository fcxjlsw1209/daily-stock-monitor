"""Render collected source pages offline; no financial calculations."""
from pathlib import Path
import hashlib
import json
import sys
OUT=Path(__file__).resolve().parent
BUNDLE=Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
sys.path.append(str(BUNDLE))
import pypdfium2 as pdfium
pdf=OUT/'FY2019.response.pdf'
assert hashlib.sha256(pdf.read_bytes()).hexdigest()=='ece287b69ac1a6a030af4c1dce5d1e76cc0c5b85cfed13302440a1a0c372714c'
doc=pdfium.PdfDocument(str(pdf))
result=[]
for number in [1,22,37,39,40]:
    page=doc[number-1]
    image=page.render(scale=1.5).to_pil()
    path=OUT/('FY2019.page'+str(number)+'.png')
    assert not path.exists()
    image.save(path)
    result.append(dict(page_1based=number,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    page.close()
doc.close()
with (OUT/'RENDERED_PAGES.json').open('x') as stream:
    json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result,indent=2))
