from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parent.parent
D=R/'outputs/monthly_profitability_extension_sources_v1'
p=D/'EXTRACTION_PLAN.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50'
plan=json.loads(p.read_text())
for rel,h in plan['pdf_runtime_hashes'].items():assert hashlib.sha256(Path(rel).read_bytes()).hexdigest()==h
sys.path.append(plan['pdf_runtime']['bundle_site_packages'])
from pypdf import PdfReader
import pypdfium2 as pdfium
A=D/'pdf_artifacts';A.mkdir(exist_ok=True)
for spec in plan['sources']:
 if spec['symbol']!='AAPL':continue
 raw=(R/spec['raw_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==spec['raw_sha256']
 import io
 reader=PdfReader(io.BytesIO(raw));pdf=pdfium.PdfDocument(raw)
 out=A/str(spec['fiscal_year']);out.mkdir(exist_ok=True)
 pages=[]
 for i,page in enumerate(reader.pages):
  if i>1:break
  text=page.extract_text(extraction_mode='layout')
  (out/('page'+str(i+1)+'.txt')).write_text(text)
  image=pdf[i].render(scale=1.25).to_pil();image.save(out/('page'+str(i+1)+'.png'))
  pages.append({'page':i+1,'lines':len(text.splitlines())})
 print(json.dumps({'year':spec['fiscal_year'],'page_count':len(reader.pages),'artifacts':pages}))
