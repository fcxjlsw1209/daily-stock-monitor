"""Derived PDF text; original .bin bytes and their checksums remain evidence."""
import sys
from pathlib import Path
from pypdf import PdfReader

for name in sys.argv[1:]:
    root=Path(__file__).resolve().parent/name/'raw'
    for f in sorted(root.glob('*.bin')):
        out=f.with_suffix('.txt')
        if out.exists() or not f.read_bytes().startswith(b'%PDF'): continue
        try:
            out.write_text('\n'.join(p.extract_text() or '' for p in PdfReader(f).pages))
        except Exception as e:
            print(f.name,str(e),flush=True)
