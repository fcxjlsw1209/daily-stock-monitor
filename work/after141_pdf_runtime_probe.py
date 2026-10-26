import importlib,sys,json
from pathlib import Path
p=Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
sys.path.append(str(p))
rows=[]
for name in ['pypdf','pdfplumber','pypdfium2']:
 try:
  m=importlib.import_module(name);rows.append({'module':name,'version':getattr(m,'__version__',str(getattr(m,'PYPDFIUM_INFO','unknown'))),'path':str(m.__file__),'ok':True})
 except Exception as e:rows.append({'module':name,'ok':False,'error':repr(e)})
print(json.dumps({'runtime':sys.version,'libraries':rows}))
