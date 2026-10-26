import json,re
from pathlib import Path
from bs4 import BeautifulSoup
from after141_amd_source_extract import table_grid,external_context
import after153_next_source_context as m
f=json.loads(Path('work/after153_next_source_feasibility.json').read_text())
e=next(e for e in f['documents'] if e['event_id']=='TMO_2022-02-02')
s=BeautifulSoup(Path(e['evidence_path']).read_bytes(),'html.parser')
for ti in (1,5):
 t=s.find_all('table')[ti];_,_,p=table_grid(t)
 print('TABLE',ti)
 for c in p.values():
  text=m.f.norm(c['text'])
  if c['row']<10 and (re.search('[A-Za-z]',text) or re.fullmatch('20[0-9]{2}',text)):
   print(c['row'],c['columns'],text if (m.temporal(text) or m.units(text) or not re.search('[0-9]',text)) else m.f.masked(text))
 print('EXTERNAL',[m.f.masked(x) for x in external_context(t)])
