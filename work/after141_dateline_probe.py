from pathlib import Path
from bs4 import BeautifulSoup
import json
import after141_amd_source_extract as a
R=Path(__file__).resolve().parent.parent
e=json.loads((R/'outputs/monthly_earnings_repair/batch_technology_industrial/AMD/events.json').read_text())
for x in e:
 if x['fiscal_quarter']==4:
  text=' '.join(BeautifulSoup((R/x['evidence_path']).read_bytes(),'html.parser').get_text(' ',strip=True).split())
  for z in x['release_date_evidence']:
   v=' '.join(z['excerpt'].split());i=text.find('SUNNYVALE');print(json.dumps({'year':x['fiscal_year'],'match':v in text,'dates':a.dates([v]),'excerpt':v,'start':text[max(0,i-100):i+150]}))
