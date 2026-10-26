"""Offline metadata-only ABBV statement audit; never parses financial values."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup
import profitability141_issuer_statements as layout

ROOT = Path(__file__).resolve().parent.parent
OUT = 'outputs/monthly_abbv_revenue156/'
META = 'outputs/monthly_revenue_coverage156/ANNUAL_REPORTS.json'
EVENTS = 'outputs/monthly_earnings_repair/batch_health_energy_foreign/events.json'
SELF = 'work/after155_abbv_revenue_audit.py'

def digest(raw): return hashlib.sha256(raw).hexdigest()
def bind(rel):
    raw = (ROOT / rel).read_bytes()
    return {'path': rel, 'sha256': digest(raw), 'bytes': len(raw)}
def read(b):
    raw = (ROOT / b['path']).read_bytes()
    if digest(raw) != b['sha256'] or len(raw) != b['bytes']: raise ValueError('Changed pin: '+b['path'])
    return raw
def write(name, data):
    p = ROOT / OUT / name
    p.parent.mkdir(exist_ok=True, parents=True)
    with p.open('x') as f: json.dump(data, f, indent=2, sort_keys=True); f.write('\n')
    return bind(OUT + name)
def freeze():
    records = [r for r in json.loads((ROOT / META).read_bytes()) if r['symbol'] == 'ABBV']
    assert [r['fiscal_year'] for r in records] == list(range(2013, 2026))
    sources = [{'metadata': r, 'raw': bind(r['path']), 'receipt': bind(r['path'].replace('.bin','.meta.json'))} for r in records]
    for s in sources: assert s['raw']['sha256'] == s['metadata']['sha256']
    print(json.dumps(write('SOURCE_SCOPE.json', {
        'created_utc':datetime.now(timezone.utc).isoformat(), 'status':'SOURCE_ONLY_BEFORE_MONETARY_EXTRACTION',
        'symbol':'ABBV', 'years':list(range(2013,2026)), 'sources':sources,
        'inputs':[bind(p) for p in [META, EVENTS, SELF, 'work/after155_multi_issuer_source_map.json', 'work/profitability141_issuer_statements.py', 'run_local.sh','local_run.py','local.sandbox.sb']],
        'policy': 'Only original annual consolidated total net revenue; current and same-report prior year separately. Metadata audit of title/annual header/label/units/receipt precedes frozen parser profile. No amounts, factors, returns, strategy admission, network or cloud reads. Ambiguous and adjusted candidates held; no segment sum or comparative backfill.'})))
def audit():
    scope=json.loads((ROOT / OUT / 'SOURCE_SCOPE.json').read_bytes())
    output=[]
    for source in scope['sources']:
        raw=read(source['raw']); receipt=json.loads(read(source['receipt']))
        soup=BeautifulSoup(raw, 'html.parser'); tables=[]
        for ti,t in enumerate(soup.find_all('table')):
            rows,grid,physical=layout.table_grid(t)
            labels=[(ri,layout.norm(c.get_text(' ',strip=True))) for ri,row in enumerate(rows) for c in row.find_all(['td','th'],recursive=False) if re.fullmatch(r'(?:total )?(?:net )?(?:revenues?|sales)',layout.norm(c.get_text(' ',strip=True)),re.I)]
            if not labels: continue
            first=min(ri for ri,_ in labels)
            previous=[]
            for p in t.find_all_previous(['p','h1','h2','h3','h4'],limit=7):
                if p.find_parent('table') is None: previous.append(layout.norm(p.get_text(' ',strip=True)))
            tables.append({'table_index':ti,'table_html_sha256':digest(str(t).encode()),
                           'labels':[{'row_index':ri,'label':label,'cell_geometry':[{'physical_cell':pi,'columns':v['columns']} for (r,pi),v in physical.items() if r==ri]} for ri,label in labels],
                           'header_rows':[[layout.norm(c.get_text(' ',strip=True)) for c in row.find_all(['td','th'],recursive=False)] for row in rows[:first]],
                           'preceding_non_table_elements':previous,
                           'nonmonetary_row_labels':[layout.norm(row.find(['td','th'],recursive=False).get_text(' ',strip=True)) for row in rows[first:] if row.find(['td','th'],recursive=False) is not None]})
        date_elements=[{'tag':t.name,'class':t.get('class'), 'text':layout.norm(t.get_text(' ',strip=True))} for t in soup.find_all(['time','span','div']) if 'date' in str(t.get('class','')).lower() and len(t.get_text(' ',strip=True))<70]
        output.append({'fiscal_year':source['metadata']['fiscal_year'],'receipt':receipt,'date_elements':date_elements,'candidate_table_metadata':tables})
    print(json.dumps(write('LAYOUT_AUDIT.json',{'scope':bind(OUT+'SOURCE_SCOPE.json'),'monetary_values_parsed':False,'reports':output})))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','audit']);a=p.parse_args()
    {'freeze':freeze,'audit':audit}[a.phase]()
