"""Bounded source heading diagnostic; no production imports."""
import json
from bs4 import BeautifulSoup, NavigableString
import after149_batch1_independent_review as a
r=a.Reader(); p=r.load(a.PLAN,a.PLAN_SHA)
for row in p['records']:
    if row['event_id'] not in ('CSCO_2011-08-10','CSCO_2019-08-14','CSCO_2023-08-16'):
        continue
    ctx=next(e for e in r.load(row['context_path'])['events'] if e['event_id']==row['event_id'])
    soup=BeautifulSoup(r.read(row['source_path'],row['source_sha256']),'html.parser')
    for x in soup(['script','style']):x.decompose()
    for field in ('annual_net_income','current_total_assets'):
        ti,ri,cols=a.source_coordinates(ctx,'CSCO',field); table=soup.find_all('table')[ti]
        print(json.dumps({"event":row["event_id"],"field":field,"headtexts":[v["text"]for v in a.grid_from_html(table)[0].values()if v["row"]<4]},ensure_ascii=False));strings=[]
        for i,node in enumerate(table.previous_elements):
            if i>70:break
            if isinstance(node,NavigableString) and a.norm(str(node)):
                strings.append({'text':a.norm(str(node))[:250], 'inside_table':node.find_parent('table')is not None,'tag':node.parent.name})
        print(json.dumps({'event':row['event_id'],'field':field,'table':ti,'external':a.external_heading(table),'previous':strings},ensure_ascii=False))
