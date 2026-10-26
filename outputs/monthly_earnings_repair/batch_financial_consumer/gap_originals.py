import json,re
import core as c
from expand_download import saveq
for s in ['HD','PYPL','ABT']:
 p=c.HERE/s/'indexed_gaps.json';items=[]
 for title,url in re.findall(r'^\s*(.*?) \((https://[^\s]+)\)',json.loads(p.read_text())['response'],re.M):
  good=s=='HD'and'2019-q4-pr.pdf'in url or s=='PYPL'and'2019-07-24-paypal-reports'in url or s=='ABT'and'Abbott Reports'in title and'Quarter'in title
  if good:items.append(dict(url=url.split('?')[0],text=title,discovery_index=c.proof(p,url)))
 saveq(s,items)
