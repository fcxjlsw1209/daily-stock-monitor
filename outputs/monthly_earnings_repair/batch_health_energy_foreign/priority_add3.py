import collect as c
from pathlib import Path

def indexed(t,u,tx,srcs,label,support=False):
 p=c.RAW/t/('index_priority3_'+c.sha(u.encode())[:20]+'.json')
 c.dump(p,dict(url=u,retrieved_at=max(c.load(s,{})['retrieved_at'] for s in srcs),source_representation='public_search_index_not_original_bytes',original_bytes_available=False,indexed_source_text=tx,upstream_captures=[dict(path=c.rel(s),sha256=c.sha(s.read_bytes())) for s in srcs]))
 f=c.HERE/(t+('_supports.json' if support else '_candidates.json'));arr=c.load(f,[]);arr.append(dict(ticker=t,url=u,label=label,evidence_file=c.rel(p),indexed_original=True));c.dump(f,list({a.get('evidence_file',a['url']):a for a in arr}.values()))
headers=c.RAW/'priority_lly_headers.json';tables=c.RAW/'priority_lly_open.json'
h={v['value']['url']:v['value']['response'] for v in c.load(headers,{})['results'] if v['status']=='fulfilled'}
for v in c.load(tables,{})['results']:
 if v['status']=='fulfilled':
  u=v['value']['url'];indexed('LLY',u,h[u]+'\n\nStatement tables of same official original:\n'+v['value']['response'],[headers,tables],'Official Lilly original release')
src=c.RAW/'priority_search5.json';indexed('ASML','https://www.asml.com/en/news/press-releases/2021/asml-reports-eur-14dot0-billion-net-sales-and-eur3dot6-billion-net-income-in-2020',c.load(src,{})['results'][1]['value'],[src],'Q4 2020 official original indexed')
src=c.RAW/'priority_search7.json';d=c.load(src,{})
indexed('UNH','https://www.sec.gov/Archives/edgar/data/731766/000073176618000015/a2018q1exhibit991.htm',d['results'][0]['value'],[src],'First Quarter 2018 official release exhibit')
indexed('SAP','https://www.sap.com/docs/download/investors/2025/sap-2025-q2-statement.pdf',d['results'][2]['value'],[src],'Q2 2025 original quarterly statement',True)
urls={
 'LLY':['https://www.prnewswire.com/news-releases/lilly-delivers-first-quarter-2021-financial-results-adjusts-2021-financial-guidance-301277287.html'],
 'SAP':['https://news.sap.com/africa/2020/07/sap-announces-second-quarter-and-half-year-2020-results/','https://news.sap.com/2025/07/sap-announces-q2-and-hy-2025-results/'],
 'ORCL':['https://s23.q4cdn.com/440135859/files/doc_financials/2024/Q2/2q24-pressrelease-December-final.pdf','https://www.oracle.com/in/news/announcement/q2fy24-earnings-release-2023-12-11/'],
 'PFE':['https://s206.q4cdn.com/795948973/files/doc_financials/2017/q1/Q1_2017_Earnings_Press_Release.pdf']}
for t,us in urls.items():
 arr=c.load(c.HERE/(t+'_candidates.json'),[])
 for u in us:
  try:
   p,m=c.fetch(u,t,'priority original release');arr.append(dict(ticker=t,url=u,label='Original issuer earnings release'));print(t,m['bytes'],u,flush=True)
  except Exception as e: print(t,e,flush=True)
 c.dump(c.HERE/(t+'_candidates.json'),list({a.get('evidence_file',a['url']):a for a in arr}.values()))
