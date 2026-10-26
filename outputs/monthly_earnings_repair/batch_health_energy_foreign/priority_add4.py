import collect as c
exec((c.HERE/'priority_add3.py').read_text().split('headers=c.RAW')[0])
src=c.RAW/'priority_search8.json';d=c.load(src,{})
indexed('ORCL','https://investor.oracle.com/investor-news/news-details/2018/Q3-FY18-Cloud-Revenues-Up-32-to-16-Billion-and-Total-Revenues-Up-6-to-98-Billion/default.aspx',d['results'][1]['value'],[src],'Third Quarter Fiscal 2018 original indexed')
src=c.RAW/'priority_search9.json';d=c.load(src,{})
# Search response contains only the requested issuer-original exhibit here.
indexed('UNH','https://www.sec.gov/Archives/edgar/data/731766/000073176618000002/q42017exhibit991.htm',d['results'][0]['value'],[src],'Fourth Quarter 2017 original exhibit indexed')
indexed('SAP','https://www.sap.com/docs/download/investors/2020/sap-2020-q2-statement.pdf',d['results'][2]['value'],[src],'Q2 2020 original quarterly statement',True)
u='https://s21.q4cdn.com/317678438/files/doc_financials/2019/q3/Q3-2019-PFE-Earnings-Press-Release-FINAL.pdf'
try:
 p,m=c.fetch(u,'PFE','priority original release');arr=c.load(c.HERE/'PFE_candidates.json',[]);arr.append(dict(ticker='PFE',url=u,label='Third Quarter 2019 official original'));c.dump(c.HERE/'PFE_candidates.json',list({a.get('evidence_file',a['url']):a for a in arr}.values()));print(m['bytes'],u)
except Exception as e:print(e)
