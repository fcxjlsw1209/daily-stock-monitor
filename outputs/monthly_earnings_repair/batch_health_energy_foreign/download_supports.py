import collect as c,re
from concurrent.futures import ThreadPoolExecutor
from extend_sources import save_tasks

def mrk():
 arr=c.load(c.HERE/'MRK_supports.json',[])
 for mp in (c.RAW/'MRK').glob('*.meta.json'):
  m=c.load(mp,{})
  if m.get('role')!='archive_json':continue
  for report in c.load(mp.with_suffix('').with_suffix('.bin'),[]):
   label=report.get('title',{}).get('rendered','')
   for d in report.get('extended_data',{}).get('acf_fields',{}).get('documents',[]):
    if d.get('document_category') not in ['10Q','10K']:continue
    u=d.get('document_path','')
    if 'sec.gov' in u or not '.pdf' in u.lower():continue
    arr.append(dict(ticker='MRK',url=u,label=label))
 for l in c.load(c.HERE/'MRK_links2.json',[]):
  if 'Form 10-Q' in l['text']:arr.append(dict(ticker='MRK',url=l['url'],label='First Quarter 2026' if '03.31' in l['url'] else 'Second Quarter 2026'))
 arr=list({a['url']:a for a in arr}.values());c.dump(c.HERE/'MRK_supports.json',arr)
 for a in arr:
  try:p,m=c.fetch(a['url'],'MRK','fiscal_period_support');c.text_of(p)
  except Exception as e:
   print('MRK',str(e),flush=True)
   if getattr(e,'code',None) in [403,429]:break
 print('MRK supports',len(arr),flush=True)
def chl():
 arr=[]
 for mp in (c.RAW/'CHL').glob('*.meta.json'):
  m=c.load(mp,{})
  if m.get('role')!='discovery' or 'hkex_announcements' not in m['url']:continue
  p=mp.with_suffix('').with_suffix('.bin')
  for l in c.links(p,m['url']):
   if re.search(r'Unaudited Key Performance Indicators.*(?:Quarter|Three Quarters)',l['text'],re.I):arr.append(dict(ticker='CHL',url=l['url'],label=l['text'],archive_url=m['url'],archive_path=c.rel(p),archive_sha256=m['sha256']))
 save_tasks('CHL',arr)
def oracle():
 # Current migrated CDN keeps observed historical document paths. Downloaded
 # original content alone determines dates, never the URL or filename.
 arr=c.load(c.HERE/'ORCL_candidates.json',[])
 for y in range(2027,2010,-1):
  for q in range(1,5):
   if y==2027 and q>1:continue
   month=['September','December','March','June'][q-1];yy=str(y)[-2:]
   for variant in [f'{q}q{yy}-pressrelease-{month}-final.pdf',f'{q}q{yy}-pressrelease-{month}.pdf',f'{q}q{yy}-pressrelease-{month.lower()}.pdf']:
    u=f'https://s23.q4cdn.com/440135859/files/doc_financials/{y}/q{q}/{variant}'
    try:
     p,m=c.fetch(u,'ORCL','earnings_candidate');c.text_of(p);arr.append(dict(ticker='ORCL',url=u,label='Original Oracle Corporation earnings release'));break
    except Exception as e:
     if getattr(e,'code',None) in [403,429] or 'source stopped' in str(e):c.dump(c.HERE/'ORCL_candidates.json',list({a['url']:a for a in arr}.values()));print('ORCL STOP',str(e),flush=True);return
 c.dump(c.HERE/'ORCL_candidates.json',list({a['url']:a for a in arr}.values()));print('ORCL all',len(arr),flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
 for f in [pool.submit(mrk),pool.submit(chl),pool.submit(oracle)]:
  try:f.result()
  except Exception as e:print(str(e),flush=True)
