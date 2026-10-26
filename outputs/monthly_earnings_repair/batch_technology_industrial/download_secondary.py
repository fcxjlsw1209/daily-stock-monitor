from common import *
import concurrent.futures

def micron():
 for p in sorted((HERE/'MU/public_index_observations').glob('*.json')):
  if p.name=='archive.json':continue
  s=json.loads(p.read_text())['rendering'];match=re.search(r'\((https?://[^\s)]+)\)',s)
  if not match:continue
  try:b,p,m=fetch('MU',match.group(1));print('MU',len(b),flush=True)
  except Exception as e:print('MU ERROR',str(e),flush=True)
def palantir():
 feed=json.loads((HERE/'PLTR/raw/8k_feed.bin').read_text())['GetEdgarFilingListResult']
 rows=json.loads((HERE/'PLTR/raw/news_feed.bin').read_text())['GetPressReleaseListResult']
 dates={dt.datetime.strptime(x['PressReleaseDate'],'%m/%d/%Y %H:%M:%S').date().isoformat() for x in rows if re.search(r'Reports|Delivers|Raises',x['Headline'])}
 for x in feed:
  d=dt.datetime.strptime(x['FilingDate'],'%m/%d/%Y %H:%M:%S').date().isoformat()
  if d not in dates:continue
  u=next((z['Url'] for z in x['DocumentList'] if z['DocumentType']=='HTML'),None)
  if not u:continue
  try:b,p,m=fetch('PLTR',u,'filing_'+str(x['FilingId']));print('PLTR',d,len(b),flush=True)
  except Exception as e:print('PLTR ERROR',str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda f:f(),[micron,palantir]))
