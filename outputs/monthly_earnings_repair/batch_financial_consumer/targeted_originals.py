import json,re,sys,urllib.parse
import core as c
from expand_download import saveq
from archive_download import links
def run(s):
 items=[]
 for p in (c.HERE/s).glob("indexed*.json"):
  if p.name.endswith(".meta.json"):continue
  for title,url in re.findall(r"^\s*(.*?) \((https://[^\s]+)\)",json.loads(p.read_text())["response"],re.M):
   if s=="MA" and "/479285134/"in url and "Earnings-Release"in url:items.append(dict(url=url,text=title,discovery_index=c.proof(p,url)))
   if s=="ABT"and"abbott.mediaroom.com/"in url and re.search("Abbott Reports.*Quarter",title,re.I):items.append(dict(url=url.split("?")[0],text=title,discovery_index=c.proof(p,url)))
   if s=="NKE"and"Fiscal 2021 Fourth"in title and "Reports"in title:items.append(dict(url=url.split("?")[0],text=title,discovery_index=c.proof(p,url)))
 if s=="MA":
  for y in range(2011,2027):
   for q in range(1,5):
    if y==2026 and q>2:continue
    u=f"https://s25.q4cdn.com/479285134/files/doc_financials/{y}/q{q}/{q}Q{y%100:02}-Earnings-Release.pdf"
    items.append(dict(url=u,discovery_basis="Alternate public issuer CDN original-release pattern; no dates inferred from filenames"))
 if s=="ABT":
  for y in range(2011,2021):
   u=f"https://abbott.mediaroom.com/press-releases?category=783&year={y}&l=100"
   try:
    items += [a for a in links(u,s,"year_"+str(y)) if re.search("Abbott Reports.*Quarter",a["text"],re.I)]
   except Exception as e:print(s,y,e,flush=True)
 saveq(s,items)
if __name__=="__main__":
 for s in sys.argv[1:]:run(s)
