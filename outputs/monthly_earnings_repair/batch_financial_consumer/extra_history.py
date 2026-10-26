import sys,json,re
import core as c
from archive_download import links
from expand_download import saveq

def run_c():
 for suffix,filename in [("","release_queue.json"),("s","supplement_queue.json")]:
  p=c.HERE/"C"/filename;old=json.loads(p.read_text());seen={a["url"]for a in old}
  for year in range(2010,2017):
   for q in range(1,5):
    if year==2010 and q<4:continue
    u=f"https://www.citigroup.com/rcs/citigpa/akpublic/storage/public/qer{q}{year%100:02}{suffix}.pdf"
    if u in seen:continue
    a=dict(url=u,discovery_basis="Public issuer archive filename pattern; issuer and all dates require original content validation")
    try:
     b,r,m=c.fetch(u,"C");a.update(evidence_path=c.rel(r),evidence_sha256=c.sha(b),meta_path=c.rel(r.with_name(r.name+".meta.json")));c.pdf_text(r)
    except Exception as e:a["error"]=str(e)
    old.append(a);c.dump(p,old);print("C",year,q,suffix,a.get("error","OK"),flush=True)
def run_t():
 ls=[]
 for p in (c.HERE/"T/raw").glob("archive_*.html"):
  m=json.loads(p.with_name(p.name+".meta.json").read_text())
  if m["status"]!=200:continue
  for a in links(m["url"],"T",p.stem):
   if re.search(r"\.pdf",a["url"],re.I)and re.search(r"earnings.release|press.release",a["url"]+" "+a["text"],re.I)and not re.search("supplement|schedule|financial.and.operat",a["url"],re.I):ls.append(a)
 saveq("T",ls)
if __name__=="__main__":
 for s in sys.argv[1:]:{ "C":run_c,"T":run_t}[s]()
