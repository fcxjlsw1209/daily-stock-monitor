"""Use independently public Disney CDN archive; blocked main IR hosts untouched."""
from budvz_io import P,fetch,dump
import json
OUT=P/'dis_supplement'
def main():
    rows=[]
    for y in range(2011,2027):
        for q in range(1,5):
            if (y,q)>(2026,3):continue
            # Archive path is only a discovery template, never date evidence.
            month={1:'02',2:'05',3:'08',4:'11'}[q]
            url=f'https://s206.q4cdn.com/979796730/files/doc_events/{y}/{month}/q{q}-fy{str(y)[2:]}-earnings.pdf'
            try:p,m=fetch('dis_supplement',url)
            except Exception as e:print(type(e).__name__,str(e),flush=True);continue
            rows.append(dict(year=y,quarter=q,url=url,path=str(p),meta=m))
            dump(OUT/'MANIFEST.json',rows)
            print(y,q,m['http_status'],len(p.read_bytes()),flush=True)
            if m['http_status'] in (403,429):break
        if rows and rows[-1]['meta']['http_status'] in (403,429):break
if __name__=='__main__':main()
