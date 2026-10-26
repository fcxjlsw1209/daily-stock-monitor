"""Original issuer documents for explicitly assigned GS/VOD training windows."""
import json,re,urllib.parse
import core as c
for s in ['GS','VOD']:
 links=json.loads((c.HERE/s/'original_links.json').read_text());out=[];seen=set()
 for a in links:
  u=a['url']
  if s=='GS':
   if '2010-q4-earnings.pdf'not in u:continue
  else:
   if 'src='in u:u=urllib.parse.urljoin('https://www.vodafone.com',urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)['src'][0])
   if not re.search('/financial-results/201[0-3]/',u)or not u.lower().endswith('.pdf')or re.search('transcript|annual|20f|acquisition|presentation',u,re.I):continue
  if u in seen:continue
  seen.add(u);a=dict(a,url=u,discovery_basis='Explicit public issuer report link; URL path itself is not date evidence')
  try:
   b,p,m=c.fetch(u,s);a.update(evidence_path=c.rel(p),evidence_sha256=c.sha(b),meta_path=c.rel(p.with_name(p.name+'.meta.json')));c.pdf_text(p)
  except Exception as e:a['error']=str(e)
  out.append(a);c.dump(c.HERE/s/'release_queue.json',out);print(s,u,a.get('error','OK'),flush=True)
