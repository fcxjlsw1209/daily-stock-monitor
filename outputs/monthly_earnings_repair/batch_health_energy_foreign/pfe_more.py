import collect as c,re
s='\n'.join(r['value'] for r in c.load(c.RAW/'pfe_all_search.json',{})['results'] if r['status']=='fulfilled');urls=list(dict.fromkeys(re.findall(r'\((https://[^)]+\.pdf)\)',s)));urls=[u for u in urls if 'q4cdn.com' in u and re.search('pfe.*earnings.*release',u,re.I)];arr=c.load(c.HERE/'PFE_candidates.json',[]);print('PFE discovered',len(urls),flush=True)
for u in urls:
 try:
  p,m=c.fetch(u,'PFE','original earnings release');arr.append(dict(ticker='PFE',url=u,label='Original Pfizer earnings PDF'));print(m['bytes'],u,flush=True)
 except Exception as e:print(e,flush=True)
c.dump(c.HERE/'PFE_candidates.json',list({a['url']:a for a in arr}.values()))
