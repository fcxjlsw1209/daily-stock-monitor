import collect as c,re
s='\n'.join(r['value'] for r in c.load(c.RAW/'orcl_all_search.json',{})['results'] if r['status']=='fulfilled');urls=list(dict.fromkeys(re.findall(r'\((https://s23.q4cdn.com/440135859/[^)]+\.pdf)\)',s)));arr=c.load(c.HERE/'ORCL_candidates.json',[]);print('ORCL discovered',len(urls),flush=True)
for u in urls:
 if re.search('presentation|transcript|10-k|10-q|annual',u,re.I):continue
 try:
  p,m=c.fetch(u,'ORCL','original earnings release');arr.append(dict(ticker='ORCL',url=u,label='Original Oracle PDF discovered in public index'));print(m['bytes'],u,flush=True)
 except Exception as e:print(e,flush=True)
c.dump(c.HERE/'ORCL_candidates.json',list({a['url']:a for a in arr}.values()))
arr=c.load(c.HERE/'PFE_candidates.json',[])
for y,q in [(y,q) for y in range(2017,2024) for q in range(1,5)]:
 file=f'Q{q}_{y}_PFE_Earnings_Release.pdf' if y==2017 else f'Q{q}-{y}-PFE-Earnings-Release.pdf';u=f'https://s206.q4cdn.com/795948973/files/doc_financials/{y}/q{q}/{file}'
 try:
  p,m=c.fetch(u,'PFE','original earnings release');arr.append(dict(ticker='PFE',url=u,label='Original Pfizer PDF, actual body required'))
 except Exception as e:print(e,flush=True)
c.dump(c.HERE/'PFE_candidates.json',list({a['url']:a for a in arr}.values()))
