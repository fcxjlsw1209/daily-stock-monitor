import collect as c,re
urls={
'NVS':['https://www.novartis.com/news/media-releases/novartis-delivered-solid-2016-performance-growth-products1-absorbing-gleevec-us-loe-innovation-momentum-continued-announces-share-buyback'],
'NVO':['https://attachment.news.eu.nasdaq.com/a7acd5ea39dadf3a089a8773f53bc50af','https://ml-eu.globenewswire.com/resource/download/aa7660a8-5b54-43f7-9100-e3dfae8b7988'],
'PFE':['https://s206.q4cdn.com/795948973/files/doc_financials/2016/q1/Q1_2016_PFE_Earnings_Press_Release_nvewlnavd.pdf','https://s206.q4cdn.com/795948973/files/doc_financials/Quarterly/2020/q1/Q1-2020-PFE-Earnings-Release.pdf']}
for t,us in urls.items():
 arr=c.load(c.HERE/(t+'_candidates.json'),[]);sup=c.load(c.HERE/(t+'_supports.json'),[])
 for u in us:
  try:
   p,m=c.fetch(u,t,'official earnings original');arr.append(dict(ticker=t,url=u,label='Original issuer earnings release'));print(t,m['bytes'],u,flush=True)
   if t=='NVS':
    for a in c.links(p,u):
     if (a['url'].lower().endswith('.pdf') and re.search('full|report|release',a['text'],re.I)):
      try:
       sp,sm=c.fetch(a['url'],t,'official release financial attachment');sup.append(dict(ticker=t,url=a['url'],label=a['text'],release_url=u));arr.append(dict(ticker=t,url=a['url'],label='Q4 2016 original full release'));print('attachment',sm['bytes'],a['url'],flush=True)
      except Exception as e:print(e,flush=True)
  except Exception as e:print(t,e,flush=True)
 c.dump(c.HERE/(t+'_candidates.json'),list({a.get('evidence_file',a['url']):a for a in arr}.values()));c.dump(c.HERE/(t+'_supports.json'),sup)
p=c.HERE/'build.py';s=p.read_text().replace(r'\s*EARNINGS',r'\s*EARNIN\s*GS');p.write_text(s)
