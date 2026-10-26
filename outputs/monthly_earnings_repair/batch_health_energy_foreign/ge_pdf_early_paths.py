import collect as c,re
# Filename templates observed in the adjacent archived GE original releases.
# No guessed date is admitted; all downloaded headers and actual tables must pass offline parsing.
a=c.load(c.HERE/'GE_candidates.json',[])
for code in ['10182013','01232015','01222016','10172014','01182013']:
 for fmt in ['ge_webcast_press_release_{d}_0.pdf','ge_webcast_pressrelease_{d}_0.pdf']:
  u='https://www.ge.com/sites/default/files/'+fmt.format(d=code)
  try:
   p,m=c.fetch(u,'GE','Candidate original archive filename template; body must independently verify all dates')
   if p.read_bytes().startswith(b'%PDF'):
    a.append(dict(ticker='GE',url=u,label='GE historical quarterly earnings original'));print('PDF',m['bytes'],u,flush=True);break
   print('NOTPDF',m['bytes'],u,flush=True)
  except Exception as ex:print(ex,flush=True)
 c.dump(c.HERE/'GE_candidates.json',list({z.get('evidence_file',z['url']):z for z in a}.values()))
