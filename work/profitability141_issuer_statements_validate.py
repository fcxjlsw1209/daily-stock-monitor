import json
import profitability141_issuer_statements as m
p=json.loads(m.read_pinned({'path':m.PLAN,'sha256':m.PLAN_SHA}));x={k:json.loads(m.read_pinned(v)) for k,v in p['inputs'].items()};ok=[];errors=[]
for s in p['source_scope']['sources']:
 try:
  a=m.extract_source(s,x['MSFT_events']+x['NVDA_events'],x['calendar_sessions']['sessions'],x['calendar_plan']['start']);ok.append([s['symbol'],s['fiscal_year'],a['currency']['USD_verified'],a['annual_period']['start']])
 except Exception as e:errors.append([s['symbol'],s['fiscal_year'],repr(e)])
print(json.dumps({'pass':len(ok),'ok':ok,'errors':errors}))
