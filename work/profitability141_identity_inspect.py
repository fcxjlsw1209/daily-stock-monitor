from pathlib import Path
import json,collections
R=Path.cwd();prior=json.loads((R/'work/monthly_profitability141_identity_review.json').read_text());inv=json.loads((R/'work/profitability141_migrated_cache_inventory_complete_20261019T202336581108Z.json').read_text());manifest=json.loads((R/'migration/top100_local_completion_20261019/FULL_LOCAL_MANIFEST_20261019T202326644648Z.json').read_text());byname=collections.defaultdict(list)
for x in manifest:byname[Path(x['path']).name].append(x['path'])
for row in prior['rows']:
 if not row.get('plan_candidates'):continue
 p=R/row['current_registrants'][0]['evidence'];s=json.loads(p.read_text());datasets=[s['filings']['recent']];oldlocal=[];missing=[]
 for old in s['filings']['files']:
  if old['filingFrom']>'2026-09-24' or old['filingTo']<'2015-01-01':continue
  ps=byname[old['name']]
  if not ps:missing.append(old['name']);continue
  fp=ps[0];oldlocal.append(fp);datasets.append(json.loads((R/fp).read_text()))
 years=set()
 for d in datasets:
  for i,f in enumerate(d.get('form',[])):
   if f=='10-K' and d.get('filingDate',['']*len(d['form']))[i]<='2026-09-24':
    y=d.get('reportDate',['']*len(d['form']))[i][:4]
    if y>='2015':years.add(y)
 matches=[x for x in inv['candidates'] if x['cik']==s['cik']]
 cp=sorted(set(y for x in matches for c in x['annual_concept_metadata'] for y in c.get('pair_coverage_by_form',{}).get('10-K',{}).get('period_end_years',[]) if y>='2015'))
 print(json.dumps({'symbol':row['symbol'],'cik':s['cik'],'name':s['name'],'tickers':s['tickers'],'former':[(n['name'],n['to'][:10]) for n in s['formerNames']],'10K_years':sorted(years),'older_local':len(oldlocal),'older_missing':len(missing),'facts_files':len(matches),'10K_pair_years':cp}))
