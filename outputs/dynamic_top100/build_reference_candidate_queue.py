"""Reference-date discovery queue, not an eligible US issuer ranking."""
from pathlib import Path
from collections import defaultdict
import json,re,csv,datetime
P=Path(__file__).resolve().parent;D=P/'additional_sources/ft_rankings'
validation=json.loads((D/'extraction_validation.json').read_text())
valid={r['file'] for r in validation if r['ranks_1_to_500_complete'] and r['cap_descending']}
rows=[r for r in json.loads((D/'reference_rankings.json').read_text()) if r['source_file'] in valid]+json.loads((D/'xls_reference_rankings.json').read_text())
repair=json.loads((D/'2013_repair_validation.json').read_text())
assert repair['complete'] and repair['cap_descending']
rows=[r for r in rows if r['reference_date']!='2013-03-28']+json.loads((D/'2013_repaired_rankings.json').read_text())
bydate=defaultdict(list)
for r in rows:bydate[r['reference_date']].append(r)
for day,rs in bydate.items():
 assert {r['global_rank'] for r in rs}==set(range(1,501)) and len(rs)==500
 assert all(any(c.isalpha() for c in r['company_name_as_printed']) for r in rs), 'Invalid reference company name'
(D/'validated_reference_rankings.json').write_text(json.dumps(rows,indent=2)+'\n')
# Printed-name normalization is for discovery only; never infer permanent identity.
def norm(s):return re.sub(r'[^a-z0-9]','',s.lower())
names=defaultdict(set)
with (P/'expanded_sec/candidate_identity_evidence.csv').open() as f:
 for r in csv.DictReader(f):names[norm(r['name'])].add(r['symbol'])
queue=defaultdict(list)
for r in rows:queue[r['company_name_as_printed']].append(r)
out=[]
for name,rs in queue.items():
 out.append({'printed_name':name,'best_global_rank':min(x['global_rank'] for x in rs),'reference_dates':sorted(x['reference_date'] for x in rs),'exact_name_candidate_symbols':sorted(names[norm(name)]),'us_listing_identity_status':'requires effective-dated verification','reference_observations':rs,'daily_top100_certified':False})
out.sort(key=lambda r:(r['best_global_rank'],r['printed_name']))
(P/'HISTORICAL_REFERENCE_CANDIDATES.json').write_text(json.dumps(out,indent=2))
summary={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete_reference_dates':sorted(bydate),'reference_observations':len(rows),'distinct_printed_names':len(out),'names_ever_global_top100':sum(r['best_global_rank']<=100 for r in out),'names_ever_global_top200':sum(r['best_global_rank']<=200 for r in out),'excluded_incomplete_pdf':None,'uncovered_years':[y for y in range(2010,2027) if not any(d.startswith(str(y)) for d in bydate)],'all_other_trading_dates_still_require_daily_reconstruction':True,'scope':'Global cap ranking, not US listing ranking. US eligibility, issuer names/aliases and publication dates unverified. Same issuer may appear under multiple names. Global rank>200 not ruled out from US Top100.','can_start_backtest':False}
(P/'REFERENCE_RANKING_PROGRESS.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
