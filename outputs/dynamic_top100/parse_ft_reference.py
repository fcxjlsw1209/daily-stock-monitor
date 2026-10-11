"""Extract reference-date market-cap evidence, never daily PIT membership."""
from pathlib import Path
import re,json,hashlib
import pdfplumber
P=Path(__file__).resolve().parent/'additional_sources/ft_rankings'
dates={'2010-03-31':'2010-03-31','2010-12-31':'2010-12-31','2011_reference':'2011-03-31','2012_reference':'2012-03-30','2013_reference':'2013-03-28'}
countries='US/UK|Argentina|UAE|Australia/UK|South Korea|Hong Kong|South Africa|Saudi Arabia|United Arab Emirates|Czech Republic|New Zealand|United States|US|UK|China|Switzerland|Brazil|Japan|Russia|France|Australia|Germany|Spain|Italy|Canada|Belgium|Netherlands/UK|Netherlands|Norway|India|Israel|Mexico|Singapore|Taiwan|Sweden|Denmark|Finland|Indonesia|Malaysia|Thailand|Chile|Colombia|Portugal|Poland|Austria|Ireland|Turkey|Luxembourg|Qatar|Kuwait|Greece|Bermuda|Peru|Hungary|Philippines|Pakistan|Morocco'
pattern=re.compile(r'^(\d+)\s+(?:(?:\d+|NEW|New|—|-)\s+)?(.+)\s+('+countries+r')\s+(.+?)\s+([\d,]+\.\d+)\s+(.+)$')
allrows=[];reports=[]
for stem,day in dates.items():
 f=P/(stem+'.pdf')
 if not f.exists():continue
 with pdfplumber.open(f) as pdf:text='\n'.join(page.dedupe_chars().extract_text() or '' for page in pdf.pages)
 (P/(stem+'.txt')).write_text(text)
 rows=[];unparsed=[]
 for line in text.splitlines():
  if not re.match(r'^\d+\s',line):continue
  m=pattern.match(line)
  if not m:unparsed.append(line);continue
  rank,name,country,sector,number,tail=m.groups()
  # December table puts price before cap; yearly tables put cap first.
  cap=tail.split()[0] if stem=='2010-12-31' else number
  try:cap=float(cap.replace(',',''))*1e6
  except ValueError:unparsed.append(line);continue
  rows.append(dict(reference_date=day,global_rank=int(rank),company_name_as_printed=name,country=country,sector_as_printed=sector,market_cap_usd=cap,source_file=f.name,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),us_exchange_eligibility_verified=False,daily_membership_usable=False))
 ranks=[r['global_rank'] for r in rows];complete=len(rows)==500 and set(ranks)==set(range(1,501))
 ordered=all(rows[i]['market_cap_usd']>=rows[i+1]['market_cap_usd'] for i in range(len(rows)-1))
 reports.append(dict(file=f.name,reference_date=day,parsed_rows=len(rows),unique_ranks=len(set(ranks)),ranks_1_to_500_complete=complete,cap_descending=ordered,unparsed_lines=unparsed,missing_ranks=sorted(set(range(1,501))-set(ranks)),issuer_identity_verified=False))
 allrows.extend(rows)
(P/'reference_rankings.json').write_text(json.dumps(allrows,indent=2));(P/'extraction_validation.json').write_text(json.dumps(reports,indent=2))
print([(r['file'],r['parsed_rows'],r['cap_descending']) for r in reports])
