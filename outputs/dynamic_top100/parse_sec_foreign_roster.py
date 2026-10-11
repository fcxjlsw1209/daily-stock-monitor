"""Parse cached official roster text; retain snapshot scope, never forward-fill eligibility."""
import json,re,hashlib,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
F=P/'additional_sources/sec_foreign_rosters'
COUNTRIES='United Kingdom|British Virgin Islands|Cayman Islands|Marshall Islands|British West Indies|Netherlands Antilles|South Africa|New Zealand|Hong Kong|United Arab Emirates|Papua New Guinea|Isle of Man|Channel Islands|United States|Luxembourg|Switzerland|Netherlands|Singapore|Australia|Argentina|Colombia|Bermuda|Belgium|Bahamas|Barbados|Portugal|Finland|Ireland|Germany|Norway|Sweden|Austria|Denmark|Canada|Mexico|Brazil|France|Israel|Taiwan|China|Japan|Korea|India|Italy|Spain|Chile|Peru|Panama|Liberia|Greece|Russia|Turkey|Cyprus|Malta|Belize|Jersey|Guernsey|Antigua|Indonesia|Philippines|Thailand|Malaysia|Hungary|Poland'
PAT=re.compile(r'^(?P<name>.+) (?P<country>'+COUNTRIES+r') (?P<market>NYSE(?:-Amex)?(?: - (?:Debt|Preferred))?|OTC(?: - (?:Debt|Preferred))?|Global Mkt|Cap\. Mkt)$')
def parse(lines):
 rows=[];unparsed=[]
 for n,t in sorted(lines.items()):
  m=PAT.fullmatch(t.strip())
  if m:
   r=m.groupdict();r.update(source_line=n,as_of='2009-12-31',exchange_equity_candidate=r['market'] in ['NYSE','NYSE-Amex','Global Mkt','Cap. Mkt'],daily_eligibility_certified=False);rows.append(r)
  elif t.strip() and not any(t.startswith(v) for v in ['FOREIGN',' FOREIGN','WITH THE','December','Alphabetical','COMPANY','Page ']):unparsed.append(dict(line=n,text=t))
 return rows,unparsed
if __name__=='__main__':
 lines={};sources=[]
 for f in sorted(F.glob('2009_web_*.json')):
  sources.append(dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
  for n,t in re.findall(r'^L(\d+)@P[^:]+: (.*)$',json.loads(f.read_text()),re.M):
   n=int(n)
   if n in lines:assert lines[n]==t,'Overlapping source extraction conflict'
   lines[n]=t
 assert set(lines)==set(range(993)),'Incomplete source extraction'
 rows,bad=parse(lines)
 report=dict(source='https://www.sec.gov/divisions/corpfin/internatl/foreignalpha2009.pdf',source_as_of='2009-12-31',publication_date=None,retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=sources,rows=rows,unparsed=bad,scope='Official snapshot candidate evidence only; no assumed continuity to March 2010, no absence-based exclusion, and exchange candidate does not exclude funds automatically.')
 (F/'2009_parsed.json').write_text(json.dumps(report,indent=2));print('parsed',len(rows),'exchange candidates',sum(r['exchange_equity_candidate'] for r in rows),'unparsed',bad)
