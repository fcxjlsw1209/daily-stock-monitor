"""Audit cached community hints; consistency does not establish index completeness."""
import sys,json,hashlib,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'vendor'))
import yaml

def main():
 d=P/'sources/n100tickers';years=[];transitions=[];symbols=set();events=[]
 for f in sorted(d.glob('*.yaml')):
  raw=f.read_bytes();prov=json.loads(f.with_suffix('.yaml.provenance.json').read_text())
  if hashlib.sha256(raw).hexdigest()!=prov['sha256']:raise ValueError('Checksum mismatch')
  j=yaml.load(raw,Loader=yaml.BaseLoader);current=set(j['tickers_on_Jan_1']);symbols|=current;initial=set(current);bad=[]
  for day,e in sorted((j.get('changes') or {}).items(),key=lambda v:str(v[0])):
   remove=set(e.get('difference') or []);add=set(e.get('union') or []);symbols|=add
   if not remove<=current or add&current:bad.append(dict(date=str(day),missing_removals=sorted(remove-current),already_present_additions=sorted(add&current)))
   current=(current-remove)|add
   events.append(dict(index='NDX',effective_date=str(day),add=sorted(add),remove=sorted(remove),discovery_file=str(f.relative_to(P)),verified=False))
  if years:transitions.append(dict(year=j['year'],missing_from_next_start=sorted(set(years[-1]['end_members'])-initial),extra_in_next_start=sorted(initial-set(years[-1]['end_members']))))
  years.append(dict(year=j['year'],initial_count=len(initial),end_members=sorted(current),chain_errors=bad))
 r=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),years=years,year_transitions=transitions,distinct_symbols=len(symbols),event_dates=len(events),verified=False)
 for name,data in [('NDX_DISCOVERY_AUDIT',r),('ndx_events_UNVERIFIED',events),('ndx_symbols_UNVERIFIED',sorted(symbols))]:(P/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
 print('years',len(years),'symbols',len(symbols),'events',len(events),'bad events',sum(len(y['chain_errors']) for y in years),'inconsistent year boundaries',sum(bool(t['missing_from_next_start'] or t['extra_in_next_start']) for t in transitions))
if __name__=='__main__':main()
