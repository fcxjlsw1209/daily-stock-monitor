"""Strict CVX release/period parsing from explicitly retained source observations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from finalize import base, clean, DATE, date
import json
import re
import datetime as dt

P=Path(__file__).resolve().parent


def parse(row,file):
    text=clean(row.get('excerpt','')); table=clean(row.get('period_excerpt',''))
    pub=re.search(r'(?:SAN RAMON|HOUSTON)[, (]+.{0,20}?'+DATE,text,re.I)
    if row.get('tool_error') or not pub or 'interim update' in text.lower():raise ValueError('No actual results dateline')
    fq=re.search(r'(?:for\s+(?:the\s+)?)(first|second|third|fourth)\s+quarter\s+(20\d{2})',text,re.I)
    if not fq:raise ValueError('No explicit current fiscal quarter/year')
    year=int(fq[2]);quarter=['first','second','third','fourth'].index(fq[1].lower())+1
    endmonth={1:'March',2:'June',3:'September',4:'December'}[quarter];day={1:31,2:30,3:30,4:31}[quarter]
    aliases=endmonth+'|'+endmonth[:3]+r'\.?'+(r'|Sept\.?' if quarter==3 else '')
    if not re.search(r'Ended.{0,80}\b(?:'+aliases+r')\s+'+str(day)+r'\b',table,re.I) or str(year) not in table:
        raise ValueError('No explicit corresponding financial table period')
    pubdate=date(pub[1]);period=dt.date(year,quarter*3,day).isoformat()
    out=base(row,file,'CVX',pubdate,period,year,quarter,'America/Chicago' if text[:20].upper().startswith('HOUSTON') else 'America/Los_Angeles')
    out['period_basis']=table;return out


def main():
    file=P/'observations.json';events={};rejected=[]
    for i,row in enumerate(json.loads(file.read_text())):
        try:
            e=parse(row,file);key=(e['fiscal_year'],e['fiscal_quarter'])
            if key in events and events[key]['release_date']!=e['release_date']:raise ValueError('Conflicting quarter dates')
            events[key]=e
        except ValueError as exc:rejected.append(dict(index=i,url=row['source_url'],reason=str(exc)))
    (P/'events.json').write_text(json.dumps(list(events.values()),indent=2)+'\n')
    missing=[(n//4,n%4+1) for n in range(2010*4+3,2026*4+2) if (n//4,n%4+1) not in events]
    (P/'AUDIT.json').write_text(json.dumps(dict(verified=len(events),missing=missing,rejected_observations=rejected),indent=2)+'\n')
    print('Verified',len(events),'Missing',missing)


if __name__=='__main__':main()
