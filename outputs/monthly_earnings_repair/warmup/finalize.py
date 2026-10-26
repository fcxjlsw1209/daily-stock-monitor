"""Normalize observed original releases; never turn search failure into an event."""
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from extend import DATE, date

P = Path(__file__).resolve().parent
DEST = P / 'verified'


def read(name):
    return json.loads((P / name).read_text())


def clean(s):
    return re.sub(r'\s+', ' ', s)


def base(row, file, ticker, pub, period, fy, fq, tz):
    lag = (dt.date.fromisoformat(pub)-dt.date.fromisoformat(period)).days
    if not 0 < lag < 100:
        raise ValueError('Publication/period interval inconsistent')
    return dict(ticker=ticker, release_date=pub, fiscal_period_end_actual=period,
        fiscal_year=fy, fiscal_quarter=fq, publication_timestamp=None,
        publication_timezone=tz, precision='date_only',
        publication_timezone_basis='Issuer original release dateline; date-only conservative issuer-local availability.',
        source_url=row['source_url'], source_kind=row['source_kind'],
        evidence_path=str(file), evidence_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        retrieved_at=row['observed_at_utc'], evidence_excerpt=row['excerpt'],
        first_public_time_verified=False,
        evidence_limitation='Public index observation of original issuer document; checksum covers observation JSON, not original document bytes.')


def tsm(row, file):
    text=clean(row.get('excerpt',''))
    pub=re.search(r'Hsi[ns][ -]?[Cc]hu.{0,55}?'+DATE, text, re.I)
    end=re.search(r'(first|second|third|fourth) quarter ended '+DATE, text, re.I)
    if row.get('tool_error') or not pub or not end:
        raise ValueError('Missing original TSM dateline/quarter end')
    period=date(end[2]); fq=['first','second','third','fourth'].index(end[1].lower())+1
    if int(period[5:7])!=fq*3: raise ValueError('TSM quarter identity mismatch')
    return base(row,file,'TSM',date(pub[1]),period,int(period[:4]),fq,'Asia/Taipei')


def msft(row,file):
    text=clean(row.get('excerpt',''))
    pub=re.search(r'REDMOND.{0,25}?'+DATE,text,re.I)
    end=re.search(r'quarter ended '+DATE,text,re.I)
    if row.get('tool_error') or not pub or not end:
        raise ValueError('Missing original Microsoft dateline/quarter end')
    period=date(end[1]); year=int(period[:4]); month=int(period[5:7])
    if month not in (3,6,9,12): raise ValueError('Microsoft nonquarter end')
    fy=year+(month>6); fq={9:1,12:2,3:3,6:4}[month]
    title=re.search(r'FY(\d{2}) Q([1-4])',row.get('title',''))
    if title and (2000+int(title[1]),int(title[2]))!=(fy,fq): raise ValueError('Microsoft fiscal title mismatch')
    if not title and not (fq==4 and 'fourth' in (row.get('title','')+text).lower()):
        raise ValueError('Missing fiscal identity confirmation')
    return base(row,file,'MSFT',date(pub[1]),period,fy,fq,'America/Los_Angeles')


def other(row,file,observations):
    text=clean(row['excerpt']);ticker=row['ticker'];fy=row['fiscal_year'];fq=row['fiscal_quarter']
    if ticker=='BRK.B':
        pub=re.search(r'On '+DATE+r', Berkshire Hathaway Inc\. posted its '+str(fy)+r' Annual Report',text)
        supplement=next(x for x in observations if x.get('kind')=='additional_period_evidence' and '/'+str(fy)+'ar/' in x['source_url'])
        if not pub or '4th Quarter '+str(fy) not in supplement['excerpt'] or 'December 31, '+str(fy) not in supplement['excerpt']:
            raise ValueError('Missing explicit actual publication or annual Q4 evidence')
        out=base(row,file,ticker,date(pub[1]),str(fy)+'-12-31',fy,fq,'America/Chicago')
        out['period_basis']='Original annual report Q4 table and December 31 fiscal-year-end; release date explicitly reported in issuer 8-K body, not filing date.'
        out['additional_evidence']=[dict(evidence_path=str(file),sha256=out['evidence_sha256'],source_url=supplement['source_url'],role='annual_Q4_period_identity')]
        return out
    marker='MOUNTAIN VIEW' if ticker=='GOOGL' else 'SEATTLE'
    pub=re.search(marker+r'.{0,40}?'+DATE,text,re.I)
    end=re.search(r'ended '+DATE,text,re.I)
    if not pub or not end: raise ValueError('Missing actual dateline/period')
    out=base(row,file,ticker,date(pub[1]),date(end[1]),fy,fq,'America/Los_Angeles')
    if ticker=='GOOGL':
        out.update(issuer_name_at_release='Google Inc.', historical_ticker='GOOG',
            issuer_identity_note='Google Inc. 2010 fiscal earnings, historical pre-class-C GOOG; research issuer lineage stored as GOOGL, not a contemporaneous GOOGL ticker claim.')
    return out


def main():
    DEST.mkdir(exist_ok=True)
    events=read('events.json')+read('extra_events.json')+read('amazon_events.json')
    rejected=[];duplicates=[]
    for name,parser in [('tsm_observations.json',tsm),('msft_observations.json',msft)]:
        file=P/name
        for idx,row in enumerate(read(name)):
            try: events.append(parser(row,file))
            except ValueError as exc: rejected.append(dict(file=name,index=idx,url=row['source_url'],reason=str(exc)))
    obs=read('other_observations.json')
    for row in obs:
        if 'ticker' in row:events.append(other(row,P/'other_observations.json',obs))
    original=json.loads((P.parent/'tsmc/events.json').read_text())
    existing={(e['ticker'],e['fiscal_year'],e['fiscal_quarter']):e for e in original}
    result={}
    for event in events:
        key=(event['ticker'],event['fiscal_year'],event['fiscal_quarter'])
        prior=existing.get(key) or result.get(key)
        if prior:
            if (event['release_date'],event['fiscal_period_end_actual'])!=(prior['release_date'],prior['fiscal_period_end_actual']):
                raise ValueError('Conflicting quarter '+str(key))
            duplicates.append(dict(key=key,url=event['source_url']));continue
        result[key]=event
    out=sorted(result.values(),key=lambda x:(x['ticker'],x['release_date']))
    (DEST/'events.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    (DEST/'OBSERVATION_AUDIT.json').write_text(json.dumps(dict(rejected_observations=rejected,duplicate_observations=duplicates),indent=2)+'\n')
    from collections import Counter
    print(dict(Counter(e['ticker'] for e in out)))
    tsm_keys={(e['fiscal_year'],e['fiscal_quarter']) for e in out+original if e['ticker']=='TSM'}
    print('TSM missing',[(n//4,n%4+1) for n in range(2010*4+3,2026*4+2) if (n//4,n%4+1) not in tsm_keys])
    print('Rejected',len(rejected),'duplicate',len(duplicates))


if __name__=='__main__':main()
