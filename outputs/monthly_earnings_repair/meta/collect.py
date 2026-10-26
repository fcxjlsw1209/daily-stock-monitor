"""Bounded, cached official Meta/Facebook earnings PDF collection.

The static official PDF archive is independent of the IR HTML challenge. No
challenge-solving, credentials, Yahoo, AV, or SEC requests are performed.
"""
import argparse
import calendar
import datetime as dt
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / 'raw'
CUTOFF = '2026-09-24'
PREFIX = 'https://s21.q4cdn.com/399680738/files/'
WORDS = {1:'First',2:'Second',3:'Third',4:'Fourth'}
RUNTIME = '/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
IDENTITY_URL = 'https://investor.atmeta.com/investor-news/press-release-details/2022/Meta-Platforms-Inc.-to-Change-Ticker-Symbol-to-META-on-June-9/default.aspx'
OLD_DAYS = {
 (2012,2):'2012-07-26',(2012,3):'2012-10-23',(2012,4):'2013-01-30',
 (2013,1):'2013-05-01',(2013,2):'2013-07-24',(2013,3):'2013-10-30',(2013,4):'2014-01-29',
 (2014,1):'2014-04-23',(2014,2):'2014-07-23',(2014,3):'2014-10-28',(2014,4):'2015-01-28',
 (2015,1):'2015-04-22',(2015,2):'2015-07-29',(2015,3):'2015-11-04',(2015,4):'2016-01-27',
}
OVERRIDES = {
 (2016,1):'doc_news/Facebook-Announces-First-Quarter-2016-Results-and-Proposal-for-New-Class-of-Stock.pdf',
 (2018,2):'doc_financials/2018/Q2/Facebook-Reports-Second-Quarter-2018-Results.pdf',
 (2018,3):'doc_financials/2018/Q3/Q3-2018-Earnings-Press-Release.pdf',
 (2018,4):'doc_financials/2018/Q4/Q4-2018-Earnings-Release.pdf',
 (2019,2):'doc_financials/2019/Q2/FB-Q2-2019-Earnings-Release.pdf',
 (2019,4):'doc_financials/2019/q4/FB-12.31.2019-Exhibit-99.1-r61_final.pdf',
 (2022,1):'doc_financials/2022/q1/Meta-03.31.2022-Exhibit-99.1_Final.pdf',
 (2023,1):'doc_financials/2023/q1/Meta-03-31-2023-Exhibit-99-1-FINAL-v2.pdf',
}

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def sha(body): return hashlib.sha256(body).hexdigest()

def expected_quarters():
    return [(y,q) for y in range(2012,2027) for q in range(1,5)
            if (y,q)>=(2012,2) and (y,q)<=(2026,2)]

def candidate(year,quarter):
    if (year,quarter) in OVERRIDES: return PREFIX+OVERRIDES[year,quarter]
    if (year,quarter) in OLD_DAYS:
        d=dt.date.fromisoformat(OLD_DAYS[year,quarter])
        return PREFIX+'doc_news/%d/FB_News_%d_%d_%d_Financial_Releases.pdf'%(d.year,d.year,d.month,d.day)
    brand='Meta' if year>=2022 or (year,quarter)==(2021,4) else 'Facebook'
    title='%s-Reports-%s-Quarter%s-%d-Results'%(brand,WORDS[quarter],'-and-Full-Year' if quarter==4 else '',year)
    ry=year+(quarter==4)
    return PREFIX+'doc_news/'+title+('-%d'%ry if ry>=2020 else '')+'.pdf'

def parse_date(s):
    s=re.sub(r'\s+',' ',s.replace('.','')).strip()
    for f in ('%B %d, %Y','%b %d, %Y'):
        try: return dt.datetime.strptime(s,f).date().isoformat()
        except ValueError: pass
    raise ValueError('Invalid date: '+s)

def parse_pdf_text(text,year,quarter,source_url,evidence_path,body):
    text=re.sub(r'\s+',' ',text)
    text=re.sub(r'\s+,',',',text)
    # A conference-call announcement or calendar forecast is not a results release.
    title=re.search(r'(Facebook|Meta) (Reports|Announces) (First|Second|Third|Fourth) Quarter(?: and Full Year)? (\d{4})\s*Results',text[:700],re.I)
    # 2016 Q1 original used "Announces First Quarter Results" and supplies 2016 in body.
    if not title and year==2016 and quarter==1:
        title=re.search(r'Facebook Announces First Quarter Results and Proposal to Create New Class of Non-Voting Stock',text[:700],re.I)
    if not title: raise ValueError('Not an observed earnings-results headline')
    if title.lastindex and title.lastindex>=4:
        if int(title.group(4))!=year or title.group(3).lower()!=WORDS[quarter].lower():
            raise ValueError('Headline fiscal year/quarter mismatch')
    datepat=r'([A-Z][a-z]+\.? \d{1,2}, \d{4})'
    dateline=re.search(r'MENLO PARK,\s*(?:Calif\.?|California)[, .]*\s*(?:[–—-]\s*)?'+datepat,text[:1800],re.I)
    if not dateline: raise ValueError('Original Menlo Park dateline or publication date absent')
    release=parse_date(dateline.group(1))
    actual=re.search(r'(?:quarter|year)[^.]{0,90}?ended '+datepat,text[dateline.end():dateline.end()+900],re.I)
    if not actual: raise ValueError('Actual quarter end absent from original opening statement')
    end=parse_date(actual.group(1))
    mm=3*quarter;expected='%d-%02d-%02d'%(year,mm,calendar.monthrange(year,mm)[1])
    if end!=expected: raise ValueError('Actual fiscal period contradicts target: '+end+' vs '+expected)
    if not end<release: raise ValueError('Release date must follow quarter end')
    ticker=re.search(r'\(\s*Nasdaq\s*:\s*(FB|META)\s*\)',text[:1800],re.I)
    if not ticker: raise ValueError('Issuer trading ticker not observed')
    historical=ticker.group(1).upper()
    expected_ticker='META' if release>='2022-06-09' else 'FB'
    if historical!=expected_ticker: raise ValueError('Ticker conflicts with verified June 9 2022 identity transition')
    excerpt=text[dateline.start():dateline.end()+420]
    return {'ticker':'research_equity_FB','historical_ticker':historical,
      'issuer_identity':'Same Nasdaq Class A issuer: Facebook, Inc. renamed Meta Platforms, Inc.; ticker FB changed to META before market open 2022-06-09; CUSIP unchanged. research_equity_FB remains the fixed research identifier.',
      'identity_source_url':IDENTITY_URL,'release_date':release,
      'publication_timestamp':None,'precision':'date_only','publication_timezone':'America/Los_Angeles',
      'publication_timezone_basis':'Conservative issuer-local calendar date policy using original Menlo Park, California dateline; PDF does not verify precise publication time or CMS timezone.',
      'publication_timezone_evidence':{'kind':'issuer_local_calendar_date_policy','observed_dateline':dateline.group(0),'dateline_excerpt':excerpt,'evidence_path':evidence_path,'geographic_mapping':'Menlo Park, California -> America/Los_Angeles','cms_timezone_observed':False,'exact_publication_time_inferred':False},
      'first_public_time_verified':False,'official_date_published_raw':dateline.group(1),
      'fiscal_period_end_actual':end,'fiscal_year':year,'fiscal_quarter':quarter,
      'source_url':source_url,'evidence_path':evidence_path,'evidence_sha256':sha(body),
      'evidence_excerpt':excerpt,'headline':title.group(0),'source_class':'issuer_original_press_release_pdf',
      'timestamp_note':'Dateline is date-only. Neither PDF creation/modification metadata nor conference-call times are publication timestamps.',
      'in_study_window':'2016-08-17'<=release<=CUTOFF}

def fetch(url):
    if not url.startswith(PREFIX): raise ValueError('Only verified official static Meta archive allowed')
    key=sha(url.encode())[:16];p=RAW/(key+'.pdf');mp=RAW/(key+'.meta.json')
    if mp.exists() and p.exists():
        m=json.loads(mp.read_text());b=p.read_bytes()
        if sha(b)!=m['sha256']: raise RuntimeError('Cached evidence checksum mismatch')
        return m,b,p
    sp=HERE/'SOURCE_STOP.json';stop=json.loads(sp.read_text()) if sp.exists() else {}
    st=stop.get('s21.q4cdn.com',{})
    now=dt.datetime.now(dt.timezone.utc)
    if st.get('utc_date')==now.date().isoformat() or st.get('stop_until','')>now.isoformat():
        raise RuntimeError('Official static archive source stopped today')
    time.sleep(1.1)
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':'application/pdf'})
    try:
        r=urllib.request.urlopen(req,timeout=30);b=r.read(8000001);status=r.status;h=r.headers;final=r.url
    except urllib.error.HTTPError as e:
        b=e.read(8000001);status=e.code;h=e.headers;final=e.url
    if len(b)>8000000: raise RuntimeError('Oversized source; not accepted')
    now=dt.datetime.now(dt.timezone.utc)
    m={'source_url':url,'final_url':final,'http_status':status,'sha256':sha(b),'size':len(b),'retrieved_at':now.isoformat(),'retry_after':h.get('Retry-After'),'public_unauthenticated':True}
    p.write_bytes(b);dump(mp,m)
    if status in (403,429):
        until=now+dt.timedelta(days=1)
        retry=m['retry_after']
        if retry:
            from email.utils import parsedate_to_datetime
            try: until=max(until,now+dt.timedelta(seconds=int(retry)))
            except ValueError:
                try: until=max(until,parsedate_to_datetime(retry).astimezone(dt.timezone.utc))
                except (ValueError,TypeError): pass
        stop['s21.q4cdn.com']={'utc_date':now.date().isoformat(),'status':status,'stop_until':until.isoformat(),'retry_after':retry,'source_url':url};dump(sp,stop)
        raise RuntimeError('Static source access denied/rate limited; no further source requests')
    return m,b,p

def pdf_text(path):
    # Pure extraction by available bundled runtime; no package installation.
    code='from pypdf import PdfReader;import sys;print("\\n".join(p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages))'
    return subprocess.check_output([RUNTIME,'-c',code,str(path)],text=True)

def validate(events):
    keys=[]
    for e in events:
        key=(e['fiscal_year'],e['fiscal_quarter']);keys.append(key)
        p=ROOT/e['evidence_path'];body=p.read_bytes()
        if sha(body)!=e['evidence_sha256']: raise ValueError('Source hash mismatch')
        rebuilt=parse_pdf_text(pdf_text(p),*key,e['source_url'],e['evidence_path'],body)
        for k in ('release_date','ticker','historical_ticker','fiscal_period_end_actual','publication_timezone','publication_timestamp'):
            if e[k]!=rebuilt[k]: raise ValueError('Event evidence field mismatch: '+k)
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate fiscal quarter')
    return {'verified_events':len(keys),'gaps':['%dQ%d'%x for x in expected_quarters() if x not in keys]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only');ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    RAW.mkdir(parents=True,exist_ok=True)
    ep=HERE/'events.json';events=json.loads(ep.read_text()) if ep.exists() else []
    if args.validate_only:
        print(json.dumps(validate(events)));return
    overridepath=HERE/'URL_OVERRIDES.json';overrides=json.loads(overridepath.read_text()) if overridepath.exists() else {}
    bykey={(x['fiscal_year'],x['fiscal_quarter']):x for x in events};manifest=[]
    for y,q in expected_quarters():
        label='%dQ%d'%(y,q)
        if args.only and label not in args.only.split(','): continue
        if (y,q) in bykey: continue
        url=overrides.get(label,candidate(y,q))
        try:
            meta,body,p=fetch(url)
            if meta['http_status']!=200 or not body.startswith(b'%PDF'):
                manifest.append({'quarter':label,'url':url,'status':meta['http_status'],'result':'not_pdf'});print(label,meta['http_status'],flush=True);continue
            t=pdf_text(p);p.with_suffix('.txt').write_text(t)
            e=parse_pdf_text(t,y,q,url,str(p.relative_to(ROOT)),body)
            if e['release_date']>CUTOFF: raise ValueError('Release is after study cutoff')
            bykey[y,q]=e;dump(ep,sorted(bykey.values(),key=lambda x:x['release_date']))
            print(label,e['release_date'],e['historical_ticker'],flush=True)
            manifest.append({'quarter':label,'url':url,'result':'verified'})
        except RuntimeError as exc:
            manifest.append({'quarter':label,'url':url,'error':str(exc),'stopped':True});print(str(exc),flush=True);break
        except Exception as exc:
            manifest.append({'quarter':label,'url':url,'error':str(exc)});print(label,type(exc).__name__,str(exc),flush=True)
    old=json.loads((HERE/'manifest.json').read_text()) if (HERE/'manifest.json').exists() else []
    dump(HERE/'manifest.json',old+manifest)
    ev=sorted(bykey.values(),key=lambda x:x['release_date']);dump(ep,ev)
    status={'asof':dt.datetime.now(dt.timezone.utc).isoformat(),'study_cutoff':CUTOFF,'expected_quarters':len(expected_quarters()),'verified_events':len(ev),'gaps':['%dQ%d'%x for x in expected_quarters() if x not in bykey],'first_public_exact_count':0,'precision_policy':'All release dates are original datelines, not exact publication timestamps; local-calendar end bound + subsequent completed XNYS session computed downstream.','identity':'research_equity_FB joins historical FB/META same issuer; ticker transition 2022-06-09.'}
    dump(HERE/'STATUS.json',status);print(json.dumps(status),flush=True)

if __name__=='__main__': main()
