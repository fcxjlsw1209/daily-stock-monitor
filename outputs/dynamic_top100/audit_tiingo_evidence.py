"""Validate recovered Tiingo bars and preserve lifecycle/identity exclusions."""
import json,hashlib
from pathlib import Path
import pandas as pd
import exchange_calendars as xc
from audit_price_archives import audit
P=Path(__file__).resolve().parent;D=P/'additional_sources/tiingo'
# Issuer/exchange suspension notices; these rows are not executable US sessions.
cutoffs={'TWTR':'2022-10-28','ATVI':'2023-10-13','CHL':'2021-01-11','CEO':'2021-03-09','PTRCY':'2022-09-09'}
# XTO/MHS effective merger days are conservatively quarantined pending intraday
# cessation verification. Heinz explicitly permits trading through June 7 close.
cutoffs.update({'AYE':'2011-02-25','BJS':'2010-04-28','MFE':'2011-02-28','NVLS':'2012-06-04','ANRZQ':'2015-07-16'})
cutoffs.update({'PBG':'2010-02-26','RSH':'2015-02-02','RSHCQ':'2015-02-02','SUNEQ':'2016-04-21'})
cutoffs.update({'XTO':'2010-06-25','MHS':'2012-04-02','HNZ':'2013-06-08'})
sources={'PTRCY':'https://www.sec.gov/Archives/edgar/data/1108329/000094787124000089/ss2990214_15f12b.htm','CHL':'https://www.sec.gov/Archives/edgar/data/1117795/000156459021000492/chl-6k_20210108.htm','CEO':'https://s2.q4cdn.com/154085107/files/doc_news/NYSE-to-Commence-Delisting-Proceedings-Against-CNOOC-Limited-CEO-to-Comply-with-Executive-Order-13959-2021.pdf','FOX_rename':'https://www.nasdaqtrader.com/TraderNews.aspx?id=dtn2019-05'}
sources.update({'AYE':'https://www.sec.gov/Archives/edgar/data/3673/000119312511047579/d8k.htm','BJS':'https://www.sec.gov/Archives/edgar/data/808362/000095012310046646/h70190e10vq.htm','MFE':'https://www.sec.gov/Archives/edgar/data/50863/000005086311000050/form8k.htm','NVLS':'https://www.sec.gov/Archives/edgar/data/836106/000119312512258808/d362127dex991.htm','ANRZQ':'https://www.sec.gov/Archives/edgar/data/1301063/000130106315000048/anr8-k07x17x2015.htm'})
sources.update({'PBG':'https://www.sec.gov/Archives/edgar/data/1076405/000095010310000535/dp16635_8k-pepbo.htm','RSH':'https://www.sec.gov/Archives/edgar/data/96289/000087666115000132/ruleprovisionnotice.htm','SUNEQ':'https://www.sec.gov/Archives/edgar/data/945436/000119312516559806/d185590d8k.htm'})
sources.update({'XTO':'https://www.sec.gov/Archives/edgar/data/868809/000095010310001867/dp18265_8k.htm','MHS':'https://www.sec.gov/Archives/edgar/data/1532063/000119312512225539/d345721dex991.htm','HNZ':'https://news.kraftheinzcompany.com/press-releases-details/2013/Berkshire-Hathaway-and-3G-Capital-Complete-Acquisition-of-HJ-Heinz-Company/default.aspx'})
cal=set(xc.get_calendar('XNYS',start='2008-01-01',end='2026-10-05').sessions.strftime('%Y-%m-%d'))-{'2025-01-09'}
results=[]
for file in sorted(D.glob('*_prices.json')):
 symbol=file.name.split('_prices')[0];raw=file.read_bytes();meta=json.loads(file.with_name(file.stem+'.provenance.json').read_text())
 assert hashlib.sha256(raw).hexdigest()==meta['sha256']
 rows=json.loads(raw)
 if not rows:
  results.append({'symbol':symbol,'rows':0,'status':'no data'});continue
 df=pd.DataFrame(rows);df['symbol']=symbol;out=audit(df,'tiingo_'+symbol)
 terminal=out.date.ge(cutoffs[symbol]) if symbol in cutoffs else pd.Series(False,index=out.index)
 nonsession=~out.date.isin(cal)
 wrong_identity=symbol in {'SI','PTR','GENZ','SII','EP','JAVA','KG','MI','PARA','PGN','SUNE'}
 # ACS begins at acquisition; NVLS metadata identifies a different ETF despite old dates.
 identity_review=symbol in {'ACS','NVLS','BTUUQ'}
 out['quarantine']=out.quarantine|terminal|nonsession|wrong_identity|identity_review
 out['terminal_or_post_us_suspension']=terminal;out['non_session']=nonsession
 out.to_csv(P/'recovered_prices'/('tiingo_'+symbol+'.csv.gz'),index=False,compression='gzip')
 good=out[~out.quarantine]
 results.append({'symbol':symbol,'rows':len(out),'quarantined':int(out.quarantine.sum()),'terminal_or_post_us_suspension':int(terminal.sum()),'non_session':int(nonsession.sum()),'wrong_identity':wrong_identity,'identity_review':identity_review,'first_candidate':good.date.min() if len(good) else None,'last_candidate':good.date.max() if len(good) else None,'identity_and_adjustments_verified':False})
report={'sources':sources,'results':results,'usable_for_backtest':False,'remaining':'Class identity, dividend/split/spinoff price basis, merger payouts and issuer caps need verification. Dates are not certification.'}
(P/'TIINGO_RECOVERY_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
