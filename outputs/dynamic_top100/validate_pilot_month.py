"""March 2010 evidence readiness; never turn reference-date ranks into PIT members."""
from pathlib import Path
import json,re,datetime
from collections import defaultdict
import pandas as pd
import exchange_calendars as xc
from reconcile_share_evidence import reconcile
P=Path(__file__).resolve().parent
def norm(s):
 s=re.sub(r'\b(incorporated|inc|corporation|corp|limited|ltd|plc|company|co)\b','',s.lower())
 return re.sub('[^a-z0-9]','',s)
ident=pd.read_csv(P/'expanded_sec/candidate_identity_evidence.csv',dtype=str)
lookup=defaultdict(set)
for r in ident.itertuples():lookup[norm(r.name)].add(r.cik.zfill(10))
for folder in ['expanded_sec','foreign_sec','priority_sec']:
 for f in (P/folder).glob('*_submissions.json'):
  j=json.loads(f.read_text())
  if not j.get('cik'):continue
  for name in [j.get('name','')]+[x['name'] for x in j.get('formerNames',[])]:
   if name:lookup[norm(name)].add(str(j['cik']).zfill(10))
facts=pd.read_csv(P/'expanded_sec/disclosure_events_unverified.csv.gz',dtype={'cik':str});facts.cik=facts.cik.str.zfill(10)
pilot=pd.read_csv(P/'pilot_identity/disclosure_events_unverified.csv.gz',dtype={'cik':str})
facts=pd.concat([facts,pilot],ignore_index=True).drop_duplicates()
facts.cik=facts.cik.str.zfill(10)
facts=reconcile(facts,json.loads((P/'pilot_identity/share_corrections.json').read_text()))
aliases=json.loads((P/'pilot_identity/month_aliases.json').read_text())
assert aliases['session_from']=='2010-03-01' and aliases['session_to']=='2010-03-31'
for name,cik in aliases['aliases'].items():lookup[norm(name)]={cik}
for name,cik in json.loads((P/'pilot_identity/original_report_aliases.json').read_text()).items():
 lookup[norm(name)]={cik}
facts['available']=pd.to_datetime(facts.available_no_earlier_than,utc=True)
facts=facts[(~facts.conflict)&facts.shares.gt(0)&(~facts.scale_review_required)]
refs=[r for r in json.loads((P/'additional_sources/ft_rankings/reference_rankings.json').read_text()) if r['reference_date']=='2010-03-31']
assert len(refs)==500
sessions=xc.get_calendar('XNYS',start='2010-03-01',end='2010-03-31').sessions
anchors=[]
for r in refs:
 ids=sorted(lookup[norm(r['company_name_as_printed'])]);anchors.append({**r,'candidate_ciks':ids,'mapping_verified':False})
daily=[];evidence=[]
for date in sessions:
 day=date.strftime('%Y-%m-%d');freeze=pd.Timestamp(day+' 20:20',tz='America/Los_Angeles').tz_convert('UTC')
 available=facts[facts.available<=freeze];known=set(available.cik)
 matched=[r for r in anchors if len(r['candidate_ciks'])==1]
 withshares=[r for r in matched if r['candidate_ciks'][0] in known]
 daily.append({'session':day,'reference_candidates':500,'unambiguous_exact_name_hints':len(matched),'hints_with_available_share_observation':len(withshares),'certified_members':0,'membership_proven':False,'blockers':['effective-dated US listing/issuer mappings not certified','no verified cap intervals for all potential boundary entrants','global annual roster does not prove complete daily eligible coverage']})
 if day=='2010-03-31':
  for r in anchors:
   ids=r['candidate_ciks'];sub=available[available.cik.isin(ids)].sort_values(['available','reported_as_of'])
   evidence.append({'global_reference_rank':r['global_rank'],'name':r['company_name_as_printed'],'country':r['country'],'reference_market_cap_usd':r['market_cap_usd'],'candidate_ciks':ids,'available_share_rows':len(sub),'latest_share_filed':str(sub.iloc[-1].filed) if len(sub) else None,'status':'identity_hint_and_share_units_need_verification' if len(ids)==1 and len(sub) else 'identity_or_share_disclosure_missing','usable_for_daily_rank':False})
report={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'month':'2010-03','sessions':len(daily),'certified_sessions':0,'purpose':'Measure actual evidence readiness, not infer membership from future month-end reference','daily':daily,'month_end_candidate_evidence':evidence,'reference_parser_warning':'Country-word truncation fixed; normalized and former-name matches are discovery only, not effective-dated identities.','can_start_backtest':False}
(P/'PILOT_2010_03.json').write_text(json.dumps(report,indent=2));print(json.dumps({'sessions':len(daily),'first':daily[0],'last':daily[-1],'top100_reference_blockers':[{k:r[k] for k in ['global_reference_rank','name','status']} for r in evidence[:100] if r['status']=='identity_or_share_disclosure_missing']},indent=2))
