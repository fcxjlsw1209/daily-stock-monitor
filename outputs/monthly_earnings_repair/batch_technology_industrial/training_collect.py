"""Targeted pre-2016 training candidates; never reuse later issuer/ticker identities."""
from join_collect import *

def training_event(k,p,m,t,public=False):
 t=clean(t)
 if k in ['IBM','QCOM']:
  packed=re.sub(r'\s+','',t)
  # Compact whitespace only; no digits or letters are added to damaged PDF text.
  city='ARMONK' if k=='IBM' else 'SANDIEGO'
  dm=re.search(city+r'[A-Za-z.,-]{0,15}('+MONTH+r')(\d{1,2}),?(20\d{2})',packed,re.I)
  if not dm:raise ValueError('Original compacted issuer dateline absent')
  release=iso(dm.group(1)+' '+dm.group(2)+' '+dm.group(3))
  if k=='IBM':
   qm=re.search(r'IBMREPORTS(20\d{2})'+ORD+r'-?QUARTER',packed,re.I)
   if not qm:raise ValueError('IBM actual result headline fiscal identity absent')
   fy=int(qm.group(1));q=QMAP[qm.group(2).lower()]
   hm=re.search(r'COMPARATIVEFINANCIALRESULTS',packed,re.I)
   if not hm:raise ValueError('IBM current financial statement heading absent')
   head=packed[hm.end():hm.end()+700]
   if not re.search('ThreeMonths',head,re.I):raise ValueError('IBM current quarterly table absent')
   md=re.search('('+MONTH+r')(\d{1,2}),?',head,re.I)
   ym=re.search(r'20\d{2}',head[md.end():]) if md else None
   if not md or not ym:raise ValueError('IBM observed current quarter date absent')
   period=iso(md.group(1)+' '+md.group(2)+' '+ym.group());pe='Whitespace-compacted original comparative financial-results table: '+head[:350]
   endmonth=12
  else:
   qm=re.search(ORD+r'Quarter(?:and)?(?:Fiscal)?(20\d{2})',packed[:1500],re.I)
   if not qm:raise ValueError('Qualcomm actual results fiscal identity absent')
   q=QMAP[qm.group(1).lower()];fy=int(qm.group(2));endmonth=9
   pm=re.search(r'quarter(?:of)?fiscal20\d{2}ended('+MONTH+r')(\d{1,2}),?(20\d{2})',packed[:3500],re.I)
   if not pm:raise ValueError('Qualcomm observed current quarter end absent')
   period=iso(pm.group(1)+' '+pm.group(2)+' '+pm.group(3));pe='Whitespace-compacted original actual results narrative: '+pm.group()
  pd=dt.date.fromisoformat(period);month=(endmonth-3*(4-q)-1)%12+1;year=fy-(month>endmonth);ref=dt.date(year,month,calendar.monthrange(year,month)[1])
  if abs((pd-ref).days)>10:raise ValueError('Issuer fiscal identity disagrees with observed current period')
  if not 0<(dt.date.fromisoformat(release)-pd).days<100:raise ValueError('Invalid actual publication lag')
  if not '2011-01-01'<=release<=CUTOFF:raise ValueError('Outside requested window')
  r=dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,dateline_excerpt='Whitespace-compacted original release dateline: '+dm.group(),period_excerpt=pe,identity_excerpt='Whitespace-compacted original result headline: '+qm.group())
 else:r=parse(k,t)
 e=event(k,p,m,r,'public_index_observation_of_original_issuer_release_not_full_bytes' if public else 'issuer_original_release_full_bytes')
 e['coverage_scope']='training_candidate_history_only' if k in ['IBM','HPQ','QCOM'] else 'monthly_and_training'
 if e['release_date']<'2011-01-01':e['training_predecessor_only']=True
 if public:e['evidence_limitation']='Saved public-index rendering of the original official release; SHA-256 identifies this observation JSON, not original HTML bytes.'
 if k=='HPQ':e['issuer_identity_note']='Pre-2015 Hewlett-Packard Company (HPQ), before the Hewlett Packard Enterprise separation; not HPE financial results.'
 return e

def build_training():
 for k in ['IBM','QCOM','HPQ']:
  out=[];pending=[]
  for mp in (HERE/k/'raw').glob('*.meta.json'):
   if mp.name.endswith('.text.meta.json'):continue
   m=json.load(open(mp));p=mp.with_name(mp.name.replace('.meta.json','.bin'))
   if m['status']!=200:continue
   if not p.read_bytes().startswith(b'%PDF'):pending.append(dict(source_url=m['url'],reason='Discovery HTML, redirect, or JavaScript shell; not admitted as a full earnings original'));continue
   try:
    e=training_event(k,p,m,extract(p))
    if k=='IBM' and not e['release_date']<='2016-07-31':continue
    if k=='QCOM' and not '2011-01-01'<=e['release_date']<='2014-06-30':continue
    out.append(e)
   except ValueError as ex:pending.append(dict(source_url=m['url'],reason=str(ex)))
  if k=='HPQ':
   for p in (HERE/k/'public_index_observations').glob('*.json'):
    j=json.load(open(p))
    for block in j['rendering'].split('--------------------------------------------------------------------------------'):
     url=re.search(r'\((https://investor\.hp\.com/news-events/news/news-details/(?:2010|2011)/[^)]+)\)',block)
     if not url:continue
     try:
      e=training_event(k,p,dict(url=url.group(1),retrieved_at=j['observed_at']),block,True)
      if (e['fiscal_year'],e['fiscal_quarter']) in [(2010,4),(2011,1)]:out.append(e)
     except ValueError as ex:pending.append(dict(source_url=url.group(1),reason=str(ex)))
  merge(k,out,pending)
 for mp in (HERE/'CSCO/raw').glob('*.meta.json'):
  if mp.name.endswith('.text.meta.json'):continue
  m=json.load(open(mp))
  if '/y2010/m11/cisco-reports-first-quarter-earnings.html' in m['url']:
   p=mp.with_name(mp.name.replace('.meta.json','.bin'));e=training_event('CSCO',p,m,extract(p));merge('CSCO',[e],[])
if __name__=='__main__':build_training()
