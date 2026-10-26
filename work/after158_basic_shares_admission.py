"""Field-specific conditional source overlay. No ratios or strategy calculations."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SELF='work/after158_basic_shares_admission.py'
TEST='work/test_after158_basic_shares_admission.py'
POLICY='work/after158_basic_shares_admission_policy.json'
OUT='outputs/after158_basic_shares_admission_v1'
AV='outputs/after158_basic_shares_availability_v1'
BR='work/after158_basic_shares_gaap_bridge_v1'
RAW='outputs/after158_basic_shares_extraction_v1/RAW_SHARES.json'
REVIEW='work/after158_basic_shares_independent_remaining_review.json'
PINS={
 POLICY:'37963c102a37588af91dff3ad43efbea3f84e136817216729cfdf40477c2648d',
 AV+'/MANIFEST.json':'b5206f4b9ca869bc30c7d1cca95df97811b649945f9647a9189492b0a36b1d3f',
 AV+'/ROWS.json':'5fef7230b84af1379b552ed2d6863488790f7f0d17628f374b5adb9ba2568dc1',
 AV+'/EVENTS.json':'d6f920c98f37643fc5aca1953e5e8bdc3d13c3b73447b7e9ee6812f1d85d3ab9',
 BR+'/PLAN.json':'5781f97c37ccd5d75e18dbb43a8c0a02474554c2797d986bc3f29aedaddf858e',
 BR+'/METADATA.json':'82955c91435fcfa91c21155490d28424455955863099e07496f298a426dd0f19',
 BR+'/COMPARISON.json':'be95678dd809c1972545a0b9e02922ac8f38f6f7599299b891e0bf9bd9f27947',
 RAW:'31f221d888f5af960e4e601b2ca68ec29f665ece6c8d215a26fb9b19f6371fa3',
 REVIEW:'71ef19f2ac2c9cd41b1e6825bc3abf9956f1a06294b9f8aecb304554fd370aef',
}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def path(p):return Path(p) if Path(p).is_absolute() else ROOT/p
def sha(p):return hashlib.sha256(path(p).read_bytes()).hexdigest()
def read(p):return json.loads(path(p).read_text())
def write(p,d):
 q=path(p);q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('x') as f:json.dump(d,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def verify(pins):
 for p,h in pins.items():require(sha(p)==h,'Digest changed: '+p)
def merge(*maps):
 out={}
 for m in maps:
  for p,h in m.items():
   p=str(path(p).resolve());require(p not in out or out[p]==h,'Conflicting digest: '+p);out[p]=h
 return out
def stamp(s):
 t=datetime.fromisoformat(s.replace('Z','+00:00'));require(t.tzinfo is not None,'Timezone required');return t
def positive(s):
 require(type(s)is str,'Original exact count string required')
 v=Decimal(s);require(v.is_finite() and v>0,'Positive finite BASIC count required');return v
def period_ok(p):
 try:
  start,end=date.fromisoformat(p['period_start']),date.fromisoformat(p['period_end'])
  n=(end-start).days+1
  return 330<=n<=400 and p['days_inclusive']==n
 except (TypeError,KeyError,ValueError):return False

def event_overlay(raw,avail,meta,bridge,clock):
 require(len({x['event_id'] for x in (raw,avail,meta,bridge)})==1,'Event axis mismatch')
 require(raw['symbol']==avail['symbol']==meta['symbol']==bridge['symbol'],'Symbol mismatch')
 require(raw['periods']==avail['periods']==meta['periods'],'Original periods changed')
 require(raw['inherited_holds']==avail['inherited_holds'],'Original holds changed')
 require(raw['identity']==avail['identity'],'Original identity changed')
 reasons=['INHERITED:'+x for x in raw['inherited_holds']['inherited_rejection_codes']]
 pair=raw['periods'];p,c=pair['comparison'],pair['current']
 normal=period_ok(p) and period_ok(c)
 if normal:normal=date.fromisoformat(c['period_start'])==date.fromisoformat(p['period_end'])+timedelta(days=1)
 if not normal:reasons.append('ORIGINAL_NORMAL_CONSECUTIVE_PERIODS_NOT_ESTABLISHED')
 unique=meta['status']==bridge['metadata_status']=='unique_same_accession_annual_pair'
 if not unique:reasons.append('UNIQUE_EARLIEST_CACHED_GAAP_PAIR_UNAVAILABLE')
 if unique:
  require(meta['selected_accession']==bridge['selected_accession'] and meta['selected_filed']==bridge['selected_filed'],'Bridge accession changed')
  for original_side,meta_side in (('current','current_selected'),('comparison','prior_selected')):
   context=meta[meta_side]
   require(context['accn']==meta['selected_accession'] and context['filed']==meta['selected_filed'],'Different accession context')
   pp=pair[original_side]
   require(context['end']==pp['period_end'],'Bridge period end mismatch')
   if pp['period_start'] is not None:require(context['start']==pp['period_start'],'Bridge period start mismatch')
 compatible=unique and all(bridge['cells'][side]['equal'] is True or
  bridge['cells'][side]['native_rounding_interval_compatible'] is True for side in ('current','comparison'))
 if not compatible:reasons.append('ORIGINAL_GAAP_DENOMINATOR_NOT_CORROBORATED')
 counts={}
 for side in ('current','comparison'):
  cell=raw['cells'][side];positive(cell['value_in_shares'])
  require(cell['unit']['share_specific'] is True,'Share-specific unit required')
  require(cell['semantic_basis'] in ('basic_eps_denominator_weighting_implicit','explicit_basic_weighted_average_shares'), 'Wrong BASIC semantic basis')
  require(cell['period_end']==pair[side]['period_end'],'Cell not original annual period')
  counts[side]=cell['value_in_shares']
 known=clock(raw['release_date']);require(known==avail['known_at'],'Original knowledge clock changed')
 usable=max((known,clock(meta['selected_filed'])),key=stamp) if meta['selected_filed'] else known
 return dict(event_id=raw['event_id'],symbol=raw['symbol'],fiscal_year=raw['fiscal_year'],
  period_end=c['period_end'],prior_period_end=p['period_end'],known_at=known,usable_at=usable,
  current_shares=counts['current'],prior_shares=counts['comparison'],identity=deepcopy(raw['identity']),
  conditional_source_ready=not reasons,same_report_annual_pair=normal,same_share_basis=compatible,
  basis_is_declared_presentation_assumption=True,unresolved_reasons=reasons,
  inherited_holds=deepcopy(raw['inherited_holds']),original_event=deepcopy(raw),
  gaap_bridge=deepcopy(bridge),selected_gaap_contexts=deepcopy(meta),
  strict_source_admitted=False,old_admission_flags_changed=False)

def map_row(row,events):
 a=row['availability'];selected=a['selected_event_id'];cutoff=stamp(row['cutoff'])
 known=[e for e in events if e['symbol']==row['symbol'] and stamp(e['known_at'])<=cutoff]
 if known:
  latest=max(e['period_end'] for e in known);chosen=[e for e in known if e['period_end']==latest]
  require(len(chosen)==1 and chosen[0]['event_id']==selected,'Latest known selection changed')
 else:require(selected is None,'Unexpected selected event')
 reasons=list(a['reasons']);e=next((e for e in events if e['event_id']==selected),None)
 if e:
  reasons.extend(e['unresolved_reasons'])
  if stamp(e['usable_at'])>cutoff:reasons.append('LATEST_GAAP_CORROBORATION_NOT_YET_USABLE')
 if row['signal_index']==0:reasons.append('INITIAL_PARTIAL_EXACT_INCUMBENT')
 return dict(signal_index=row['signal_index'],candidate_index=row['candidate_index'],symbol=row['symbol'],
  cutoff=row['cutoff'],selected_event_id=selected,conditional_source_ready=bool(e) and not reasons,
  reasons=list(dict.fromkeys(reasons)),original_availability_row=deepcopy(row),
  strict_source_admitted=False,old_admission_flags_changed=False)

def summary(events,rows):
 s={'events':len(events),'conditionally_ready_events':sum(e['conditional_source_ready'] for e in events),
    'event_blockers':dict(Counter(x for e in events for x in e['unresolved_reasons']))}
 for name,start,end in (('all122',0,122),('early53',0,53),('recent35',53,88),('first88',0,88),('later34',88,122)):
  good=[r for r in rows if start<=r['signal_index']<end and r['conditional_source_ready']]
  s[name]=dict(slots=len(good),by_symbol=dict(Counter(r['symbol'] for r in good)),
   distinct_events=len({r['selected_event_id'] for r in good}),
   months_with_at_least_two=sum(sum(r['signal_index']==i for r in good)>=2 for i in range(start,end)))
 return s

def freeze():
 verify(PINS);v=read(AV+'/MANIFEST.json');b=read(BR+'/PLAN.json');r=read(REVIEW)
 require(r['passed'] and r['checks']['all_44_source_pairs_and_units_verified'],'Independent full raw review required')
 require(r['summary']['explicit_different_current_prior_basis_observed']==0,'Unresolved explicit split-basis conflict')
 maps=[PINS,v['input_hashes'],v['outputs'],r['input_hashes']]
 # The bridge owns and freezes its own transitive inputs.
 maps.append(b['input_hashes'])
 pins=merge(*maps);verify(pins)
 tests=read(OUT+'/TESTS.json');implementation={SELF:sha(SELF),TEST:sha(TEST)}
 require(tests['passed'] and tests['implementation_hashes']==implementation,'Fresh source-gate synthetic tests required')
 write(OUT+'/PLAN.json',dict(schema='basic_shares_conditional_admission_plan_v1',created_utc=datetime.now(timezone.utc).isoformat(),
  policy=read(POLICY),input_hashes=pins,implementation_hashes=implementation,
  tests_sha256=sha(OUT+'/TESTS.json'),ratios_computed=0,real_strategy_trials=0))
 print(json.dumps({'plan_sha256':sha(OUT+'/PLAN.json')}))

def run(expected):
 require(sha(OUT+'/PLAN.json')==expected,'Explicit frozen plan SHA required');p=read(OUT+'/PLAN.json')
 verify(p['input_hashes']);verify(p['implementation_hashes']);require(sha(OUT+'/TESTS.json')==p['tests_sha256'],'Changed tests')
 import after158_basic_shares_availability as av
 sessions,closes=av.calendar();clock=lambda d:av.known_clock(d,sessions,closes)['iso']
 raw=read(RAW)['events'];ae=read(AV+'/EVENTS.json')['events'];bm=read(BR+'/METADATA.json')['records'];br=read(BR+'/COMPARISON.json')['records']
 require(len(raw)==len(ae)==len(bm)==len(br)==44,'44 event axis required')
 events=[event_overlay(*parts,clock) for parts in zip(raw,ae,bm,br)]
 original=read(AV+'/ROWS.json');require(len(original)==5978,'Original full grid required')
 rows=[map_row(row,events) for row in original];s=summary(events,rows)
 verify(p['input_hashes']);verify(p['implementation_hashes'])
 write(OUT+'/EVENTS.json',{'events':events,'strict_source_admitted':False})
 write(OUT+'/ROWS.json',rows);write(OUT+'/COVERAGE.json',s)
 outputs={OUT+'/'+n:sha(OUT+'/'+n) for n in ('EVENTS.json','ROWS.json','COVERAGE.json')}
 write(OUT+'/MANIFEST.json',dict(schema='basic_shares_conditional_admission_manifest_v1',plan_sha256=expected,
  input_hashes=merge(p['input_hashes'],p['implementation_hashes'],{OUT+'/PLAN.json':expected}),outputs=outputs,
  summary=s,strict_source_admitted=False,old_admission_flags_changed=False,numerical_experiment_registered=False,
  activity={'ratios':0,'ranks':0,'scores':0,'orders':0,'returns':0},limitations=p['policy']['source_limitations']))
 print(json.dumps({'summary':s,'manifest_sha256':sha(OUT+'/MANIFEST.json')},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--plan-sha');args=ap.parse_args()
 if args.freeze:freeze()
 else:run(args.plan_sha)
