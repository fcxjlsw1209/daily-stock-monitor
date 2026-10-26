"""Metadata-only basic-share availability map; never open share cells as JSON."""
import argparse
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
SELF='work/after158_basic_shares_availability.py'
TEST='work/after158_basic_shares_availability_tests.py'
OUT='outputs/after158_basic_shares_availability_v1'
CONTRACT='work/after158_basic_shares_availability_contract.json'
EXTRACT_CONTRACT='work/after158_basic_shares_extraction_contract.json'
EXTRACT_MANIFEST='outputs/after158_basic_shares_extraction_v1/MANIFEST.json'
REVIEW='work/after158_basic_shares_independent_review.json'
META='outputs/monthly_local_linear70/META.json'
ROWS='outputs/monthly_revenue_pilot_source156/ROWS.json'
CAL='outputs/monthly_paired_target143/CALENDAR.json'
CAL_CODE='outputs/monthly_earnings_repair/research_calendar.py'
PINS={EXTRACT_CONTRACT:'59d7b44098b3c03118783295c36c470c361fab2a8ddf3dd7867f56e1bef85f49',EXTRACT_MANIFEST:'1930bcb6979e92aa47b82297665e4cb82e75edd0be0f0bb1e5f98910a833fc2b',REVIEW:'d0a108bbd11dbe79715d9ad63abcb3be05dbd04c597b7f78ffd491a27269a1f3',META:'3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',ROWS:'1732c3611fdbd14b00ef39d370a95493aec91f1a2c3ee60a2ce2ec3e27080196',CAL:'8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'}
HOLD_KEYS=('certified_daily_security_master','complete_correction_history_verified','fact_admitted','first_public_version_verified','inherited_rejection_codes','model_admitted','prior_field_admitted','strategy_admitted')
EVENT_KEYS=('original_index','event_id','symbol','fiscal_year','source','identity','release_date','available_from','periods')
PROHIBITED_JSON={'RAW_SHARES.json','RAW_FACTS.json'}
ERAS=(('all122',0,122),('first88',0,88),('later34',88,122),('early53',0,53),('recent35',53,88))


def require(ok,msg):
 if not ok:raise ValueError(msg)
def encoded(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resolve(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 require(p.resolve()==p,'Symlink or noncanonical input path');return p
def read(path):
 p=resolve(path);require(p.name not in PROHIBITED_JSON,'Financial cells may only be hashed, never parsed');return json.loads(p.read_text())
def verify(pins):
 for name,h in pins.items():require(sha(resolve(name))==h,'Hash mismatch: '+name)
def merge(*maps):
 result={}
 for mapping in maps:
  for p,h in mapping.items():
   name=str(resolve(p));require(name not in result or result[name]==h,'Conflicting input pin');result[name]=h
 return result
def write_once(path,obj):
 p=resolve(path);p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
 with os.fdopen(fd,'wb')as f:f.write(encoded(obj))
def code_pins():return {p:sha(resolve(p))for p in (SELF,TEST,'AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb')}
def day(value):
 require(type(value)is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}',value),'Strict ISO date required');d=date.fromisoformat(value);require(d.isoformat()==value,'Invalid date');return d
def ns(value):
 require(type(value)is str,'Timestamp string required');x=datetime.fromisoformat(value.replace('Z','+00:00'));require(x.tzinfo is not None,'Timezone required');delta=x.astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc);return (delta.days*86400+delta.seconds)*10**9+delta.microseconds*1000


def rules():
 return dict(scope='Exactly 44 frozen annual events and all original 122x49 candidate slots, including ETFs and every out-of-scope issuer.',read_boundary='No share-cell JSON parsing or share quantities; only hash financial artifacts. The independently reviewed complete 44/44 extraction manifest establishes all 44 event extraction-success flags.',clock='Known_at is the actual close of the second XNYS session strictly after release_date. Early-close sessions are complete scheduled sessions. Usable_at initially equals known_at; future corroboration requires a separate version taking max.',cutoff='For each original signal-day position t, use actual close of full-calendar session t-42. Preserve original signal decision day/close separately.',selection='Among symbol-matching annual events with known_at <= cutoff, choose maximum current period_end before any extraction, identity, staleness, split or hold test. Duplicate maximum periods block; no older-source fallback.',age='Maximum 550 calendar days from selected current annual period_end to cutoff date, inclusive. Prior period must precede current; no ratios or year-length adjustment.',identity='Selected original security_id must equal exact candidate symbol, issuer_id must be present, and original inclusive identity_valid_from/through must cover cutoff date. Never match ETFs or other symbols.',holds='Preserve every original hold and flag. Split review required for every event; no missing-keyword inference. Metadata alignment is conditional only, never source admission. Old CFO-specific holds remain listed separately and do not erase extracted share metadata.',output='Keep every original revenue row unchanged inside original_row, append a separate availability mapping. No original eligibility/financial/strategy flag is changed. No eligibility arrays or share values are consumed.',activity={'share_values_parsed':0,'ratios':0,'ranks':0,'features':0,'orders':0,'returns':0,'source_admissions':0})


def collect():
 verify(PINS);contract=read(EXTRACT_CONTRACT);manifest=read(EXTRACT_MANIFEST);review=read(REVIEW);cal=read(CAL)
 require(review['passed']is True and review['source_admission_changed']is False,'Independent extraction review required')
 summary=manifest['summary'];require(summary['events']==summary['raw_annual_pairs_extracted']==44 and summary['failures']=={},'Full 44/44 success needed for value-free status proof')
 require(review['summary']['events']==review['summary']['raw_annual_pairs_extracted']==44,'Independent population mismatch')
 require(len(contract['events'])==44 and len({e['event_id']for e in contract['events']})==44,'44 distinct original events')
 pins=merge(PINS,contract['input_hashes'],manifest['input_hashes'],manifest['implementation_hashes'],manifest['outputs'],review['input_hashes'],cal['provenance']['source_hashes']);verify(pins)
 require(len(cal['provenance']['source_hashes'])==9,'All nine calendar dependencies required')
 events=[]
 for e in contract['events']:
  item={k:deepcopy(e[k])for k in EVENT_KEYS};item['inherited_holds']={k:deepcopy(e[k])for k in HOLD_KEYS}
  item.update(extraction_success=True,extraction_success_basis='Bound independently reviewed manifest: exact original 44 events, 44 extracted pairs, zero failures; no share values opened.',split_review_required=True,same_share_basis=None,conditional_source_ready=False,source_admission_changed=False)
  # Dates and identity are structural; a malformed event is not silently dropped.
  day(item['release_date']);day(item['periods']['current']['period_end']);day(item['periods']['comparison']['period_end'])
  events.append(item)
 result=dict(schema='after158_basic_shares_availability_collection_v1',events=events,input_hashes=pins,calendar_source_hashes=cal['provenance']['source_hashes'],activity=rules()['activity'],source_admission_changed=False)
 write_once(OUT+'/COLLECTION.json',result);print(json.dumps({'collection_sha256':sha(resolve(OUT+'/COLLECTION.json')),'events':44,'input_pins':len(pins)}))


def freeze():
 c=read(OUT+'/COLLECTION.json');verify(c['input_hashes']);tests=read(OUT+'/TESTS.json');code=code_pins()
 require(tests['passed']is True and tests['synthetic_only']is True and tests['implementation_hashes']==code,'Current passing synthetic tests required')
 doc=dict(schema='after158_basic_shares_availability_contract_v1',created_utc=datetime.now(timezone.utc).isoformat(),rules=rules(),collection={'path':OUT+'/COLLECTION.json','sha256':sha(resolve(OUT+'/COLLECTION.json'))},input_hashes=c['input_hashes'],calendar_source_hashes=c['calendar_source_hashes'],implementation_hashes=code,tests_sha256=sha(resolve(OUT+'/TESTS.json')),actual_grid_mapped=False,source_admission_changed=False)
 write_once(CONTRACT,doc);print(json.dumps({'contract':CONTRACT,'sha256':sha(resolve(CONTRACT))}))


def calendar():
 frozen=read(CAL);spec=importlib.util.spec_from_file_location('basic_share_availability_calendar',resolve(CAL_CODE));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 schedule=module.market_calendar(frozen['start'],frozen['end']).schedule
 sessions=[str(x.date())for x in schedule.index];require(sessions==frozen['sessions'],'Frozen full XNYS axis changed')
 closes={d:{'iso':stamp.isoformat(),'ns':int(stamp.value)}for d,stamp in zip(sessions,schedule['close'])}
 require(all(ns(v['iso'])==v['ns']for v in closes.values()),'Close timestamp mismatch');return sessions,closes


def known_clock(release,sessions,closes):
 day(release);i=bisect_right(sessions,release)+1;require(i<len(sessions),'Second following XNYS session missing');d=sessions[i]
 return {'session':d,'iso':closes[d]['iso'],'ns':closes[d]['ns']}


def enrich(events,sessions,closes):
 result=[]
 for original in events:
  e=deepcopy(original);clock=known_clock(e['release_date'],sessions,closes)
  e.update(known_at=clock['iso'],known_at_ns=clock['ns'],known_session=clock['session'],usable_at=clock['iso'],usable_at_ns=clock['ns'],clock_is_declared_date_only_reconstruction=True)
  result.append(e)
 return result


def select(events,symbol,cutoff,cutoff_ns,is_etf=False):
 result=dict(selected_event_id=None,known_event_ids=[],metadata_alignment_pass=False,conditional_source_ready=False,source_admitted=False,reasons=[],unresolved_holds=[])
 if is_etf:result['reasons']=['ETF_NOT_APPLICABLE'];return result
 candidates=[e for e in events if e['symbol']==symbol]
 if not candidates:result['reasons']=['ISSUER_OUTSIDE_FIXED_44_EVENT_SCOPE'];return result
 known=[e for e in candidates if e['known_at_ns']<=cutoff_ns];result['known_event_ids']=[e['event_id']for e in known]
 if not known:result['reasons']=['NO_KNOWN_ORIGINAL_ANNUAL_EVENT'];return result
 latest=max(e['periods']['current']['period_end']for e in known);selected=[e for e in known if e['periods']['current']['period_end']==latest]
 if len(selected)!=1:result['reasons']=['AMBIGUOUS_LATEST_ORIGINAL_ANNUAL_EVENT'];return result
 e=selected[0];identity=e['identity'];age=(day(cutoff)-day(latest)).days
 result.update(selected_event_id=e['event_id'],selected_original_index=e['original_index'],period_end=latest,prior_period_end=e['periods']['comparison']['period_end'],known_at=e['known_at'],usable_at=e['usable_at'],known_at_ns=e['known_at_ns'],usable_at_ns=e['usable_at_ns'],age_days=age,identity=deepcopy(identity),inherited_holds=deepcopy(e['inherited_holds']),extraction_success=e['extraction_success'],split_review_required=e['split_review_required'],same_share_basis=e['same_share_basis'])
 problems=result['reasons']
 if e['usable_at_ns']<e['known_at_ns']:problems.append('USABLE_PRECEDES_KNOWN')
 if e['usable_at_ns']>cutoff_ns:problems.append('LATEST_NOT_YET_USABLE')
 if e['extraction_success']is not True:problems.append('LATEST_EXTRACTION_FAILED')
 if age<0:problems.append('LATEST_PERIOD_END_AFTER_CUTOFF')
 if age>550:problems.append('LATEST_PERIOD_OLDER_THAN_550_DAYS')
 if day(e['periods']['comparison']['period_end'])>=day(latest):problems.append('ANNUAL_PERIOD_ORDER_INVALID')
 try:
  interval=day(identity['identity_valid_from'])<=day(cutoff)<=day(identity['identity_valid_through'])
 except (ValueError,KeyError,TypeError):interval=False
 if identity.get('security_id')!=symbol or not identity.get('issuer_id') or not interval:problems.append('ORIGINAL_IDENTITY_DOES_NOT_COVER_CUTOFF')
 result['metadata_alignment_pass']=not problems
 result['unresolved_holds']=['SPLIT_AND_SAME_SHARE_BASIS_REVIEW_REQUIRED']
 for key in ('certified_daily_security_master','complete_correction_history_verified','first_public_version_verified'):
  if e['inherited_holds'].get(key)is not True:result['unresolved_holds'].append(key.upper()+'_NOT_VERIFIED')
 result['unresolved_holds'] += ['INHERITED:'+x for x in e['inherited_holds']['inherited_rejection_codes']]
 return result


def map_grid(meta,original,events,sessions,closes):
 require(len(meta['dates'])==len(meta['symbols'])==122 and len(original)==5978,'Original 122x49 grid required');require(len(meta['etfs'])==17,'Original ETF axis')
 pos={d:i for i,d in enumerate(sessions)};out=[]
 for i,(signal,names)in enumerate(zip(meta['dates'],meta['symbols'])):
  require(len(names)==len(set(names))==49 and signal in pos and pos[signal]>=42,'Complete original candidate/session axis');cutoff=sessions[pos[signal]-42]
  for j,symbol in enumerate(names):
   old=original[49*i+j];require((old['signal_index'],old['candidate_index'],old['symbol'],old['decision_date'],old['decision_cutoff_ns'])==(i,j,symbol,signal,closes[signal]['ns']),'Original row/decision close changed');require(ns(old['decision_cutoff'])==closes[signal]['ns'],'Original decision ISO close changed')
   mapped=select(events,symbol,cutoff,closes[cutoff]['ns'],symbol in meta['etfs'])
   out.append(dict(signal_index=i,candidate_index=j,symbol=symbol,decision_date=signal,decision_cutoff=old['decision_cutoff'],signal_calendar_position=pos[signal],cutoff_session_position=pos[signal]-42,cutoff_date=cutoff,cutoff=closes[cutoff]['iso'],cutoff_ns=closes[cutoff]['ns'],is_etf=symbol in meta['etfs'],original_row=deepcopy(old),availability=mapped,old_flags_changed=False,source_admission_changed=False))
 return out


def coverage(rows):
 result={}
 for name,start,end in ERAS:
  rr=[r for r in rows if start<=r['signal_index']<end];aligned=[r for r in rr if r['availability']['metadata_alignment_pass']]
  result[name]={'candidate_slots':len(rr),'stock_slots':sum(not r['is_etf']for r in rr),'selected_event_slots':sum(r['availability']['selected_event_id']is not None for r in rr),'conditional_metadata_aligned_slots':len(aligned),'conditional_metadata_aligned_by_symbol':dict(Counter(r['symbol']for r in aligned)),'distinct_selected_events':len({r['availability']['selected_event_id']for r in rr if r['availability']['selected_event_id']}),'months_with_two_metadata_aligned':sum(sum(r['signal_index']==i for r in aligned)>=2 for i in range(start,end)),'source_ready_slots':sum(r['availability']['conditional_source_ready']for r in rr),'source_admitted_slots':sum(r['availability']['source_admitted']for r in rr),'blocking_reasons':dict(Counter(reason for r in rr for reason in r['availability']['reasons']))}
 return result


def run(contract_sha):
 require(re.fullmatch('[0-9a-f]{64}',contract_sha or''),'Explicit contract SHA required');require(sha(resolve(CONTRACT))==contract_sha,'Contract digest changed');doc=read(CONTRACT)
 require(doc['rules']==rules(),'Contract rules changed');verify(doc['input_hashes']);verify(doc['implementation_hashes']);require(sha(resolve(doc['collection']['path']))==doc['collection']['sha256'],'Collection changed');require(sha(resolve(OUT+'/TESTS.json'))==doc['tests_sha256'],'Tests changed')
 collected=read(doc['collection']['path']);sessions,closes=calendar();events=enrich(collected['events'],sessions,closes);meta=read(META);original=read(ROWS);rows=map_grid(meta,original,events,sessions,closes);summary=coverage(rows)
 verify(doc['input_hashes']);verify(doc['implementation_hashes'])
 write_once(OUT+'/EVENTS.json',{'schema':'basic_share_availability_events_v1','events':events,'source_admission_changed':False})
 write_once(OUT+'/ROWS.json',rows);write_once(OUT+'/COVERAGE.json',summary)
 outputs={OUT+'/'+n:sha(resolve(OUT+'/'+n))for n in ('EVENTS.json','ROWS.json','COVERAGE.json')}
 manifest=dict(schema='basic_share_availability_manifest_v1',contract={'path':CONTRACT,'sha256':contract_sha},input_hashes=merge(doc['input_hashes'],doc['implementation_hashes'],{CONTRACT:contract_sha,doc['collection']['path']:doc['collection']['sha256'],OUT+'/TESTS.json':doc['tests_sha256']}),outputs=outputs,summary=summary,source_admission_changed=False,activity=rules()['activity'],limitations=['All conditional counts are metadata alignment only; no event is source ready or admitted.','Date-only second-session timing is a declared conservative research reconstruction, not certified first-public timestamp.','All old provenance/identity/correction-history flags and same-share-basis review remain.','Original eligibility arrays are not read here; later stages must apply unchanged eligibility and initial-partial policy.'])
 write_once(OUT+'/MANIFEST.json',manifest);print(json.dumps({'summary':summary,'outputs':outputs,'manifest_sha256':sha(resolve(OUT+'/MANIFEST.json'))},indent=2))


def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--collect',action='store_true');g.add_argument('--freeze',action='store_true');g.add_argument('--contract-sha');a=p.parse_args()
 if a.collect:collect()
 elif a.freeze:freeze()
 else:run(a.contract_sha)
if __name__=='__main__':main()
