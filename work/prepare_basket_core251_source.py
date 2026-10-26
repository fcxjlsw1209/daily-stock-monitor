"""Build251 price-only source metadata; hash source bytes without decoding bars.

Run via run_local.sh. The launcher alone relocates historical JSON path strings.
No panel loader, model, source array, price CSV/JSON or strategy is imported.
"""
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/basket_core251_SOURCE_CONTRACT.json'
SCHEMA='monthly_basket_core251_metadata_source_review_v1'
CLOCK_KEYS=('query_index','date','initial_partial','numeric_evaluation_required','symbols','query_position',
 'next_official_session','price_start','price_end','source_start_position','source_end_position',
 'price_count','return_count','block_count','block_sessions','block_boundary_dates')
IDENTITY_KEYS=('query_index','column_index','date','symbol','issuer_key','not_before','available_from',
 'bound_visible','price_start','price_end','price_window_dates_before_bound')
EXPECTED={
 'work/after250_basket_core251_definition_v1.json':'c5709c65767a0929d5316ab4fe38a02cc795335819ec70ba72349190d354b4bc',
 'work/after250_basket_core251_definition_v2.json':'3fd3acdcff6e5f0ce782b5647ed77b96921a82717ad247c91832a61a7d476a17',
 'work/basket_core251_DEFINITION_REVIEW.json':'0b31b574290c01939c74a91a6846f16865202fbdeb47b6c22df651cb339b1e4f',
 'work/basket_core251_MATH_REVIEW_v2.json':'1e74e5cd49a612f66758ca8b5a9086b7c4083454e09e7d02a8b62b1d0398824f',
 'work/basket_core251_SOURCE_REVIEW.json':'d9f7ecf71cbcd2a648b4b212546409ca81b442879c993bd36d6a0fff2e74e296',
 'work/calendar_clock248_ORIGIN_DIAGNOSTIC.json':'d250fc4d42b6089c62f4b1c4a032b872db7a1dd3cc99f9c133b8c8421c8c0eaf',
 'outputs/monthly_local_linear70/PLAN.json':'0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96',
 'outputs/monthly_local_linear70/META.json':'3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
 'outputs/monthly_paired_target143/CALENDAR.json':'8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
 'outputs/index_union/alias_resolved_inputs/MANIFEST.json':'b48b3e337c9789bd6effbdb5ee533fddfbcfe124dfb2a07d330b9f1da43ee5ce'}

def need(value,label):
    if not value:raise ValueError(label)

def path(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    need(p.is_relative_to(ROOT) and 'CloudStorage' not in p.parts,'Local metadata only')
    need(p.name!='CONTINUATION_STATUS.json','Mutable central is not an input pin')
    return p

def sha(value):
    h=hashlib.sha256()
    with path(value).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(value):
    p=path(value);need(p.suffix=='.json','Metadata JSON only')
    return json.loads(p.read_text())

def bind(pins,value,expected=None):
    p=path(value);key=str(p.relative_to(ROOT));h=sha(p)
    need(expected is None or h==expected,'Digest mismatch '+key)
    need(key not in pins or pins[key]==h,'Conflicting digest '+key)
    pins[key]=h;return h

def main():
    need(platform.python_version()=='3.9.6','Use pinned run_local.sh')
    need(not OUT.exists(),'Source contract write-once')
    pins={}
    for p,h in EXPECTED.items():bind(pins,p,h)
    reviews={k:read('work/basket_core251_'+v+'.json') for k,v in
             (('definition','DEFINITION_REVIEW'),('math','MATH_REVIEW_v2'),('source','SOURCE_REVIEW'))}
    for r in reviews.values():
        need(r['passed'] is True and r['unresolved_blockers']==[],'Prospective reviews passed')
        for p,h in r['input_sha256'].items():bind(pins,p,h)
    s=reviews['source'];need(s['metadata_only'] is True and s['source_admitted'] is False,'Metadata-only source precedent')
    bind(pins,Path(__file__))
    meta_path='outputs/monthly_local_linear70/META.json'
    cal_path='outputs/monthly_paired_target143/CALENDAR.json'
    manifest_path='outputs/index_union/alias_resolved_inputs/MANIFEST.json'
    meta=read(meta_path);cal=read(cal_path)['sessions'];pos={d:i for i,d in enumerate(cal)}
    need(cal==sorted(set(cal)) and all(date.fromisoformat(d).isoformat()==d for d in cal),'Canonical official axis')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122,'122 metadata events')
    need(meta['dates']==sorted(set(meta['dates'])) and meta['ends'][:-1]==meta['dates'][1:],'Ordered monthly intervals')
    need(meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed evaluation endpoints')
    need(len(meta['etfs'])==len(set(meta['etfs']))==17,'17 frozen ETFs')
    need(all(len(x)==len(set(x))==49 and set(meta['etfs']).issubset(x) for x in meta['symbols']),'Full32+17 inventory')
    original=read('outputs/monthly_local_linear70/PLAN.json')
    raw={str(path(p)):h for p,h in original['raw_source_hashes'].items()}
    need(len(raw)==64 and raw==s['raw_source_hashes'],'Original64 source map')
    for p,h in raw.items():bind(pins,p,h)
    origin=read('work/calendar_clock248_ORIGIN_DIAGNOSTIC.json');manifest=read(manifest_path)
    need(origin['actual_origin']==s['source_origin']['panel_first_session']=='2008-01-02','Actual original panel axis origin')
    need(origin['raw_sources_exact'] is True and origin['raw_sources']==64,'Origin64 byte proof')
    need(manifest['start']==s['source_origin']['reported_manifest_start']=='2010-01-04','Reported manifest start distinct from origin')
    clocks=[]
    for q,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=pos[day];partial=day=='2016-08-17'
        need(t>=294 and t+1<len(cal) and (not partial or q==0),'Available official metadata clock')
        need(partial or cal[t+1][:7]!=day[:7],'Formal month-end witness')
        dates=cal[t-294:t-41];boundaries=dates[::21]
        need(len(dates)==253 and len(boundaries)==13 and boundaries[-1]==dates[-1],'253 prices and13 block endpoints')
        c=dict(query_index=q,date=day,initial_partial=partial,numeric_evaluation_required=not partial,
          symbols=deepcopy(names),query_position=t,next_official_session=cal[t+1],price_start=dates[0],price_end=dates[-1],
          source_start_position=t-294,source_end_position=t-42,price_count=253,return_count=252,
          block_count=12,block_sessions=21,block_boundary_dates=boundaries)
        need(tuple(c)==CLOCK_KEYS,'Clock schema')
        prior=s['calendar']['clock_events'][q]
        need(all(c[k]==prior[k] for k in ('query_index','date','initial_partial','next_official_session','price_start','price_end','source_start_position','source_end_position')),'Independent metadata clock comparison')
        need(prior['query_session_position']==t and prior['prices']==253 and prior['return_intervals']==252,'Independent clock scope')
        clocks.append(c)
    bounds=deepcopy(s['listing_bounds']);by={b['symbol']:b for b in bounds}
    need(len(bounds)==len(by)==2 and set(by)=={'SPCX','SKHY'},'Two reviewed visible listing bounds')
    prior_ids={(x['query_index'],x['column_index']):x for x in s['listing_identity_cells']}
    expected={(q,j) for q,names in enumerate(meta['symbols']) for j,x in enumerate(names) if x in by}
    need(len(prior_ids)==5 and set(prior_ids)==expected,'All five reviewed inventory occurrences')
    identities=[]
    for q,j in sorted(expected):
        c=clocks[q];symbol=c['symbols'][j];b=by[symbol];old=prior_ids[q,j]
        x=dict(query_index=q,column_index=j,date=c['date'],symbol=symbol,issuer_key=b['issuer_key'],
          not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=c['date'],
          price_start=c['price_start'],price_end=c['price_end'],
          price_window_dates_before_bound=sum(d<b['not_before'] for d in cal[c['source_start_position']:c['source_end_position']+1]))
        need(tuple(x)==IDENTITY_KEYS and all(old[k]==v for k,v in x.items()),'Independent issuer/clock metadata identity')
        identities.append(x)
    result=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
      status='METADATA_CONTRACT_COMPLETE_ACTUAL_REQUIRED_PRICE_ADMISSION_PENDING',passed=True,metadata_only=True,
      source_admitted=False,strict_pit=False,registered_strategy=False,prices_loaded=False,volumes_loaded=False,npz_decoded=False,
      actual_features_orders_returns_computed=False,actual_features_computed=False,actual_orders_computed=False,actual_returns_computed=False,
      raw_source_count=64,raw_source_hashes=raw,raw_source_bytes_rehashed=True,raw_source_values_decoded=False,
      raw_source_verification='Original64 byte hashes verified only; including protected baseline volume/sector/opening files does not make them new basket-statistic inputs.',
      panel_first_session=origin['actual_origin'],panel_axis_end=origin['actual_end'],panel_axis_rows=origin['actual_rows'],
      panel_origin_evidence=deepcopy(s['source_origin']['origin_evidence']),manifest_reported_start=manifest['start'],manifest_reported_start_is_panel_origin=False,
      panel_manifest_path=manifest_path,panel_manifest_sha256=pins[manifest_path],
      calendar_path=cal_path,calendar_sha256=pins[cal_path],metadata_path=meta_path,metadata_sha256=pins[meta_path],
      events=122,width=49,formal_monthends=121,etfs=deepcopy(meta['etfs']),price_sessions=253,return_intervals=252,block_count=12,intervals_per_block=21,
      clock_event_keys=list(CLOCK_KEYS),clock_events=clocks,identity_cell_keys=list(IDENTITY_KEYS),listing_bounds=bounds,identity_cells=identities,
      listing_scope='All inventory metadata occurrences checked; new positive-price contradictions apply only to formal Eold-true required253price windows and query-visible exact issuer bounds. Original source/core checks retain their separate scope.',
      unknown_other_listing_bounds=True,actual_Eold_counts='UNKNOWN_NOT_DECODED',actual_cell_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',
      original_U0_and_core_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE',actual_price_continuity='UNKNOWN_INHERITED_ADJUSTED_PRICE_ASSUMPTIONS',
      source_scope=deepcopy(s['source_scope']),source_policy=dict(
        required_domain='Formal event Eold-true names, before M/ranking; not onlyTop6. All49 metadata preserved.',
        original_control='Original full49/fullpast core reconstruction and frozen Local67 score/order parity must pass separately. Eold-false is unconsumed only for NEW253-price statistics, not for inherited control verification.',
        partial='Named2016-08-17 copies exact incumbent and performs no new numerical source/statistic evaluation.',
        ordinary_missing='Any required missing symbol, official row, scalar or issuer evidence is hard failure; no imputation, dropping, ranking replacement, source-mask reuse or whole-month incumbent fallback.',
        invalid='Required zero,negative,NaN,Inf,bool,nonreal,failed/overflowing cast or nonzero value casting tozero is hard failure. New exact arithmetic begins only after all required253 prices validate.',
        identity='Apply only same symbol AND issuer with available_from<=query; any required positive pre-bound price is a hard identity contradiction. No future evidence or alternate-security splice.',
        source_axis='Query and full original consumed axes must match frozen control requirements; new253 source dates must equal the official clock exactly. No reindex repair, compression or forward fill.',
        zero_variance='Only complete-source exact zero VJ orVD excludes a triple from the common objective set; it is not source missing. Emptycommon set takes rawMTop3.',
        transition='No volume-unit or250zero/missing transition-fallback policy is imported. Existing adjusted-price/alias/provider assumptions remain unverified; a known required identity contradiction hardfails.',
        numeric_prefix='Source_review may remain full for logical/physical prefixes. Validate consumed q<n clock/identity records; ignore future numeric inputs. Full-inventory metadata is not a computed E mask.'),
      transition_policy=dict(new_transition_rules=[],volume_transition_policy_used=False,ordinary_source_fallback_allowed=False,empty_list_certifies_continuity=False,scope='Inherited original adjusted-price identity/vintage limitations only; no new transition-based availability rule.'),
      metadata_review_scope='Official dates, identities, fixed source manifests and byte hashes only. No actual Eold mask or new253 price coverage read; prior250 support is not inherited.',
      limitations=deepcopy(s['listing_limits'])+['Original price adjustments and estimated pool remain declared assumptions; strictPIT false.','No volume,sector,VOX or universalSPY source requirement in the new basket objective.','No original source availability or originalcore parity certificate is issued by this metadata builder.'])
    for key,p in [('definition','work/after250_basket_core251_definition_v2.json'),('definition_review','work/basket_core251_DEFINITION_REVIEW.json'),('math_review','work/basket_core251_MATH_REVIEW_v2.json'),('source_review','work/basket_core251_SOURCE_REVIEW.json')]:
        result[key+'_path']=p;result[key+'_sha256']=pins[p]
    result['input_sha256']=pins
    for p,h in pins.items():need(sha(p)==h,'Input changed during metadata build '+p)
    with OUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='SOURCE_METADATA_READY_NOT_ADMITTED',path=str(OUT.relative_to(ROOT)),sha256=sha(OUT),input_pins=len(pins),raw_source_count=64,events=122,identity_cells=5)),flush=True)

if __name__=='__main__':main()
