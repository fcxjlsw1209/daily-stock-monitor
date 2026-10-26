"""263 metadata-only clocks/identity contract. Never decode prices or NPZ."""
from copy import deepcopy
from datetime import date
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SCHEMA='risk_clock263_metadata_source_contract_v1'
INITIAL_PARTIAL='2016-08-17'
ORIGIN='2008-01-02'
PINS={
 'work/calendar_clock248_v2_SOURCE_REVIEW.json':'26b8f28a7c467788a55a891d5985945a8c1f5b4c82342dafad33e67501bbbac0',
 'work/peer239_maturity_policy.json':'3bbcbcbe141fb0d2d96cd64ca74f2ee1e9450e12c1f2594c7cb84fb98bf64862',
 'work/risk_clock263_source_feasibility.REVIEW_DRAFT.json':'9db948b27031a14b07ec15c474a6af023e1d47e93adf56673f198ff611556f90',
 'outputs/monthly_local_linear70/META.json':'3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
 'outputs/monthly_paired_target143/CALENDAR.json':'8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'}
CLOCK_KEYS=('query_index','date','initial_partial','potential_numeric_evaluation','symbols','query_position','next_official_session','numerator_positions','numerator_dates','current_positions','current_dates','lag_positions','lag_dates','required_positions','required_dates','required_count','return_count')
IDENTITY_KEYS=('query_index','column_index','date','symbol','issuer_key','not_before','available_from','bound_visible')

def need(ok,label):
    if not ok:raise ValueError(label)

def read_pin(path):
    p=ROOT/path;raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==PINS[path],'Metadata pin '+path);return json.loads(raw)

def dependency_files():return tuple(ROOT/p for p in PINS)

def calendar_days(calendar):
    need(isinstance(calendar,(list,tuple)),'Calendar sessions list');out=[]
    for d in calendar:
        need(type(d) is str,'ISO session string')
        dt=date.fromisoformat(d);need(dt.isoformat()==d and dt.weekday()<5 and (not out or d>out[-1]),'Ordered unique official weekdays');out.append(d)
    return out

def clock(q,day,names,sessions):
    need(day in sessions,'Query official session');t=sessions.index(day);partial=day==INITIAL_PARTIAL
    need(t+1<len(sessions),'Next official session witness')
    if not partial:need(t>=294 and sessions[t+1][:7]!=day[:7],'Formal monthend and full clock')
    ep=[] if partial else [t-294,t-42];cp=[] if partial else list(range(t-63,t+1));lp=[] if partial else list(range(t-105,t-41));rp=sorted(set(ep+cp+lp));ds=lambda ps:[sessions[i] for i in ps]
    out=dict(query_index=q,date=day,initial_partial=partial,potential_numeric_evaluation=not partial,symbols=deepcopy(names),query_position=t,next_official_session=sessions[t+1],numerator_positions=ep,numerator_dates=ds(ep),current_positions=cp,current_dates=ds(cp),lag_positions=lp,lag_dates=ds(lp),required_positions=rp,required_dates=ds(rp),required_count=len(rp),return_count=0 if partial else 126)
    need(tuple(out)==CLOCK_KEYS,'Clock schema');return out

def identities(meta,bounds,n):
    out=[]
    for q in range(n):
        for j,s in enumerate(meta['symbols'][q]):
            for b in bounds:
                if s==b['symbol']:
                    d=meta['dates'][q];out.append(dict(query_index=q,column_index=j,date=d,symbol=s,issuer_key=b['issuer_key'],not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=d))
    return out

def build_metadata_contract():
    old=read_pin('work/calendar_clock248_v2_SOURCE_REVIEW.json');meta=read_pin('outputs/monthly_local_linear70/META.json');sessions=calendar_days(read_pin('outputs/monthly_paired_target143/CALENDAR.json')['sessions']);bounds=read_pin('work/peer239_maturity_policy.json')['securities'];read_pin('work/risk_clock263_source_feasibility.REVIEW_DRAFT.json')
    need(old['panel_first_session']==ORIGIN and old['raw_source_count']==64,'Original64 origin')
    n=len(meta['dates']);width=len(meta['symbols'][0]);need(n==122 and width==49 and all(len(s)==width for s in meta['symbols']),'Frozen monthly axes')
    pins=dict(old['input_sha256']);pins.update(PINS)
    return dict(schema=SOURCE_SCHEMA,passed=True,metadata_only=True,source_admitted=False,strict_pit=False,panel_first_session=ORIGIN,origin_session_position=sessions.index(ORIGIN),events=n,width=width,formal_monthends=n-1,raw_source_count=64,raw_source_hashes=deepcopy(old['raw_source_hashes']),metadata_path=old['metadata_path'],metadata_sha256=old['metadata_sha256'],calendar_path=old['calendar_path'],calendar_sha256=old['calendar_sha256'],panel_manifest_path=old['panel_manifest_path'],panel_manifest_sha256=old['panel_manifest_sha256'],listing_bounds=deepcopy(bounds),identity_cells=identities(meta,bounds,n),clock_events=[clock(q,meta['dates'][q],meta['symbols'][q],sessions) for q in range(n)],potential_clock_is_not_state_or_U0_admission=True,actual107_slot_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',original_U0_and_core_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE',prices_loaded=False,npz_decoded=False,actual_features_orders_returns_computed=False,registered_strategy=False,input_sha256=pins,source_policy=dict(original_fullpast='Full official grid from2008-01-02; original pandas missing semantics; never infer U0 fromC/E.',new_required='All107 slots for each noncalm ordinary originalU0; strictly positive finite, no source shrink/fill.',listing='Original239 visible whole-issuer bounds only; no262/252 actual-trade overrides or warmup admission.',zero='Only after all required source and applicable risk arithmetic, either zeroSD anywhere yields both whole-query incumbent.',calm_partial='No new107 numerical consumption.',additional_provider=False,additional_sector_mapping=False,additional_vox=False))
