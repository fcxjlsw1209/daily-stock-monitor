"""Independently reconstruct clocks for all consumed event metadata; no prices."""
import argparse
from bisect import bisect_left
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_consumed_clock156'
PINS={
 'outputs/monthly_earnings_repair/EVENTS.json':'1e027fc8b933af9753241c15a2e0dcc5a32daa13205ba4d74f90b64db6e2e103',
 'outputs/monthly_event_patch156/ADDED_EVENTS.json':'1d107c783d93a78ced29572383c55f55c50d24108014aee417dc3772fef2a5e3',
 'outputs/monthly_event_overlay_scope156/MANIFEST.json':'926f869fb85cc423c60b9995a7fd7a60c07b7662409603eb04563c27006d67e6',
 'outputs/monthly_paired_target143/CALENDAR.json':'8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
}


def check(ok,msg):
    if not ok:raise ValueError(msg)


def local(path):
    p=Path(path)
    if not p.is_absolute():p=ROOT/p
    check(p.is_relative_to(ROOT) and p.resolve()==p,'Nonlocal path')
    return p


def sha(path):return hashlib.sha256(local(path).read_bytes()).hexdigest()
def read(path):return json.loads(local(path).read_text())
def write(name,data):
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/name).open('x') as stream:
        json.dump(data,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def reconstruct(event,days,closes,pd):
    """Independent stdlib date/zone path, not event_schema.normalize_event."""
    check(event['precision'] in ('date_only','exact'),'Unknown precision')
    day=date.fromisoformat(event['publication_date'])
    check(date.fromisoformat(event['official_actual_period_end'])<=day,'Future fiscal period')
    zone=ZoneInfo(event['publication_timezone'])
    if event['precision']=='date_only':
        next_midnight=datetime.combine(day+timedelta(days=1),time.min,tzinfo=zone)
        upper=pd.Timestamp(next_midnight.astimezone(timezone.utc)).value-1
        idx=bisect_left(closes,upper+1)
        while idx<len(days) and days[idx]<=day.isoformat():idx+=1
        check(idx<len(days),'Calendar exhausted')
        return dict(available_session=days[idx],known_ns=closes[idx],upper_ns=upper)
    stamp=datetime.fromisoformat(event['publication_at'].replace('Z','+00:00'))
    check(stamp.tzinfo is not None and stamp.astimezone(zone).date()==day,'Invalid exact stamp')
    known=pd.Timestamp(stamp.astimezone(timezone.utc)).value
    idx=bisect_left(closes,known)
    check(idx<len(days),'Calendar exhausted')
    return dict(available_session=days[idx],known_ns=known,upper_ns=None)


def synthetic(pd):
    days=['2024-03-11','2024-11-04','2024-11-29','2024-12-02','2025-01-08','2025-01-10']
    closes=[pd.Timestamp(t).value for t in ['2024-03-11T20:00Z','2024-11-04T21:00Z',
        '2024-11-29T18:00Z','2024-12-02T21:00Z','2025-01-08T21:00Z','2025-01-10T21:00Z']]
    def ev(day):return dict(precision='date_only',publication_date=day,publication_timezone='America/Los_Angeles',official_actual_period_end='2023-12-31')
    a=reconstruct(ev('2024-03-10'),days,closes,pd)
    check(a['upper_ns']==pd.Timestamp('2024-03-11T07:00Z').value-1,'Spring DST local-day end')
    a=reconstruct(ev('2024-11-03'),days,closes,pd)
    check(a['upper_ns']==pd.Timestamp('2024-11-04T08:00Z').value-1,'Fall DST local-day end')
    check(reconstruct(ev('2024-11-28'),days,closes,pd)['known_ns']==closes[2],'Early close')
    check(reconstruct(ev('2024-11-29'),days,closes,pd)['available_session']=='2024-12-02','Friday to Monday')
    check(reconstruct(ev('2025-01-08'),days,closes,pd)['available_session']=='2025-01-10','Official closure')
    e=ev('2024-11-29');e.update(precision='exact',publication_at='2024-11-29T18:20:00+00:00')
    check(reconstruct(e,days,closes,pd)['available_session']=='2024-12-02','Exact after early close')
    e['publication_at']='2024-11-29T17:50:00+00:00'
    check(reconstruct(e,days,closes,pd)['available_session']=='2024-11-29','Exact before early close')
    e['publication_at']='2024-11-29T17:50:00'
    try:reconstruct(e,days,closes,pd)
    except ValueError:pass
    else:raise ValueError('Naive exact stamp accepted')
    return 8


def validate(pins):
    check(all(sha(p)==h for p,h in pins.items()),'Source hash mismatch')
    check(all(pins.get(p)==h for p,h in PINS.items()),'Pinned source version mismatch')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run','verify']);parser.add_argument('--spec-sha');args=parser.parse_args()
    if args.phase=='freeze':
        pins=dict(PINS)
        for name in ['SLOT_AUDIT.json','INTERVAL_AUDIT.json','AUDIT.json','REMAINING_ITEMS.json']:
            p='outputs/monthly_event_overlay_scope156/'+name;pins[p]=sha(p)
        closure=read('outputs/monthly_paired_target143/CALENDAR.json')['provenance']['source_hashes']
        for p,h in closure.items():pins[str(local(p).relative_to(ROOT))]=h
        pins['work/after155_consumed_clock_audit.py']=sha(__file__)
        validate(pins)
        write('SPEC.json',dict(stage='FROZEN_CONSUMED_EVENT_CLOCK_AUDIT',runtime=sys.version,hashes=pins,
            created_utc=datetime.now(timezone.utc).isoformat(),scope='All1681 consumed identities after independently reviewed four-event overlay; preserve every original slot.',
            price_reads=0,new_strategy_trials=0,source_date_truth_reauthenticated=False))
        print(json.dumps({'phase':'freeze','spec_sha256':sha(OUT/'SPEC.json')}));return
    check(sha(OUT/'SPEC.json')==args.spec_sha,'Exact spec hash required');spec=read(OUT/'SPEC.json');validate(spec['hashes'])
    check(spec['runtime']==sys.version,'Runtime mismatch')
    import pandas as pd
    cp=local('outputs/monthly_earnings_repair/research_calendar.py')
    ms=importlib.util.spec_from_file_location('independent_calendar156',cp);calmod=importlib.util.module_from_spec(ms);ms.loader.exec_module(calmod)
    frozen=read('outputs/monthly_paired_target143/CALENDAR.json');cal=calmod.market_calendar(frozen['start'],frozen['end'])
    days=[str(d.date()) for d in cal.schedule.index];closes=[x.value for x in cal.schedule['close']]
    check(days==frozen['sessions'],'Frozen session axis mismatch')
    checks=synthetic(pd)
    all_events=read('outputs/monthly_earnings_repair/EVENTS.json')+read('outputs/monthly_event_patch156/ADDED_EVENTS.json')
    event_map={e['event_id']:e for e in all_events};check(len(event_map)==len(all_events)==3775,'Unique event identities')
    slots=read('outputs/monthly_event_overlay_scope156/SLOT_AUDIT.json')
    check(len(slots)==3904,'Original stock grid')
    ids=sorted({eid for s in slots for eid in s['after']['event_ids']});check(len(ids)==1681,'Consumed unique count')
    rows=[]
    for eid in ids:
        e=event_map[eid];expected=reconstruct(e,days,closes,pd)
        actual=dict(available_session=e['available_session'],known_ns=pd.Timestamp(e['known_by_at']).value,
            upper_ns=None if e.get('date_only_upper_bound_at') is None else pd.Timestamp(e['date_only_upper_bound_at']).value)
        rows.append(dict(event_id=eid,precision=e['precision'],matches=expected==actual,expected=expected,recorded=actual))
    failures=[r for r in rows if not r['matches']]
    summary=dict(stage='CONSUMED_CLOCK_INDEPENDENT_RECONSTRUCTION',passed=not failures,spec_sha256=args.spec_sha,
        checked=len(rows),precision_counts=dict(Counter(r['precision'] for r in rows)),mismatches=failures,
        synthetic_checks=checks,source_date_truth_reauthenticated=False,
        calendar_completeness_verified=False,factor_admitted=False,
        limitations=['Reconstructs clocks from existing publication evidence; not an independent authentication of all original publication dates.',
            'Exact instants remain inherited evidence; no first-public time or full-news-calendar claim.',
            'Does not certify interval enumeration or factor readiness.'],
        financial_values_extracted=0,price_values_read=0,factor_calculations=0,model_fits=0,orders=0,returns=0,new_strategy_trials=0)
    validate(spec['hashes'])
    if args.phase=='run':
        write('EVENT_CLOCKS.json',rows);write('SUMMARY.json',summary)
        write('MANIFEST.json',dict(spec_sha256=args.spec_sha,outputs={n:sha(OUT/n) for n in ['EVENT_CLOCKS.json','SUMMARY.json']}))
    else:
        check(rows==read(OUT/'EVENT_CLOCKS.json') and summary==read(OUT/'SUMMARY.json'),'Reproduction failed')
        manifest=read(OUT/'MANIFEST.json');check(manifest['spec_sha256']==args.spec_sha,'Manifest spec mismatch')
        check(all(sha(OUT/n)==h for n,h in manifest['outputs'].items()),'Output hash mismatch')
    print(json.dumps({'phase':args.phase,**summary},ensure_ascii=False))
    check(not failures,'Consumed event clock mismatch; review before any factor use')


if __name__=='__main__':main()
