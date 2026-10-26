"""Versioned VOX full-range collection, explicitly authorized separately from research.

One prices GET maximum; metadata is a pinned cached verification, never a request.
The CLI fixes the repository/output/symbol/dates. Dependency injection is solely
for offline tests. No strategy, price joining, source admission or PIT claim.
"""
import argparse
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import urllib.request
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
HELPER_PATH=ROOT/'work/collect_ownsector238_vox.py'
HELPER_SHA='3d885637e5cf848827550080f1e8906a3a1664aaa1f0f390df63b4881a08bf10'
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()!=HELPER_SHA:
    raise RuntimeError('frozen_helper_sha256_mismatch')
_spec=importlib.util.spec_from_file_location('vox2014_frozen238',HELPER_PATH)
h=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(h)
AcquisitionError=h.AcquisitionError
require=h.require
SCHEMA='residual246_vox2014_acquisition_v1'
REVIEW_SCHEMA='residual246_vox2014_source_authorization_v1'
OLD_DIR='outputs/monthly_ownsector238_vox_sources'
OUT_REL='outputs/monthly_residual246_vox2014_sources'
OLD_STOP=OLD_DIR+'/STOPPED.json'
HOST='api.tiingo.com'
SYMBOL='VOX'
START,END='2014-06-18','2026-09-24'
METADATA_PINS={
 'metadata.raw.json':'087a0d420a47c2f1ade00e224b20e78e9b3df077d9ef07f5e2eace12450059c1',
 'metadata.receipt.json':'24c5e8cdb601318bb602cf1f0b7f63f055d30ff088910bb9af3f5c794ac45a88',
 'metadata.attempt.json':'85e496e326c4d8f96d3f44ce5e3f1092ef3b1ba12cfa7b0db9b86653ffe5a942'}
FLAGS=('metadata_exchange_reviewed','known_provider_stop_scope_reviewed',
       'local_research_no_concurrent_collection','local_research_exclusive_execution_confirmed')


def descriptor():
    params={'startDate':START,'endDate':END,'format':'json','resampleFreq':'daily'}
    endpoint='https://'+HOST+'/tiingo/daily/'+SYMBOL+'/prices'
    return dict(phase='prices',symbol=SYMBOL,endpoint=endpoint,parameters=params,
        request_url=endpoint+'?'+h.urllib.parse.urlencode(params),method='GET',
        redirects_permitted=False,retries_permitted=False,maximum_gets=1)


def local_path(root,relative):
    require(isinstance(relative,str) and bool(relative),'invalid_local_path')
    p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts,'invalid_local_path')
    target=root/p
    h.safe_directory(target.parent)
    require(not target.is_symlink(),'symlink_forbidden')
    return target


def metadata_screen(obj):
    screen=h.metadata_screen(obj)  # fixed old schema/identity, then stricter earlier date
    require(obj['startDate']<=START and obj['endDate']>=END,'metadata_history_insufficient')
    return screen


def cached_metadata(root):
    directory=root/OLD_DIR
    h.safe_directory(directory)
    for name,digest in METADATA_PINS.items():
        require(h.sha(h.read_bytes(directory/name))==digest,'metadata_cache_sha256_mismatch')
    result=h.verify_cache(directory,'metadata',HELPER_SHA,None)
    screen=metadata_screen(h.parse_json(h.read_bytes(directory/'metadata.raw.json')))
    require(result['raw_sha256']==METADATA_PINS['metadata.raw.json'] and
            result['receipt_sha256']==METADATA_PINS['metadata.receipt.json'],
            'metadata_cache_binding_mismatch')
    return dict(directory=OLD_DIR,file_sha256=dict(METADATA_PINS),screen=screen,
        metadata_http_requests_this_invocation=0,metadata_cached_verification=True,
        price_rows_or_coverage_certified=False,source_admitted=False)


def _aware(value):
    require(isinstance(value,str),'invalid_review_clock')
    try:
        dt=datetime.fromisoformat(value)
    except ValueError:
        raise AcquisitionError('invalid_review_clock') from None
    require(dt.tzinfo is not None,'invalid_review_clock')
    return dt.astimezone(timezone.utc)


def check_review(root,relative,digest,now,request_planned):
    require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None,
            'source_review_sha256_required')
    raw=h.read_bytes(local_path(root,relative));require(h.sha(raw)==digest,'source_review_sha256_mismatch')
    review=h.parse_json(raw)
    require(isinstance(review,dict),'source_review_invalid')
    expected=dict(schema=REVIEW_SCHEMA,provider_host=HOST,symbol=SYMBOL,
                  requested_start=START,requested_end=END,metadata_file_sha256=METADATA_PINS)
    require(all(review.get(k)==v for k,v in expected.items()),'source_review_scope_mismatch')
    require(all(review.get(k) is True for k in FLAGS),'source_review_confirmation_missing')
    require(review.get('external_account_activity_unknown') is True,'external_account_scope_misrepresented')
    require(review.get('collector_sha256')==h.sha(h.read_bytes(Path(__file__).resolve())) and
            review.get('frozen_helper_sha256')==HELPER_SHA,'source_review_code_binding_mismatch')
    preflight=review.get('source_preflight')
    require(isinstance(preflight,dict) and set(preflight)=={'path','sha256'},'preflight_binding_missing')
    preflight_raw=h.read_bytes(local_path(root,preflight['path']))
    require(h.sha(preflight_raw)==preflight['sha256'],'preflight_sha256_mismatch')
    checked=_aware(review.get('reviewed_at_utc'))
    require(checked<=now,'source_review_from_future')
    utcday=checked.date().isoformat();laday=checked.astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat()
    require(review.get('review_date_utc')==utcday and review.get('review_date_america_los_angeles')==laday,
            'source_review_dates_mismatch')
    if request_planned:
        require(utcday==now.astimezone(timezone.utc).date().isoformat() and
                laday==now.astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat(),
                'source_review_not_today')
    records=review.get('known_stop_files')
    require(isinstance(records,list) and bool(records),'known_stop_inventory_missing')
    seen=set()
    for item in records:
        require(isinstance(item,dict) and set(item)=={'path','sha256'},'known_stop_record_invalid')
        name=item['path'];path=local_path(root,name)
        require(name not in seen,'duplicate_known_stop');seen.add(name)
        exists=h.exists(path)
        if item['sha256'] is None:
            require(not exists,'known_stop_appeared')
            continue
        require(isinstance(item['sha256'],str) and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,
                'known_stop_sha256_invalid')
        require(exists,'known_stop_missing')
        stopraw=h.read_bytes(path);require(h.sha(stopraw)==item['sha256'],'known_stop_changed')
        stop=h.parse_json(stopraw)
        require(isinstance(stop,dict) and stop.get('provider_host')==HOST,'known_stop_unrecognized')
        # The original acquisition directory is conservatively never bypassed.
        require(name!=OLD_STOP,'original_directory_stopped')
        for field in ('stopped_date_utc','stopped_date_america_los_angeles'):
            require(isinstance(stop.get(field),str),'known_stop_date_missing')
            try: date.fromisoformat(stop[field])
            except ValueError:raise AcquisitionError('known_stop_date_invalid') from None
        same_day=(stop['stopped_date_utc']==now.astimezone(timezone.utc).date().isoformat() or
                  stop['stopped_date_america_los_angeles']==now.astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat())
        require(not (same_day and (stop.get('http_status')==429 or stop.get('provider_stop_today_required') is True)),
                'known_provider_stop_today')
        if stop.get('retry_after_earliest_utc') is not None:
            require(_aware(stop['retry_after_earliest_utc'])<=now,'known_provider_retry_after_active')
    require(OLD_STOP in seen,'original_stop_check_missing')
    return dict(path=relative,sha256=digest,known_stop_paths=sorted(seen),source_preflight=preflight,
                review_is_operator_attestation=True,external_account_activity_unknown=True,
                shared_global_stop_mechanism=False)


def binding(script_sha,review_ref):
    return dict(descriptor(),collector_sha256=script_sha,frozen_helper_sha256=HELPER_SHA,
                source_review=review_ref,metadata_file_sha256=METADATA_PINS)


def scope(fixed):
    return dict(schema=SCHEMA,binding=fixed,provider_host=HOST,source_admitted=False,
        metadata_request_permitted=False,price_splicing_permitted=False,
        shared_stop_ledger_modified=False,other_scripts_automatically_blocked=False)


def stopped(out,script_sha,reason,clock,details):
    now=clock();headers=details.get('safe_response_headers',{});retry=headers.get('Retry-After')
    record=dict(schema=SCHEMA,collector_sha256=script_sha,frozen_helper_sha256=HELPER_SHA,
        provider_host=HOST,phase='prices',reason_code=reason,stopped_utc=h.stamp(now),
        stopped_date_utc=now.astimezone(timezone.utc).date().isoformat(),
        stopped_date_america_los_angeles=now.astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat(),
        http_status=details.get('http_status'),safe_response_headers=headers,retry_after_raw=retry,
        retry_after_earliest_utc=h.retry_after_time(retry,now),permanent_directory_stop=True,
        further_requests_permitted=False,provider_stop_today_required=details.get('http_status')==429,
        provider_stop_scope='Tiingo provider/account for recorded UTC and America/Los_Angeles day; external operator coordination',
        other_scripts_automatically_blocked=False,shared_stop_ledger_modified=False,
        request=descriptor(),raw_response_sha256=details.get('raw_response_sha256'),source_admitted=False)
    if not h.exists(out/'STOPPED.json'):h.write_new(out/'STOPPED.json',record)


def verify_cache(out,fixed,metadata):
    ar=h.read_bytes(out/'prices.attempt.json');a=h.parse_json(ar)
    raw=h.read_bytes(out/'prices.raw.json');rr=h.read_bytes(out/'prices.receipt.json');r=h.parse_json(rr)
    require(isinstance(a,dict) and a.get('binding')==fixed and a.get('attempt_budget_consumed') is True,
            'cache_attempt_binding_mismatch')
    require(isinstance(r,dict) and r.get('binding')==fixed and r.get('attempt_sha256')==h.sha(ar)
            and r.get('raw_sha256')==h.sha(raw) and r.get('raw_bytes')==len(raw),'cache_digest_mismatch')
    screen=h.prices_screen(h.parse_json(raw))
    require(r.get('screen')==screen and r.get('metadata_cached_verification')==metadata and
            r.get('outcome')=='ACQUIRED_NOT_ADMITTED' and r.get('http_status')==200 and
            r.get('request_count')==1 and r.get('source_admitted') is False,'cache_receipt_invalid')
    return dict(outcome='CACHE_VERIFIED_NOT_ADMITTED',raw_sha256=h.sha(raw),receipt_sha256=h.sha(rr),
                http_requests_this_invocation=0,metadata_http_requests_this_invocation=0,screen=screen,source_admitted=False)


def collect(*,expected_sha256,source_review,source_review_sha256,repository=ROOT,
            keychain=None,request=None,clock=None):
    """CLI fixes repository; injectable repository/HTTP/clock/credential are offline test seams."""
    script_sha=h.sha(h.read_bytes(Path(__file__).resolve()))
    require(expected_sha256==script_sha,'collector_sha256_mismatch')
    root=Path(repository).absolute();out=h.safe_directory(root/OUT_REL)
    keychain,request,clock=keychain or h.read_keychain,request or h.request_once,clock or h.utc
    now=clock();h.stamp(now)
    created=False
    try:out.mkdir(mode=0o700);created=True
    except FileExistsError:require(out.is_dir(),'output_not_directory')
    lock=out/'.lock'
    try:lock.mkdir(mode=0o700)
    except FileExistsError:raise AcquisitionError('directory_locked_no_request') from None
    details={};secret=None
    try:
        require(not h.exists(out/'STOPPED.json'),'directory_permanently_stopped')
        allowed={'SCOPE.json','.lock','prices.attempt.json','prices.raw.json','prices.receipt.json'}
        require({p.name for p in out.iterdir()}<=allowed,'unexpected_directory_member')
        present=[h.exists(out/('prices'+suffix)) for suffix in ('.attempt.json','.raw.json','.receipt.json')]
        require(not any(present) or all(present),'partial_phase_permanent_stop')
        review_ref=check_review(root,source_review,source_review_sha256,now,not all(present))
        fixed=binding(script_sha,review_ref)
        if created:h.write_new(out/'SCOPE.json',scope(fixed))
        require(h.parse_json(h.read_bytes(out/'SCOPE.json'))==scope(fixed),'scope_mismatch')
        metadata=cached_metadata(root)
        if all(present):return verify_cache(out,fixed,metadata)
        started=h.stamp(now)
        attempt_sha=h.write_new(out/'prices.attempt.json',dict(schema=SCHEMA,binding=fixed,
            attempt_started_utc=started,attempt_budget_consumed=True,http_gets_maximum=1))
        # Check again immediately before credentials/request to catch an appeared known STOP.
        check_review(root,source_review,source_review_sha256,clock(),True)
        try:secret=keychain()
        except Exception:raise AcquisitionError('credential_unavailable') from None
        require(isinstance(secret,str) and re.fullmatch(r'[A-Za-z0-9_.-]{16,512}',secret) is not None,
                'credential_format_invalid')
        req=urllib.request.Request(fixed['request_url'],method='GET',headers={
            'Authorization':'Token '+secret,'Accept':'application/json','Accept-Encoding':'identity',
            'User-Agent':'LocalBoundedSourceResearch/246-VOX2014-v1'})
        check_review(root,source_review,source_review_sha256,clock(),True)
        raw,obj=h.fetch(req,request,secret,details)
        details['raw_response_sha256']=h.write_new(out/'prices.raw.json',raw)
        screen=h.prices_screen(obj)
        receipt=dict(schema=SCHEMA,binding=fixed,attempt_sha256=attempt_sha,
            request_started_utc=started,request_finished_utc=h.stamp(clock()),outcome='ACQUIRED_NOT_ADMITTED',
            http_status=200,safe_response_headers=details['safe_response_headers'],raw_sha256=h.sha(raw),
            raw_bytes=len(raw),request_count=1,metadata_http_requests_this_invocation=0,
            metadata_cached_verification=metadata,redirects_followed=0,screen=screen,source_admitted=False,
            strict_pit_certified=False,credential_persisted=False,features_computed=False,
            orders_computed=False,returns_computed=False,old_price_cache_read=False,price_splicing_performed=False)
        digest=h.write_new(out/'prices.receipt.json',receipt)
        return dict(outcome=receipt['outcome'],raw_sha256=h.sha(raw),receipt_sha256=digest,
            http_requests_this_invocation=1,metadata_http_requests_this_invocation=0,screen=screen,source_admitted=False)
    except BaseException as exc:
        # Expose only our fixed safe reason codes, never exception/request/credential text.
        text=str(exc)
        reason=(text if isinstance(exc,AcquisitionError) and re.fullmatch(r'[a-z][a-z0-9_]{0,80}',text)
                and (secret is None or secret not in text) else 'acquisition_failed')
        try:stopped(out,script_sha,reason,clock,details)
        except BaseException:raise AcquisitionError('stop_persistence_failed_directory_unusable') from None
        raise AcquisitionError(reason) from None
    finally:lock.rmdir()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prices',));p.add_argument('--script-sha256',required=True)
    p.add_argument('--source-review',required=True);p.add_argument('--source-review-sha256',required=True)
    args=p.parse_args(argv)
    try:result=collect(expected_sha256=args.script_sha256,source_review=args.source_review,
                       source_review_sha256=args.source_review_sha256)
    except AcquisitionError as exc:
        print(json.dumps(dict(outcome='STOPPED_OR_BLOCKED',reason_code=str(exc)),sort_keys=True));return 2
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
