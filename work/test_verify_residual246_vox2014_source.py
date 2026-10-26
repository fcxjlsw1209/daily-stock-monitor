"""Fake, offline inputs only: never loads any historical or newly collected raw cache."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
import verify_residual246_vox2014_source as v

DAYS = ['2014-06-18','2014-06-19','2015-01-02','2015-01-05','2026-09-24']
AUTH = 'work/fake_vox2014_AUTHORIZATION.json'
PREFLIGHT = 'work/fake_vox2014_SOURCE_PREFLIGHT.json'
NOW = '2026-10-24T15:30:00+00:00'


def enc(x):
    return json.dumps(x, sort_keys=True, allow_nan=False).encode()


def row(day, close=100.0):
    return dict(date=day + 'T00:00:00.000Z', open=close-1, high=close+2, low=close-2, close=close,
                adjOpen=(close-1)/2, adjHigh=(close+2)/2, adjLow=(close-2)/2, adjClose=close/2,
                volume=100, adjVolume=200, divCash=0.0, splitFactor=1.0)


def fixture(new_mutate=None, old_mutate=None, metadata_mutate=None, calendar=None, metadata_finish=None, preflight_mutate=None):
    """Build a byte-exact fake chain independently of collector implementation."""
    rows = [row(d, 100.0+i) for i, d in enumerate(DAYS)]
    old = copy.deepcopy(rows[2:])
    if new_mutate: new_mutate(rows)
    if old_mutate: old_mutate(old)
    meta = dict(ticker='VOX', name='Vanguard Communication Services ETF', exchangeCode='NYSE',
                startDate='2004-09-29', endDate='2026-10-23')
    if metadata_mutate: metadata_mutate(meta)
    blobs = {v.HELPER: b'fake frozen collector, never imported', v.COLLECTOR: b'fake new collector, never imported',
             v.CALENDAR: enc({'sessions': DAYS if calendar is None else calendar}),
             PREFLIGHT: enc({'schema':'residual246_vox2014_source_preflight_v1', 'source_only':True, 'strict_pit':False,
                             'source_admitted':False,'strategy_registered':False})}
    if preflight_mutate:
        p=json.loads(blobs[PREFLIGHT]);preflight_mutate(p);blobs[PREFLIGHT]=enc(p)
    def put(p, x):
        blobs[p] = enc(x)
        return v.digest(blobs[p])
    def dh(p): return v.digest(blobs[p])
    def desc(phase, start):
        endpoint='https://api.tiingo.com/tiingo/daily/VOX' + ('/prices' if phase=='prices' else '')
        params={} if phase=='metadata' else {'startDate':start,'endDate':v.END,'format':'json','resampleFreq':'daily'}
        url=endpoint if not params else endpoint+'?startDate='+start+'&endDate='+v.END+'&format=json&resampleFreq=daily'
        return dict(phase=phase,symbol='VOX',endpoint=endpoint,parameters=params,request_url=url,method='GET',
                    redirects_permitted=False,retries_permitted=False,maximum_gets=1)
    def ps(rs):
        return dict(rows_acquired=len(rs),adjClose_key_present_in_each_row=True,numeric_values_verified=False,
                    dates_unique_verified=False,date_window_verified=False,session_coverage_verified=False,
                    adjustment_vintage_verified=False,source_admitted=False)
    ms=dict(ticker=meta['ticker'],name=meta['name'],exchangeCode=meta['exchangeCode'],startDate=meta['startDate'],
            endDate=meta['endDate'],identity_history_screen_passed=True,exchange_listing_certified=False,
            history_rows_verified=False,source_admitted=False)
    def acquire(directory,phase,data,binding,schema,screen):
        raw=directory+'/'+phase+'.raw.json'; attempt=directory+'/'+phase+'.attempt.json'
        put(raw,data)
        minute=28 if phase=='metadata' else (29 if directory==v.OLD else 30)
        begin='2026-10-24T15:'+str(minute)+':00+00:00'
        finish='2026-10-24T15:'+str(minute)+':01+00:00'
        if phase=='metadata' and metadata_finish is not None:finish=metadata_finish
        a=dict(schema=schema,binding=binding,attempt_started_utc=begin,attempt_budget_consumed=True,http_gets_maximum=1)
        put(attempt,a)
        r=dict(schema=schema,binding=binding,attempt_sha256=dh(attempt),request_started_utc=begin,
               request_finished_utc=finish,outcome='ACQUIRED_NOT_ADMITTED',http_status=200,
               safe_response_headers={'Content-Type':'application/json','Content-Length':str(len(blobs[raw]))},
               raw_sha256=dh(raw),raw_bytes=len(blobs[raw]),request_count=1,redirects_followed=0,screen=screen,
               source_admitted=False,credential_persisted=False,features_computed=False,orders_computed=False,returns_computed=False)
        put(directory+'/'+phase+'.receipt.json',r)
        return r
    mb=dict(desc('metadata',v.OLD_START),collector_sha256=dh(v.HELPER),operator_metadata_review_sha256=None)
    acquire(v.OLD,'metadata',meta,mb,v.OLD_SCHEMA,ms)
    ob=dict(desc('prices',v.OLD_START),collector_sha256=dh(v.HELPER),operator_metadata_review_sha256=dh(v.OLD+'/metadata.raw.json'))
    acquire(v.OLD,'prices',old,ob,v.OLD_SCHEMA,ps(old))
    mp={p:dh(v.OLD+'/'+p) for p in ('metadata.raw.json','metadata.receipt.json','metadata.attempt.json')}
    preflight=dict(path=PREFLIGHT,sha256=dh(PREFLIGHT))
    auth=dict(schema='residual246_vox2014_source_authorization_v1',provider_host='api.tiingo.com',symbol='VOX',
        requested_start=v.START,requested_end=v.END,metadata_file_sha256=mp,metadata_exchange_reviewed=True,
        known_provider_stop_scope_reviewed=True,local_research_no_concurrent_collection=True,local_research_exclusive_execution_confirmed=True,
        external_account_activity_unknown=True,collector_sha256=dh(v.COLLECTOR),frozen_helper_sha256=dh(v.HELPER),
        source_preflight=preflight,reviewed_at_utc=NOW,review_date_utc='2026-10-24',review_date_america_los_angeles='2026-10-24',
        known_stop_files=[{'path':v.OLD+'/STOPPED.json','sha256':None}])
    put(AUTH,auth)
    ref=dict(path=AUTH,sha256=dh(AUTH),known_stop_paths=[v.OLD+'/STOPPED.json'],source_preflight=preflight,
             review_is_operator_attestation=True,external_account_activity_unknown=True,shared_global_stop_mechanism=False)
    nb=dict(desc('prices',v.START),collector_sha256=dh(v.COLLECTOR),frozen_helper_sha256=dh(v.HELPER),
            source_review=ref,metadata_file_sha256=mp)
    r=acquire(v.NEW,'prices',rows,nb,v.NEW_SCHEMA,ps(rows))
    r.update(metadata_http_requests_this_invocation=0,
        metadata_cached_verification=dict(directory=v.OLD,file_sha256=mp,screen=ms,metadata_http_requests_this_invocation=0,
            metadata_cached_verification=True,price_rows_or_coverage_certified=False,source_admitted=False),
        strict_pit_certified=False,old_price_cache_read=False,price_splicing_performed=False)
    put(v.NEW+'/prices.receipt.json',r)
    put(v.NEW+'/SCOPE.json',dict(schema=v.NEW_SCHEMA,binding=nb,provider_host='api.tiingo.com',source_admitted=False,
        metadata_request_permitted=False,price_splicing_permitted=False,shared_stop_ledger_modified=False,
        other_scripts_automatically_blocked=False))
    manifest=dict(schema='residual246_vox2014_verification_inputs_v1',authorization_path=AUTH,source_preflight_path=PREFLIGHT,
                  input_sha256={p:dh(p) for p in blobs})
    return blobs,manifest


def changed(args,path,fn):
    b,m=copy.deepcopy(args)
    obj=json.loads(b[path]);fn(obj);b[path]=enc(obj);m['input_sha256'][path]=v.digest(b[path])
    return b,m


class SourceVerificationTests(unittest.TestCase):
    def test_valid_complete_bundle_and_scope(self):
        out=v.verify_source(*fixture())
        self.assertTrue(out['passed'])
        self.assertEqual(out['new_source']['rows'],5)
        self.assertEqual(out['old_source']['rows'],3)
        self.assertEqual(out['newly_covered_sessions'],DAYS[:2])
        self.assertEqual(set(out['flags']),set(v.FLAGS))
        self.assertTrue(all(out['flags'].values()))
        for k in ('source_admitted','strategy_admitted','strict_pit','currency_certified','strategy_features_computed',
                  'orders_computed','strategy_returns_computed','sources_spliced','missing_rows_filled'):
            self.assertIs(out[k],False)
        self.assertEqual(out['strategy_trials'],0)
        self.assertIsNone(out['currency'])

    def test_no_input_mutation(self):
        args=fixture();before=copy.deepcopy(args);v.verify_source(*args);self.assertEqual(args,before)

    def test_missing_new_or_old_official_day_fails(self):
        for kwargs in ({'new_mutate':lambda r:r.pop(2)}, {'old_mutate':lambda r:r.pop(1)}):
            with self.subTest(kwargs=tuple(kwargs)),self.assertRaisesRegex(ValueError,'coverage'):
                v.verify_source(*fixture(**kwargs))

    def test_duplicates_and_unsorted_dates_fail(self):
        for mutate in (lambda r:r.insert(1,copy.deepcopy(r[0])),lambda r:r.reverse()):
            with self.subTest(),self.assertRaisesRegex(ValueError,'unique_ordered'):
                v.verify_source(*fixture(new_mutate=mutate))

    def test_invalid_date_and_timezone_fail(self):
        for value in ('2014-06-18T01:00:00Z','2014-06-18T00:00:00-07:00','2014-06-18','2014-06-18T00:00:00.000001Z'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                v.verify_source(*fixture(new_mutate=lambda r:r[0].update(date=value)))

    def test_wrong_requested_day_fails(self):
        with self.assertRaisesRegex(ValueError,'outside_request'):
            v.verify_source(*fixture(new_mutate=lambda r:r[0].update(date='2014-06-17T00:00:00Z')))

    def test_calendar_bad_axis_or_scope_fails(self):
        for cal in ([],DAYS[1:],DAYS[:-1],list(reversed(DAYS)),DAYS+['2026-09-26']):
            with self.subTest(cal=cal),self.assertRaises(ValueError): v.verify_source(*fixture(calendar=cal))

    def test_prices_types_positive_and_finite(self):
        for value in (True,'99',None,0,-1,10**400):
            with self.subTest(value=str(value)[:20]),self.assertRaises(ValueError):
                v.verify_source(*fixture(new_mutate=lambda r:r[0].update(open=value)))
        for value in (float('inf'),float('-inf'),float('nan')):
            rs=[row(d) for d in DAYS];rs[0]['open']=value
            with self.subTest(value=value),self.assertRaises(ValueError):v.inspect_rows(rs,DAYS,v.START)

    def test_both_raw_and_adjusted_extrema(self):
        for field,value in (('high',98),('low',101),('adjHigh',49),('adjLow',51)):
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'extrema'):
                v.verify_source(*fixture(new_mutate=lambda r:r[0].update({field:value})))

    def test_adjustment_factor_unit_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError,'factor_inconsistent'):
            v.verify_source(*fixture(new_mutate=lambda r:r[0].update(adjOpen=r[0]['adjOpen']*(1+1e-7))))
        out=v.verify_source(*fixture(new_mutate=lambda r:r[0].update(adjOpen=r[0]['adjOpen']*(1+1e-9))))
        self.assertGreater(out['new_source']['maximum_within_row_factor_relative_spread'],0)
        self.assertEqual(out['new_source']['adjustment_factor_relative_tolerance'],1e-8)

    def test_volume_and_corporate_action_domains(self):
        for field,value in (('volume',-1),('volume',1.5),('volume',True),('adjVolume',-1),('divCash',-0.1),
                            ('divCash','0'),('splitFactor',0),('splitFactor',False)):
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                v.verify_source(*fixture(new_mutate=lambda r:r[0].update({field:value})))

    def test_dividend_split_fraction_and_zero_volume_recorded(self):
        out=v.verify_source(*fixture(new_mutate=lambda r:r[0].update(divCash=.5,splitFactor=.2,volume=0,adjVolume=0)))
        self.assertEqual(out['new_source']['dividend_dates'],[DAYS[0]])
        self.assertEqual(out['new_source']['split_dates'],[DAYS[0]])
        self.assertEqual(out['new_source']['zero_raw_volume_dates'],[DAYS[0]])
        self.assertFalse(out['complete_corporate_action_adjustment_certified'])

    def test_equal_overlap_counts_all_fields_and_returns(self):
        o=v.verify_source(*fixture())['overlap']
        self.assertEqual(o['compared_dates'],DAYS[2:])
        self.assertFalse(o['difference_review_required'])
        self.assertEqual(set(o['fields']),set(v.FIELDS)|{'adjustment_factor'})
        self.assertEqual(o['source_daily_return_consistency']['close']['observations'],2)
        self.assertEqual(o['source_daily_return_consistency']['adjClose']['changed_count'],0)

    def test_vintage_constant_adjustment_is_diagnostic_not_strategy(self):
        def mutate(rs):
            for r in rs:
                for k in ('adjOpen','adjHigh','adjLow','adjClose'):r[k]*=2
        o=v.verify_source(*fixture(new_mutate=mutate))['overlap']
        self.assertTrue(o['difference_review_required'])
        self.assertEqual(o['fields']['close']['changed_count'],0)
        self.assertEqual(o['fields']['adjClose']['changed_count'],3)
        self.assertEqual(o['adjustment_factor_new_over_old_min'],2)
        self.assertEqual(o['source_daily_return_consistency']['adjClose']['changed_count'],0)

    def test_nonuniform_price_revisions_show_daily_difference(self):
        def mutate(rs):
            for k in v.PRICE_FIELDS:rs[3][k]*=1.1
            rs[3]['volume']=101
        o=v.verify_source(*fixture(new_mutate=mutate))['overlap']
        self.assertEqual(o['fields']['close']['changed_dates'],[DAYS[3]])
        self.assertEqual(o['source_daily_return_consistency']['close']['changed_count'],2)
        self.assertEqual(o['fields']['volume']['changed_count'],1)

    def test_zero_old_difference_relative_stat_explicit(self):
        o=v.verify_source(*fixture(new_mutate=lambda r:r[3].update(divCash=.25)))['overlap']['fields']['divCash']
        self.assertEqual(o['changed_count'],1);self.assertEqual(o['zero_old_count'],3)
        self.assertEqual(o['maximum_absolute_difference'],.25)
        self.assertEqual(o['maximum_relative_difference_nonzero_old'],0)

    def test_metadata_wrong_security_exchange_or_history(self):
        for key,value in (('ticker','SPY'),('name','Other Fund'),('exchangeCode','UNKNOWN'),('startDate','2015-01-01'),('endDate','2026-08-31')):
            with self.subTest(key=key),self.assertRaises(ValueError):
                v.verify_source(*fixture(metadata_mutate=lambda m:m.update({key:value})))

    def test_row_explicit_ticker_rejects_substitute(self):
        with self.assertRaisesRegex(ValueError,'row_security'):
            v.verify_source(*fixture(new_mutate=lambda r:r[0].update(ticker='XLC')))

    def test_no_currency_certificate_even_metadata_field(self):
        o=v.verify_source(*fixture(metadata_mutate=lambda m:m.update(currency='USD')))
        self.assertEqual(o['currency'],'USD');self.assertFalse(o['currency_certified'])

    def test_raw_or_manifest_digest_tampering(self):
        args=fixture();args[0][v.NEW+'/prices.raw.json']+=b' '
        with self.assertRaisesRegex(ValueError,'digest'):v.verify_source(*args)
        args=fixture();args[1]['input_sha256'][v.HELPER]='0'*64
        with self.assertRaisesRegex(ValueError,'digest'):v.verify_source(*args)

    def test_single_get_receipt_types_and_scope(self):
        for field,value in (('request_count',2),('request_count',True),('redirects_followed',1),('http_status',201),
                            ('source_admitted',True),('features_computed',True),('strict_pit_certified',True),
                            ('old_price_cache_read',True),('metadata_http_requests_this_invocation',1)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                v.verify_source(*changed(fixture(),v.NEW+'/prices.receipt.json',lambda r:r.update({field:value})))

    def test_request_binding_dates_host_and_method(self):
        for field,value in (('method','POST'),('endpoint','https://other.example/VOX'),('maximum_gets',2),('retries_permitted',True)):
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'binding'):
                v.verify_source(*changed(fixture(),v.NEW+'/prices.receipt.json',lambda r:r['binding'].update({field:value})))
        with self.assertRaisesRegex(ValueError,'binding'):
            v.verify_source(*changed(fixture(),v.NEW+'/prices.receipt.json',lambda r:r['binding']['parameters'].update(startDate='2015-01-01')))

    def test_receipt_raw_size_digest_and_attempt_chain(self):
        for field,value in (('raw_sha256','0'*64),('attempt_sha256','0'*64),('raw_bytes',1)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                v.verify_source(*changed(fixture(),v.NEW+'/prices.receipt.json',lambda r:r.update({field:value})))

    def test_attempt_and_scope_tampering(self):
        with self.assertRaises(ValueError):
            v.verify_source(*changed(fixture(),v.NEW+'/prices.attempt.json',lambda a:a.update(attempt_budget_consumed=False)))
        with self.assertRaisesRegex(ValueError,'scope'):
            v.verify_source(*changed(fixture(),v.NEW+'/SCOPE.json',lambda a:a.update(metadata_request_permitted=True)))

    def test_authorization_and_preflight_tampering(self):
        for path,fn in ((AUTH,lambda a:a.update(external_account_activity_unknown=False)),
                        (AUTH,lambda a:a.update(collector_sha256='0'*64)),
                        (AUTH,lambda a:a['source_preflight'].update(sha256='0'*64)),
                        (PREFLIGHT,lambda a:a.update(changed=True))):
            with self.subTest(path=path),self.assertRaises(ValueError):v.verify_source(*changed(fixture(),path,fn))

    def test_cached_metadata_attestation_and_screen(self):
        for fn in (lambda r:r['metadata_cached_verification'].update(price_rows_or_coverage_certified=True),
                   lambda r:r['screen'].update(numeric_values_verified=True)):
            with self.subTest(),self.assertRaises(ValueError):
                v.verify_source(*changed(fixture(),v.NEW+'/prices.receipt.json',fn))

    def test_cached_metadata_must_precede_price_collection(self):
        with self.assertRaisesRegex(ValueError,'metadata_not_prior'):
            v.verify_source(*fixture(metadata_finish='2026-10-24T15:40:00+00:00'))
        for key,value in (('source_only',False),('source_admitted',True),('strategy_registered',True),('strict_pit',True)):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'preflight_scope'):
                v.verify_source(*fixture(preflight_mutate=lambda p:p.update({key:value})))

    def test_time_chain_and_unsafe_headers(self):
        for fn in (lambda r:r.update(request_finished_utc='2026-10-23T15:30:00+00:00'),
                   lambda r:r['safe_response_headers'].update(Authorization='not-a-secret'),
                   lambda r:r['safe_response_headers'].update({'Content-Length':'1'}),
                   lambda r:r['safe_response_headers'].update({'Content-Type':'text/html'})):
            with self.subTest(),self.assertRaises(ValueError):
                v.verify_source(*changed(fixture(),v.NEW+'/prices.receipt.json',fn))

    def test_manifest_rejects_extra_inputs_and_path_escape(self):
        args=fixture();args[0]['work/unrelated.json']=b'{}';args[1]['input_sha256']['work/unrelated.json']=v.digest(b'{}')
        with self.assertRaisesRegex(ValueError,'input_set'):v.verify_source(*args)
        args=fixture();args[1]['authorization_path']='work/../outputs/unsafe.json'
        with self.assertRaisesRegex(ValueError,'evidence_path'):v.verify_source(*args)

    def test_strict_json_duplicate_and_nonfinite(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',b'\xff'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):v.parse(raw)

    def test_cli_refuses_fake_historical_pins_without_loading_rows(self):
        _,manifest=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'work').mkdir();raw=enc(manifest);(root/'work/manifest.json').write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'frozen_historical_pin'):
                v.run_offline('work/manifest.json',v.digest(raw),'work/residual246_vox2014_SOURCE_AUDIT.json',root=root)
            self.assertEqual(sorted(p.name for p in (root/'work').iterdir()),['manifest.json'])

    def test_symlink_and_absolute_read_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'link').symlink_to(root/'missing')
            for p in ('/etc/passwd','../outside','link'):
                with self.subTest(p=p),self.assertRaises(ValueError):v.safe_path(root,p)

    def test_missing_row_field_and_derived_failure(self):
        with self.assertRaisesRegex(ValueError,'row_schema'):
            v.verify_source(*fixture(new_mutate=lambda r:r[0].pop('splitFactor')))
        rows=[row(d) for d in DAYS]
        for k in ('open','high','low','close'):rows[0][k]=5e-324
        for k in ('adjOpen','adjHigh','adjLow','adjClose'):rows[0][k]=1e308
        with self.assertRaisesRegex(ValueError,'derived'):v.inspect_rows(rows,DAYS,v.START)


if __name__=='__main__':
    unittest.main(verbosity=2)
