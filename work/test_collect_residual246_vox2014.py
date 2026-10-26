"""Offline fake HTTP/Keychain tests. Only pinned old metadata bytes are copied.
No price cache, network, Keychain, frozen-module monkeypatch or research execution.
"""
from datetime import datetime,timezone,timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import collect_residual246_vox2014 as c

NOW=datetime(2026,10,25,2,3,4,tzinfo=timezone.utc)
TOKEN='synthetic_token_only_not_a_credential'

def encode(x):return json.dumps(x,sort_keys=True).encode()

class Response(io.BytesIO):
    def __init__(self,body,url,status=200,headers=None):
        super().__init__(body);self.url=url;self.status=status
        self.headers={'Content-Length':str(len(body)),'Content-Type':'application/json'}
        self.headers.update(headers or {})
    def geturl(self):return self.url
    def getcode(self):return self.status

class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='vox2014-offline-',dir='/private/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'outputs').mkdir();(self.root/'work').mkdir()
        old=self.root/c.OLD_DIR;old.mkdir()
        for name in c.METADATA_PINS:(old/name).write_bytes((c.ROOT/c.OLD_DIR/name).read_bytes())
        self.out=self.root/c.OUT_REL;self.digest=c.h.sha(Path(c.__file__).read_bytes())
        preflight=self.root/'work/preflight.json';preflight.write_bytes(encode({'synthetic_source_only':True}))
        self.review={
            'schema':c.REVIEW_SCHEMA,'provider_host':c.HOST,'symbol':c.SYMBOL,
            'requested_start':c.START,'requested_end':c.END,'metadata_file_sha256':dict(c.METADATA_PINS),
            'collector_sha256':self.digest,'frozen_helper_sha256':c.HELPER_SHA,
            'source_preflight':{'path':'work/preflight.json','sha256':c.h.sha(preflight.read_bytes())},
            'reviewed_at_utc':NOW.isoformat(),'review_date_utc':'2026-10-25',
            'review_date_america_los_angeles':'2026-10-24','external_account_activity_unknown':True,
            'known_stop_files':[{'path':c.OLD_STOP,'sha256':None}],**{k:True for k in c.FLAGS}}
        self.save_review();self.key=Mock(return_value=TOKEN);self.requests=[];self.responses=[]
        self.clock=lambda:NOW
    def save_review(self):
        self.review_path=self.root/'work/authorization.json';self.review_path.write_bytes(encode(self.review))
        self.review_sha=c.h.sha(self.review_path.read_bytes())
    def respond(self,obj=None,**kw):
        raw=kw.pop('raw',None)
        if raw is None:raw=encode([{'date':'2014-06-18','adjClose':100.}] if obj is None else obj)
        self.responses.append(Response(raw,kw.pop('url',c.descriptor()['request_url']),**kw));return raw
    def request(self,req,timeout):
        self.requests.append((req,timeout));return self.responses.pop(0)
    def collect(self,**kw):
        options=dict(expected_sha256=self.digest,source_review='work/authorization.json',
            source_review_sha256=self.review_sha,repository=self.root,keychain=self.key,
            request=self.request,clock=self.clock);options.update(kw);return c.collect(**options)
    def file(self,name):return json.loads((self.out/name).read_text())
    def stopped(self,reason):
        with self.assertRaisesRegex(c.AcquisitionError,'^'+reason+'$'):self.collect()
        stop=self.file('STOPPED.json');self.assertEqual(stop['reason_code'],reason)
        count=(len(self.requests),self.key.call_count)
        with self.assertRaisesRegex(c.AcquisitionError,'directory_permanently_stopped'):self.collect()
        self.assertEqual(count,(len(self.requests),self.key.call_count));self.assertEqual(stop,self.file('STOPPED.json'))
        self.assertTrue(stop['permanent_directory_stop']);self.assertFalse(stop['other_scripts_automatically_blocked'])
        return stop
    def add_stop(self,name='outputs/other_STOP.json',**kw):
        obj={'provider_host':c.HOST,'http_status':429,'provider_stop_today_required':True,
            'stopped_date_utc':'2026-10-25','stopped_date_america_los_angeles':'2026-10-24'};obj.update(kw)
        p=self.root/name;p.write_bytes(encode(obj));self.review['known_stop_files'].append({'path':name,'sha256':c.h.sha(p.read_bytes())});self.save_review()
        return p
    def test_fixed_full_range_single_GET_cached_metadata_only(self):
        raw=self.respond();r=self.collect();self.assertEqual(len(self.requests),1);self.key.assert_called_once()
        req,timeout=self.requests[0];self.assertEqual(req.get_method(),'GET');self.assertEqual(timeout,45)
        self.assertEqual(req.full_url,'https://api.tiingo.com/tiingo/daily/VOX/prices?startDate=2014-06-18&endDate=2026-09-24&format=json&resampleFreq=daily')
        self.assertEqual(req.get_header('Authorization'),'Token '+TOKEN)
        self.assertEqual((self.out/'prices.raw.json').read_bytes(),raw);self.assertFalse(r['source_admitted'])
        receipt=self.file('prices.receipt.json');self.assertEqual(receipt['metadata_http_requests_this_invocation'],0)
        self.assertEqual(receipt['metadata_cached_verification']['file_sha256'],c.METADATA_PINS)
        self.assertFalse(receipt['old_price_cache_read']);self.assertFalse(receipt['price_splicing_performed'])
        self.assertFalse(receipt['strict_pit_certified']);self.assertFalse((self.out/'metadata.raw.json').exists())
        for p in self.out.iterdir():self.assertNotIn(TOKEN.encode(),p.read_bytes())
    def test_repeat_success_is_zero_GET_and_preserves_bytes(self):
        self.respond();self.collect();before={p.name:p.read_bytes() for p in self.out.iterdir()}
        self.clock=lambda:NOW+timedelta(days=1)
        r=self.collect();self.assertEqual(r['http_requests_this_invocation'],0);self.assertEqual(len(self.requests),1)
        self.key.assert_called_once();self.assertEqual(before,{p.name:p.read_bytes() for p in self.out.iterdir()})
    def test_durable_attempt_precedes_keychain_and_request(self):
        def key():
            self.assertTrue((self.out/'prices.attempt.json').exists());return TOKEN
        self.key=key
        def req(r,t):
            self.assertTrue((self.out/'.lock').exists())
            with self.assertRaisesRegex(c.AcquisitionError,'directory_locked_no_request'):self.collect()
            return self.request(r,t)
        self.respond();self.collect(request=req);self.assertEqual(len(self.requests),1)
    def test_partial_attempt_blocks_without_second_GET(self):
        self.out.mkdir();(self.out/'prices.attempt.json').write_bytes(b'{}')
        self.stopped('partial_phase_permanent_stop');self.key.assert_not_called();self.assertEqual(self.requests,[])
    def test_any_partial_raw_or_receipt_blocks(self):
        for name in ('prices.raw.json','prices.receipt.json'):
            with self.subTest(name=name):
                # Each case gets an isolated directory; never remove a stopped real directory.
                self.setUp();self.out.mkdir();(self.out/name).write_bytes(b'{}');self.stopped('partial_phase_permanent_stop')
    def test_cached_tampering_is_permanent_stop(self):
        self.respond();self.collect();(self.out/'prices.raw.json').write_bytes(b'[]')
        self.stopped('cache_digest_mismatch');self.assertEqual(len(self.requests),1)
    def test_success_cache_cannot_bypass_local_stop(self):
        self.respond();self.collect();(self.out/'STOPPED.json').write_bytes(encode({'reason_code':'manual_stop'}))
        with self.assertRaisesRegex(c.AcquisitionError,'directory_permanently_stopped'):self.collect()
        self.assertEqual(len(self.requests),1)
    def test_metadata_pin_tampering_never_requests(self):
        (self.root/c.OLD_DIR/'metadata.raw.json').write_bytes(b'{}');self.stopped('metadata_cache_sha256_mismatch')
        self.key.assert_not_called()
    def test_metadata_stricter_earlier_range(self):
        obj=c.h.parse_json((c.ROOT/c.OLD_DIR/'metadata.raw.json').read_bytes());obj['startDate']='2014-07-01'
        with self.assertRaisesRegex(c.AcquisitionError,'metadata_history_insufficient'):c.metadata_screen(obj)
    def test_review_wrong_digest_prevents_request(self):
        self.review_sha='0'*64;self.stopped('source_review_sha256_mismatch');self.key.assert_not_called()
    def test_stale_review_and_external_account_false_refused(self):
        self.clock=lambda:NOW+timedelta(days=1);self.stopped('source_review_not_today')
        self.setUp();self.review['external_account_activity_unknown']=False;self.save_review()
        self.stopped('external_account_scope_misrepresented')
    def test_review_flags_and_code_pins_are_binding(self):
        self.review[c.FLAGS[1]]=False;self.save_review();self.stopped('source_review_confirmation_missing')
        self.setUp();self.review['frozen_helper_sha256']='0'*64;self.save_review();self.stopped('source_review_code_binding_mismatch')
    def test_preflight_digest_is_binding(self):
        (self.root/'work/preflight.json').write_bytes(b'{}');self.stopped('preflight_sha256_mismatch')
    def test_original_stop_inventory_mandatory(self):
        self.review['known_stop_files']=[{'path':'outputs/other_STOP.json','sha256':None}];self.save_review()
        self.stopped('original_stop_check_missing')
    def test_original_or_other_stop_appeared_blocks(self):
        (self.root/c.OLD_STOP).write_bytes(b'{}');self.stopped('known_stop_appeared');self.key.assert_not_called()
    def test_current_provider_429_stops_across_UTC_LA_day(self):
        self.add_stop(stopped_date_utc='2026-10-24')
        self.stopped('known_provider_stop_today');self.key.assert_not_called()
    def test_old_stop_retry_after_still_active(self):
        self.add_stop(stopped_date_utc='2026-10-23',stopped_date_america_los_angeles='2026-10-23',
                      retry_after_earliest_utc=(NOW+timedelta(hours=1)).isoformat())
        self.stopped('known_provider_retry_after_active')
    def test_expired_other_provider_stop_does_not_claim_global_clearance(self):
        self.add_stop(stopped_date_utc='2026-10-23',stopped_date_america_los_angeles='2026-10-23')
        self.respond();self.collect();ref=self.file('prices.receipt.json')['binding']['source_review']
        self.assertTrue(ref['external_account_activity_unknown']);self.assertFalse(ref['shared_global_stop_mechanism'])
    def test_stop_appears_during_credential_read_blocks_GET(self):
        def key():
            (self.root/c.OLD_STOP).write_bytes(b'{}');return TOKEN
        with self.assertRaisesRegex(c.AcquisitionError,'known_stop_appeared'):self.collect(keychain=key)
        self.assertEqual(self.requests,[]);self.assertTrue((self.out/'prices.attempt.json').exists())
    def test_http_failures_permanent_and_429_retryafter(self):
        for code in (301,401,403,429,500):
            with self.subTest(code=code):
                self.setUp();self.respond(status=code,headers={'Retry-After':'120','Set-Cookie':TOKEN})
                stop=self.stopped('http_status_not_200');self.assertEqual(stop['http_status'],code)
                self.assertEqual(stop['retry_after_raw'],'120');self.assertEqual(stop['retry_after_earliest_utc'],(NOW+timedelta(seconds=120)).isoformat())
                self.assertEqual(stop['provider_stop_today_required'],code==429);self.assertNotIn('Set-Cookie',stop['safe_response_headers'])
                self.assertEqual(len(self.requests),1)
    def test_redirect_url_rejected(self):
        self.respond(url='https://other.example/prices');self.stopped('redirect_forbidden')
    def test_empty_array_saved_then_stopped_not_admitted(self):
        raw=self.respond([]);self.stopped('prices_empty_array');self.assertEqual((self.out/'prices.raw.json').read_bytes(),raw)
        self.assertFalse((self.out/'prices.receipt.json').exists())
    def test_missing_adjclose_stopped_but_numeric_check_deferred(self):
        self.respond([{'date':'x'}]);self.stopped('prices_adjclose_missing')
        self.setUp();self.respond([{'adjClose':'not numeric','date':'x'}]);r=self.collect()
        self.assertFalse(r['screen']['numeric_values_verified']);self.assertFalse(r['screen']['date_window_verified'])
    def test_partial_body_transport_failure_no_retry_no_secret(self):
        response=Response(b'abc',c.descriptor()['request_url'])
        def broken_read(_):raise RuntimeError(TOKEN)
        response.read=broken_read;self.responses.append(response);self.stopped('acquisition_failed')
        self.assertEqual(len(self.requests),1);self.assertFalse((self.out/'prices.raw.json').exists())
        self.assertNotIn(TOKEN,(self.out/'STOPPED.json').read_text())
    def test_content_length_mismatch_is_permanent(self):
        self.respond(headers={'Content-Length':'999'});self.stopped('response_length_mismatch')
    def test_credential_body_header_and_exception_never_saved(self):
        for mode in ('body','header','key'):
            with self.subTest(mode=mode):
                self.setUp()
                if mode=='body':self.respond([{'adjClose':1,'token':TOKEN}]);reason='credential_echo_in_body'
                elif mode=='header':self.respond(headers={'ETag':TOKEN});reason='credential_echo_in_headers'
                else:self.key=Mock(side_effect=c.AcquisitionError(TOKEN));reason='credential_unavailable'
                self.stopped(reason)
                for p in self.out.iterdir():self.assertNotIn(TOKEN.encode(),p.read_bytes())
    def test_unsafe_path_and_symlink_blocked(self):
        self.review['known_stop_files'].append({'path':'../outside','sha256':None});self.save_review();self.stopped('invalid_local_path')
        self.setUp();(self.root/'work/preflight.json').unlink();(self.root/'work/preflight.json').symlink_to(self.review_path)
        self.stopped('symlink_forbidden')
    def test_script_pin_and_output_lock_no_request(self):
        with self.assertRaisesRegex(c.AcquisitionError,'collector_sha256_mismatch'):self.collect(expected_sha256='0'*64)
        self.assertFalse(self.out.exists());self.out.mkdir();(self.out/'.lock').mkdir()
        with self.assertRaisesRegex(c.AcquisitionError,'directory_locked_no_request'):self.collect()
        self.key.assert_not_called();self.assertEqual(self.requests,[])
    def test_cache_scope_and_request_dates_cannot_change(self):
        self.respond();self.collect();scope=self.file('SCOPE.json');scope['binding']['parameters']['startDate']='2015-01-01'
        (self.out/'SCOPE.json').write_bytes(encode(scope));self.stopped('scope_mismatch');self.assertEqual(len(self.requests),1)

if __name__=='__main__':unittest.main(verbosity=2)
