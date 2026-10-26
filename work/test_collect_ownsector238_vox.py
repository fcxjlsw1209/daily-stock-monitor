"""Offline synthetic tests; HTTP and Keychain are always mocked/injected."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

import collect_ownsector238_vox as c

TOKEN = 'synthetic_test_token_not_a_credential'
NOW = datetime(2026, 10, 24, 2, 3, 4, tzinfo=timezone.utc)
METADATA = {'ticker': 'VOX', 'name': 'Vanguard Communication Services Index Fund',
            'exchangeCode': 'NYSE ARCA', 'startDate': '2004-09-23', 'endDate': '2026-10-23'}


def encoded(obj):
    return json.dumps(obj, indent=2).encode()


class Response(io.BytesIO):
    def __init__(self, raw, url, status=200, headers=None):
        super().__init__(raw)
        self.url, self.status = url, status
        self.headers = {'Content-Length': str(len(raw)), 'Content-Type': 'application/json'}
        self.headers.update(headers or {})
    def geturl(self):
        return self.url
    def getcode(self):
        return self.status


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vox-offline-', dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'sources'
        self.digest = c.sha(Path(c.__file__).read_bytes())
        self.keychain = Mock(return_value=TOKEN)
        self.requests = []
        self.responses = []
        # Catch any accidental use of the real network/credential implementations.
        self.real_key = patch.object(c.subprocess, 'run', side_effect=AssertionError('real subprocess forbidden'))
        self.real_http = patch.object(c.urllib.request, 'build_opener', side_effect=AssertionError('real HTTP forbidden'))
        self.real_key.start()
        self.real_http.start()
        self.addCleanup(self.real_key.stop)
        self.addCleanup(self.real_http.stop)

    def respond(self, obj=None, *, phase='metadata', raw=None, status=200, headers=None, url=None):
        body = encoded(METADATA if obj is None else obj) if raw is None else raw
        self.responses.append(Response(body, url or c.descriptor(phase)['request_url'], status, headers))
        return body

    def request(self, req, timeout):
        self.requests.append((req, timeout))
        self.assertEqual(req.get_method(), 'GET')
        self.assertEqual(timeout, 45)
        return self.responses.pop(0)

    def collect(self, phase='metadata', metadata_sha=None, **kwargs):
        return c.collect(phase, expected_sha256=self.digest, metadata_sha256=metadata_sha,
                         output=self.out, keychain=self.keychain, request=self.request,
                         clock=lambda: NOW, **kwargs)

    def json_file(self, name):
        return json.loads((self.out / name).read_text())

    def metadata(self):
        self.respond()
        return self.collect()['raw_sha256']

    def assert_stopped(self, reason, *, phase='metadata', metadata_sha=None):
        with self.assertRaisesRegex(c.AcquisitionError, '^' + reason + '$'):
            self.collect(phase, metadata_sha)
        stop = self.json_file('STOPPED.json')
        self.assertEqual(stop['reason_code'], reason)
        self.assertTrue(stop['permanent_directory_stop'])
        self.assertFalse(stop['further_requests_permitted'])
        self.assertFalse(stop['other_scripts_automatically_blocked'])
        before = (len(self.requests), self.keychain.call_count)
        for later in ('metadata', 'prices'):
            with self.assertRaisesRegex(c.AcquisitionError, 'directory_permanently_stopped'):
                self.collect(later, metadata_sha if later == 'prices' and metadata_sha else ('a' * 64 if later == 'prices' else None))
        self.assertEqual(before, (len(self.requests), self.keychain.call_count))
        self.assertEqual(stop, self.json_file('STOPPED.json'))
        return stop

    def test_metadata_is_one_get_raw_preserved_and_no_prices(self):
        raw = self.respond()
        result = self.collect()
        self.assertEqual(len(self.requests), 1)
        req, _ = self.requests[0]
        self.assertEqual(req.full_url, c.BASE)
        self.assertEqual(req.get_header('Authorization'), 'Token ' + TOKEN)
        self.assertNotIn(TOKEN, req.full_url)
        self.assertEqual((self.out / 'metadata.raw.json').read_bytes(), raw)
        self.assertEqual(result['raw_sha256'], c.sha(raw))
        self.assertFalse(result['source_admitted'])
        self.assertFalse(result['screen']['exchange_listing_certified'])
        self.assertFalse((self.out / 'prices.attempt.json').exists())
        for path in self.out.iterdir():
            self.assertNotIn(TOKEN.encode(), path.read_bytes())

    def test_repeated_metadata_and_prices_only_verify_cache(self):
        metadata_sha = self.metadata()
        self.respond([{'date': '2015-01-02', 'adjClose': 100}], phase='prices')
        prices = self.collect('prices', metadata_sha)
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        self.assertEqual(self.collect()['http_requests_this_invocation'], 0)
        self.assertEqual(self.collect('prices', metadata_sha)['raw_sha256'], prices['raw_sha256'])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.keychain.call_count, 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})
        self.assertEqual(self.requests[1][0].full_url, c.descriptor('prices')['request_url'])
        receipt = self.json_file('prices.receipt.json')
        self.assertEqual(receipt['binding']['parameters'], c.PRICE_PARAMS)
        self.assertEqual(receipt['binding']['operator_metadata_review_sha256'], metadata_sha)

    def test_prerequisite_no_metadata_never_accesses_credentials(self):
        self.assert_stopped('metadata_success_required', phase='prices', metadata_sha='a' * 64)
        self.keychain.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_mismatched_review_sha_stops_before_prices(self):
        self.metadata()
        self.assert_stopped('metadata_review_sha256_mismatch', phase='prices', metadata_sha='b' * 64)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.keychain.call_count, 1)

    def test_metadata_screen_accepts_telecommunication_no_etf(self):
        for name in ('VANGUARD TELECOMMUNICATION SERVICES INDEX FUND', 'Vanguard Communication Services ETF'):
            screen = c.metadata_screen(dict(METADATA, name=name))
            self.assertEqual(screen['name'], name)
            self.assertFalse(screen['source_admitted'])

    def test_metadata_identity_exchange_history_all_hard_fail(self):
        cases = [('ticker', 'vox', 'metadata_ticker_mismatch'),
                 ('name', 'Vanguard Energy ETF', 'metadata_name_mismatch'),
                 ('name', 'Other Communication ETF', 'metadata_name_mismatch'),
                 ('exchangeCode', ' ', 'metadata_exchange_missing'),
                 ('startDate', None, 'metadata_history_date_invalid'),
                 ('startDate', '2015-1-01', 'metadata_history_date_invalid'),
                 ('startDate', '2015-02-30', 'metadata_history_date_invalid'),
                 ('startDate', '2015-01-02', 'metadata_history_insufficient'),
                 ('endDate', '2026-09-23', 'metadata_history_insufficient')]
        for key, value, reason in cases:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(c.AcquisitionError, reason):
                c.metadata_screen(dict(METADATA, **{key: value}))

    def test_invalid_identity_saves_raw_then_permanent_stop(self):
        raw = self.respond(dict(METADATA, ticker='WRONG'))
        stop = self.assert_stopped('metadata_ticker_mismatch')
        self.assertEqual((self.out / 'metadata.raw.json').read_bytes(), raw)
        self.assertEqual(stop['raw_response_sha256'], c.sha(raw))
        self.assertFalse((self.out / 'metadata.receipt.json').exists())

    def test_all_non200_statuses_stop_one_get_and_hide_error_body(self):
        for status in (301, 302, 307, 308, 401, 403, 404, 429, 500, 503):
            with self.subTest(status=status):
                self.out = Path(self.temp.name) / str(status)
                self.respond(raw=TOKEN.encode(), status=status, headers={'Authorization': TOKEN, 'Set-Cookie': TOKEN})
                before = len(self.requests)
                stop = self.assert_stopped('http_status_not_200')
                self.assertEqual(stop['http_status'], status)
                self.assertEqual(len(self.requests), before + 1)
                self.assertEqual(stop['provider_stop_today_required'], status == 429)
                self.assertFalse((self.out / 'metadata.raw.json').exists())
                for path in self.out.iterdir():
                    self.assertNotIn(TOKEN.encode(), path.read_bytes())

    def test_retry_after_seconds_and_both_dates(self):
        self.respond(status=429, headers={'Retry-After': '120'})
        stop = self.assert_stopped('http_status_not_200')
        self.assertEqual(stop['retry_after_raw'], '120')
        self.assertEqual(stop['retry_after_earliest_utc'], '2026-10-24T02:05:04+00:00')
        self.assertEqual(stop['stopped_date_utc'], '2026-10-24')
        self.assertEqual(stop['stopped_date_america_los_angeles'], '2026-10-23')
        self.assertEqual(stop['provider_host'], 'api.tiingo.com')

    def test_retry_after_http_date_invalid_negative_overflow(self):
        self.assertEqual(c.retry_after_time('Sat, 24 Oct 2026 02:06:00 GMT', NOW), '2026-10-24T02:06:00+00:00')
        for raw in (None, 'tomorrow', '-1', '999999999999999999999999'):
            self.assertIsNone(c.retry_after_time(raw, NOW))

    def test_redirect_response_not_followed(self):
        self.respond(url='https://other.example/VOX')
        self.assert_stopped('redirect_forbidden')
        self.assertEqual(len(self.requests), 1)
        self.assertIsNone(c.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.example'))

    def test_network_exception_is_generic_and_stop_durable(self):
        def fail(req, timeout):
            self.requests.append((req, timeout))
            raise OSError('failed Authorization: Token ' + TOKEN)
        self.request = fail
        self.assert_stopped('acquisition_failed')
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn(TOKEN, (self.out / 'STOPPED.json').read_text())
        self.assertTrue((self.out / 'metadata.attempt.json').exists())

    def test_keychain_failure_consumes_attempt_and_never_gets(self):
        self.keychain.side_effect = OSError(TOKEN)
        self.assert_stopped('acquisition_failed')
        self.assertEqual(self.requests, [])
        self.assertTrue((self.out / 'metadata.attempt.json').exists())
        self.assertNotIn(TOKEN, (self.out / 'STOPPED.json').read_text())

    def test_credential_echo_plain_and_json_escaped_not_persisted(self):
        for escaped in (False, True):
            with self.subTest(escaped=escaped):
                self.out = Path(self.temp.name) / ('escaped' if escaped else 'plain')
                raw = encoded(dict(METADATA, extra=TOKEN))
                if escaped:
                    raw = raw.replace(TOKEN.encode(), ''.join('\\u%04x' % ord(ch) for ch in TOKEN).encode())
                self.respond(raw=raw)
                self.assert_stopped('credential_echo_in_body')
                self.assertFalse((self.out / 'metadata.raw.json').exists())
                for path in self.out.iterdir():
                    self.assertNotIn(TOKEN.encode(), path.read_bytes())

    def test_credential_echo_in_safe_header_is_not_recorded(self):
        self.respond(headers={'Retry-After': TOKEN})
        stop = self.assert_stopped('credential_echo_in_headers')
        self.assertEqual(stop['safe_response_headers'], {})
        self.assertNotIn(TOKEN, (self.out / 'STOPPED.json').read_text())

    def test_bad_json_and_duplicate_keys_stop(self):
        for raw, reason in ((b'not json', 'invalid_json'), (b'{"a":1,"a":2}', 'duplicate_json_key'),
                            (b'[NaN]', 'nonstandard_json_constant')):
            with self.subTest(reason=reason):
                self.out = Path(self.temp.name) / reason
                self.respond(raw=raw)
                self.assert_stopped(reason)

    def test_empty_prices_save_response_and_stop(self):
        metadata_sha = self.metadata()
        raw = self.respond([], phase='prices')
        self.assert_stopped('prices_empty_array', phase='prices', metadata_sha=metadata_sha)
        self.assertEqual((self.out / 'prices.raw.json').read_bytes(), raw)
        self.assertFalse((self.out / 'prices.receipt.json').exists())

    def test_prices_require_adjclose_key_only_numeric_and_dates_deferred(self):
        metadata_sha = self.metadata()
        unusual = [{'date': 'wrong', 'adjClose': None}, {'date': 'wrong', 'adjClose': 'not a number'}]
        self.respond(unusual, phase='prices')
        result = self.collect('prices', metadata_sha)
        self.assertEqual(result['screen']['rows_acquired'], 2)
        for field in ('numeric_values_verified', 'dates_unique_verified', 'date_window_verified',
                      'session_coverage_verified', 'source_admitted'):
            self.assertFalse(result['screen'][field])
        for obj, reason in (({}, 'prices_not_array'), ([{'close': 3}], 'prices_adjclose_missing')):
            with self.assertRaisesRegex(c.AcquisitionError, reason):
                c.prices_screen(obj)

    def test_content_length_and_body_limits(self):
        cases = [(b'{}', {'Content-Length': '1'}, 'response_length_mismatch'),
                 (b'{}', {'Content-Length': '-1'}, 'response_length_invalid'),
                 (b'{}', {'Content-Length': str(c.MAX_BYTES + 1)}, 'response_length_invalid'),
                 (b'{}', {'Content-Encoding': 'gzip'}, 'response_encoding_unsupported')]
        for idx, (raw, headers, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                self.out = Path(self.temp.name) / ('length%d' % idx)
                self.respond(raw=raw, headers=headers)
                self.assert_stopped(reason)
        self.out = Path(self.temp.name) / 'streamlimit'
        response = Response(b'12345', c.BASE)
        del response.headers['Content-Length']
        self.responses.append(response)
        # Scope/attempt reads stay below full size; only fetch sees this smaller bound.
        real_fetch = c.fetch
        def small_fetch(*args):
            with patch.object(c, 'MAX_BYTES', 4):
                return real_fetch(*args)
        with patch.object(c, 'fetch', side_effect=small_fetch):
            self.assert_stopped('response_too_large')

    def test_orphan_attempt_stops_without_second_request(self):
        self.out.mkdir()
        c.write_new(self.out / 'SCOPE.json', c.scope(self.digest))
        c.write_new(self.out / 'metadata.attempt.json', {'old': 'incomplete'})
        before = (self.out / 'metadata.attempt.json').read_bytes()
        self.assert_stopped('partial_phase_permanent_stop')
        self.assertEqual(before, (self.out / 'metadata.attempt.json').read_bytes())
        self.keychain.assert_not_called()

    def test_missing_receipt_and_mutated_cache_are_never_refetched(self):
        for kind in ('missing', 'mutated'):
            with self.subTest(kind=kind):
                self.out = Path(self.temp.name) / kind
                self.metadata()
                if kind == 'missing':
                    (self.out / 'metadata.receipt.json').unlink()
                    reason = 'partial_phase_permanent_stop'
                else:
                    (self.out / 'metadata.raw.json').write_bytes(encoded(dict(METADATA, exchangeCode='changed')))
                    reason = 'cache_digest_mismatch'
                before = len(self.requests)
                self.assert_stopped(reason)
                self.assertEqual(len(self.requests), before)

    def test_cache_request_identity_and_scope_are_bound(self):
        for kind in ('receipt', 'attempt', 'scope'):
            with self.subTest(kind=kind):
                self.out = Path(self.temp.name) / kind
                self.metadata()
                name = 'SCOPE.json' if kind == 'scope' else 'metadata.' + kind + '.json'
                obj = self.json_file(name)
                if kind == 'scope':
                    obj['requested_start'] = '2008-01-01'
                    reason = 'scope_mismatch'
                else:
                    obj['binding']['request_url'] = c.BASE + '/other'
                    reason = 'cache_' + kind + '_binding_mismatch'
                (self.out / name).write_bytes(encoded(obj))
                self.assert_stopped(reason)

    def test_stale_or_concurrent_lock_blocks_without_request(self):
        self.out.mkdir()
        (self.out / '.lock').mkdir()
        with self.assertRaisesRegex(c.AcquisitionError, 'directory_locked_no_request'):
            self.collect()
        self.assertTrue((self.out / '.lock').is_dir())
        self.keychain.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_reentrant_phase_cannot_spend_second_attempt(self):
        self.respond()
        real = self.request
        def nested(req, timeout):
            with self.assertRaisesRegex(c.AcquisitionError, 'directory_locked_no_request'):
                self.collect()
            return real(req, timeout)
        self.request = nested
        self.collect()
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.keychain.call_count, 1)

    def test_symlink_and_hardlink_are_rejected(self):
        external = Path(self.temp.name) / 'external'
        external.mkdir()
        self.out.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(c.AcquisitionError, 'symlink_forbidden'):
            self.collect()
        self.assertEqual(list(external.iterdir()), [])
        self.out.unlink()
        self.metadata()
        path = self.out / 'metadata.raw.json'
        backup = Path(self.temp.name) / 'saved.raw'
        path.rename(backup)
        path.symlink_to(backup)
        self.assert_stopped('acquisition_failed')
        self.assertEqual(json.loads(backup.read_text()), METADATA)

    def test_exclusive_write_never_overwrites_existing(self):
        p = Path(self.temp.name) / 'existing'
        p.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            c.write_new(p, b'replacement')
        self.assertEqual(p.read_bytes(), b'keep')

    def test_wrong_script_hash_and_missing_review_hash_no_directory(self):
        with self.assertRaisesRegex(c.AcquisitionError, 'collector_sha256_mismatch'):
            c.collect('metadata', expected_sha256='a' * 64, output=self.out)
        with self.assertRaisesRegex(c.AcquisitionError, 'metadata_review_sha256_required'):
            self.collect('prices')
        self.assertFalse(self.out.exists())
        self.keychain.assert_not_called()

    def test_unowned_existing_directory_is_not_adopted(self):
        self.out.mkdir()
        (self.out / 'unrelated').write_text('preserve')
        self.assert_stopped('acquisition_failed')
        self.assertEqual((self.out / 'unrelated').read_text(), 'preserve')
        self.keychain.assert_not_called()

    def test_http_transport_has_no_redirect_proxy_or_retry(self):
        fake = Mock()
        response = Response(encoded(METADATA), c.BASE)
        fake.open.return_value = response
        with patch.object(c.urllib.request, 'build_opener', return_value=fake) as builder:
            req = urllib.request.Request(c.BASE)
            self.assertIs(c.request_once(req, 45), response)
            fake.open.assert_called_once_with(req, timeout=45)
            handlers = builder.call_args.args
            self.assertIsInstance(handlers[0], urllib.request.ProxyHandler)
            self.assertEqual(handlers[0].proxies, {})
            self.assertIsInstance(handlers[1], c.NoRedirect)
            err = urllib.error.HTTPError(c.BASE, 429, 'limited', {}, io.BytesIO(b'error'))
            fake.open.side_effect = err
            self.assertIs(c.request_once(req, 45), err)
            self.assertEqual(fake.open.call_count, 2)

    def test_keychain_uses_existing_service_and_captured_output(self):
        with patch.object(c.subprocess, 'run', return_value=Mock(returncode=0, stdout=TOKEN + '\n')) as run:
            self.assertEqual(c.read_keychain(), TOKEN)
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[0], '/usr/bin/security')
            self.assertIn('daily-stock-monitor.tiingo', cmd)
            self.assertTrue(run.call_args.kwargs['capture_output'])
            self.assertNotIn(TOKEN, str(run.call_args))
        with patch.object(c.subprocess, 'run', return_value=Mock(returncode=1, stdout=TOKEN)):
            with self.assertRaisesRegex(c.AcquisitionError, 'credential_unavailable'):
                c.read_keychain()


if __name__ == '__main__':
    unittest.main(verbosity=2)
