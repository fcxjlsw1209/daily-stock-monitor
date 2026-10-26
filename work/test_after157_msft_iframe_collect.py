"""Synthetic tests only; never invoke a live HTTP transport."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import after157_msft_iframe_collect as c


class Response(io.BytesIO):
    def __init__(self, body=b'<html>fixture</html>', status=200, headers=None, url=c.URL):
        super().__init__(body)
        self.status, self.headers, self.url = status, headers or {}, url

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.net = patch.object(c.base, 'transport', side_effect=AssertionError('Live network forbidden'))
        self.net.start()
        self.addCleanup(self.net.stop)
        self.put(c.SELF, Path(c.__file__).read_bytes())
        self.put(c.BASE, Path(c.base.__file__).read_bytes())
        self.put(c.PARENT_RAW, ('\n' * 874 + '<iframe src="' + c.URL + '"></iframe>\n').encode())
        self.put(c.PARENT_SCOPE, b'{"fixture":"scope"}\n')
        self.put(c.PARENT_REVIEW, b'{"fixture":"review"}\n')
        self.put(c.PARENT_ATTEMPT, b'{"fixture":"attempt"}\n')
        self.put_json(c.PARENT_RECEIPT, dict(sha256=self.hash(c.PARENT_RAW),
            bytes=len((self.root / c.PARENT_RAW).read_bytes()), http_status=200, outcome='success',
            scope_sha256=self.hash(c.PARENT_SCOPE), helper_sha256=c.BASE_SHA,
            attempt_sha256=self.hash(c.PARENT_ATTEMPT)))
        self.put_json(c.BRIDGE + '/COLLECTION.json', dict(all_successful=True, stopped_providers=[],
            targets=[dict(id='msft2015_sec_filings_page', receipt_sha256=self.hash(c.PARENT_RECEIPT))]))
        names = [c.BASE, c.PARENT_RAW, c.PARENT_RECEIPT, c.PARENT_ATTEMPT,
                 c.PARENT_SCOPE, c.PARENT_REVIEW, c.BRIDGE + '/COLLECTION.json']
        self.scope = dict(schema=c.SCHEMA, url=c.URL, allowed_host=c.HOST, output_dir=c.OUT,
            body_file=c.RAW, receipt_file=c.RECEIPT, max_bytes=c.MAX_BYTES, timeout_seconds=c.TIMEOUT,
            allow_redirects=False, source_admission_changed=False, abort_if_exists=c.STOP_GUARDS,
            helper_sha256=self.hash(c.SELF), input_hashes={n: self.hash(n) for n in names},
            observed_link=dict(raw_path=c.PARENT_RAW, raw_line_1based=875, url=c.URL))
        self.pin()

    def put(self, name, raw):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)

    def put_json(self, name, value):
        self.put(name, (json.dumps(value, sort_keys=True) + '\n').encode())

    def hash(self, name):
        return c.base.sha((self.root / name).read_bytes())

    def pin(self):
        self.put_json(c.SCOPE, self.scope)
        self.scope_sha = self.hash(c.SCOPE)

    def request(self, req, timeout):
        self.calls.append((req, timeout))
        return Response()

    def run_collection(self, request=None):
        return c.collect(self.scope_sha, self.root, request or self.request)

    def read_output(self, name):
        return json.loads((self.root / c.OUT / name).read_bytes())

    def test_success_and_exact_cache_reuse_without_request(self):
        result = self.run_collection()
        before = {p.name: p.read_bytes() for p in (self.root / c.OUT).iterdir()}
        self.assertEqual(result, self.run_collection(lambda *a, **k: self.fail('Retried success')))
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / c.OUT).iterdir()})
        self.assertTrue(result['all_successful'])
        self.assertFalse(result['source_admission_changed'])
        self.assertEqual(len(self.calls), 1)
        req, timeout = self.calls[0]
        self.assertEqual((req.full_url, req.get_method(), timeout), (c.URL, 'GET', 35))
        self.assertNotIn('Authorization', req.headers)
        self.assertNotIn('Cookie', req.headers)
        receipt = self.read_output(c.RECEIPT)
        self.assertEqual(receipt['sha256'], self.hash(c.OUT + '/' + c.RAW))
        self.assertEqual(receipt['attempt_sha256'], self.hash(c.OUT + '/.attempt.json'))
        self.assertEqual(receipt['redirect_requests'], 0)

    def test_all_http_failures_stop_no_retry(self):
        for status in (401, 403, 429, 500, 503):
            with self.subTest(status=status):
                body, result = c.fetch(lambda *a, **k: Response(b'denied', status, {'Retry-After': '3600'}))
                self.assertEqual((body, result['http_status'], result['headers']['Retry-After']),
                                 (b'denied', status, '3600'))
                self.assertTrue(result['stop_provider'])
        result = self.run_collection(lambda *a, **k: Response(b'limited', 429, {'Retry-After': '3600'}))
        self.assertFalse(result['all_successful'])
        stop = self.read_output('STOPPED.json')
        self.assertFalse(stop['further_requests_permitted'])
        self.assertEqual(stop['retry_after'], '3600')
        self.assertEqual(result, self.run_collection(lambda *a, **k: self.fail('Retried failure')))

    def test_no_redirects_even_same_host_or_sec(self):
        for location in ('/financial-information/other', c.URL, 'https://www.sec.gov/x',
                         'https://evil.example/x', 'http://microsoft.gcs-web.com/x'):
            with self.subTest(location=location):
                calls = []
                def request(req, timeout):
                    calls.append(req.full_url)
                    return Response(b'moved', 302, {'Location': location})
                raw, receipt = c.fetch(request)
                self.assertEqual(calls, [c.URL])
                self.assertEqual(receipt['outcome'], 'redirect_not_followed')
                self.assertEqual(receipt['headers']['Location'], location)
                self.assertTrue(receipt['stop_provider'])

    def test_implicit_transport_redirect_rejected_and_reported(self):
        _, result = c.fetch(lambda *a, **k: Response(url='https://www.sec.gov/x'))
        self.assertEqual(result['outcome'], 'request_error')
        self.assertEqual(result['final_url'], 'https://www.sec.gov/x')

    def test_network_error_retained_and_no_retry(self):
        def unavailable(*a, **k):
            raise TimeoutError('synthetic timeout')
        result = self.run_collection(unavailable)
        self.assertFalse(result['all_successful'])
        self.assertEqual(self.read_output(c.RECEIPT)['error_type'], 'TimeoutError')
        self.assertEqual(result, self.run_collection(lambda *a, **k: self.fail('Retried error')))

    def test_interrupted_attempt_forbids_retry(self):
        def interrupted(*a, **k):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_collection(interrupted)
        self.assertTrue((self.root / c.OUT / '.attempt.json').exists())
        self.assertFalse((self.root / c.OUT / '.collection.lock').exists())
        with self.assertRaisesRegex(ValueError, 'retry forbidden'):
            self.run_collection(lambda *a, **k: self.fail('Retried interruption'))

    def test_limits_and_incomplete_bodies_fail_closed(self):
        for headers, body in (({'Content-Length': str(c.MAX_BYTES + 1)}, b'x'),
                              ({'Content-Length': 'bogus'}, b'x'),
                              ({'Content-Length': '3'}, b'xx'), ({}, b'')):
            with self.subTest(headers=headers, body=body):
                _, result = c.fetch(lambda *a, **k: Response(body, headers=headers))
                self.assertTrue(result['stop_provider'])
        with patch.object(c, 'MAX_BYTES', 9):
            raw, result = c.fetch(lambda *a, **k: Response(b'0123456789'))
        self.assertEqual(raw, b'012345678')
        self.assertTrue(result['body_truncated_at_limit'])
        self.assertEqual(result['outcome'], 'request_error')

    def test_bad_exact_scope_fields_fail_before_fetch(self):
        for key, value in [('url', c.URL + '?year=2015'), ('url', 'https://www.sec.gov/x'),
                           ('url', 'https://user:pass@microsoft.gcs-web.com/x'),
                           ('allowed_host', 'www.microsoft.com'), ('output_dir', '../escape'),
                           ('body_file', '../escape'), ('receipt_file', '/tmp/escape'),
                           ('max_bytes', c.MAX_BYTES + 1), ('timeout_seconds', 36),
                           ('allow_redirects', True), ('source_admission_changed', True),
                           ('abort_if_exists', [])]:
            original = copy.deepcopy(self.scope)
            with self.subTest(key=key, value=value):
                self.scope[key] = value
                self.pin()
                with self.assertRaises(ValueError):
                    self.run_collection()
                self.assertEqual(self.calls, [])
            self.scope = original
        self.pin()

    def test_wrong_scope_digest_or_helper_digest_fail(self):
        with self.assertRaisesRegex(ValueError, 'Scope SHA'):
            c.collect('0' * 64, self.root, self.request)
        self.scope['helper_sha256'] = '0' * 64
        self.pin()
        with self.assertRaisesRegex(ValueError, 'Helper SHA'):
            self.run_collection()
        self.assertEqual(self.calls, [])

    def test_input_hash_changed_or_removed_fails(self):
        self.put(c.PARENT_RAW, b'changed')
        with self.assertRaisesRegex(ValueError, 'Input SHA'):
            self.run_collection()
        del self.scope['input_hashes'][c.PARENT_RAW]
        self.pin()
        with self.assertRaisesRegex(ValueError, 'Missing parent'):
            self.run_collection()
        self.assertEqual(self.calls, [])

    def test_parent_receipt_and_collection_chain_fail(self):
        for name, field, value in [(c.PARENT_RECEIPT, 'http_status', 403),
                                  (c.BRIDGE + '/COLLECTION.json', 'all_successful', False)]:
            with self.subTest(name=name):
                original = (self.root / name).read_bytes()
                obj = json.loads(original)
                obj[field] = value
                self.put_json(name, obj)
                self.scope['input_hashes'][name] = self.hash(name)
                self.pin()
                with self.assertRaises(ValueError):
                    self.run_collection()
                self.put(name, original)
                self.scope['input_hashes'][name] = self.hash(name)
                self.pin()
        self.assertEqual(self.calls, [])

    def test_observed_link_must_match_fixed_raw_line(self):
        self.scope['observed_link']['raw_line_1based'] = 874
        self.pin()
        with self.assertRaisesRegex(ValueError, 'Observed-link'):
            self.run_collection()
        self.assertEqual(self.calls, [])

    def test_current_provider_stop_prevents_attempt(self):
        self.put(c.STOP_GUARDS[0], b'{"status":403}')
        with self.assertRaisesRegex(ValueError, 'Existing Microsoft source stop'):
            self.run_collection()
        self.assertFalse((self.root / c.OUT).exists())
        self.assertEqual(self.calls, [])

    def test_symlink_and_traversal_inputs_rejected(self):
        p = self.root / c.PARENT_RAW
        p.unlink()
        p.symlink_to(self.root / c.PARENT_SCOPE)
        with self.assertRaisesRegex(ValueError, 'Symlink forbidden'):
            self.run_collection()
        self.scope['input_hashes']['../outside'] = '0' * 64
        self.pin()
        with self.assertRaises(ValueError):
            self.run_collection()
        self.assertEqual(self.calls, [])

    def test_unowned_output_directory_rejected(self):
        (self.root / c.OUT).mkdir()
        with self.assertRaises(FileNotFoundError):
            self.run_collection()
        self.assertEqual(self.calls, [])

    def test_changed_cached_body_rejected_without_retry(self):
        self.run_collection()
        self.put(c.OUT + '/' + c.RAW, b'changed')
        with self.assertRaisesRegex(ValueError, 'Body SHA'):
            self.run_collection(lambda *a, **k: self.fail('Retried changed body'))

    def test_cached_receipt_tamper_rejected(self):
        self.run_collection()
        receipt = self.read_output(c.RECEIPT)
        receipt['source_url'] = 'https://www.sec.gov/x'
        self.put_json(c.OUT + '/' + c.RECEIPT, receipt)
        with self.assertRaisesRegex(ValueError, 'Receipt binding'):
            self.run_collection(lambda *a, **k: self.fail('Retried altered receipt'))

    def test_missing_cached_body_does_not_refetch(self):
        self.run_collection()
        (self.root / c.OUT / c.RAW).unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_collection(lambda *a, **k: self.fail('Retried missing body'))

    def test_changed_input_during_request_cannot_publish_summary(self):
        def request(req, timeout):
            self.put(c.PARENT_REVIEW, b'changed during request')
            return Response()
        with self.assertRaisesRegex(ValueError, 'Input SHA'):
            self.run_collection(request)
        self.assertFalse((self.root / c.OUT / 'COLLECTION.json').exists())
        self.assertTrue((self.root / c.OUT / c.RECEIPT).exists())

    def test_interrupted_post_receipt_summary_can_recover_without_request(self):
        result = self.run_collection()
        (self.root / c.OUT / 'COLLECTION.json').unlink()
        self.assertEqual(result, self.run_collection(lambda *a, **k: self.fail('Retried complete receipt')))

    def test_failed_receipt_stop_can_recover_without_request(self):
        result = self.run_collection(lambda *a, **k: Response(b'denied', 403))
        (self.root / c.OUT / 'STOPPED.json').unlink()
        self.assertEqual(result, self.run_collection(lambda *a, **k: self.fail('Retried failure')))
        self.assertTrue((self.root / c.OUT / 'STOPPED.json').exists())

    def test_no_environment_or_frozen_host_changes(self):
        before = dict(os.environ)
        self.run_collection()
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(c.base.HOSTS, {'microsoft': 'www.microsoft.com',
                                     'coca_cola': 'investors.coca-colacompany.com'})

    def test_validation_only_has_no_output(self):
        _, sha = c.validate(self.root, self.scope_sha)
        self.assertEqual(sha, self.scope['helper_sha256'])
        self.assertFalse((self.root / c.OUT).exists())
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
