"""Offline synthetic collector regression tests; no HTTP transport is used."""
import copy
from email.message import Message
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import after157_annual_body_collect as c


class Response(io.BytesIO):
    def __init__(self, url, data=b'original bytes', status=200, headers=None):
        super().__init__(data)
        self.url, self.status = url, status
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = str(value)

    def geturl(self):
        return self.url

    def getcode(self):
        return self.status


class Transport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request.full_url, timeout, dict(request.header_items())))
        if not self.replies:
            raise AssertionError('Unexpected request')
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'outputs').mkdir()
        (self.root / 'work').mkdir()
        (self.root / 'work/input.json').write_bytes(b'{"pinned":true}\n')
        self.scope = dict(schema=c.SCHEMA, helper_sha256=c.sha(Path(c.__file__).read_bytes()),
                          output_dir='outputs/new_collection', max_bytes=c.MAX_BYTES,
                          timeout_seconds=35,
                          input_hashes={'work/input.json': c.sha(b'{"pinned":true}\n')},
                          targets=[self.target('one')])
        self.out = self.root / self.scope['output_dir']
        # Even a regression falling through to the real transport cannot make HTTP.
        self.no_real = patch.object(c, 'transport', side_effect=AssertionError('No real transport'))
        self.no_real.start()
        self.addCleanup(self.no_real.stop)

    def target(self, key, provider='microsoft'):
        return dict(id=key, provider=provider, url='https://' + c.HOSTS[provider] + '/' + key,
                    body_file=key + '.bin', receipt_file=key + '.receipt.json')

    def save(self):
        raw = json.dumps(self.scope, sort_keys=True).encode()
        (self.root / 'work/scope.json').write_bytes(raw)
        return c.sha(raw)

    def run_collect(self, replies=()):
        fake = Transport(replies)
        result = c.collect('work/scope.json', self.save(), root=self.root, request=fake)
        return result, fake

    def receipt(self, key='one'):
        return json.loads((self.out / (key + '.receipt.json')).read_bytes())

    def test_success_exact_bytes_receipt_and_readonly_reuse(self):
        url = self.scope['targets'][0]['url']
        data = b'\x00exact\xff\r\n'
        result, fake = self.run_collect([Response(url, data, headers={'Content-Length': len(data)})])
        self.assertTrue(result['all_successful'])
        self.assertEqual((self.out / 'one.bin').read_bytes(), data)
        rec = self.receipt()
        self.assertEqual(rec['sha256'], c.sha(data))
        self.assertEqual((rec['source_url'], rec['final_url'], rec['http_status']), (url, url, 200))
        self.assertTrue(rec['retrieved_utc'].endswith('+00:00'))
        self.assertEqual(fake.calls[0][1], 35)
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        second, fake2 = self.run_collect()
        self.assertEqual(result, second)
        self.assertEqual(fake2.calls, [])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_reject_scope_and_helper_and_input_hash_mismatch_before_io(self):
        good = self.save()
        with self.assertRaisesRegex(ValueError, 'Scope SHA'):
            c.collect('work/scope.json', '0' * 64, root=self.root, request=Transport([]))
        for field in ('helper_sha256', 'input_hashes'):
            old = copy.deepcopy(self.scope[field])
            self.scope[field] = '0' * 64 if field == 'helper_sha256' else {'work/input.json': '0' * 64}
            with self.assertRaises(ValueError):
                self.run_collect()
            self.scope[field] = old
        self.assertFalse(self.out.exists())

    def test_reject_scope_traversal_absolute_and_symlink(self):
        self.save()
        for path in ('../scope.json', '/tmp/scope.json', 'work/../work/scope.json', 'work//scope.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                c.collect(path, self.save(), root=self.root, request=Transport([]))
        (self.root / 'work/link.json').symlink_to(self.root / 'work/scope.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            c.collect('work/link.json', self.save(), root=self.root, request=Transport([]))

    def test_reject_output_traversal_monitor_and_existing_unowned_directory(self):
        for dest in ('../escape', 'outputs/a/../b', 'outputs/frozen_monitor', 'work/output', 'outputs/a/b'):
            self.scope['output_dir'] = dest
            with self.subTest(dest=dest), self.assertRaises(ValueError):
                self.run_collect()
        self.scope['output_dir'] = 'outputs/new_collection'
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, 'not this pinned'):
            self.run_collect()

    def test_reject_target_names_duplicate_ids_urls_and_files(self):
        original = copy.deepcopy(self.scope)
        mutations = [lambda s: s['targets'][0].update(body_file='../escape'),
                     lambda s: s['targets'][0].update(receipt_file='/tmp/escape'),
                     lambda s: s['targets'][0].update(body_file='COLLECTION.json'),
                     lambda s: s['targets'][0].update(body_file='STOPPED_microsoft.json'),
                     lambda s: s['targets'][0].update(id='../escape'),
                     lambda s: s['targets'].append(copy.deepcopy(s['targets'][0])),
                     lambda s: s['targets'].append(dict(self.target('two'), body_file='one.bin')),
                     lambda s: s['targets'].append(dict(self.target('two'), url=s['targets'][0]['url']))]
        for mutate in mutations:
            self.scope = copy.deepcopy(original)
            mutate(self.scope)
            with self.assertRaises(ValueError):
                self.run_collect()

    def test_reject_hosts_credentials_downgrade_fragments_and_port(self):
        for url in ('https://www.sec.gov/Archives/a', 'https://www.microsoft.com.evil.test/x',
                    'http://www.microsoft.com/x', 'https://user:pass@www.microsoft.com/x',
                    'https://www.microsoft.com:444/x', 'https://www.microsoft.com/x#frag',
                    'https://www.microsoft.com/\nx', 'https://investors.coca-colacompany.com/x'):
            self.scope['targets'][0]['url'] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.run_collect()

    def test_fixed_limits_and_eight_target_ceiling(self):
        for key, bad in [('max_bytes', c.MAX_BYTES + 1), ('timeout_seconds', 36),
                         ('allowed_hosts', ['www.sec.gov'])]:
            original = copy.deepcopy(self.scope)
            self.scope[key] = bad
            with self.assertRaises(ValueError):
                self.run_collect()
            self.scope = original
        self.scope['targets'] = [self.target('x' + str(n)) for n in range(9)]
        with self.assertRaisesRegex(ValueError, 'eight'):
            self.run_collect()

    def test_allowed_relative_redirect_is_recorded_before_success(self):
        url = self.scope['targets'][0]['url']
        result, fake = self.run_collect([Response(url, status=302, headers={'Location': '/final'}),
                                        Response('https://www.microsoft.com/final', b'linked')])
        self.assertTrue(result['all_successful'])
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(self.receipt()['redirect_chain'][0]['http_status'], 302)
        self.assertEqual(self.receipt()['final_url'], 'https://www.microsoft.com/final')

    def test_unsafe_redirect_not_requested_and_provider_stopped(self):
        for destination in ('https://www.sec.gov/x', 'http://www.microsoft.com/x',
                            'https://investors.coca-colacompany.com/x'):
            with self.subTest(destination=destination):
                target = self.target('one')
                fake = Transport([Response(target['url'], status=302, headers={'Location': destination})])
                body, rec = c.fetch(target, fake)
                self.assertEqual(len(fake.calls), 1)
                self.assertTrue(rec['stop_provider'])
                self.assertEqual(rec['outcome'], 'request_error')

    def test_implicit_redirect_transport_fails_closed(self):
        result, fake = self.run_collect([Response('https://www.sec.gov/x')])
        self.assertFalse(result['all_successful'])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(self.receipt()['outcome'], 'request_error')

    def test_redirect_loop_no_repeat_request(self):
        target = self.target('one')
        fake = Transport([Response(target['url'], status=302, headers={'Location': target['url']})])
        _, rec = c.fetch(target, fake)
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(rec['stop_provider'])

    def test_redirect_count_is_bounded(self):
        target = self.target('one')
        urls = [target['url']] + ['https://www.microsoft.com/hop' + str(n) for n in range(6)]
        fake = Transport([Response(urls[n], status=302, headers={'Location': urls[n + 1]})
                          for n in range(6)])
        _, rec = c.fetch(target, fake)
        self.assertEqual(len(fake.calls), 6)
        self.assertEqual(len(rec['redirect_chain']), 5)
        self.assertTrue(rec['stop_provider'])

    def test_real_transport_configuration_disables_proxy_and_automatic_redirect(self):
        self.no_real.stop()
        try:
            with patch.object(c.urllib.request, 'build_opener') as build:
                c.transport()
                handlers = build.call_args.args
                self.assertEqual(handlers[0].proxies, {})
                self.assertIsInstance(handlers[1], c.NoRedirect)
                self.assertIsNone(handlers[1].redirect_request(None, None, 302, '', {}, 'https://www.sec.gov'))
                build.return_value.open.assert_not_called()
        finally:
            self.no_real.start()

    def test_environment_is_unchanged(self):
        before = dict(os.environ)
        self.run_collect([Response(self.target('one')['url'])])
        self.assertEqual(dict(os.environ), before)

    def test_provider_failure_stops_only_that_provider_and_preserves_retry_after(self):
        self.scope['targets'] += [self.target('two'), self.target('three', 'coca_cola')]
        one, _, three = self.scope['targets']
        result, fake = self.run_collect([Response(one['url'], b'denied', 429, {'Retry-After': '3600'}),
                                        Response(three['url'], b'other issuer')])
        self.assertEqual([x[0] for x in fake.calls], [one['url'], three['url']])
        self.assertEqual(self.receipt('two')['outcome'], 'skipped_provider_stopped')
        self.assertFalse((self.out / 'two.bin').exists())
        stop = json.loads((self.out / 'STOPPED_microsoft.json').read_bytes())
        self.assertEqual(stop['retry_after'], '3600')
        self.assertEqual(self.receipt()['headers']['Retry-After'], '3600')
        again, retry = self.run_collect()
        self.assertEqual(again, result)
        self.assertFalse(retry.calls)

    def test_all_required_http_failure_codes_stop(self):
        for status in (401, 403, 429, 500, 503, 599):
            target = self.target('one')
            fake = Transport([Response(target['url'], b'error', status)])
            _, rec = c.fetch(target, fake)
            self.assertTrue(rec['stop_provider'], status)

    def test_404_failure_is_never_retried_but_next_target_can_run(self):
        self.scope['targets'].append(self.target('two'))
        one, two = self.scope['targets']
        result, fake = self.run_collect([Response(one['url'], b'not found', 404), Response(two['url'])])
        self.assertEqual(len(fake.calls), 2)
        self.assertFalse(result['all_successful'])
        self.assertFalse(self.receipt()['stop_provider'])
        _, repeated = self.run_collect()
        self.assertFalse(repeated.calls)

    def test_network_failure_receipt_and_stop_survive(self):
        self.scope['targets'].append(self.target('two'))
        result, fake = self.run_collect([TimeoutError('synthetic timeout')])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(self.receipt()['error_type'], 'TimeoutError')
        self.assertEqual((self.out / 'one.bin').read_bytes(), b'')
        self.assertTrue((self.out / 'STOPPED_microsoft.json').exists())
        self.assertEqual(self.receipt('two')['outcome'], 'skipped_provider_stopped')

    def test_oversize_declared_and_streamed_and_exact_limit(self):
        with patch.object(c, 'MAX_BYTES', 8):
            for data, headers, success in [(b'123456789', {}, False),
                                           (b'x', {'Content-Length': '9'}, False),
                                           (b'12345678', {}, True)]:
                target = self.target('one')
                body, rec = c.fetch(target, Transport([Response(target['url'], data, headers=headers)]))
                self.assertLessEqual(len(body), 8)
                self.assertEqual(rec['outcome'] == 'success', success)

    def test_truncated_empty_and_partial_response_not_success(self):
        target = self.target('one')
        for reply in (Response(target['url'], b'short', headers={'Content-Length': '10'}),
                      Response(target['url'], b''), Response(target['url'], b'part', 206)):
            _, rec = c.fetch(target, Transport([reply]))
            self.assertNotEqual(rec['outcome'], 'success')
            self.assertTrue(rec['stop_provider'])

    def test_body_without_receipt_fails_without_overwrite(self):
        self.run_collect([Response(self.target('one')['url'])])
        (self.out / 'one.receipt.json').unlink()
        before = (self.out / 'one.bin').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Body without receipt'):
            self.run_collect()
        self.assertEqual(before, (self.out / 'one.bin').read_bytes())

    def test_changed_body_and_receipt_binding_rejected(self):
        self.run_collect([Response(self.target('one')['url'])])
        body = self.out / 'one.bin'
        original = body.read_bytes()
        body.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            self.run_collect()
        body.write_bytes(original)
        rec = self.receipt()
        rec['source_url'] += 'changed'
        (self.out / 'one.receipt.json').write_text(json.dumps(rec))
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            self.run_collect()

    def test_successful_receipt_without_body_cannot_refetch(self):
        self.run_collect([Response(self.target('one')['url'])])
        (self.out / 'one.bin').unlink()
        with self.assertRaisesRegex(ValueError, 'body missing'):
            self.run_collect()

    def test_changed_scope_never_adopts_existing_output(self):
        self.run_collect([Response(self.target('one')['url'])])
        self.scope['note'] = 'different scope bytes'
        with self.assertRaisesRegex(ValueError, 'not this pinned'):
            self.run_collect()

    def test_output_symlinks_rejected_and_lock_exclusive(self):
        self.run_collect([Response(self.target('one')['url'])])
        body = self.out / 'one.bin'
        body.unlink()
        body.symlink_to(self.root / 'work/input.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.run_collect()
        body.unlink()
        (self.out / '.collection.lock').write_text('other process')
        with self.assertRaises(FileExistsError):
            self.run_collect()
        self.assertEqual((self.out / '.collection.lock').read_text(), 'other process')

    def test_crash_after_failure_receipt_recovers_stop_before_new_request(self):
        self.scope['targets'].append(self.target('two'))
        one, two = self.scope['targets']
        self.run_collect([Response(one['url'], b'denied', 403)])
        for name in ('STOPPED_microsoft.json', 'two.receipt.json', 'COLLECTION.json'):
            (self.out / name).unlink()
        result, fake = self.run_collect()
        self.assertFalse(fake.calls)
        self.assertTrue((self.out / 'STOPPED_microsoft.json').exists())
        self.assertEqual(self.receipt('two')['outcome'], 'skipped_provider_stopped')

    def test_completed_old_receipts_verified_before_new_target_request(self):
        self.scope['targets'].append(self.target('two'))
        self.run_collect([Response(self.target('one')['url']), Response(self.target('two')['url'])])
        (self.out / 'one.bin').unlink()
        (self.out / 'one.receipt.json').unlink()
        (self.out / '.attempt_one.json').unlink()
        (self.out / 'two.bin').write_bytes(b'corrupt later evidence')
        fake = Transport([Response(self.target('one')['url'])])
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            c.collect('work/scope.json', self.save(), root=self.root, request=fake)
        self.assertFalse(fake.calls)

    def test_pinned_inputs_checked_again_after_requests(self):
        def mutate(request, timeout):
            (self.root / 'work/input.json').write_bytes(b'changed during request')
            return Response(request.full_url)
        with self.assertRaisesRegex(ValueError, 'Input SHA mismatch'):
            c.collect('work/scope.json', self.save(), root=self.root, request=mutate)
        self.assertFalse((self.out / 'COLLECTION.json').exists())
        self.assertTrue((self.out / 'one.receipt.json').exists())

    def test_interrupted_inflight_attempt_is_persisted_and_never_retried(self):
        fake = Transport([KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            c.collect('work/scope.json', self.save(), root=self.root, request=fake)
        self.assertTrue((self.out / '.attempt_one.json').is_file())
        self.assertFalse((self.out / 'one.receipt.json').exists())
        with self.assertRaisesRegex(ValueError, 'Interrupted attempt'):
            self.run_collect()

    def test_changed_attempt_marker_rejected(self):
        self.run_collect([Response(self.target('one')['url'])])
        marker = self.out / '.attempt_one.json'
        value = json.loads(marker.read_bytes())
        value['attempt_started_utc'] = 'modified'
        marker.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Attempt marker binding'):
            self.run_collect()


if __name__ == '__main__':
    unittest.main(verbosity=2)
