"""Synthetic only: no original PDFs, rendering library, or financial admission."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import roa_field_scope270_pdf_rerender as m


class Fixture:
    def __init__(self, root):
        self.root = root
        self.pins = {}
        self.docs = {}
        old, pdf, visual, specs = [], [], [], []
        for i in range(10):
            raw = ('%%PDF-FAKE %s' % i).encode()
            source_path = 'inputs/report%d/raw/doc.pdf' % i
            receipt_path = 'inputs/report%d/raw/doc.meta.json' % i
            source = {'path': source_path, 'sha256': m.sha(raw),
                      'url': 'https://example.test/report%d.pdf' % i}
            self.add(source_path, raw)
            receipt = {'url': source['url'], 'status': 200,
                       'retrieved_at': '2026-10-01T12:00:00+00:00',
                       'response_sha256': m.sha(raw), 'response_path': 'raw/doc.pdf'}
            source['receipt'] = self.document(receipt_path, receipt)
            event = {'event_id': 'FAKE:%d' % i, 'source': source,
                     'periods': {'current': {'period_end': '2020-12-31'},
                                 'comparison': {'period_end': '2019-12-31'}},
                     'available_from': '2021-02-02', 'fields': {'current': {}, 'comparison': {}}}
            for field in m.FIELDS:
                side, name = field.split('.')
                locator = {'page_1based': 3 if name == 'net_income' else 2,
                           'financial_column_index_0based': 1 if side == 'comparison' else 0,
                           'period_end': event['periods'][side]['period_end'],
                           'line_1based': 2, 'row_label': 'Net income' if name == 'net_income' else 'Total assets'}
                event['fields'][side][name] = {'locator': locator, 'value_decimal': '100',
                                              'source': deepcopy(source), 'scale': '1000000'}
            old.append(deepcopy(event)); pdf.append(deepcopy(event))
            visual.append({'event_id': event['event_id'], 'source': deepcopy(source),
                           'cash_flow_page_1based': 3, 'balance_sheet_page_1based': 2,
                           'period_end_dates': ['2020-12-31', '2019-12-31']})
            specs.append({'event_id': event['event_id'], 'source': deepcopy(source), 'pages_1based': [2, 3]})
        old.extend({'event_id': 'HTML:%d' % i} for i in range(34))
        inputs = {}
        for role, doc in (('dataset', {'events': old}), ('pdf_data', {'events': pdf}),
                          ('visual_review', {'events': visual}), ('three_review', {'events': []}),
                          ('runtime_gap', {'actual_assess_calls': 0}), ('pypi_metadata', {})):
            self.docs[role] = doc
            inputs[role] = self.document('inputs/' + role + '.json', doc)
        runtime_files = {}
        for i in range(126):
            ref = self.add(m.RUNTIME_SITE + '/package/file%d' % i, str(i).encode())
            runtime_files[ref['path']] = ref['sha256']
        inputs['wheel'] = self.add('inputs/wheel.whl', b'SYNTHETIC WHEEL')
        acquisition = {'members': dict(list(runtime_files.items())[:2]),
                       'wheel_sha256': inputs['wheel']['sha256'],
                       'metadata_sha256': inputs['pypi_metadata']['sha256']}
        inputs['acquisition'] = self.document('inputs/acquisition.json', acquisition)
        runtime = {'pdf_runtime': {'bundle_site_packages': m.RUNTIME_SITE},
                   'acquisition': inputs['acquisition'], 'pdf_runtime_hashes': runtime_files,
                   'official_new_binary_sha256': 'a' * 64,
                   'old_plan': self.document('inputs/old_plan.json', {'historical': True})}
        inputs['runtime'] = self.document('inputs/runtime.json', runtime)
        inputs['runtime_loader'] = self.add('inputs/loader.py', b'# Never imported in fixture\n')
        codes = {name: self.add(name, b'# synthetic code pin\n')['sha256'] for name in m.CODE}
        encoder = {}
        for directory in m.ENCODER_DIRS:
            ref = self.add(directory + '/test_member', b'SYNTHETIC ENCODER')
            encoder[ref['path']] = ref['sha256']
        self.plan = {'schema': m.SCHEMA, 'status': 'FROZEN_AUTHORIZED_RENDER_ONLY',
                     'execution_authorized': True, 'source_only': True,
                     'financial_admission_authorized': False, 'old_runtime_recovered': False,
                     'output_directory': m.OUT, 'render': deepcopy(m.RENDER),
                     'inputs': inputs, 'code_sha256': codes, 'input_sha256': self.pins,
                     'limits': {'events': 10, 'pages': 20, 'fields': 30}, 'events': specs,
                     'runtime': {'pdfium_sha256': 'a' * 64, 'png_encoder_sha256': encoder,
                                 'python_version': '3.9.6',
                                 'executable_path': str(Path(sys.executable).resolve()),
                                 'executable_sha256': m.sha(Path(sys.executable).read_bytes())}}

    def add(self, path, raw):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        self.pins[path] = m.sha(raw)
        return {'path': path, 'sha256': self.pins[path]}

    def document(self, path, value):
        return self.add(path, m.canonical(value))

    def verify(self):
        ref = self.document('proposal.json', self.plan)
        return m.verify_proposal(self.root, ref['path'], ref['sha256'])


class FakeBackend:
    kind = 'synthetic_test_double'
    def render_pages(self, raw, pages):
        return [{'page_1based': n, 'document_page_count': 3,
                 'width_pixels': 10, 'height_pixels': 20, 'png': m.PNG + bytes([n]),
                 'pypdf_layout_text': 'HEADER\nTotal assets 100 90\nNet income 50 40',
                 'pdfium_text': 'HEADER\nTotal assets 100 90\nNet income 50 40',
                 'pdfium_character_count': 1,
                 'nonwhitespace_character_boxes': [{'index_0based': 0, 'text': 'H',
                                                    'bbox_pdf_units': [0, 0, 1, 1]}]} for n in pages]


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='roa270-render-synthetic-')
        self.root = Path(self.temp.name).resolve()
        self.f = Fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_fake_metadata_authentication(self):
        p, ev, docs, events, raw = self.f.verify()
        self.assertEqual(len(events), 10)
        self.assertEqual(len(raw), 10)
        self.assertEqual(sum(len(x['fields']) for x in events), 30)
        self.assertTrue(all(x.startswith(b'%PDF-') for x in raw.values()))

    def test_draft_rejected_before_input_read_or_runtime(self):
        self.f.plan['status'] = 'DRAFT_NO_EXECUTION'
        self.f.plan['execution_authorized'] = False
        with patch.object(m.Evidence, 'read', side_effect=AssertionError('input body read')):
            with self.assertRaisesRegex(ValueError, 'not authorized'):
                self.f.verify()

    def test_false_admission_scope_required(self):
        self.f.plan['financial_admission_authorized'] = True
        with self.assertRaisesRegex(ValueError, 'Source-only'):
            self.f.verify()

    def test_old_runtime_recovery_claim_rejected(self):
        self.f.plan['old_runtime_recovered'] = True
        with self.assertRaises(ValueError):
            self.f.verify()

    def test_proposal_digest_tamper(self):
        ref = self.f.document('proposal.json', self.f.plan)
        with self.assertRaisesRegex(ValueError, 'Proposal byte'):
            m.verify_proposal(self.root, ref['path'], '0' * 64)

    def test_runtime_member_tamper(self):
        (self.root / m.RUNTIME_SITE / 'package/file50').write_text('tamper')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            self.f.verify()

    def test_encoder_inventory_must_be_complete(self):
        (self.root / m.ENCODER_DIRS[0] / 'new.py').write_text('unbound')
        with self.assertRaisesRegex(ValueError, 'Complete PNG encoder'):
            self.f.verify()

    def test_raw_pdf_tamper(self):
        (self.root / self.f.plan['events'][4]['source']['path']).write_bytes(b'%PDF-CHANGED')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            self.f.verify()

    def test_rich_source_mutation_rejected(self):
        self.f.plan['events'][0]['source']['url'] = 'https://other.test/report.pdf'
        with self.assertRaisesRegex(ValueError, 'Full original PDF source'):
            self.f.verify()

    def test_fixed_scope_and_page_inventory(self):
        for change in ('count', 'page', 'order'):
            with self.subTest(change=change):
                original = deepcopy(self.f.plan['events'])
                if change == 'count': self.f.plan['events'].pop()
                elif change == 'page': self.f.plan['events'][0]['pages_1based'] = [2, 3, 4]
                else: self.f.plan['events'].reverse()
                with self.assertRaises(ValueError): self.f.verify()
                self.f.plan['events'] = original

    def test_old_cell_full_object_mismatch(self):
        self.f.docs['pdf_data']['events'][0]['fields']['current']['net_income']['extra'] = True
        self.f.plan['inputs']['pdf_data'] = self.f.document('inputs/pdf_data.json', self.f.docs['pdf_data'])
        with self.assertRaisesRegex(ValueError, 'Complete original/repaired cell'):
            self.f.verify()

    def test_receipt_status_url_body_time_and_path(self):
        source = self.f.plan['events'][0]['source']
        raw_receipt = json.loads((self.root / source['receipt']['path']).read_text())
        for key, value in [('status', True), ('url', 'https://wrong.test/a.pdf'),
                           ('response_sha256', '0' * 64), ('retrieved_at', '2020-01-01'),
                           ('response_path', 'raw/other.pdf')]:
            with self.subTest(key=key):
                bad = dict(raw_receipt, **{key: value})
                ref = self.f.document(source['receipt']['path'], bad)
                src = deepcopy(source); src['receipt'] = ref
                with self.assertRaises(ValueError):
                    m.source_bytes(src, m.Evidence(self.root, self.f.pins))
        self.f.document(source['receipt']['path'], raw_receipt)

    def test_unsafe_paths_and_symlink(self):
        for path in ('../escape', '/etc/passwd', 'work/CloudStorage/x'):
            with self.assertRaises(ValueError): m.local(self.root, path)
        (self.root / 'link').symlink_to(self.root / 'inputs')
        with self.assertRaises(ValueError): m.local(self.root, 'link/dataset.json')

    def test_no_global_runtime_fallback(self):
        self.f.plan['runtime']['executable_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Python executable'):
            self.f.verify()

    def test_duplicate_json_and_nonfinite_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError): m.decode(raw)

    def test_authorized_run_imports_only_after_all_gates(self):
        self.f.plan['execution_authorized'] = False
        ref = self.f.document('proposal.json', self.f.plan)
        with patch.object(m, 'load_backend', side_effect=AssertionError('loaded')):
            with self.assertRaisesRegex(ValueError, 'not authorized'):
                m.run(ref['path'], ref['sha256'], self.root)

    def test_fake_render_complete_evidence_is_not_admission(self):
        p, ev, docs, events, raw = self.f.verify()
        before = deepcopy(events)
        result = m.execute_bundle(p, '0' * 64, ev, events, raw, FakeBackend(), {'synthetic': True})
        out = self.root / m.OUT
        evidence = json.loads((out / 'EVIDENCE.json').read_text())
        self.assertEqual(events, before)
        self.assertEqual(len(result['output_sha256']), 42)
        self.assertEqual(len(list(out.glob('*.png'))), 20)
        self.assertEqual(evidence['counts']['actual_assess_calls'], 0)
        self.assertTrue(evidence['synthetic_backend'])
        self.assertFalse(evidence['field_admitted'])
        self.assertFalse(result['old_runtime_verified'])
        self.assertTrue(result['independent_review_pending'])
        for path, digest in result['output_sha256'].items():
            self.assertEqual(m.sha((out / path).read_bytes()), digest)
        field = evidence['events'][0]['fields']['current.net_income']
        self.assertFalse(field['new_text_locator_evidence']['amount_revalidated'])
        self.assertEqual(field['new_text_locator_evidence']['pypdf_label_candidates'][0]['line_1based'], 3)

    def test_write_once_no_overwrite(self):
        p, ev, docs, events, raw = self.f.verify()
        out = self.root / m.OUT
        out.mkdir(parents=True)
        (out / 'keep').write_bytes(b'original')
        with self.assertRaisesRegex(ValueError, 'Write-once'):
            m.execute_bundle(p, '0' * 64, ev, events, raw, FakeBackend(), {})
        self.assertEqual((out / 'keep').read_bytes(), b'original')

    def test_failure_leaves_no_success_manifest(self):
        p, ev, docs, events, raw = self.f.verify()
        bad = FakeBackend()
        bad.render_pages = lambda *_: []
        with self.assertRaisesRegex(ValueError, 'fixed page scope'):
            m.execute_bundle(p, '0' * 64, ev, events, raw, bad, {})
        self.assertFalse((self.root / m.OUT / 'MANIFEST.json').exists())

    def test_post_render_input_mutation_refused(self):
        p, ev, docs, events, raw = self.f.verify()
        bad = FakeBackend()
        original = bad.render_pages
        def changed(*args):
            (self.root / events[0]['source']['path']).write_bytes(b'%PDF-MUTATED')
            return original(*args)
        bad.render_pages = changed
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            m.execute_bundle(p, '0' * 64, ev, events, raw, bad, {})
        self.assertFalse((self.root / m.OUT / 'MANIFEST.json').exists())

    def test_ambiguous_or_missing_text_remains_pending(self):
        field = {'original_locator': {'row_label': 'Net income', 'line_1based': 100}}
        loc = m.locator_evidence(field, {'pypdf_layout_text': 'Net income 1\nNet income 2',
                                       'pdfium_text': 'unrelated'})
        self.assertEqual(len(loc['pypdf_label_candidates']), 2)
        self.assertEqual(loc['pdfium_label_candidates'], [])
        self.assertIsNone(loc['new_pypdf_text_at_old_line'])
        self.assertTrue(loc['independent_visual_review_pending'])


class BackendTests(unittest.TestCase):
    def test_real_backend_code_against_fake_pdf_objects_only(self):
        closed, rendered = [], []
        class Image:
            width, height = 10, 20
            def save(self, stream, format):
                self_test.assertEqual(format, 'PNG'); stream.write(m.PNG + b'fake')
            def close(self): closed.append('image')
        class Bitmap:
            def to_pil(self): return Image()
            def close(self): closed.append('bitmap')
        class TextPage:
            def get_text_range(self, *args, **kwargs): return 'A' if args else 'A B'
            def count_chars(self): return 1
            def get_charbox(self, i): return (0.0, 0.0, 1.0, 1.0)
            def close(self): closed.append('textpage')
        class Page:
            def get_textpage(self): return TextPage()
            def render(self, **kwargs): rendered.append(kwargs); return Bitmap()
            def close(self): closed.append('page')
        class Doc:
            def __len__(self): return 3
            def __getitem__(self, i): return Page()
            def close(self): closed.append('doc')
        self_test = self
        reader = lambda _: types.SimpleNamespace(pages=[types.SimpleNamespace(
            extract_text=lambda **kwargs: 'Net income 1') for _ in range(3)])
        backend = m.PdfBackend(reader, types.SimpleNamespace(PdfDocument=lambda _: Doc()))
        pages = backend.render_pages(b'not parsed by a real PDF library', [2, 3])
        self.assertEqual([x['page_1based'] for x in pages], [2, 3])
        self.assertEqual(rendered, [{'scale': 1.6, 'rotation': 0, 'crop': (0, 0, 0, 0),
                                     'may_draw_forms': True}] * 2)
        self.assertEqual(closed.count('image'), 2)
        self.assertEqual(closed.count('bitmap'), 2)
        self.assertEqual(closed.count('textpage'), 2)
        self.assertEqual(closed.count('page'), 2)
        self.assertEqual(closed.count('doc'), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
