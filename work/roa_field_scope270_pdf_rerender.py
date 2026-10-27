"""Prospective, source-only ROA PDF evidence renderer. No financial admission.

Nothing renders on import. The command requires a separately approved frozen
proposal and byte pins before importing the isolated CAPEX193 PDF toolchain.
Historical extraction/visual records are anchors, not newly certified old runtime.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'roa_field_scope270_pdf_rerender_proposal_v1'
OUT = 'outputs/monthly_roa_field_scope270_pdf_rerender_v1'
FIELDS = ('current.net_income', 'current.total_assets', 'comparison.total_assets')
ROLES = ('dataset', 'pdf_data', 'visual_review', 'three_review', 'runtime_gap',
         'runtime', 'acquisition', 'runtime_loader', 'pypi_metadata', 'wheel')
CODE = ('work/roa_field_scope270_pdf_rerender.py',
        'work/test_roa_field_scope270_pdf_rerender.py')
RUNTIME_SITE = 'work/capex193_pdf_runtime/site-packages'
ENCODER_DIRS = ('work/venv/lib/python3.9/site-packages/PIL',
                'work/venv/lib/python3.9/site-packages/pillow-11.3.0.dist-info')
RENDER = {'scale': 1.6, 'rotation': 0, 'crop': [0, 0, 0, 0],
          'may_draw_forms': True, 'image_format': 'PNG',
          'pypdf_extraction_mode': 'layout', 'pdfium_text_errors': 'strict'}
PNG = b'\x89PNG\r\n\x1a\n'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def check_sha(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'SHA256 required')


def local(root, name):
    need(type(name) is str and name and '..' not in Path(name).parts, 'Safe local path required')
    p = Path(name)
    p = p if p.is_absolute() else root / p
    need(p.is_relative_to(root) and p.resolve() == p
         and not any(s in str(p).lower() for s in ('cloudstorage', 'onedrive')),
         'Local nonsymlink path required')
    return p


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


class Evidence:
    def __init__(self, root, pins):
        self.root = Path(root).resolve()
        need(type(pins) is dict and pins, 'Nonempty pinned input map required')
        self.pins = deepcopy(pins)
        self.verified = {}

    def read(self, name, expected=None):
        need(name in self.pins, 'Unpinned input: ' + str(name))
        bound = self.pins[name]
        check_sha(bound)
        need(expected is None or expected == bound, 'Conflicting input hash')
        p = local(self.root, name)
        need(p.is_file(), 'Regular input required')
        before = p.stat()
        raw = p.read_bytes()
        after = p.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_ino, after.st_size, after.st_mtime_ns), 'Input mutated while reading')
        need(sha(raw) == bound, 'SHA mismatch: ' + name)
        self.verified[name] = bound
        return raw

    def document(self, ref):
        need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'Exact file reference required')
        return decode(self.read(ref['path'], ref['sha256']))

    def stable(self):
        for path, digest in list(self.verified.items()):
            self.read(path, digest)


def https_url(value):
    need(type(value) is str, 'HTTPS URL required')
    u = urlsplit(value)
    need(u.scheme == 'https' and u.hostname and not u.username and not u.password
         and not u.fragment, 'HTTPS URL without credentials/fragment required')


def source_bytes(source, evidence):
    required = {'path', 'sha256', 'url', 'receipt'}
    need(type(source) is dict and required <= set(source) <= required | {'receipt_entry_key'},
         'Full original source shape required')
    https_url(source['url'])
    receipt = evidence.document(source['receipt'])
    key = source.get('receipt_entry_key')
    if key is not None:
        need(type(key) is str and key and key in receipt, 'Retrieval entry required')
        receipt = receipt[key]
    need(type(receipt) is dict and receipt.get('url') == source['url'], 'Retrieval URL mismatch')
    status = receipt.get('http_status', receipt.get('status'))
    need(type(status) is int and status == 200, 'Retrieval HTTP 200 required')
    time = receipt.get('retrieved_at')
    need(type(time) is str, 'Real retrieval timestamp required')
    parsed = datetime.fromisoformat(time.replace('Z', '+00:00'))
    need(parsed.tzinfo is not None, 'Timezone-aware retrieval timestamp required')
    body_sha = receipt.get('sha256', receipt.get('response_sha256'))
    need(body_sha == source['sha256'], 'Retrieval body digest mismatch')
    if 'response_path' in receipt:
        rp = receipt['response_path']
        need(type(rp) is str and rp and '..' not in Path(rp).parts, 'Safe response path required')
        if not Path(rp).is_absolute() and Path(rp).parts[0] == 'raw':
            resolved = local(evidence.root, source['receipt']['path']).parent.parent / rp
        else:
            resolved = local(evidence.root, rp)
        need(resolved == local(evidence.root, source['path']), 'Retrieval response path mismatch')
    final = receipt.get('final_url')
    if final is not None:
        https_url(final)
    raw = evidence.read(source['path'], source['sha256'])
    need(raw.startswith(b'%PDF-'), 'PDF magic required')
    return raw


def index_events(document):
    result = {}
    for event in document['events']:
        key = event['event_id']
        need(type(key) is str and key not in result, 'Unique event IDs required')
        result[key] = event
    return result


def event_inventory(docs, expected_events):
    """Read only saved metadata, never evaluate a financial field or admission."""
    old = index_events(docs['dataset'])
    raw = index_events(docs['pdf_data'])
    visual = index_events(docs['visual_review'])
    need(len(old) == 44 and len(raw) == len(visual) == 10, 'Fixed 44/10 source scope')
    need(type(expected_events) is list and len(expected_events) == 10
         and [x['event_id'] for x in expected_events] == list(raw) == list(visual),
         'Exact ordered ten-report inventory required')
    selected = []
    for supplied in expected_events:
        key = supplied['event_id']
        event, repaired, seen = old[key], raw[key], visual[key]
        need(supplied['source'] == event['source'] == repaired['source'] == seen['source'],
             'Full original PDF source mismatch')
        need(event['periods'] == repaired['periods'] and event['available_from'] == repaired['available_from'],
             'Original period/clock changed')
        need(supplied['pages_1based'] == [2, 3], 'Only fixed pages 2 and 3')
        fields = {}
        for field in FIELDS:
            side, name = field.split('.')
            cell = event['fields'][side][name]
            need(cell == repaired['fields'][side][name], 'Complete original/repaired cell mismatch')
            locator = cell['locator']
            page = 3 if name == 'net_income' else 2
            col = 1 if side == 'comparison' else 0
            need(type(locator['page_1based']) is int and locator['page_1based'] == page
                 and type(locator['financial_column_index_0based']) is int
                 and locator['financial_column_index_0based'] == col, 'Fixed physical cell location')
            need(locator['period_end'] == event['periods'][side]['period_end'], 'Cell endpoint mismatch')
            need(type(locator['line_1based']) is int and locator['line_1based'] > 0
                 and type(locator['row_label']) is str and locator['row_label'], 'Original text locator required')
            visual_page = seen['cash_flow_page_1based' if name == 'net_income' else 'balance_sheet_page_1based']
            need(visual_page == page and seen['period_end_dates'][col] == locator['period_end'],
                 'Saved visual location disagrees')
            fields[field] = {'original_cell_sha256': sha(canonical(cell)),
                             'original_cell': deepcopy(cell), 'original_locator': deepcopy(locator)}
        selected.append({'event_id': key, 'original_event_sha256': sha(canonical(event)),
                         'source': deepcopy(event['source']), 'pages_1based': [2, 3], 'fields': fields})
    return selected


def verify_proposal(root, proposal_path, expected_sha):
    root = Path(root).resolve()
    check_sha(expected_sha)
    proposal_raw = local(root, proposal_path).read_bytes()
    need(sha(proposal_raw) == expected_sha, 'Proposal byte digest mismatch')
    p = decode(proposal_raw)
    need(p.get('schema') == SCHEMA, 'Proposal schema')
    # Check authorization BEFORE reading input PDFs or importing a rendering module.
    need(p.get('status') == 'FROZEN_AUTHORIZED_RENDER_ONLY'
         and p.get('execution_authorized') is True, 'Prospective draft: actual rendering not authorized')
    need(p.get('source_only') is True and p.get('financial_admission_authorized') is False
         and p.get('old_runtime_recovered') is False, 'Source-only scope required')
    need(p['output_directory'] == OUT and p['render'] == RENDER, 'Fixed new output/render parameters')
    need(set(p['inputs']) == set(ROLES) and set(p['code_sha256']) == set(CODE), 'Complete roles/code pins')
    need(p['limits'] == {'events': 10, 'pages': 20, 'fields': 30}, 'Fixed minimal PDF scope')
    ev = Evidence(root, p['input_sha256'])
    for path, digest in p['code_sha256'].items():
        ev.read(path, digest)
    docs = {name: ev.document(ref) for name, ref in p['inputs'].items()
            if name not in ('runtime_loader', 'wheel')}
    ev.read(p['inputs']['runtime_loader']['path'], p['inputs']['runtime_loader']['sha256'])
    ev.read(p['inputs']['wheel']['path'], p['inputs']['wheel']['sha256'])
    runtime = docs['runtime']
    need(runtime['acquisition'] == p['inputs']['acquisition'], 'Acquisition binding mismatch')
    need(runtime['official_new_binary_sha256'] == p['runtime']['pdfium_sha256'], 'Isolated binary binding')
    need(local(root, runtime['pdf_runtime']['bundle_site_packages']) == root / RUNTIME_SITE,
         'Isolated runtime location required')
    need(len(runtime['pdf_runtime_hashes']) == 126, 'Complete saved CAPEX193 manifest required')
    for name, digest in runtime['pdf_runtime_hashes'].items():
        need(local(root, name).is_relative_to(root / RUNTIME_SITE), 'Unexpected runtime member')
        ev.read(name, digest)
    acquisition = docs['acquisition']
    for name, digest in acquisition['members'].items():
        ev.read(name, digest)
    need(acquisition['wheel_sha256'] == p['inputs']['wheel']['sha256']
         and acquisition['metadata_sha256'] == p['inputs']['pypi_metadata']['sha256'], 'Wheel/metadata binding')
    # Loader also verifies the old pypdf source-copy hashes from this immutable document.
    ev.document(runtime['old_plan'])
    encoder = p['runtime']['png_encoder_sha256']
    actual_names = set()
    for directory in ENCODER_DIRS:
        base = local(root, directory)
        need(base.is_dir(), 'PNG encoder directory missing')
        for path in base.rglob('*'):
            need(not path.is_symlink(), 'PNG encoder symlink forbidden')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                actual_names.add(str(path.relative_to(root)))
    need(set(encoder) == actual_names and encoder, 'Complete PNG encoder inventory required')
    for name, digest in encoder.items():
        ev.read(name, digest)
    need(sys.version.split()[0] == p['runtime']['python_version'] == '3.9.6', 'Pinned Python version')
    executable = Path(sys.executable).resolve()
    need(str(executable) == p['runtime']['executable_path']
         and sha(executable.read_bytes()) == p['runtime']['executable_sha256'], 'Pinned Python executable')
    events = event_inventory(docs, p['events'])
    # All raw PDFs and retrieval evidence are authenticated before any render starts.
    raw = {e['event_id']: source_bytes(e['source'], ev) for e in events}
    return p, ev, docs, events, raw


class PdfBackend:
    kind = 'isolated_capex193_pypdf_pdfium_pillow'

    def __init__(self, reader, pdfium):
        self.reader, self.pdfium = reader, pdfium

    def render_pages(self, raw, pages):
        parsed = self.reader(io.BytesIO(raw))
        doc = self.pdfium.PdfDocument(raw)
        try:
            need(len(parsed.pages) == len(doc) and max(pages) <= len(doc) <= 100,
                 'Consistent bounded PDF page count')
            result = []
            for number in pages:
                page = doc[number - 1]
                textpage = None
                bitmap = None
                image = None
                try:
                    layout = parsed.pages[number - 1].extract_text(extraction_mode='layout')
                    need(type(layout) is str, 'Page layout text required')
                    textpage = page.get_textpage()
                    text = textpage.get_text_range(errors='strict')
                    characters = []
                    count = textpage.count_chars()
                    for i in range(count):
                        char = textpage.get_text_range(i, 1, errors='strict')
                        if not char or char.isspace():
                            continue
                        need(len(char) == 1, 'Unmapped PDFium character')
                        box = list(textpage.get_charbox(i))
                        need(len(box) == 4 and all(math.isfinite(x) for x in box)
                             and box[2] > box[0] and box[3] > box[1], 'Unlocated PDFium character')
                        characters.append({'index_0based': i, 'text': char, 'bbox_pdf_units': box})
                    bitmap = page.render(scale=1.6, rotation=0, crop=(0, 0, 0, 0), may_draw_forms=True)
                    image = bitmap.to_pil()
                    stream = io.BytesIO()
                    image.save(stream, format='PNG')
                    result.append({'page_1based': number, 'document_page_count': len(doc),
                                   'width_pixels': image.width, 'height_pixels': image.height,
                                   'png': stream.getvalue(), 'pypdf_layout_text': layout,
                                   'pdfium_text': text, 'pdfium_character_count': count,
                                   'nonwhitespace_character_boxes': characters})
                finally:
                    if image is not None:
                        image.close()
                    if bitmap is not None:
                        bitmap.close()
                    if textpage is not None:
                        textpage.close()
                    page.close()
            return result
        finally:
            doc.close()


def load_backend(p, evidence):
    """Reuse the reviewed 193 wheel/RECORD runtime loader, not its finance parser."""
    name = p['inputs']['runtime_loader']['path']
    evidence.read(name, p['inputs']['runtime_loader']['sha256'])
    need(not any(k in sys.modules for k in ('pypdf', 'pypdfium2', 'pypdfium2_raw')),
         'PDF module already imported before isolated runtime gate')
    spec = importlib.util.spec_from_file_location('_roa270_capex193_runtime_loader', local(evidence.root, name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pypdf, pdfium, proof = module.runtime()
    # PNG encoding is separately pinned; 193's PDF manifest does not cover Pillow.
    from PIL import Image, PngImagePlugin
    for imported in (Image, PngImagePlugin):
        need(Path(imported.__file__).resolve().is_relative_to(evidence.root / ENCODER_DIRS[0]),
             'PNG encoder imported outside pinned Pillow')
    evidence.stable()
    return PdfBackend(pypdf.PdfReader, pdfium), {
        'capex193_runtime_verification': proof,
        'pdf_runtime_hashes': deepcopy(module.load(module.RUNTIME)['pdf_runtime_hashes']),
        'png_encoder_sha256': deepcopy(p['runtime']['png_encoder_sha256']),
        'python_version': sys.version, 'executable_path': str(Path(sys.executable).resolve()),
        'executable_sha256': sha(Path(sys.executable).read_bytes()),
        'system_libraries_and_whole_os_pinned': False}


def locator_evidence(field, page):
    old = field['original_locator']
    label = ' '.join(old['row_label'].casefold().split())
    def candidates(text):
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            normalized = ' '.join(line.casefold().split())
            if normalized.startswith(label) and (len(normalized) == len(label)
                                                 or not normalized[len(label)].isalpha()):
                rows.append({'line_1based': i, 'text': line})
        return rows
    layout_lines = page['pypdf_layout_text'].splitlines()
    old_line = old['line_1based']
    return {'original_locator': deepcopy(old),
            'pypdf_label_candidates': candidates(page['pypdf_layout_text']),
            'pdfium_label_candidates': candidates(page['pdfium_text']),
            'new_pypdf_text_at_old_line': layout_lines[old_line - 1] if old_line <= len(layout_lines) else None,
            'original_line_number_revalidated': False,
            'selected_column_revalidated': False, 'amount_revalidated': False,
            'independent_visual_review_pending': True}


def publish(path, data):
    with path.open('xb') as stream:
        stream.write(data)
    return sha(data)


def execute_bundle(p, proposal_sha, evidence, events, raw, backend, runtime_proof):
    """Execution seam for synthetic fake backends; CLI always uses load_backend."""
    need(len(events) == 10 and len(raw) == 10, 'Exact ten-report execution scope')
    output = local(evidence.root, p['output_directory'])
    need(not output.exists(), 'Write-once new output directory already exists')
    evidence.stable()
    output.mkdir(parents=True, exist_ok=False)
    started = timestamp()
    members = {}
    rendered_events = []
    for index, event in enumerate(events):
        need(sha(raw[event['event_id']]) == event['source']['sha256'], 'Raw bytes changed after authentication')
        began = timestamp()
        pages = backend.render_pages(raw[event['event_id']], [2, 3])
        ended = timestamp()
        need([x['page_1based'] for x in pages] == [2, 3], 'Backend changed fixed page scope')
        records = {}
        for page in pages:
            need(type(page['png']) is bytes and page['png'].startswith(PNG), 'PNG bytes required')
            need(type(page['width_pixels']) is int and page['width_pixels'] > 0
                 and type(page['height_pixels']) is int and page['height_pixels'] > 0, 'Positive image dimensions')
            stem = 'event_%02d_page_%d' % (index, page['page_1based'])
            image_name, text_name = stem + '.png', stem + '.json'
            members[image_name] = publish(output / image_name, page['png'])
            text = {k: v for k, v in page.items() if k != 'png'}
            text.update(event_id=event['event_id'], source=deepcopy(event['source']),
                        rendered_started_utc=began, rendered_ended_utc=ended,
                        image={'path': image_name, 'sha256': members[image_name]},
                        render_parameters=deepcopy(RENDER), independent_visual_review_pending=True)
            members[text_name] = publish(output / text_name, canonical(text) + b'\n')
            records[page['page_1based']] = {'image': {'path': image_name, 'sha256': members[image_name]},
                                          'text_geometry': {'path': text_name, 'sha256': members[text_name]}}
        fields = {}
        for name in FIELDS:
            field = deepcopy(event['fields'][name])
            number = field['original_locator']['page_1based']
            page = next(x for x in pages if x['page_1based'] == number)
            field['render_evidence'] = records[number]
            field['new_text_locator_evidence'] = locator_evidence(field, page)
            field['field_admitted'] = False
            fields[name] = field
        rendered_events.append({'event_id': event['event_id'],
                                'original_event_sha256': event['original_event_sha256'],
                                'source': deepcopy(event['source']), 'fields': fields,
                                'pages': [records[2], records[3]], 'independent_review_pending': True})
    evidence.stable()
    members['RUNTIME.json'] = publish(output / 'RUNTIME.json', canonical(runtime_proof) + b'\n')
    result = {'schema': 'roa_field_scope270_pdf_render_evidence_v1',
              'source_only': True, 'render_started_utc': started, 'render_finished_utc': timestamp(),
              'proposal_sha256': proposal_sha, 'events': rendered_events,
              'counts': {'events': 10, 'pages_rendered': 20, 'required_fields': 30,
                         'html_events_reextracted': 0, 'actual_assess_calls': 0},
              'backend': backend.kind, 'synthetic_backend': backend.kind != PdfBackend.kind,
              'input_sha256': deepcopy(evidence.verified),
              'runtime_manifest': {'path': 'RUNTIME.json', 'sha256': members['RUNTIME.json']},
              'independent_visual_review_pending': True, 'field_admitted': False,
              'ratios_computed': False, 'chain_admitted': False, 'strategy_admitted': False,
              'old_runtime_verified': False, 'strict_pit': False,
              'first_public_version_verified': False, 'iso_currency_certified': False}
    members['EVIDENCE.json'] = publish(output / 'EVIDENCE.json', canonical(result) + b'\n')
    manifest = {'schema': 'roa_field_scope270_pdf_render_manifest_v1',
                'completed_utc': timestamp(), 'proposal_sha256': proposal_sha,
                'output_sha256': members, 'input_sha256': deepcopy(evidence.verified),
                'render_complete': True, 'independent_review_pending': True,
                'financial_admission': False, 'old_runtime_verified': False,
                'synthetic_backend': result['synthetic_backend']}
    publish(output / 'MANIFEST.json', canonical(manifest) + b'\n')
    return manifest


def run(proposal_path, proposal_sha, root=ROOT):
    p, evidence, docs, events, raw = verify_proposal(root, proposal_path, proposal_sha)
    backend, runtime = load_backend(p, evidence)
    result = execute_bundle(p, proposal_sha, evidence, events, raw, backend, runtime)
    return {'output_directory': p['output_directory'], 'render_complete': result['render_complete'],
            'independent_review_pending': True, 'financial_admission': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', required=True)
    parser.add_argument('--proposal-sha', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.proposal, args.proposal_sha), sort_keys=True))


if __name__ == '__main__':
    main()
