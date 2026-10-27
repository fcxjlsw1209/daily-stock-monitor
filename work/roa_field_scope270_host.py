"""Policy-bound, source-only ROA census host. No ratio, chain, model or price I/O.

Run only after an independently frozen policy and synthetic test receipt exist.
Historical reviews are checked as saved evidence, not claimed as fresh extraction.
"""
import argparse
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import urlsplit

import roa_field_scope270_v2 as scope

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_roa_field_scope270_source_v1'
SCHEMA = 'roa_field_scope270_source_policy_v1'
CODE_FILES = ('work/roa_field_scope270_host.py', 'work/test_roa_field_scope270_host.py',
              'work/roa_field_scope270_v2.py', 'work/test_roa_field_scope270_v2.py')
ROLES = ('dataset', 'legacy_policy', 'legacy_admission', 'legacy_review', 'union_data',
         'union_plan', 'union_manifest', 'union_review', 'sessions', 'calendar_plan',
         'accounting', 'baseline_data', 'baseline_plan', 'baseline_manifest', 'html_review',
         'html_repair_data', 'html_repair_review', 'pdf_data', 'pdf_review', 'pdf_visual',
         'three_review')
HASH_GROUPS = ('input_sha256', 'input_hashes', 'source_hashes', 'source_code_hashes',
               'review_code_hashes', 'producer_code_hashes', 'pdf_runtime_hashes')
ZERO = ('ratios_computed', 'chain_admissions', 'features_computed', 'models_fitted',
        'prices_read', 'NPZ_read', 'returns_computed', 'network_requests', 'observer_changes')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def same(a, b, message):
    need(scope.canonical_sha256(a) == scope.canonical_sha256(b), message)


def unique(rows):
    need(type(rows) is list, 'Event list required')
    result = {}
    for row in rows:
        key = row.get('event_id')
        need(type(key) is str and key and key not in result, 'Unique event ID required')
        result[key] = row
    return result


def one(rows, **keys):
    matches = [row for row in rows if all(row.get(k) == v for k, v in keys.items())]
    need(len(matches) == 1, 'Exactly one evidence row required: ' + repr(keys))
    return matches[0]


class Evidence:
    """Only policy-pinned local files may be read; external runtime pins are hash-only."""
    def __init__(self, root, pins):
        self.root = Path(root).resolve()
        self.pins = dict(pins)
        self.verified = {}

    def path(self, name, external=False):
        need(type(name) is str and name and '..' not in Path(name).parts, 'Safe local path required')
        need(not any(x in name.lower() for x in ('onedrive', 'cloudstorage')), 'Cloud input forbidden')
        p = Path(name)
        p = p if p.is_absolute() else self.root / p
        allowed = p.is_relative_to(self.root) or (external and str(p).startswith(
            '/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/'))
        need(allowed and p.resolve() == p, 'External path or symlink forbidden')
        return p

    def hash(self, name, expected=None):
        need(name in self.pins, 'File omitted from policy: ' + name)
        expected = self.pins[name] if expected is None else expected
        scope._sha(expected)
        need(self.pins[name] == expected, 'Conflicting policy pin: ' + name)
        p = self.path(name, external=True)
        before = p.stat()
        need(stat.S_ISREG(before.st_mode), 'Regular input file required')
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        after = p.stat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')
        need(all(getattr(before, k) == getattr(after, k) for k in fields), 'Input changed while hashing')
        need(h.hexdigest() == expected, 'SHA mismatch: ' + name)
        self.verified[name] = expected
        return expected

    def read(self, ref):
        scope._reference(ref)
        name = ref['path']
        self.hash(name, ref['sha256'])
        raw = self.path(name).read_bytes()
        need(digest(raw) == ref['sha256'], 'Input changed before JSON read')
        def pairs(items):
            value = {}
            for key, item in items:
                need(key not in value, 'Duplicate JSON key')
                value[key] = item
            return value
        def invalid(value):
            raise ValueError('Nonfinite JSON token: ' + value)
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)

    def declared(self, document):
        for group in HASH_GROUPS:
            if group in document:
                pins = document[group]
                need(type(pins) is dict, 'Declared hash map required')
                for path, sha in pins.items():
                    self.hash(path, sha)

    def stable(self):
        for path, sha in list(self.verified.items()):
            self.hash(path, sha)


def verify_receipt(source, ev):
    """Read retrieval evidence without treating retrieval time as publication time."""
    scope._source(source)
    ev.hash(source['path'], source['sha256'])
    receipt = ev.read(source['receipt'])
    entry = source.get('receipt_entry_key')
    if entry is not None:
        need(entry in receipt, 'Missing retrieval manifest entry')
        receipt = receipt[entry]
    need(type(receipt) is dict, 'Retrieval receipt object required')
    need(receipt.get('url') == source['url'], 'Receipt URL mismatch')
    status = receipt.get('http_status', receipt.get('status'))
    need(type(status) is int and status == 200, 'Raw source HTTP 200 required')
    stamp = receipt.get('retrieved_at')
    need(type(stamp) is str, 'Retrieval timestamp required')
    parsed = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    need(parsed.tzinfo is not None and parsed.utcoffset() is not None, 'Timezone-aware retrieval timestamp required')
    if 'sha256' in receipt:
        body_sha = receipt['sha256']
    else:
        need('response_sha256' in receipt, 'Raw response digest required')
        body_sha = receipt['response_sha256']
        if 'response_path' in receipt:
            # Historical absolute locations are translated by the pinned launcher only.
            rp = receipt['response_path']
            need(type(rp) is str and '..' not in Path(rp).parts, 'Safe response path required')
            source_path = ev.path(source['path'])
            if not Path(rp).is_absolute() and Path(rp).parts[0] == 'raw':
                # AAPL collection receipts explicitly use collection-root-relative raw/name.
                resolved = ev.path(source['receipt']['path']).parent.parent / rp
            else:
                resolved = ev.path(rp)
            need(resolved == source_path, 'Raw response path mismatch')
    need(body_sha == source['sha256'], 'Receipt body SHA mismatch')
    if 'final_url' in receipt:
        final = urlsplit(receipt['final_url'])
        need(final.scheme == 'https' and final.hostname and final.username is None
             and final.password is None and not final.fragment, 'Invalid bound final URL')
    return dict(raw_source_sha256=source['sha256'], receipt_sha256=source['receipt']['sha256'],
                receipt_entry_key=entry, url=source['url'], http_status=200, retrieved_at=stamp,
                raw_bytes_verified=True, receipt_bytes_verified=True,
                publication_time_inferred_from_retrieval=False,
                publisher_identity_scope='Inherited pinned publisher/identity evidence; not first-public certification')


def policy_check(policy, ev):
    need(policy.get('schema') == SCHEMA and policy.get('source_only') is True, 'Source-only policy schema')
    need(policy.get('output_directory') == OUT and policy.get('expected_events') == 44
         and policy.get('expected_old_admitted') == 31, 'Fixed output and 44/31 census policy')
    need(set(policy['inputs']) == set(ROLES), 'Exact source roles required')
    need(set(policy['code_sha256']) == set(CODE_FILES), 'Complete host/scope/test code inventory')
    for name, sha in policy['code_sha256'].items():
        ev.hash(name, sha)
    need(policy['runtime']['python'] == sys.version, 'Pinned Python runtime required')
    need(policy['runtime']['executable_sha256'] == digest(Path(sys.executable).read_bytes()),
         'Pinned Python executable required')
    tests = ev.read(policy['tests'])
    need(tests.get('passed') is True and tests.get('synthetic_only') is True, 'Passing synthetic receipt required')
    same(tests['code_sha256'], policy['code_sha256'], 'Synthetic receipt code mismatch')
    need(type(tests['tests']) is int and tests['tests'] > 0 and tests['failures'] == 0
         and tests['errors'] == 0 and type(tests['terminal']['exit_code']) is int
         and tests['terminal']['exit_code'] == 0
         and type(tests['terminal']['chunk_id']) is str and tests['terminal']['chunk_id'],
         'Observed clean synthetic terminal required')
    # All declared source files are authenticated before any assessment occurs.
    for name, sha in policy['input_sha256'].items():
        ev.hash(name, sha)
    return {role: ev.read(ref) for role, ref in policy['inputs'].items()}


def verify_package(policy, docs, ev):
    refs = policy['inputs']
    for role, document in docs.items():
        ev.declared(document)
    old, legacy, admission = docs['dataset'], docs['legacy_policy'], docs['legacy_admission']
    need(old['policy_sha256'] == admission['policy_sha256'] == refs['legacy_policy']['sha256']
         and admission['data_sha256'] == refs['dataset']['sha256'] and admission['passed'] is True,
         'Legacy DATASET/POLICY/ADMISSION binding')
    for role, key in (('union_data', 'union_data'), ('union_plan', 'union_plan'),
                      ('union_manifest', 'union_manifest'), ('union_review', 'union_independent'),
                      ('sessions', 'sessions'), ('accounting', 'accounting')):
        same(refs[role], legacy['inputs'][key], 'Legacy input reference mismatch: ' + role)
    for ref in legacy['inputs'].values():
        ev.hash(ref['path'], ref['sha256'])
    ev.hash('work/cash_quality_source_admission_v1.py', legacy['implementation_sha256'])
    legacy_tests = ev.read(legacy['inputs']['tests'])
    need(legacy_tests['passed'] is True and legacy_tests['source_sha256'] == legacy['implementation_sha256'],
         'Legacy implementation/tests binding')
    legacy_code_review = ev.read(legacy['independent_code_review'])
    need(legacy_code_review['passed'] is True, 'Legacy independent code review')
    ev.declared(legacy_code_review)
    same(admission['source_hashes'], {r['path']: r['sha256'] for r in legacy['inputs'].values()},
         'Legacy admission source closure')
    li = docs['legacy_review']
    need(li['passed'] is True and li['policy_sha256'] == refs['legacy_policy']['sha256']
         and li['data_sha256'] == refs['dataset']['sha256'] and li['admitted_events'] == 31
         and li['held_events'] == 13, 'Legacy independent review binding')
    union, ur, up, um = (docs[k] for k in ('union_data', 'union_review', 'union_plan', 'union_manifest'))
    need(ur['passed'] is True and ur['data_sha256'] == refs['union_data']['sha256']
         and ur['plan_sha256'] == union['plan_sha256'] == um['plan_sha256'] == refs['union_plan']['sha256']
         and ur['source_manifest_sha256'] == refs['union_manifest']['sha256']
         and um['output_hashes']['RAW_FACTS.json'] == refs['union_data']['sha256'], 'Union data/plan/manifest/review binding')
    for role in ('union_data', 'union_plan', 'union_manifest'):
        need(ur['input_hashes'][refs[role]['path']] == refs[role]['sha256'], 'Union independent input closure')
    old_events, raw_events = unique(old['events']), unique(union['events'])
    need(len(old_events) == 44 and list(old_events) == list(raw_events), 'Exact ordered 44-event census')
    accounting = unique(docs['accounting']['events'])
    need(set(accounting) == set(old_events), 'Accounting census mismatch')
    admitted = []
    for key, event in old_events.items():
        need(set(event) == set(raw_events[key]) | {'cash_quality_admission'}, 'Only legacy admission may be added')
        same({k: v for k, v in event.items() if k != 'cash_quality_admission'}, raw_events[key],
             'Full RAW_FACTS event preservation: ' + key)
        p = event['union_provenance']
        same(p['accounting_event_review'], accounting[key], 'Accounting event binding')
        same(p['accounting_review_reference'], refs['accounting'], 'Accounting source binding')
        if event['cash_quality_admission']['admitted'] is True:
            admitted.append(key)
    need(len(admitted) == 31 and admitted == admission['admitted_event_ids']
         and admission['events'] == 44 and admission['admitted_events'] == 31
         and admission['held_events'] == 13, 'Legacy admission census changed')
    same(admission['holds'], [dict(event_id=k, **v['cash_quality_admission'])
                             for k, v in old_events.items() if k not in admitted], 'Legacy holds changed')
    calendar = docs['sessions']
    sessions = calendar['sessions']
    need(sessions and sessions == sorted(set(sessions)) and all(date.fromisoformat(s).isoformat() == s for s in sessions),
         'Strict pinned session axis required')
    need(calendar['count'] == len(sessions) and calendar['first'] == sessions[0] and calendar['last'] == sessions[-1]
         and calendar['plan_sha256'] == refs['calendar_plan']['sha256'], 'Calendar metadata provenance')
    hr = docs['html_review']
    need(hr['passed'] is True and hr['data_sha256'] == refs['baseline_data']['sha256']
         and hr['plan_sha256'] == refs['baseline_plan']['sha256']
         and hr['source_manifest_sha256'] == refs['baseline_manifest']['sha256'], 'Baseline HTML review binding')
    need(docs['baseline_data']['plan_sha256'] == docs['baseline_manifest']['plan_sha256'] == refs['baseline_plan']['sha256']
         and docs['baseline_manifest']['output_hashes']['RAW_FACTS.json'] == refs['baseline_data']['sha256'],
         'Baseline RAW_FACTS closure')
    for review_role, data_role in (('html_repair_review', 'html_repair_data'), ('pdf_review', 'pdf_data')):
        r = docs[review_role]
        need(r['passed'] is True and r['input_hashes'][refs[data_role]['path']] == refs[data_role]['sha256'],
             'Repaired source review binding')
    pv = docs['pdf_visual']
    need(pv['plan_sha256'] == refs['baseline_plan']['sha256'], 'Visual source plan binding')
    for row in pv['events']:
        for ref in row['inspected_images']:
            ev.hash(ref['path'], ref['sha256'])
    tr = docs['three_review']
    need(tr['passed'] is True and tr['stage'] == 'independent_three_event_raw_source_cell_fact_review_not_admission'
         and tr['original_dataset'] == {'path': refs['dataset']['path'], 'byte_sha256': refs['dataset']['sha256']},
         'New three-event review binding')
    new = unique(tr['events'])
    need(len(new) == 3 and set(new) <= set(old_events) and not set(new).intersection(admitted), 'Exactly three previously held reviews')
    for role in ('dataset', 'sessions'):
        need(tr['input_sha256'].get(refs[role]['path']) == refs[role]['sha256'], 'Three-event required input binding')
    for row in new.values():
        source = row['source']
        for ref in ({k: source[k] for k in ('path', 'sha256')}, source['receipt']):
            need(tr['input_sha256'].get(ref['path']) == ref['sha256'], 'Three-event raw/receipt input binding')
    return old_events, sessions, new


def amount_equal(a, b, message):
    need(type(a) in (str, int) and type(b) in (str, int), message)
    aa, bb = Decimal(a), Decimal(b)
    need(aa.is_finite() and bb.is_finite() and aa == bb, message)


def base_cell(event, side, field, data):
    old = one(data['events'], event_id=event['event_id'])
    cell = event['fields'][side][field]
    same(cell, old['fields'][side][field], 'Complete source cell differs from bound RAW_FACTS')
    same(event['source'], old['source'], 'Source record differs from bound RAW_FACTS')
    need(event['available_from'] == old['available_from'] and event['periods'] == old['periods'], 'Source period/clock mismatch')
    return cell


def html_baseline_cell(event, side, field, docs):
    # The union retains the complete HTML-repair cell, including its two
    # normalization annotations and (where needed) an explicit table segment.
    # First bind that complete object; then bind EVERY old key to the baseline.
    # This is not a generic removal of unknown fields or a fresh raw extraction.
    cell = base_cell(event, side, field, docs['html_repair_data'])
    baseline = one(docs['baseline_data']['events'], event_id=event['event_id'])
    previous = baseline['fields'][side][field]
    same(previous, event['union_provenance']['frozen_v1_existing_cells'][side][field],
         'Frozen baseline cell provenance')
    same(event['source'], baseline['source'], 'Baseline source record mismatch')
    need(event['available_from'] == baseline['available_from'] and event['periods'] == baseline['periods'],
         'Baseline period/clock mismatch')
    additions = {'normalized_monetary_literal', 'comma_whitespace_normalized'}
    need(type(previous) is dict and not additions.intersection(previous)
         and set(cell) == set(previous) | additions, 'Exact HTML normalization field extension')
    display = cell['displayed_amount']
    need(type(display) is str and type(cell['normalized_monetary_literal']) is str
         and cell['normalized_monetary_literal'] == re.sub(r'\s*,\s*', ',', display)
         and type(cell['comma_whitespace_normalized']) is bool
         and cell['comma_whitespace_normalized'] == (cell['normalized_monetary_literal'] != display),
         'HTML comma normalization annotations mismatch')
    projected = {k: deepcopy(cell[k]) for k in previous}
    loc, old_loc = projected['locator'], previous['locator']
    if 'statement_segment' not in old_loc and 'statement_segment' in loc:
        segment = loc.pop('statement_segment')
        need(type(segment) is dict and set(segment) == {'table_index_zero_based', 'start_row', 'end_row_exclusive', 'title'},
             'Exact HTML statement segment fields')
        need(all(type(segment[k]) is int for k in ('table_index_zero_based', 'start_row', 'end_row_exclusive'))
             and segment['table_index_zero_based'] == loc['table_index_zero_based']
             and 0 <= segment['start_row'] <= loc['row_index_zero_based'] < segment['end_row_exclusive']
             and (segment['title'] is None or type(segment['title']) is str), 'HTML statement segment bounds')
    same(projected, previous, 'Complete baseline cell projection differs')
    row = one(docs['html_review']['events'], event_id=event['event_id'])
    same(row['source'], event['source'], 'HTML independent source mismatch')
    check = one(row['cells'], side=side, field=field)
    need(check['matched'] is True and check['status'] == 'matched' and check['failures'] == []
         and check['independent_candidate_count'] == 1, 'HTML independent cell not uniquely matched')
    observed, loc = check['independent_source'], cell['locator']
    for key in ('value_decimal', 'scale'):
        amount_equal(observed[key], cell[key], 'HTML amount/scale mismatch')
    need(observed['currency_literal'] == cell['currency_literal'] == loc['currency_literal'], 'HTML currency mismatch')
    for a, b in (('table_index', 'table_index_zero_based'), ('row_index', 'row_index_zero_based'),
                 ('physical_cell_indices', 'physical_cell_indices_zero_based'), ('logical_columns', 'logical_columns'),
                 ('headers', 'header_literals'), ('row_label', 'row_label'), ('title', 'statement_title')):
        same(observed[a], loc[b], 'HTML independent locator mismatch: ' + a)
    same(observed['parsed_cell_sha256'], cell['parsed_cell_sha256'], 'HTML physical markup binding')
    need(loc['period_end'] == cell['period']['period_end'] == event['periods'][side]['period_end'], 'HTML endpoint mismatch')
    return 'html_review'


def html_repair_cell(event, side, field, docs):
    cell = base_cell(event, side, field, docs['html_repair_data'])
    row = one(docs['html_repair_review']['checked_positions'], event_id=event['event_id'], side=side, field=field)
    need(row['matched'] is True and row['current_release'] == event['release_date']
         and row['available_from'] == event['available_from'], 'Repair review clock')
    same(row['source'], event['source'], 'Repair source mismatch')
    same(row['period'], cell['period'], 'Repair annual period mismatch')
    observed, loc = row['independent_reference'], cell['locator']
    for key in ('value_decimal', 'scale'):
        amount_equal(observed[key], cell[key], 'Repair amount/scale mismatch')
    for key in ('table_index_zero_based', 'row_index_zero_based', 'period_end'):
        same(observed[key], loc[key], 'Repair locator mismatch')
    same([observed['physical_cell_index_zero_based']], loc['physical_cell_indices_zero_based'], 'Repair column mismatch')
    need(observed['raw_markup_sha256'] in cell['parsed_cell_sha256'], 'Repair markup digest mismatch')
    need(observed['ni_start_row'] == loc['row_index_zero_based'] and loc['ni_scope'] == 'indirect_method_consolidated_start',
         'Repair consolidated NI start mismatch')
    return 'html_repair_review'


def pdf_cell(event, side, field, docs):
    cell = base_cell(event, side, field, docs['pdf_data'])
    check = one(docs['pdf_review']['checks'], event_id=event['event_id'], side=side, field=field)
    need(check['visual_match'] is True, 'PDF repaired visual check failed')
    amount_equal(check['value_decimal'], cell['value_decimal'], 'PDF review amount mismatch')
    need(check['displayed_amount'] == cell['displayed_amount'], 'PDF displayed amount mismatch')
    visual = one(docs['pdf_visual']['events'], event_id=event['event_id'])
    same(visual['source'], event['source'], 'PDF visual source mismatch')
    need(visual['column_order'] == ['current', 'comparison'] and visual['monetary_scale'] == 'millions',
         'PDF visual columns/scale')
    col = visual['column_order'].index(side)
    loc = cell['locator']
    need(loc['financial_column_index_0based'] == col and loc['period_end'] == visual['period_end_dates'][col]
         == cell['period']['period_end'] == event['periods'][side]['period_end'], 'PDF date/column mismatch')
    page_key = 'cash_flow_page_1based' if field == 'net_income' else 'balance_sheet_page_1based'
    amount_key = 'visually_observed_cash_flow_starting_net_income' if field == 'net_income' else 'visually_observed_total_assets'
    need(loc['page_1based'] == visual[page_key] and any(p['page_1based'] == loc['page_1based'] for p in visual['inspected_images']),
         'PDF visual page binding')
    amount_equal(cell['value_decimal'], visual[amount_key][col], 'PDF independently observed amount mismatch')
    amount_equal(cell['scale'], '1000000', 'PDF observed million scale mismatch')
    need(cell['currency_literal'] == '$' and loc['currency_literal'] == '$', 'PDF literal dollar unit')
    # Line number and full locator are bound by exact RAW_FACTS; pixels do not claim a line-number observation.
    need(type(loc['line_1based']) is int and loc['line_1based'] > 0, 'Bound PDF physical line required')
    return 'pdf_review'


def new_review_event(event, reviewed, refs):
    same(reviewed['source'], event['source'], 'Three-event full source mismatch')
    need(reviewed['original_event_sha256'] == scope.canonical_sha256(event)
         and reviewed['legacy_cash_quality_admission_sha256'] == scope.canonical_sha256(event['cash_quality_admission']),
         'Three-event complete event/legacy digest mismatch')
    same(reviewed['inherited_rejection_codes_retained'], event['inherited_rejection_codes'], 'Three-event inherited holds mismatch')
    clock = reviewed['clock']
    need(clock['release_date'] == event['release_date'] and clock['available_from'] == event['available_from']
         and clock['calendar_ref'] == refs['sessions']['path'] and clock['first_pinned_session_strictly_after_release'] is True
         and clock['first_publication_time_certified'] is False, 'Three-event clock evidence mismatch')
    period = reviewed['current_period_review']
    for key in ('period_start', 'period_end', 'days_inclusive'):
        same(period[key], event['periods']['current'][key], 'Three-event annual period mismatch')
    need(period['annual_header_observed'] is True and period['opening_is_current_start_minus_one_day'] is True
         and period['opening_assets_as_of'] == event['periods']['comparison']['period_end']
         and period['closing_assets_as_of'] == event['periods']['current']['period_end'], 'Three-event endpoints mismatch')
    need(set(reviewed['fields']) == set(scope.CELL_NAMES), 'Three-event exact field set')
    for side, field in scope.REQUIRED:
        key, cell = side + '.' + field, event['fields'][side][field]
        row = reviewed['fields'][key]
        need(row['original_field_path'] == 'fields.' + key and row['original_cell_sha256'] == scope.canonical_sha256(cell)
             and row['source_path'] == event['source']['path'] and row['source_sha256'] == event['source']['sha256']
             and row['matched_true'] is True and row['same_report_source_and_available_from_match'] is True
             and row['source_amount_sign_scale_and_endpoint_match'] is True, 'Three-event field binding mismatch')
        observed, loc = row['observed'], cell['locator']
        for k in ('value_decimal', 'scale'):
            amount_equal(observed[k], cell[k], 'Three-event observed amount/scale mismatch')
        need(observed['currency_literal'] == cell['currency_literal'] == '$' and observed['currency_iso_certified'] is False
             and observed['as_of'] == loc['period_end'] == cell['period']['period_end'], 'Three-event unit/endpoint mismatch')
        keys = ('page_1based', 'financial_column_index_0based') if loc['format'] == 'pdf' else (
            'table_index_zero_based', 'row_index_zero_based', 'physical_cell_indices_zero_based')
        for k in keys:
            same(observed[k], loc[k], 'Three-event observed physical locator mismatch')
        need(observed['row_label'] == loc['row_label'] and observed['header_literals']
             and observed['period_kind'] == ('duration' if field == 'net_income' else 'instant'), 'Three-event row/period interpretation')


def field_review(event, docs, refs, new_reviews):
    used, mapping = [], []
    if event['event_id'] in new_reviews:
        new_review_event(event, new_reviews[event['event_id']], refs)
        used = ['three_review']
        mapping = [dict(field=name, route='three_review', full_cell_sha256=scope.canonical_sha256(event['fields'][s][f]))
                   for (s, f), name in zip(scope.REQUIRED, scope.CELL_NAMES)]
    elif event['cash_quality_admission']['admitted'] is True:
        for side, name in scope.REQUIRED:
            cell = event['fields'][side][name]
            need(type(cell) is dict, 'Old admitted required cell missing')
            if cell['locator']['format'] == 'pdf':
                route = pdf_cell(event, side, name, docs)
                used.extend(('pdf_data', 'pdf_visual'))
            elif event['union_provenance']['frozen_v1_existing_cells'][side].get(name) is not None:
                route = html_baseline_cell(event, side, name, docs)
                used.extend(('baseline_data', 'html_repair_data'))
            else:
                route = html_repair_cell(event, side, name, docs)
                used.append('html_repair_data')
            used.append(route)
            mapping.append(dict(field=side + '.' + name, route=route, full_cell_sha256=scope.canonical_sha256(cell)))
    else:
        return None, dict(review_available=False, reason='No bound three-field review for this held event', field_mapping=[])
    used = list(dict.fromkeys(used))
    result = dict(schema=scope.REVIEW_SCHEMA, event_id=event['event_id'], event_sha256=scope.canonical_sha256(event),
                  raw_source={k: event['source'][k] for k in ('path', 'sha256')},
                  cell_sha256={s + '.' + f: scope.canonical_sha256(event['fields'][s][f]) for s, f in scope.REQUIRED},
                  review_receipts=[deepcopy(refs[k]) for k in used], independently_reviewed=True)
    return result, dict(review_available=True, field_mapping=mapping,
                       review_sources=[deepcopy(refs[k]) for k in used],
                       scope='Saved exact field mapping; no fresh raw extraction or accounting/chain certification')


def assess_census(old_events, sessions, new_reviews, docs, refs, ev):
    rows, results, regressions = [], [], []
    for key, event in old_events.items():
        before = scope.canonical_sha256(event)
        auth = verify_receipt(event['source'], ev)
        review, mapping = field_review(event, docs, refs, new_reviews)
        decision = scope.assess(event, sessions, review)
        need(scope.canonical_sha256(event) == before, 'Source event mutation')
        if review is None:
            need(decision['admitted'] is False, 'Unreviewed held event must remain held')
        if event['cash_quality_admission']['admitted'] is True and not decision['admitted']:
            regressions.append(dict(event_id=key, reasons=decision['reasons']))
        row = deepcopy(event)
        row['roa_field_scope270'] = decision
        row['roa_field_scope270_evidence'] = dict(authentication=auth, mapping=mapping, field_review=review)
        same({k: v for k, v in row.items() if k not in ('roa_field_scope270', 'roa_field_scope270_evidence')},
             event, 'Output event changed original bytes semantics')
        rows.append(row)
        results.append(dict(event_id=key, prior_admitted=event['cash_quality_admission']['admitted'],
                            admitted=decision['admitted'], reasons=decision['reasons'],
                            review_available=review is not None, event_sha256=before))
    return rows, results, regressions


def write_outputs(root, dataset, assessment, policy_ref, inputs):
    directory = Path(root).resolve() / OUT
    need(not directory.exists(), 'Write-once output directory already exists')
    need(directory.resolve() == directory, 'Output path must not traverse a symlink')
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, value in (('DATASET.json', dataset), ('ASSESSMENT.json', assessment)):
        raw = encoded(value)
        with (directory / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        hashes[name] = digest(raw)
    manifest = dict(schema='roa_field_scope270_source_manifest_v1', policy=policy_ref,
                    input_sha256=inputs, output_sha256=hashes, source_only=True,
                    completed=True, assessment_passed=assessment['passed'])
    with (directory / 'MANIFEST.json').open('xb') as stream:
        stream.write(encoded(manifest)); stream.flush(); os.fsync(stream.fileno())
    return manifest


def run(root, policy_path, policy_sha):
    bootstrap = Evidence(root, {policy_path: policy_sha})
    policy_ref = dict(path=policy_path, sha256=policy_sha)
    policy = bootstrap.read(policy_ref)
    pins = dict(policy['input_sha256'])
    for path, sha in policy['code_sha256'].items():
        need(path not in pins or pins[path] == sha, 'Input/code hash conflict')
        pins[path] = sha
    for ref in (policy['tests'], policy_ref):
        need(ref['path'] not in pins or pins[ref['path']] == ref['sha256'], 'Policy/tests hash conflict')
        pins[ref['path']] = ref['sha256']
    ev = Evidence(root, pins)
    need(not (Path(root) / OUT).exists(), 'Write-once output already exists')
    docs = policy_check(policy, ev)
    events, sessions, new = verify_package(policy, docs, ev)
    rows, results, regressions = assess_census(events, sessions, new, docs, policy['inputs'], ev)
    ev.stable()
    dataset = deepcopy(docs['dataset'])
    dataset['events'] = rows
    dataset['roa_field_scope270_source'] = dict(schema='roa_field_scope270_source_dataset_v1', policy=policy_ref,
                                              original_dataset=policy['inputs']['dataset'], source_only=True,
                                              strict_pit=False, ratios_computed=False, chain_admitted=False)
    assessment = dict(schema='roa_field_scope270_source_assessment_v1', policy=policy_ref,
                      passed=not regressions, events=len(results),
                      admitted_events=sum(r['admitted'] for r in results),
                      held_events=sum(not r['admitted'] for r in results),
                      old_admitted_regressions=regressions, event_results=results,
                      original_event_preservation=True, source_only=True, strict_pit=False,
                      original_legacy_admissions_modified=False, first_public_version_verified=False,
                      source_scope='Raw/receipt byte authentication plus saved independent field evidence; no fresh extraction',
                      counts={k: 0 for k in ZERO}, input_sha256=dict(ev.verified))
    manifest = write_outputs(root, dataset, assessment, policy_ref, dict(ev.verified))
    return dict(passed=assessment['passed'], events=assessment['events'],
                admitted_events=assessment['admitted_events'], held_events=assessment['held_events'],
                old_admitted_regressions=regressions, manifest_sha256=digest(encoded(manifest)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--policy-sha', required=True)
    args = parser.parse_args()
    result = run(ROOT, args.policy, args.policy_sha)
    print(json.dumps(result, sort_keys=True))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
