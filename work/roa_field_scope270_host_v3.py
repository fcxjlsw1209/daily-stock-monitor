"""Additive source-only consumer: exact historical PDF edges, new PDF facts.

No PDF renderer/parser is imported. Frozen scope_v2 and HTML helpers are reused
without changing their globals. A separately hashed policy must authorize census.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import math
import os
import sys

import roa_field_scope270_host as old

scope = old.scope
need, same, digest, encoded, unique, one = old.need, old.same, old.digest, old.encoded, old.unique, old.one
ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_roa_field_scope270_source_v3'
SCHEMA = 'roa_field_scope270_source_policy_v3'
CODE_FILES = old.CODE_FILES + ('work/roa_field_scope270_host_v3.py',
    'work/test_roa_field_scope270_host_v3.py', 'work/prepare_roa_field_scope270_policy_v3.py')
NEW_ROLES = ('runtime_gap', 'pdf_authorized', 'pdf_manifest', 'pdf_evidence',
             'pdf_runtime', 'pdf_terminal', 'pdf_primary', 'pdf_independent_visual')
ROLES = old.ROLES + NEW_ROLES
EDGE_KEYS = ('declaring_document', 'declaration_map', 'declaration_key',
             'declaration_json_pointer', 'runtime_path', 'expected_sha256',
             'disposition', 'replacement_chain_id')


def pointer(group, path):
    return '/' + group + '/' + path.replace('~', '~0').replace('/', '~1')


def edge_key(ref, group, path, sha):
    return (ref['path'], ref['sha256'], group, path, sha)


def edges_check(edges):
    need(type(edges) is list and edges, 'Explicit historical edges required')
    seen = set()
    for e in edges:
        need(type(e) is dict and set(e) == set(EDGE_KEYS), 'Exact historical edge fields')
        scope._reference(e['declaring_document']); scope._sha(e['expected_sha256'])
        need(e['declaration_map'] in old.HASH_GROUPS and e['declaration_key'] == e['runtime_path']
             and e['declaration_json_pointer'] == pointer(e['declaration_map'], e['runtime_path'])
             and e['disposition'] == 'historical_anchor_only'
             and e['replacement_chain_id'] == 'new_pdf_render_and_two_reviews', 'Historical edge semantics')
        key = edge_key(e['declaring_document'], e['declaration_map'], e['runtime_path'], e['expected_sha256'])
        need(key not in seen, 'Duplicate historical edge'); seen.add(key)
    return seen


class Evidence(old.Evidence):
    def __init__(self, root, pins, edges=()):
        super().__init__(root, pins)
        self.edges = deepcopy(list(edges))
        self.allowed = edges_check(self.edges) if edges else set()
        self.historical_paths = {e['runtime_path'] for e in self.edges}
        self.historical_seen = set()
        self.documents = {}

    def path(self, name, external=False):
        # The new render records this exact interpreter; other external paths
        # retain the frozen evidence reader's narrow runtime-root restriction.
        if external and name == str(Path(sys.executable).resolve()):
            return Path(name)
        return super().path(name, external=external)

    def read(self, ref):
        doc = super().read(ref)
        self.documents[id(doc)] = (doc, deepcopy(ref), scope.canonical_sha256(doc))
        return doc

    def declared(self, doc):
        entry = self.documents.get(id(doc))
        need(entry is not None and entry[0] is doc and entry[2] == scope.canonical_sha256(doc),
             'Authenticated unchanged declaring document required')
        ref = entry[1]
        for group in old.HASH_GROUPS:
            if group not in doc:
                continue
            need(type(doc[group]) is dict, 'Declared hash map required')
            for path, sha in doc[group].items():
                key = edge_key(ref, group, path, sha)
                if key in self.allowed:
                    self.historical_seen.add(key)
                else:
                    # Same target in any other document is ACTIVE, not exempt.
                    self.hash(path, sha)

    def complete_history(self):
        need(self.historical_seen == self.allowed, 'Missing declared historical edge')
        return deepcopy(self.edges)

    def nested_refs(self, value):
        """Hash explicit references; never expand referenced JSON transitively."""
        if type(value) is dict:
            if 'path' in value and 'sha256' in value:
                self.hash(value['path'], value['sha256'])
            for child in value.values(): self.nested_refs(child)
        elif type(value) is list:
            for child in value: self.nested_refs(child)


def design_check(policy, design, ev):
    ev.declared(design)
    need(design['schema'] == 'roa_field_scope270_source_v3_design_v1' and design['source_only'] is True
         and design['execution_authorized'] is False and design['financial_admission'] is False,
         'Reviewed source-only design is not itself execution/admission authorization')
    same(design['historical_runtime']['edges'], ev.edges, 'Exact design historical edges')
    same(design['role_refs'], policy['inputs'], 'Design role byte bindings')
    same(design['role_map'], {k:r['path'] for k,r in policy['inputs'].items()}, 'Design role paths')
    need(design['role_count'] == 29 and len(ev.allowed) == 574
         and len({x[0] for x in ev.allowed}) == 7 and len(ev.historical_paths) == 82,
         'Fixed historical 7-document/82-path/574-edge inventory')
    chain = design['replacement_chains']['new_pdf_render_and_two_reviews']
    same(chain, {k:policy['inputs'][k] for k in NEW_ROLES if k != 'runtime_gap'}, 'Exact replacement chain')
    inventory = design['declared_document_inventory']; names = set()
    need(type(inventory) is list and len(inventory) == 29, 'Exact declaring inventory')
    for row in inventory:
        ref = row['document']; need(ref['path'] not in names, 'Duplicate declaring document'); names.add(ref['path'])
        doc = ev.read(ref)
        same(row['expand_top_level_maps'], [k for k in old.HASH_GROUPS if k in doc], 'Exact declared map inventory')
        ev.declared(doc); ev.nested_refs(doc)
    need({policy['inputs'][k]['path'] for k in old.ROLES} <= names, 'All legacy role declarations required')
    ev.complete_history()
    copies = design['new_runtime_source_copy_table']
    doc = ev.read(copies['document'])  # byte anchor, only listed 55 expectations consumed below
    need(copies['map'] == 'pdf_runtime_hashes' and len(copies['entries']) == 55, 'Exact pypdf copy table')
    keys, targets = set(), set()
    for row in copies['entries']:
        key, target, sha = row['declaration_key'], row['active_copy_path'], row['expected_sha256']
        need(key not in keys and target not in targets, 'Duplicate runtime copy')
        keys.add(key); targets.add(target)
        need('/pypdf/' in key and target == 'work/capex193_pdf_runtime/site-packages/pypdf/' + key.split('/pypdf/', 1)[1],
             'Exact isolated pypdf copy path')
        need(doc[copies['map']][key] == sha, 'Historical copy expectation')
        ev.hash(target, sha)
    need(keys == {k for k in doc[copies['map']] if '/pypdf/' in k}, 'Complete original pypdf copy expectations')


def policy_check(policy, ev):
    need(policy.get('schema') == SCHEMA and policy.get('source_only') is True
         and policy.get('execution_authorized') is True, 'Authorized v3 source-only policy required')
    need(policy['output_directory'] == OUT and policy['expected_events'] == 44
         and policy['expected_old_admitted'] == 31, 'Fixed census/output')
    need(set(policy['inputs']) == set(ROLES) and set(policy['code_sha256']) == set(CODE_FILES),
         'Complete v3 roles/code inventory')
    need(policy['runtime']['python'] == sys.version
         and policy['runtime']['executable_sha256'] == digest(Path(sys.executable).read_bytes()), 'Pinned Python runtime')
    for path, sha in policy['input_sha256'].items():
        ev.hash(path, sha)
    for path, sha in policy['code_sha256'].items(): ev.hash(path, sha)
    design = ev.read(policy['design'])
    design_check(policy, design, ev)
    tests, review = ev.read(policy['tests']), ev.read(policy['independent_code_review'])
    need(tests['passed'] is True and tests['synthetic_only'] is True
         and type(tests['tests']) is int and tests['tests'] > 0
         and tests['failures'] == tests['errors'] == 0
         and type(tests['terminal']['exit_code']) is int and tests['terminal']['exit_code'] == 0
         and tests['terminal']['chunk_id'], 'Observed clean synthetic terminal required')
    same(tests['code_sha256'], policy['code_sha256'], 'Tests code binding')
    need(review['passed'] is True and review['unresolved_blockers'] == [], 'Independent code review required')
    same(review['code_sha256'], policy['code_sha256'], 'Review code binding')
    docs = {role: ev.read(ref) for role, ref in policy['inputs'].items()}
    return docs


def ref_equal(a, b, message):
    same(a, b, message)


def pdf_package(policy, docs, ev):
    refs = policy['inputs']; auth = docs['pdf_authorized']; m = docs['pdf_manifest']
    e, v, p, t = (docs[k] for k in ('pdf_evidence', 'pdf_independent_visual', 'pdf_primary', 'pdf_terminal'))
    need(auth['execution_authorized'] is True and auth['financial_admission_authorized'] is False,
         'Separate render authorization required')
    need(m['schema'] == 'roa_field_scope270_pdf_render_manifest_v1'
         and m['render_complete'] is True and m['synthetic_backend'] is False
         and m['financial_admission'] is False and m['old_runtime_verified'] is False, 'Render manifest scope')
    need(m['proposal_sha256'] == e['proposal_sha256'] == refs['pdf_authorized']['sha256'], 'Render proposal binding')
    base = str(Path(refs['pdf_manifest']['path']).parent)
    expected = {'EVIDENCE.json', 'RUNTIME.json'} | {
        'event_%02d_page_%d.%s' % (i, page, ext) for i in range(10) for page in (2, 3) for ext in ('json', 'png')}
    need(set(m['output_sha256']) == expected, 'Exact 42 render members')
    for name, sha in m['output_sha256'].items(): ev.hash(base + '/' + name, sha)
    for role, name in (('pdf_evidence', 'EVIDENCE.json'), ('pdf_runtime', 'RUNTIME.json')):
        ref_equal(refs[role], dict(path=base + '/' + name, sha256=m['output_sha256'][name]), 'Render member reference')
    need(e['schema'] == 'roa_field_scope270_pdf_render_evidence_v1' and e['source_only'] is True
         and e['synthetic_backend'] is False and e['counts'] == dict(actual_assess_calls=0, events=10,
             html_events_reextracted=0, pages_rendered=20, required_fields=30), 'Render evidence scope')
    for flag in ('chain_admitted', 'field_admitted', 'ratios_computed', 'iso_currency_certified', 'old_runtime_verified',
                 'strategy_admitted', 'strict_pit', 'first_public_version_verified'):
        need(e[flag] is False, 'Render evidence claim: ' + flag)
    need(e['runtime_manifest'] == dict(path='RUNTIME.json', sha256=refs['pdf_runtime']['sha256']), 'New runtime binding')
    for role in NEW_ROLES:
        ev.declared(docs[role])
    for name, sha in docs['pdf_runtime']['png_encoder_sha256'].items(): ev.hash(name, sha)
    runtime = docs['pdf_runtime']; ev.hash(runtime['executable_path'], runtime['executable_sha256'])
    for key, role in (('manifest', 'pdf_manifest'), ('evidence', 'pdf_evidence')):
        ref_equal(t[key], refs[role], 'Render terminal binding')
        ref_equal(v[key], refs[role], 'Visual package binding')
    need(t['render_complete'] is True and t['financial_admission'] is False
         and type(t['terminal']['exit_code']) is int and t['terminal']['exit_code'] == 0
         and t['terminal']['chunk_id'] and type(t['terminal']['session_id']) is int, 'Actual render terminal required')
    ev.hash(t['log']['path'], t['log']['sha256'])
    ref_equal(v['owner_render_terminal_receipt'], refs['pdf_terminal'], 'Visual terminal receipt')
    same(v['owner_render_terminal'], t['terminal'], 'Visual actual terminal')
    need(v['schema'] == 'roa_field_scope270_pdf_rerender_visual_review_v1'
         and v['passed'] is True and v['source_only'] is True
         and v['unresolved_field_fact_discrepancies'] == []
         and v['independence']['pixel_review_performed_by_reviewer'] is True
         and v['independence']['other_agent_primary_review_used'] is False
         and v['independence']['parser_text_used_as_substitute_for_viewing'] is False
         and v['independence']['producer_passed_flags_used_as_visual_evidence'] is False, 'Independent visual facts required')
    for flag in ('financial_admission', 'ratios_computed', 'chains_admitted', 'iso_currency_certified', 'strict_pit'):
        need(v[flag] is False, 'Visual scope claim: ' + flag)
    for key, count in (('events', 10), ('images_actually_viewed', 20), ('required_fields', 30),
                       ('visually_matched_fields', 30), ('canonical_full_cells_checked', 30)):
        need(v['counts'][key] == count, 'Visual counts')
    need(p['primary_checks_passed'] is True and p['review_role'] == 'producer_side_self_review'
         and p['independently_reviewed'] is False and p['financial_admission'] is False, 'Primary review limited scope')
    ref_equal(p['evidence_chain']['render_manifest'], refs['pdf_manifest'], 'Primary manifest binding')
    ref_equal(p['evidence_chain']['render_evidence'], refs['pdf_evidence'], 'Primary evidence binding')
    old_events = unique(docs['dataset']['events'])
    rendered, visual, primary, authorized = unique(e['events']), unique(v['events']), unique(p['events']), unique(auth['events'])
    pdf_ids = [key for key, row in old_events.items() if row['fields']['current']['net_income'] is not None
               and row['fields']['current']['net_income']['locator']['format'] == 'pdf']
    need(len(pdf_ids) == 10 and list(rendered) == list(visual) == list(primary) == list(authorized) == pdf_ids,
         'Exact original ten PDF reports')
    need(len(v['pages']) == 20, 'Exact twenty visual pages')
    for key in pdf_ids:
        event, rr, vr = old_events[key], rendered[key], visual[key]
        same(authorized[key]['source'], event['source'], 'Authorized original source')
        same(authorized[key]['pages_1based'], [2,3], 'Authorized exact pages')
        need(rr['original_event_sha256'] == vr['event_sha256'] == scope.canonical_sha256(event), 'PDF full event hash')
        same(rr['source'], event['source'], 'Rendered original source'); same(vr['source'], event['source'], 'Visual original source')
        need(set(rr['fields']) == set(scope.CELL_NAMES)
             and {x['field_path'] for x in vr['cells']} == set(scope.CELL_NAMES) and len(vr['cells']) == 3,
             'Exact PDF three fields')
        for page in (2, 3):
            pr = one(v['pages'], event_id=key, page_1based=page)
            need(pr['pixel_review_performed'] is True and pr['required_headers_legible'] is True
                 and pr['required_value_rows_legible'] is True and pr['required_value_or_date_contradiction_found'] is False
                 and pr['financial_admission'] is False, 'Complete independent page facts')
            dates = [old.date.fromisoformat(event['periods'][s]['period_end']) for s in ('current','comparison')]
            headers = [x.strftime('%B ') + str(x.day) + x.strftime(', %Y') for x in dates]
            same(pr['observed_header_dates_left_to_right'], headers, 'Visual endpoint headers')
            need(pr['observed_unit_literal'] == 'In millions' and pr['required_currency_glyph_pixels'] == '$'
                 and pr['annual_header_directly_observed'] == ('Twelve Months Ended' if page == 3 else None), 'Visual annual/unit header')
            selected = one(vr['cells'], field_path='current.net_income' if page == 3 else 'current.total_assets')
            same(pr['image'], selected['new_image'], 'Page image/cell consistency')
            same(pr['text_geometry'], selected['new_text_geometry'], 'Page text/cell consistency')
            page_doc = ev.read(pr['text_geometry'])
            for textkey, lines in pr['dual_parser_header_lines'].items():
                need(textkey in ('pypdf_layout_text','pdfium_text') and type(lines) is list and lines, 'Header parser scope')
                body = page_doc[textkey].splitlines()
                for line in lines:
                    j = line['line_1based']
                    need(type(j) is int and 1 <= j <= len(body) and body[j-1] == line['literal_text'], 'Saved header text location')
            need(set(pr['dual_parser_header_lines']) == {'pypdf_layout_text','pdfium_text'}, 'Both parser headers required')
        for side, field in scope.REQUIRED:
            pdf_cell(event, side, field, docs, refs, ev)
    return rendered, visual


def pdf_cell(event, side, field, docs, refs, ev):
    cell = old.base_cell(event, side, field, docs['pdf_data'])
    key = side + '.' + field; sha = scope.canonical_sha256(cell); loc = cell['locator']
    rr = one(docs['pdf_evidence']['events'], event_id=event['event_id'])['fields'][key]
    vr = one(docs['pdf_independent_visual']['events'], event_id=event['event_id'])
    c = one(vr['cells'], field_path=key)
    primary = one(docs['pdf_primary']['events'], event_id=event['event_id'])
    need(primary['original_event_sha256'] == scope.canonical_sha256(event)
         and set(primary['fields']) == set(scope.CELL_NAMES), 'Primary event and field scope')
    same(primary['source'], event['source'], 'Primary original source')
    pc = primary['fields'][key]
    need(rr['original_cell_sha256'] == vr['cell_sha256'][key] == c['original_cell_sha256'] == sha, 'PDF cell canonical binding')
    same(rr['original_cell'], cell, 'Complete PDF cell preservation')
    same(rr['original_locator'], loc, 'Complete PDF locator preservation')
    need(c['matched'] is True and c['observed_from_pixels'] is True
         and c['physical_row_and_column_visually_matched'] is True
         and c['dual_parser_amount_tokens_match_visual'] is True
         and c['consolidated_scope_observed'] is True, 'PDF observed fact assertions')
    old.amount_equal(c['value_decimal'], cell['value_decimal'], 'New PDF amount')
    old.amount_equal(c['scale'], cell['scale'], 'New PDF scale')
    need(c['observed_amount_literal'] == cell['displayed_amount'] and c['scale'] == '1000000'
         and c['observed_unit_literal'] == 'In millions' and c['currency_literal'] == cell['currency_literal'] == '$'
         and c['iso_currency'] is None and cell['currency_iso'] is None and c['iso_currency_certified'] is False,
         'New PDF literal units')
    col, page = (1 if side == 'comparison' else 0), (3 if field == 'net_income' else 2)
    need(c['financial_column_index_0based'] == loc['financial_column_index_0based'] == col
         and c['page_1based'] == loc['page_1based'] == page
         and c['period_end'] == loc['period_end'] == cell['period']['period_end'] == event['periods'][side]['period_end']
         and c['observed_row_label'] == loc['row_label'], 'New PDF physical position/endpoint')
    need(c['period_kind'] == ('annual_duration' if field == 'net_income' else 'instant')
         and c['exact_period_start_directly_printed_and_reverified'] is False
         and c['field_admitted'] is False and c['chain_admitted'] is False, 'New PDF period/scope')
    need(pc['original_cell_sha256'] == sha and pc['observed_value_decimal'] == c['value_decimal']
         and pc['displayed_amount'] == c['observed_amount_literal'] and pc['scale'] == c['scale']
         and pc['currency_literal'] == '$' and pc['currency_iso'] is None
         and pc['page_1based'] == page and pc['financial_column_index_0based'] == col
         and pc['observed_endpoint'] == c['period_end'] and pc['instant_or_duration'] == c['period_kind']
         and pc['independently_reviewed'] is False and pc['field_admitted'] is False
         and pc['currency_iso_certified'] is False, 'Primary field facts/boundary')
    same(pc['stored_period_preserved'], cell['period'], 'Primary original period preserved')
    same(pc['header_endpoints'], [event['periods'][s]['period_end'] for s in ('current','comparison')], 'Primary two endpoints')
    for flag in ('amount_matches_original','physical_column_matches_original','units_match_original',
                 'endpoint_matches_original','annual_or_instant_semantics_match','complete_original_cell_equal_to_dataset'):
        need(pc[flag] is True, 'Primary cell fact: '+flag)
    if field == 'net_income':
        need(c['observed_duration_header'] == 'Twelve Months Ended' and c['net_income_scope']
             and loc['ni_scope'] == 'indirect_method_consolidated_start', 'Annual indirect NI scope')
    else:
        need(c['observed_duration_header'] is None and c['net_income_scope'] is None, 'Instant asset scope')
    base = str(Path(refs['pdf_manifest']['path']).parent)
    for k, name in (('new_image', 'image'), ('new_text_geometry', 'text_geometry')):
        ref = rr['render_evidence'][name]
        same(pc[name], ref, 'Primary render member binding')
        ref_equal(c[k], dict(path=base + '/' + ref['path'], sha256=ref['sha256']), 'New visual page binding')
        ev.hash(c[k]['path'], c[k]['sha256'])
    # Check saved parser rows, not merely the review's passed bit. No new PDF decoding.
    page_doc = ev.read(c['new_text_geometry'])
    need(page_doc['event_id'] == event['event_id'] and page_doc['page_1based'] == page, 'Page JSON identity')
    same(page_doc['image'], rr['render_evidence']['image'], 'Page JSON image binding')
    geometry_check(pc, page_doc, col, c['pdfium_row']['numeric_literals_in_original_row_order'])
    for source, text_key in (('pypdf_row', 'pypdf_layout_text'), ('pdfium_row', 'pdfium_text')):
        r = c[source]; lines = page_doc[text_key].splitlines()
        need(type(r['line_1based']) is int and 1 <= r['line_1based'] <= len(lines)
             and lines[r['line_1based'] - 1] == r['literal_text'], 'Actual saved parser row binding')
        tokens = old.re.findall(r'\d+(?:,\d{3})*', r['literal_text'])
        need(tokens == r['numeric_literals_in_original_row_order'] and len(tokens) == 2
             and tokens[col] == c['observed_amount_literal'], 'Saved amount token/column')
    need(pc['raw_pdfium_row'] == c['pdfium_row']['literal_text'], 'Primary PDFium row binding')
    # The primary's optional new_pypdf_line_matches_old claim is not consumed.
    # We check the independent review's location in the NEW bound text above,
    # without claiming a fresh validation of the historical locator line.
    return 'pdf_independent_visual'


def geometry_check(primary_cell, page, column, tokens):
    """Bind saved producer geometry to saved renderer boxes, not a new visual measurement."""
    boxes = {}
    for box in page['nonwhitespace_character_boxes']:
        i = box['index_0based']
        need(type(i) is int and i >= 0 and i not in boxes, 'Unique page character index')
        boxes[i] = box
    columns = primary_cell['geometry_columns']
    need(type(columns) is list and len(columns) == 2, 'Exactly two physical amount columns')
    used = set()
    for item, token in zip(columns, tokens):
        indices = item['char_indices']
        need(type(indices) is list and indices and all(type(i) is int and i in boxes for i in indices)
             and indices == sorted(set(indices)) and not used.intersection(indices), 'Unique valid amount character indices')
        used.update(indices)
        chars = [boxes[i] for i in indices]
        need(all(type(x['text']) is str and x['text'] for x in chars)
             and ''.join(x['text'] for x in chars) == item['text'] == token, 'Saved amount geometry text')
        for x in chars:
            b = x['bbox_pdf_units']
            need(type(b) is list and len(b) == 4 and all(type(v) in (float,int) and math.isfinite(v) for v in b)
                 and b[0] <= b[2] and b[1] <= b[3], 'Finite character box')
        expected = [min(x['bbox_pdf_units'][0] for x in chars), min(x['bbox_pdf_units'][1] for x in chars),
                    max(x['bbox_pdf_units'][2] for x in chars), max(x['bbox_pdf_units'][3] for x in chars)]
        same(item['bbox'], expected, 'Saved amount enclosing box')
    need(columns[0]['bbox'][2] < columns[1]['bbox'][0], 'Current/comparison physical left-right order')
    same(primary_cell['selected_geometry'], columns[column], 'Selected physical column geometry')


def field_review(event, docs, refs, new_reviews, ev):
    is_pdf = any(type(event['fields'][s].get(f)) is dict and event['fields'][s][f]['locator']['format'] == 'pdf'
                 for s, f in scope.REQUIRED)
    if not is_pdf:
        return old.field_review(event, docs, refs, new_reviews)
    if event['event_id'] in new_reviews:
        old.new_review_event(event, new_reviews[event['event_id']], refs)
    else:
        need(event['cash_quality_admission']['admitted'] is True, 'No unsupported PDF review route')
    for side, field in scope.REQUIRED: pdf_cell(event, side, field, docs, refs, ev)
    used = ['pdf_data', 'pdf_authorized', 'pdf_manifest', 'pdf_evidence', 'pdf_runtime',
            'pdf_terminal', 'pdf_primary', 'pdf_independent_visual']
    if event['event_id'] in new_reviews: used.append('three_review')
    review = dict(schema=scope.REVIEW_SCHEMA, event_id=event['event_id'], event_sha256=scope.canonical_sha256(event),
        raw_source={k:event['source'][k] for k in ('path','sha256')},
        cell_sha256={s+'.'+f:scope.canonical_sha256(event['fields'][s][f]) for s,f in scope.REQUIRED},
        review_receipts=[deepcopy(refs[k]) for k in used], independently_reviewed=True)
    mapping = dict(review_available=True, field_mapping=[dict(field=s+'.'+f, route='pdf_independent_visual',
        full_cell_sha256=scope.canonical_sha256(event['fields'][s][f])) for s,f in scope.REQUIRED],
        review_sources=deepcopy(review['review_receipts']), scope='New PDF fact review only; retained clock/identity and scope_v2 rules')
    return review, mapping


def assess_census(events, sessions, new_reviews, docs, refs, ev):
    rows, results, regressions, prepared = [], [], [], []
    # Complete ALL raw/receipt/mapping checks before the FIRST scope assessment.
    # A late bad HTML report or receipt must never leave partial assessments.
    for key, event in events.items():
        before = scope.canonical_sha256(event)
        auth = old.verify_receipt(event['source'], ev)
        review, mapping = field_review(event, docs, refs, new_reviews, ev)
        need(before == scope.canonical_sha256(event), 'Original event mutation during evidence preflight')
        prepared.append((key,event,before,auth,review,mapping))
    ev.stable()
    for key,event,before,auth,review,mapping in prepared:
        need(before == scope.canonical_sha256(event), 'Original event mutation before assessment')
        decision = scope.assess(event, sessions, review)
        need(before == scope.canonical_sha256(event), 'Original event mutation')
        if review is None: need(decision['admitted'] is False, 'Unreviewed event must remain held')
        if event['cash_quality_admission']['admitted'] is True and not decision['admitted']:
            regressions.append(dict(event_id=key, reasons=decision['reasons']))
        row = deepcopy(event); row['roa_field_scope270_v3'] = decision
        row['roa_field_scope270_v3_evidence'] = dict(authentication=auth, mapping=mapping, field_review=review)
        same({k:v for k,v in row.items() if k not in ('roa_field_scope270_v3','roa_field_scope270_v3_evidence')}, event,
             'Full original event preservation')
        rows.append(row); results.append(dict(event_id=key, prior_admitted=event['cash_quality_admission']['admitted'],
            admitted=decision['admitted'], reasons=decision['reasons'], review_available=review is not None, event_sha256=before))
    return rows, results, regressions


def run(root, policy_path, policy_sha):
    pref = dict(path=policy_path, sha256=policy_sha)
    policy = old.Evidence(root, {policy_path:policy_sha}).read(pref)
    pins = dict(policy['input_sha256'])
    for path, sha in policy['code_sha256'].items():
        need(path in pins and pins[path] == sha, 'Code must be active policy input')
    pins[policy_path] = policy_sha
    ev = Evidence(root, pins, policy['historical_dependency_edges'])
    need(not (Path(root)/OUT).exists(), 'Write-once output directory')
    docs = policy_check(policy, ev)
    pdf_package(policy, docs, ev)
    events, sessions, new = old.verify_package(policy, {k:docs[k] for k in old.ROLES}, ev)
    history = ev.complete_history()
    rows, results, regressions = assess_census(events, sessions, new, docs, policy['inputs'], ev)
    ev.stable()
    dataset = deepcopy(docs['dataset']); dataset['events'] = rows
    dataset['roa_field_scope270_source_v3'] = dict(policy=pref, original_dataset=policy['inputs']['dataset'],
        source_only=True, strict_pit=False, ratios_computed=False, chain_admitted=False)
    assessment = dict(schema='roa_field_scope270_source_assessment_v3', policy=pref, passed=not regressions,
        events=len(results), admitted_events=sum(r['admitted'] for r in results), held_events=sum(not r['admitted'] for r in results),
        old_admitted_regressions=regressions, event_results=results, original_event_preservation=True,
        old_runtime_verified=False, historical_dependency_edges=history, input_sha256=dict(ev.verified),
        pdf_parser_warning_review=deepcopy(docs['pdf_independent_visual'].get('parser_warning_review')),
        source_only=True, strict_pit=False, counts={k:0 for k in old.ZERO})
    out = Path(root)/OUT; need(out.resolve() == out, 'No output symlink'); out.mkdir(parents=True, exist_ok=False)
    members = {}
    for name, doc in (('DATASET.json',dataset),('ASSESSMENT.json',assessment)):
        raw = encoded(doc)
        with (out/name).open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        members[name] = digest(raw)
    manifest = dict(schema='roa_field_scope270_source_manifest_v3', policy=pref, input_sha256=dict(ev.verified),
        historical_dependency_edges=history, output_sha256=members, source_only=True, completed=True,
        old_runtime_verified=False, assessment_passed=not regressions)
    with (out/'MANIFEST.json').open('xb') as f: f.write(encoded(manifest)); f.flush(); os.fsync(f.fileno())
    return dict(passed=not regressions, events=len(results), admitted_events=assessment['admitted_events'],
        held_events=assessment['held_events'], old_admitted_regressions=regressions, manifest_sha256=digest(encoded(manifest)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True); parser.add_argument('--policy-sha', required=True)
    a = parser.parse_args(); result = run(ROOT, a.policy, a.policy_sha)
    print(json.dumps(result, sort_keys=True))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
