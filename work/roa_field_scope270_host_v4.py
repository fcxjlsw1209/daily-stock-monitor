"""Additive metadata correction for three frozen union-PLAN schema examples.

Financial/PDF/HTML/clock/assessment functions are the frozen v3 functions.
Only exact authenticated schema-example objects are not file references.
No generic invalid-digest, placeholder, null or global-path exemption exists.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import os
import sys

import roa_field_scope270_host_v3 as v3

old, scope = v3.old, v3.scope
need, same, digest, encoded = v3.need, v3.same, v3.digest, v3.encoded
ROOT = v3.ROOT
OUT = 'outputs/monthly_roa_field_scope270_source_v4'
SCHEMA = 'roa_field_scope270_source_policy_v4'
CODE_FILES = v3.CODE_FILES + ('work/roa_field_scope270_host_v4.py',
    'work/test_roa_field_scope270_host_v4.py', 'work/prepare_roa_field_scope270_policy_v4.py')
ROLES, NEW_ROLES = v3.ROLES, v3.NEW_ROLES
# Reuse, do not copy or monkeypatch, every financial/source gate.
design_check, pdf_package, assess_census = v3.design_check, v3.pdf_package, v3.assess_census
UNION_PLAN = dict(path='outputs/monthly_cashflow_source_union_v1/PLAN.json',
    sha256='61abd53de8f2c99c2e979d77ff7712ddd9f3536d8fdedb370949240eb5ee75d3')
ACTUAL_BINDING = dict(path='work/cashflow_source_union_v1_bindings.json',
    sha256='88308f4cf4ac5ddaf001be4aec5ab470c01e9cb7d6126058e2f16a97ca282395')
CLASSIFICATION = dict(path='work/roa_field_scope270_schema_examples_v4.json',
    sha256='098aefbba0167c084b110adc79d50620712456af7131c3734f55a5630c92a5f6')
EXAMPLES = (
    ('/policy/binding_document/accounting_review',
     dict(path='work/cashflow_accounting_review_v1.json', sha256='<64hex>')),
    ('/policy/binding_document/producers/html/independent_reviews/0',
     dict(path='<local-relative-review.json>', sha256='<64hex>')),
    ('/policy/binding_document/producers/pdf/independent_reviews/0',
     dict(path='<local-relative-review.json>', sha256='<64hex>')),
)


def example_rows():
    return [dict(json_pointer=p, complete_object=deepcopy(obj),
                 disposition='non_reference_schema_example') for p,obj in EXAMPLES]


def is_schema_example(ref, pointer, obj):
    return ref == UNION_PLAN and any(pointer == p and obj == value for p,value in EXAMPLES)


class Evidence(v3.Evidence):
    def __init__(self, root, pins, edges=()):
        super().__init__(root, pins, edges)
        self.examples_seen = set()

    def nested_refs(self, document):
        entry = self.documents.get(id(document))
        need(entry is not None and entry[0] is document
             and entry[2] == scope.canonical_sha256(document),
             'Authenticated unchanged reference-bearing document required')
        ref = entry[1]
        def walk(value, pointer):
            if type(value) is dict:
                if 'path' in value and 'sha256' in value:
                    if is_schema_example(ref, pointer, value):
                        self.examples_seen.add(pointer)
                        return
                    # _sha remains strict, including None and unknown placeholders.
                    scope._sha(value['sha256'])
                    self.hash(value['path'], value['sha256'])
                for key, child in value.items():
                    walk(child, pointer + '/' + key.replace('~','~0').replace('/','~1'))
            elif type(value) is list:
                for i, child in enumerate(value): walk(child, pointer + '/' + str(i))
        walk(document, '')

    def complete_examples(self):
        need(self.examples_seen == {p for p,_ in EXAMPLES}, 'Complete three schema examples required')
        return dict(declaring_document=deepcopy(UNION_PLAN), examples=example_rows(),
                    disposition='non_reference_schema_examples_only')


def classification_check(policy, ev):
    same(policy['schema_examples'], CLASSIFICATION, 'Exact versioned schema classification byte binding')
    doc = ev.read(CLASSIFICATION)
    need(doc['schema'] == 'roa_field_scope270_schema_examples_v4' and doc['source_only'] is True
         and doc['execution_authorized'] is False and doc['financial_admission'] is False,
         'Classification is not execution or financial authorization')
    same(doc['declaring_document'], UNION_PLAN, 'Exact example declaring document')
    same(doc['examples'], example_rows(), 'Only three exact complete schema examples')
    same(doc['actual_binding'], ACTUAL_BINDING, 'Actual binding remains a distinct active file')
    # The complete_object examples are explanatory data in this pinned list,
    # not a second exemption for arbitrary recursive reference scanning.
    for key in ('schema_generator','failed_v3_attempt'):
        ref=doc['basis'][key]; scope._reference(ref); ev.hash(ref['path'],ref['sha256'])
    return doc


def actual_binding_check(policy, docs, ev):
    same(policy['inputs']['union_plan'], UNION_PLAN, 'Original union PLAN binding')
    plan=docs['union_plan']
    same(dict(path=plan['binding_path'],sha256=plan['binding_sha256']), ACTUAL_BINDING,
         'Actual union binding is not the schema example')
    hashes=plan['source_hashes']
    need(hashes.get(ACTUAL_BINDING['path']) == ACTUAL_BINDING['sha256'], 'Actual binding in PLAN source closure')
    binding=ev.read(ACTUAL_BINDING)
    need(binding['schema_version'] == 1 and binding['stage'] == 'reviewed_source_union_inputs'
         and binding['source_union_authorized'] is True and binding['admission_authorized'] is False,
         'Original binding scope')
    def active(ref):
        scope._reference(ref)
        need(hashes.get(ref['path']) == ref['sha256'], 'Real binding reference must remain in source_hashes')
        ev.hash(ref['path'], ref['sha256'])
    same(binding['accounting_review'],policy['inputs']['accounting'],'Actual accounting review')
    active(binding['accounting_review'])
    need(set(binding['producers']) == {'html','pdf'}, 'Two original source producers')
    for kind, role in (('html','html_repair_review'),('pdf','pdf_review')):
        row=binding['producers'][kind]
        same(row['independent_reviews'],[policy['inputs'][role]], 'Exact actual independent review: '+kind)
        for ref in row['independent_reviews']: active(ref)
        data_role='html_repair_data' if kind=='html' else 'pdf_data'
        need(row['directory'] == str(Path(policy['inputs'][data_role]['path']).parent), 'Original producer directory')
        for name,key in (('PLAN.json','plan_sha256'),('MANIFEST.json','manifest_sha256')):
            active(dict(path=row['directory']+'/'+name,sha256=row[key]))
    ev.nested_refs(binding)
    return dict(actual_binding=deepcopy(ACTUAL_BINDING), real_references_actively_verified=True)


def policy_check(policy, ev):
    need(policy.get('schema') == SCHEMA and policy.get('source_only') is True
         and policy.get('execution_authorized') is True, 'Authorized v4 source-only policy required')
    need(policy['output_directory'] == OUT and policy['expected_events'] == 44
         and policy['expected_old_admitted'] == 31, 'Fixed census/output')
    need(set(policy['inputs']) == set(ROLES) and set(policy['code_sha256']) == set(CODE_FILES),
         'Complete v4 roles/code inventory')
    need(policy['runtime']['python'] == sys.version
         and policy['runtime']['executable_sha256'] == digest(Path(sys.executable).read_bytes()), 'Pinned Python runtime')
    for path, sha in policy['input_sha256'].items():
        ev.hash(path, sha)
    for path, sha in policy['code_sha256'].items(): ev.hash(path, sha)
    classification_check(policy, ev)
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
    actual_binding_check(policy, docs, ev)
    ev.complete_examples()
    return docs


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
    examples = ev.complete_examples()
    rows, results, regressions = assess_census(events, sessions, new, docs, policy['inputs'], ev)
    ev.stable()
    dataset = deepcopy(docs['dataset']); dataset['events'] = rows
    dataset['roa_field_scope270_source_v4'] = dict(policy=pref, original_dataset=policy['inputs']['dataset'],
        source_only=True, strict_pit=False, ratios_computed=False, chain_admitted=False,
        event_scope_fields='Frozen v3 per-event assessment/evidence keys retained verbatim')
    assessment = dict(schema='roa_field_scope270_source_assessment_v4', policy=pref, passed=not regressions,
        events=len(results), admitted_events=sum(r['admitted'] for r in results), held_events=sum(not r['admitted'] for r in results),
        old_admitted_regressions=regressions, event_results=results, original_event_preservation=True,
        old_runtime_verified=False, historical_dependency_edges=history, schema_example_classification=examples, input_sha256=dict(ev.verified),
        pdf_parser_warning_review=deepcopy(docs['pdf_independent_visual'].get('parser_warning_review')),
        source_only=True, strict_pit=False, counts={k:0 for k in old.ZERO})
    out = Path(root)/OUT; need(out.resolve() == out, 'No output symlink'); out.mkdir(parents=True, exist_ok=False)
    members = {}
    for name, doc in (('DATASET.json',dataset),('ASSESSMENT.json',assessment)):
        raw = encoded(doc)
        with (out/name).open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        members[name] = digest(raw)
    manifest = dict(schema='roa_field_scope270_source_manifest_v4', policy=pref, input_sha256=dict(ev.verified),
        historical_dependency_edges=history, schema_example_classification=examples, output_sha256=members, source_only=True, completed=True,
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
