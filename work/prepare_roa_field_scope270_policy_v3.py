"""Build a new source-only policy from an explicitly reviewed edge design.

Metadata and byte hashing only. Does not call assess, render, or run a census.
No recursive expansion of arbitrary referenced JSON or old runtime manifests.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

import roa_field_scope270_host_v3 as h

OUT = 'work/roa_field_scope270_SOURCE_POLICY_v3.json'


class Collector(h.Evidence):
    """Discover active pins only through the explicit reviewed document scopes."""
    def hash(self, name, expected=None):
        h.need(expected is not None, 'Collection requires a declared expected SHA')
        h.scope._sha(expected)
        h.need(name not in self.pins or self.pins[name] == expected, 'Conflicting active pin: ' + name)
        self.pins[name] = expected
        return super().hash(name, expected)


def collect(root, design_ref, tests_ref, review_ref):
    # Root supplies the independently reviewed design byte digest on the CLI.
    start = Collector(root, {})
    design = start.read(design_ref)
    ev = Collector(root, {}, design['historical_runtime']['edges'])
    design = ev.read(design_ref)
    policy = dict(schema=h.SCHEMA, source_only=True, execution_authorized=True,
        output_directory=h.OUT, expected_events=44, expected_old_admitted=31,
        inputs=deepcopy(design['role_refs']), design=deepcopy(design_ref),
        historical_dependency_edges=deepcopy(ev.edges))
    h.need(set(policy['inputs']) == set(h.ROLES), 'Complete source roles')
    h.design_check(policy, design, ev)
    docs = {k:ev.read(r) for k,r in policy['inputs'].items()}
    for k in h.NEW_ROLES: ev.declared(docs[k])
    # The only document-relative paths are the named renderer output members.
    base = str(Path(policy['inputs']['pdf_manifest']['path']).parent)
    for name, sha in docs['pdf_manifest']['output_sha256'].items():
        h.need(Path(name).name == name, 'Flat render member')
        ev.hash(base + '/' + name, sha)
    rt = docs['pdf_runtime']
    for name, sha in rt['png_encoder_sha256'].items(): ev.hash(name, sha)
    ev.hash(rt['executable_path'], rt['executable_sha256'])
    log = docs['pdf_terminal']['log']; ev.hash(log['path'], log['sha256'])
    tests, review = ev.read(tests_ref), ev.read(review_ref)
    for doc in (tests, review): ev.declared(doc); ev.nested_refs(doc)
    code = {}
    for name in h.CODE_FILES:
        sha = h.digest(ev.path(name).read_bytes()); ev.hash(name, sha); code[name] = sha
    h.need(tests['passed'] is True and tests['synthetic_only'] is True
           and tests['code_sha256'] == code and tests['failures'] == tests['errors'] == 0
           and tests['terminal']['exit_code'] == 0, 'Final synthetic code receipt required')
    h.need(review['passed'] is True and review['unresolved_blockers'] == []
           and review['code_sha256'] == code, 'Final independent code review required')
    # These receipts are byte anchors; their old temporary logs are not new
    # financial sources. Never expand their optional incidental log references.
    for ref in design.get('ancillary_byte_anchors', []): ev.hash(ref['path'], ref['sha256'])
    policy.update(created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=code, tests=deepcopy(tests_ref), independent_code_review=deepcopy(review_ref),
        runtime=dict(python=sys.version, executable_sha256=h.digest(Path(sys.executable).read_bytes())),
        input_sha256=dict(sorted(ev.pins.items())), old_runtime_verified=False,
        strict_pit=False, financial_ratios_computed=False, chain_admitted=False,
        admission_counts_not_forced=True,
        evidence_scope='Historical edges retained verbatim; all other declared dependencies actively hashed. '
            'Ten PDF reports use new render and independent pixel facts; HTML routes and scope_v2 unchanged.')
    # Exercise exactly the runtime policy closure, without source assessment.
    h.policy_check(policy, h.Evidence(root, policy['input_sha256'], ev.edges))
    ev.stable()
    return policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'tests', 'review'):
        p.add_argument('--' + name, required=True); p.add_argument('--' + name + '-sha', required=True)
    p.add_argument('--authorize-census', action='store_true', required=True)
    a = p.parse_args()
    dest = h.ROOT / OUT
    h.need(not dest.exists(), 'Write-once new policy')
    refs = [dict(path=getattr(a, k), sha256=getattr(a, k + '_sha')) for k in ('design', 'tests', 'review')]
    policy = collect(h.ROOT, *refs)
    with dest.open('xb') as f: f.write(h.encoded(policy))
    print(h.old.json.dumps(dict(path=OUT, sha256=h.digest(dest.read_bytes()),
        active_inputs=len(policy['input_sha256']), historical_edges=len(policy['historical_dependency_edges']),
        actual_assess_calls=0)))


if __name__ == '__main__': main()
