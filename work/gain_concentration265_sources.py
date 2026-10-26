"""Unregistered265 metadata wrapper; no prices, arrays, returns, or admission."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / 'work/gain_survival264_sources.py'
PARENT_SHA = '2e4c821c1f6caba3ad9366a470a4c61b0740b792947535be0e80e87712054a05'
PROPOSAL_PATH = ROOT / 'work/after264_block_gain_concentration265_PROPOSAL.json'
PROPOSAL_SHA = '04332d6e170cd8d8dcc9fa4297e1136327810cb47d453496b9ffd135fa09a92f'
SOURCE_SCHEMA = 'gain_concentration265_metadata_source_contract_v1'


def _load_parent(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Pinned metadata helper changed')
    spec = importlib.util.spec_from_file_location('gain265_metadata_parent264', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_parent = _load_parent(PARENT_PATH, PARENT_SHA)
INITIAL_PARTIAL, ORIGIN = _parent.INITIAL_PARTIAL, _parent.ORIGIN
PINS = dict(_parent.PINS)
CLOCK_KEYS, IDENTITY_KEYS = _parent.CLOCK_KEYS, _parent.IDENTITY_KEYS
need = _parent.need
read_pin = _parent.read_pin
calendar_days = _parent.calendar_days
clock = _parent.clock
identities = _parent.identities


def dependency_files():
    return tuple(dict.fromkeys((*_parent.dependency_files(), PARENT_PATH,
                                PROPOSAL_PATH, Path(__file__).resolve())))


def build_metadata_contract():
    need(hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest() == PARENT_SHA,
         'Parent metadata helper remains pinned')
    need(hashlib.sha256(PROPOSAL_PATH.read_bytes()).hexdigest() == PROPOSAL_SHA,
         'Prospective geometry remains pinned')
    old = _parent.build_metadata_contract()
    need(old['schema'] == _parent.SOURCE_SCHEMA and old['passed'] is True
         and old['metadata_only'] is True, 'Parent metadata-only evidence')
    for key in ('source_admitted', 'strict_pit', 'prices_loaded', 'npz_decoded',
                'actual_features_orders_returns_computed', 'registered_strategy'):
        need(old[key] is False, 'No actual admission inherited: ' + key)
    for key in ('actual253_slot_coverage', 'original_U0_and_core_parity'):
        need(old[key] == 'UNKNOWN_UNTIL_REGISTERED_PREPARE',
             'Actual coverage remains unverified: ' + key)
    result = deepcopy(old)
    result['schema'] = SOURCE_SCHEMA
    result['parent_metadata_helper'] = dict(
        path=str(PARENT_PATH.relative_to(ROOT)), sha256=PARENT_SHA,
        schema=_parent.SOURCE_SCHEMA, actual264_evidence_used=False)
    result['prospective_proposal'] = dict(
        path=str(PROPOSAL_PATH.relative_to(ROOT)), sha256=PROPOSAL_SHA)
    result['input_sha256'][str(PARENT_PATH.relative_to(ROOT))] = PARENT_SHA
    result['input_sha256'][str(PROPOSAL_PATH.relative_to(ROOT))] = PROPOSAL_SHA
    result['source_policy']['new_statistic'] = (
        'Identical253 formation closes; positive binary64 simple-gain mass '
        'daily concentration versus fixed12x21 block concentration. '
        'No source admission, source shrink, fill, new risk estimator or fit.')
    return result


if __name__ == '__main__':
    target = ROOT / 'work/gain_concentration265_SOURCE_CONTRACT_DRAFT.json'
    need(not target.exists(), 'Metadata draft write-once')
    value = build_metadata_contract()
    with target.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status='UNREGISTERED_SOURCE_METADATA_DRAFT',
                         path=str(target),
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         source_admitted=False, prices_loaded=False,
                         npz_decoded=False)))
