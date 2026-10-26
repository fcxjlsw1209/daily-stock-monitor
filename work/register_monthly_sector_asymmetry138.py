"""Exclusive metadata/hash registration; never computes new state or returns."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

import monthly_sector_asymmetry138_runner as r

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_SHA = '70d04635a591b4bf44de7d9c8574232d22d195edb905d8dff080a367fe8d30e7'
BASE_PLAN_SHA = 'ee5bdeae6fdc81c24394713456641c81df3661895d778ae496817c42c410e683'
LATEST_COMPLETION_SHA = '9a4fba7230426fe351cf426fda4e79c6a5f9f3817cc64a657401382b5a181ff0'
CALENDAR_SHA = '4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'


def build(root, prepare_review, outcome_review):
    root = Path(root).resolve()
    out = root/'outputs/monthly_sector_asymmetry138'
    r.require(not (out/'PLAN.json').exists(), 'Registration already exists')
    hashes = {}

    def add(path, expected=None):
        p = Path(path)
        p = (p if p.is_absolute() else root/p).resolve()
        h = r.sha(p)
        r.require(expected is None or h == expected, 'Changed input: '+str(p))
        r.require(str(p) not in hashes or hashes[str(p)] == h, 'Concurrent input change')
        hashes[str(p)] = h
        return p

    def mapping(m):
        r.require(isinstance(m, dict) and bool(m), 'Nonempty dependency map required')
        for path, digest in m.items():
            add(path, digest)

    state = r.read(root/'outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    r.require(len(state['completed']) == 137 and
              sum(x.get('trials', 0) for x in state['completed']) == 910 and
              not state['registered_queue'], 'Unexpected continuation state')
    add('outputs/monthly_directional_tail137/COMPLETION.json', LATEST_COMPLETION_SHA)
    base_path = add('outputs/monthly_directional_tail137/PLAN.json', BASE_PLAN_SHA)
    old = r.read(base_path)
    mapping(old['hashes'])
    mapping(old['raw_source_hashes'])
    mapping(r.source_files(root))
    add(Path(__file__))
    novelty_path = add('work/monthly_sector_asymmetry138_novelty.json')
    mapping(r.read(novelty_path)['source_hashes'])
    calendar_path = add('outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json', CALENDAR_SHA)
    mapping(r.read(calendar_path)['bindings'])
    inputs = {key: Path(old['inputs'][key]['path']) for key in
              ('meta', 'policy', 'original_orders', 'prices', 'protected_manifest')}
    inputs['proposal'] = add('work/monthly_sector_asymmetry138_proposal.json', PROPOSAL_SHA)
    inputs['sessions'] = add('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json')
    r.require(r.read(inputs['sessions'])['plan_sha256'] == CALENDAR_SHA, 'Session provenance mismatch')
    r.require(set(inputs) == set(r.INPUTS), 'Runner/registrar input schema mismatch')
    refs = {key: dict(path=str(add(path)), sha256=r.sha(path)) for key, path in inputs.items()}
    independent = {}
    reviews = []
    for name, path in (('prepare', prepare_review), ('outcome', outcome_review)):
        path = add(path)
        review = r.read(path)
        r.require(review.get('passed') is True, 'Incomplete '+name+' prereview')
        mapping(review['source_hashes'])
        for section in ('input_hashes', 'implementation_hashes'):
            if review.get(section):
                mapping(review[section])
        expected = 'monthly_sector_asymmetry138_'+name+'_review.py'
        r.require(any(Path(p).name == expected for p in review['source_hashes']), 'Reviewer implementation absent')
        independent[name] = review['source_hashes']
        reviews.append(str(path))
    return dict(round=138, status='REGISTERED', registered_utc=datetime.now(timezone.utc).isoformat(),
        arms=list(r.NAMES), primary_challenger=r.NAMES[2], output_directory=str(out),
        inputs=refs, hashes=hashes, raw_source_hashes=old['raw_source_hashes'],
        independent_prepare_source_hashes=independent['prepare'],
        independent_outcome_source_hashes=independent['outcome'], prerequisite_reviews=reviews,
        runtime=dict(python=sys.version, numpy=r.np.__version__, pandas=r.pd.__version__),
        tolerances=dict(r.TOLERANCES), actual_model_fits_before_registration=0,
        actual_feature_calculations_before_registration=0, actual_returns_before_registration=0,
        new_market_downloads=0, no_observer_writes=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-review', required=True)
    p.add_argument('--outcome-review', required=True)
    a = p.parse_args()
    plan = build(ROOT, a.prepare_review, a.outcome_review)
    out = Path(plan['output_directory'])
    out.mkdir(exist_ok=True, parents=True)
    r.dump(out/'PLAN.json', plan)
    print(json.dumps(dict(path=str(out/'PLAN.json'), sha256=r.sha(out/'PLAN.json'),
                         dependencies=len(plan['hashes']))))


if __name__ == '__main__':
    main()
