"""Explicit dependency-injected ETF-universe control; source modules unchanged."""
from pathlib import Path
import argparse
import sys

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O / 'monthly_adaptive_core'))
import adaptive as a
ORIGINAL_OUTPUT = a.P
a.P = P


def prepare():
    assert not (P / 'PLAN.json').exists()
    a.initialize()
    original_loader = a.e.load_prices
    original = original_loader()
    theme = sorted(original[2]['theme17'][1])
    core = sorted(original[2]['core17'][1])
    assert len(core) == len(theme) == 17
    plan = a.read(P / 'PLAN.json')
    plan.update(task='ETF-universe sensitivity: rerun identical adaptive-core experiment with broad/sector core17 instead of theme17.',
                pool='Same reconstructed Top32/Top35, no stock additions; core17 replaces theme17 uniformly in every month.',
                etfs=core, replaced_theme_etfs=theme,
                source_limit='Neither full ETF17 list is confirmed by original author. This control is an alternative assumption, not recovered source data.',
                dependency_injection='Only this Python process: temporarily supply a copy of members whose theme17 slot contains core17. Restore loader immediately after preparing packs; output directory redirected to this isolated folder.')
    a.dump('PLAN.json', plan)

    def core_loader():
        px, raw, members, periods, sources, pools = original_loader()
        members = dict(members)
        members['theme17'] = members['core17']
        return px, raw, members, periods, sources, pools

    a.e.load_prices = core_loader
    try:
        a.prepare()
    finally:
        a.e.load_prices = original_loader
    hashes = a.read(P / 'SOURCE_HASHES.json')
    hashes[str(Path(__file__).resolve())] = a.sha(Path(__file__))
    a.dump('SOURCE_HASHES.json', hashes)
    validate_pool()


def validate_pool():
    meta = a.read(P / 'INPUTS.json')
    plan = a.read(P / 'PLAN.json')
    pools = a.read(O / 'monthly_top35_extension/POOLS.json')
    for key, d in meta['packs'].items():
        for p, ss in zip(meta['periods'], d['symbols']):
            expected = {a.e.actual(s) for s in pools[str(d['pool'])][p['start']]} | set(plan['etfs'])
            assert set(ss) == expected
    print('All276 pool/normalization/month lists match uniform core17 control', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['prepare', 'train', 'evaluate', 'validate_pool'])
    phase = ap.parse_args().phase
    if phase in ['train', 'evaluate']:
        getattr(a, phase)()
    else:
        globals()[phase]()
