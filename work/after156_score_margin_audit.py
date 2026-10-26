"""Read completed156 scores only: explain scope and size, not a new strategy trial."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/monthly_revenue_persistence156'
OUT = ROOT / 'work/after156_score_margin_audit_v2.json'
PINS = {
    'COMPLETION.json': '97177e6413d2227cb727e9efcc027b0cff68084a8c8e8a899f97d02b573665e2',
    'INDEPENDENT_PREPARE.json': '71cde0326c69f1d6e3b8e37f54cc20935bd0ddab5e2cd497fba390c8e54c0604',
}
ARMS = ('incumbent_local67', 'matched_latest_revenue_growth',
        'matched_annualized_total_revenue_growth', 'weakest_annual_revenue_growth')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(name):
    return json.loads((STUDY / name).read_text())

def summarize(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, min=None, median=None, max=None)
    require(np.isfinite(a).all(), 'Nonfinite diagnostic')
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)), max=float(a.max()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    inputs = {}
    for name, expected in PINS.items():
        inputs[name] = sha(STUDY / name)
        require(inputs[name] == expected, 'Pinned completed evidence changed: ' + name)
    review = read('INDEPENDENT_PREPARE.json')
    require(review['passed'] is True, 'Independent preparation not passed')
    inputs['CACHE.json'] = sha(STUDY / 'CACHE.json')
    require(inputs['CACHE.json'] == review['cache_sha256'], 'Reviewed cache changed')
    cache = read('CACHE.json')
    for name in ('DATA.npz', 'META.json', 'PRETRAIN_ORDERS.json', 'SCORES.npz'):
        inputs[name] = sha(STUDY / name)
        require(inputs[name] == cache[name], 'Completed artifact changed: ' + name)
    meta, orders = read('META.json'), read('PRETRAIN_ORDERS.json')
    with np.load(STUDY / 'DATA.npz', allow_pickle=False) as z:
        eligible = z['eligible'].copy()
        available = z['common_available'].copy()
    with np.load(STUDY / 'SCORES.npz', allow_pickle=False) as z:
        scores = {key: z[key].copy() for key in ARMS}
    require(len(meta['dates']) == 122 and eligible.shape == available.shape == (122, 49), 'Original axes')
    require(eligible.dtype == available.dtype == bool, 'Boolean masks')
    require(all(value.shape == (122, 49) for value in scores.values()), 'Score axes')
    base = scores[ARMS[0]]
    require(np.isfinite(base[eligible]).all(), 'Eligible base scores must be finite')
    decision_keys = ('date', 'end', 'signal_date', 'symbols')
    baseline_decisions = [{k: row[k] for k in decision_keys} for row in orders[ARMS[0]]]
    require(len(baseline_decisions) == 122, 'Original decision count')
    for name in ARMS[1:]:
        decisions = [{k: row[k] for k in decision_keys} for row in orders[name]]
        require(decisions == baseline_decisions, 'Completed156 decisions differ; revise diagnosis')
    output = []
    for start, end, label in ((0, 53, 'early53'), (53, 88, 'recent35'), (88, 122, 'later34')):
        active = available[start:end] & eligible[start:end]
        records = []
        gaps = []
        selected_source_cells = 0
        for i in range(start, end):
            symbols = meta['symbols'][i]
            held = orders[ARMS[0]][i]['symbols']
            selected_source_cells += sum(bool(available[i, symbols.index(s)]) for s in held)
            # Saved post-episode matrix ranks; not the underlying pre-episode score.
            ranked = sorted(np.flatnonzero(eligible[i]).tolist(), key=lambda j: (-base[i, j], symbols[j]))
            if len(ranked) >= 4:
                gaps.append(float(base[i, ranked[2]] - base[i, ranked[3]]))
            for j in np.flatnonzero(available[i] & eligible[i]):
                records.append(dict(date=meta['dates'][i], symbol=symbols[j],
                                    saved_post_episode_score_rank=ranked.index(int(j)) + 1,
                                    selected=symbols[j] in held))
        arms = {}
        for name in ARMS[1:]:
            used = scores[name][start:end][active] - base[start:end][active]
            # The saved score matrix is not asserted to be a complete counterfactual response map.
            require(np.isfinite(used).all(), 'Used score changes must be finite')
            arms[name] = dict(changed_order_months=0,
                              nonzero_active_score_cells=int(np.count_nonzero(used)),
                              absolute_saved_score_change=summarize(np.abs(used)))
        output.append(dict(segment=label, months=end-start,
                           available_source_cells=int(available[start:end].sum()),
                           eligible_available_cells=int(active.sum()),
                           original_selected_source_cells=selected_source_cells,
                           eligible_source_saved_ranks=records,
                           saved_post_episode_rank3_minus_rank4_gap=summarize(gaps), arms=arms))
    for name, expected in inputs.items():
        require(sha(STUDY / name) == expected, 'Concurrent artifact modification')
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='posthoc_diagnostic_of_completed156_no_new_counterfactual_orders',
        actual_new_strategy_trials=0, actual_new_factor_calculations=0,
        actual_new_returns_calculated=0, actual_new_model_fits=0,
        completed_orders_only=True, original_cache_artifacts_verified=True,
        diagnostic_repair='First diagnostic attempt incorrectly compared entire saved rows including intentionally different scores; it failed before writing. This version compares ordered symbols and decision dates, while auditing scores separately.',
        supersedes_diagnostic={'path': 'work/after156_score_margin_audit.json',
                               'sha256': sha(ROOT / 'work/after156_score_margin_audit.json'),
                               'reason': 'Independent review clarified saved matrices are post-episode; numerical outputs unchanged, terminology and provenance corrected.'},
        runtime={'python': sys.version, 'numpy': np.__version__, 'network': 'disabled_by_local_wrapper'},
        execution_hashes={name: sha(ROOT/name) for name in
            ('work/after156_score_margin_audit.py', 'run_local.sh', 'local_run.py', 'local.sandbox.sb')},
        caveats=['Saved matrices already include episode effects. Their rank gaps are descriptive, not original pre-score margins or a complete counterfactual selection threshold.',
                 'Saved score changes may include episode candidate-set effects and cannot be attributed solely to direct revenue score additions.',
                 'No proof that a larger dose helps; no new dose or variant was evaluated.',
                 'Sparse source coverage limits this experiment; zero changed holdings does not refute all revenue factors.'],
        input_hashes={str((STUDY/name).relative_to(ROOT)): value for name,value in inputs.items()},
        segments=output)
    if args.write:
        with OUT.open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps(dict(path=str(OUT) if args.write else None,
        segments=[{k:v for k,v in row.items() if k != 'eligible_source_saved_ranks'} for row in output]),
        ensure_ascii=False, allow_nan=False))

if __name__ == '__main__':
    main()
