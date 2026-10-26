"""One-row, no-fit/no-return audit of frozen round178 score interaction.

Run only through ./run_local.sh. Reads fixed registered bytes; writes only a
new score_review/REVIEW.json after exact comparison with the saved orders.
"""
import hashlib
import ast
import importlib.util
import json
import os
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_vol_inflation178'
PREP = OUT / 'prepared'
REPORT = OUT / 'score_review/REVIEW.json'
DATE = '2025-12-31'
PLAN_SHA = '46755ec3f65e62303dced85ba14ce7c9b379303fccafbe1858263aa408691122'
PREP_SHA = '35444e22c1b6ed2d8230eb6b23b6a0d69b4a1e68867193bfe262fa6ce6056c2b'
ARMS = ('incumbent_local67', 'formation_vol_inflation', 'current_vol_inflation')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    require(not REPORT.exists(), 'Append-only audit report already exists')
    require(sha(OUT / 'PLAN.json') == PLAN_SHA, 'Round178 PLAN mismatch')
    require(sha(PREP / 'COMPLETION.json') == PREP_SHA, 'Prepared completion mismatch')
    plan, complete = read(OUT / 'PLAN.json'), read(PREP / 'COMPLETION.json')
    runtime = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    require(runtime == plan['runtime'], 'Registered runtime mismatch')
    require(complete['status'] == 'COMPLETE' and complete['plan_sha256'] == PLAN_SHA,
            'Prepared completion must bind this PLAN')
    registered = {str(Path(p).resolve()): h for p, h in plan['input_sha256'].items()}
    bound = {str(OUT / 'PLAN.json'): PLAN_SHA, str(PREP / 'COMPLETION.json'): PREP_SHA}

    def bind(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, 'SHA mismatch: ' + str(path))
        bound[str(path)] = expected

    # Check code before executing imports. No host gate or archived phase runs.
    for name, digest in registered.items():
        if name.endswith('.py'):
            bind(name, digest)
    for name in ('DATA.npz', 'META.json', 'INCUMBENT_POLICY.json'):
        path = ROOT / 'outputs/monthly_local_linear70' / name
        require(str(path) in registered, 'Unregistered incumbent input')
        bind(path, registered[str(path)])
    for name, digest in complete['artifacts_sha256'].items():
        bind(PREP / name, digest)

    executed_sources = set()
    runtime_sources = {}
    script = Path(__file__).resolve()

    def guard(event, args):
        if event == 'exec':
            filename = args[0].co_filename
            if filename.startswith(str(ROOT)) and filename.endswith('.py'):
                path = str(Path(filename).resolve())
                if path != str(script):
                    library_roots = (ROOT / 'work/venv', ROOT / 'work/monthly-decision-deps',
                                     ROOT / 'work/research-plot-deps')
                    if any(Path(path).is_relative_to(p) for p in library_roots):
                        runtime_sources[path] = sha(path)
                    else:
                        require(path in bound, 'Unbound local code execution: ' + path)
                        executed_sources.add(path)
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            mode, flags = args[1], args[2]
            writes = (isinstance(mode, str) and any(c in mode for c in 'wax+'))
            writes = writes or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writes:
                require(Path(args[0]).resolve() == REPORT, 'Audit attempted an unrelated write')
        if event == 'os.mkdir':
            require(Path(args[0]).resolve() == REPORT.parent, 'Unexpected directory creation')

    sys.addaudithook(guard)
    extracted = {}

    def frozen_functions(relative, names, namespace):
        """Compile only named, unchanged AST nodes from registered source bytes.

        This avoids importing the legacy drivers' unrelated historical imports,
        data loaders, training entry points, and return-calculation functions.
        """
        path = ROOT / relative
        require(str(path) in bound, 'Unbound function source')
        nodes, found = [], set()
        for node in ast.parse(path.read_text(), filename=str(path)).body:
            targets = ({node.name} if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       else {x.id for x in node.targets if isinstance(x, ast.Name)}
                       if isinstance(node, ast.Assign) else set())
            if targets & set(names):
                require(targets <= set(names), 'Ambiguous multiple-assignment extraction')
                nodes.append(node)
                found |= targets
        require(found == set(names), 'Missing frozen AST definitions: ' + str(set(names) - found))
        env = dict(np=np, pd=pd, **namespace)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
        extracted[relative] = list(names)
        return SimpleNamespace(**{name: env[name] for name in names})

    spec = importlib.util.spec_from_file_location('score_review178_frozen_adapter',
                    ROOT / 'outputs/monthly_path70_research/incumbent_adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    ranker = frozen_functions('outputs/monthly_quality_research/explore.py', ['ranks'], {})
    ep = frozen_functions('outputs/monthly_daily_episodes/episodes.py', ['select_boundary'], {})
    ep.old = SimpleNamespace(prior=ranker)
    target_model = frozen_functions('outputs/monthly_target_definition70/model.py', ['TARGETS'], {})
    target = frozen_functions('outputs/monthly_target_definition70/study.py',
        ['TARGETS', 'POLICY_FIELDS', 'configs', 'modified_pre_score'],
        dict(model=target_model, adapter=adapter))
    scope = frozen_functions('outputs/monthly_model_scope70/study.py',
        ['SCOPES', 'POLICY_FIELDS', 'configs', 'modified_pre_score'],
        dict(upstream=target, adapter=adapter))
    study = frozen_functions('outputs/monthly_local_linear70/study.py',
        ['ARMS', 'POLICY_FIELDS', 'configs', 'source_spec', '_shape', 'original_rank_inputs',
         'ranked_forecasts', 'score_delta', 'modified_pre_score'],
        dict(scope=scope, adapter=adapter, ep=ep))
    study.adapter, study.ep = adapter, ep
    study.rp = frozen_functions('outputs/monthly_return_priority/return_priority.py', ['make_orders'], {})

    meta = read(PREP / 'META.json')
    require(meta == read(ROOT / 'outputs/monthly_local_linear70/META.json'), 'Metadata mismatch')
    i = meta['dates'].index(DATE)
    require(i == 113 and len(meta['dates']) == 122, 'Unexpected target row')
    with np.load(ROOT / 'outputs/monthly_local_linear70/DATA.npz', allow_pickle=False) as data:
        require(len(data.files) == 27, 'Expected all 27 frozen original arrays')
        original = {}
        for name in data.files:
            value = data[name]
            require(value.ndim > 0 and value.shape[0] == 122, 'Unexpected axis: ' + name)
            original[name] = value[i:i + 1].copy()
    snapshots = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in original.items()}
    one_meta = dict(meta, dates=[DATE], ends=[meta['ends'][i]], symbols=[meta['symbols'][i]])
    require(original['baseline'].shape == (1, 49), 'Expected one original 49-name row')
    pre = study.modified_pre_score(study.configs()[4], original, 1)
    require(pre.shape == (1, 49), 'Pre-score was not restricted to one event')
    policy = read(ROOT / 'outputs/monthly_local_linear70/INCUMBENT_POLICY.json')
    require(all(policy.get(k) == v for k, v in study.POLICY_FIELDS.items()), 'Policy mismatch')
    orders = read(PREP / 'ORDERS.json')
    audit_row = read(PREP / 'FEATURE_AUDIT.json')['event_history'][i]
    require(audit_row['date'] == DATE and audit_row['symbols'] == one_meta['symbols'][0],
            'Saved feature audit row mismatch')
    raw_by_symbol = {cell['symbol']: cell for cell in audit_row['cells']}
    with np.load(PREP / 'TRANSFORMS_DELTAS.npz', allow_pickle=False) as saved:
        deltas = {ARMS[0]: np.zeros((1, 49))}
        for arm, key in ((ARMS[1], 'candidate'), (ARMS[2], 'control')):
            deltas[arm] = saved['delta_' + arm][i:i + 1].copy()
            transformed = saved['transform_' + arm][i:i + 1]
            require(np.array_equal(deltas[arm], .01 * transformed), 'Saved delta scale mismatch')
            for j, symbol in enumerate(one_meta['symbols'][0]):
                cell = raw_by_symbol[symbol]
                expected = cell[key] if cell['available'] else 0.
                require(transformed[0, j] == expected, 'Saved raw/delta mismatch')

    rows, callback_calls = {}, []
    for arm in ARMS:
        changed_pre = pre.copy()
        mask = deltas[arm] != 0.
        changed_pre[mask] += deltas[arm][mask]
        captured = {}

        def captured_selector(score, eligible, symbols, effects, support_inputs, row):
            callback_calls.append(arm)
            require(score.shape == (1, 49), 'Only the requested event may be selected')
            require(np.array_equal(score, changed_pre), 'Adapter altered pre-score')
            require(np.array_equal(eligible, original['eligible']), 'Eligibility changed')
            require(np.array_equal(effects, original['effects']), 'Episode effect changed')
            for name in ('baseline', 'core', 'state_calm10', 'state_weak_narrow'):
                require(np.array_equal(support_inputs[name], original[name]),
                        'Original support/state changed: ' + name)
            cw = np.broadcast_to(original['core_weight'], (1, 49))
            require(np.array_equal(support_inputs['core_weight'], cw), 'Core weights changed')
            support = (support_inputs['baseline'] - support_inputs['core']) / (1. - cw)
            old_budget = (.025 * support_inputs['state_calm10'] +
                          .05 * support_inputs['state_weak_narrow'])
            budget = np.minimum(row['budget'], np.maximum(0., .10 - old_budget))
            rank = sorted(np.flatnonzero(eligible[0]), key=lambda j: (-score[0, j], symbols[0][j]))
            require(len(rank) >= 3, 'Unexpected undersized eligible universe')
            pool = [j for j in rank[2:row['ceiling']]
                    if score[0, rank[2]] - score[0, j] <= row['gap']]
            adjustments = np.zeros((1, 49))
            for j in pool:
                target = np.clip(support[0, j] + effects[0, j], -.5, .5)
                adjustments[0, j] = budget[0] * (target - support[0, j])
            chosen, final = study.ep.select_boundary(score, eligible, symbols, effects,
                                                    support_inputs, row)
            require(np.array_equal(final, score + adjustments), 'Episode algebra mismatch')
            for j in chosen[0]:
                require(j in rank[:2] or j in pool, 'Unexpected chosen seat')
            captured.update(budget=float(budget[0]), eligible_count=len(rank),
                locked_top_two=[symbols[0][j] for j in rank[:2]],
                boundary_pool=[symbols[0][j] for j in pool],
                cells=[dict(symbol=symbols[0][j], eligible=bool(eligible[0, j]),
                    pre_episode_rank=(rank.index(j) + 1 if j in rank else None),
                    original_pre_score=float(pre[0, j]), direct_delta=float(deltas[arm][0, j]),
                    modified_pre_score=float(score[0, j]),
                    original_support=float(support[0, j]), original_episode_effect=float(effects[0, j]),
                    episode_adjustment=float(adjustments[0, j]),
                    episode_applied=j in pool, selected=j in chosen[0],
                    final_score=float(final[0, j])) for j in range(49)])
            return chosen, final

        chosen, final = study.adapter.score_and_choose('incumbent66', 0.,
            np.full((1, 49), np.nan), original, one_meta, 1,
            pre_score=changed_pre, effects=original['effects'], episode_policy=policy,
            boundary_selector=captured_selector)
        reconstructed = study.rp.make_orders(chosen, final, one_meta)[0]
        require(reconstructed == orders[arm][i], 'Order/score bytes differ from frozen result: ' + arm)
        captured.update(order=reconstructed, frozen_order_exact=True,
                        saved_selected_score_max_abs_error=0.)
        rows[arm] = captured

    require(callback_calls == list(ARMS), 'Exactly one episode call per arm required')
    require(all((v.dtype.str, v.shape, v.tobytes()) == snapshots[k]
                for k, v in original.items()), 'Original single-row inputs mutated')
    summaries = []
    for symbol in ('BABA', 'MU'):
        cell = {arm: next(c for c in rows[arm]['cells'] if c['symbol'] == symbol) for arm in ARMS}
        changes = {}
        for arm in ARMS[1:]:
            score_change = cell[arm]['final_score'] - cell[ARMS[0]]['final_score']
            episode_change = cell[arm]['episode_adjustment'] - cell[ARMS[0]]['episode_adjustment']
            residual = score_change - (cell[arm]['direct_delta'] + episode_change)
            require(abs(residual) < 1e-15, 'Score decomposition mismatch')
            changes[arm] = dict(final_score_change=score_change,
                direct_delta=cell[arm]['direct_delta'], episode_adjustment_change=episode_change,
                decomposition_abs_error=abs(residual))
        summaries.append(dict(symbol=symbol, by_arm=cell, changes=changes))
    require(all(rows[arm]['locked_top_two'] == ['GOOGL', 'BABA'] for arm in ARMS[1:]),
            'Expected mechanism was not established')
    require(rows[ARMS[0]]['locked_top_two'] == ['GOOGL', 'MU'], 'Unexpected incumbent locked seats')
    baba = next(z['by_arm'] for z in summaries if z['symbol'] == 'BABA')
    require(baba[ARMS[0]]['episode_adjustment'] < 0., 'BABA had no original negative adjustment')
    require(all(baba[arm]['direct_delta'] == 0. and baba[arm]['episode_adjustment'] == 0.
                for arm in ARMS[1:]), 'BABA zero-delta/removed-adjustment mechanism did not hold')
    require(all(set(rows[arm]['order']['symbols']) == set(rows[ARMS[0]]['order']['symbols'])
                for arm in ARMS), 'Selected set changed')
    for name, digest in bound.items():
        require(sha(name) == digest, 'Bound input changed during inspection: ' + name)
    report = dict(schema='round178-single-event-score-review-v1', passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
        command='./run_local.sh work/monthly_vol_inflation178_score_review.py',
        script_sha256=sha(script), plan_sha256=PLAN_SHA, prepared_completion_sha256=PREP_SHA,
        registered_runtime=runtime, additionally_imported_runtime_file_sha256=runtime_sources,
        input_sha256=bound, executed_registered_source_files=sorted(executed_sources),
        extracted_unchanged_ast_definitions=extracted,
        preflight_attempts=['First guard stopped before scoring on a venv library import; '
                            'runtime libraries are now classified separately.',
                            'Second guard stopped before scoring on an unrelated unbound legacy '
                            'driver import. Final audit compiles only the named unchanged scoring '
                            'AST definitions from registered source bytes; no guard was relaxed.'],
        event_date=DATE, event_row=i, evaluated_events=1, evaluated_arms=list(ARMS),
        original_array_count=27, original_array_bytes_unchanged=True,
        frozen_inputs_unchanged=True, episode_calls=callback_calls,
        original_eligibility_support_effects_unchanged=True, all_orders_and_scores_exact=True,
        saved_selected_score_max_abs_error=0., policy={k: policy[k] for k in study.POLICY_FIELDS},
        arms=rows, attribution=summaries,
        conclusion=('MU receives a direct volatility penalty and falls below BABA in pre-episode rank. '
                    'BABA moves from boundary third seat to the locked second seat, so its original '
                    'negative episode adjustment no longer applies despite its direct delta being zero. '
                    'MU enters the boundary pool; the original episode contribution is separately '
                    'reported for each name. The selected '
                    'set remains GOOGL/MU/BABA; only its saved ordering and scores change.'),
        limitations=['One fixed historical event only; not a new strategy trial or predictive validation.',
                     'Runtime versions match the PLAN; additionally imported package source hashes '
                     'are observed evidence, distinct from registered strategy source pins.',
                     'Uses previously frozen model forecasts/effects; no fitting, factor recomputation, '
                     'execution replay, prices, portfolio returns, or performance metrics computed.'],
        new_model_fits=0, new_factor_calculations=0, portfolio_returns_calculated=False,
        closed_host_gate_called=False)
    REPORT.parent.mkdir(exist_ok=True)
    with REPORT.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, report=str(REPORT), sha256=sha(REPORT), attribution=summaries,
                          exact_orders_and_scores=True), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
