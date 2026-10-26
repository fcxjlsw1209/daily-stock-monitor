"""Three fixed annual-lag arms; real phases require a registered, pinned PLAN.

No market or source data is loaded on import. The two-issuer source pilot does
not inherit round157 breadth gates. There is exactly one first88 selection;
prefix checks concern feature/score/order invariance, never model selection.
"""
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_formation142_runner as prior
import monthly_annual_lag158_model as model

require, sha, read, dump = base.require, base.sha, base.read, base.dump
TRAIN, MONTHS, WIDTH, DOSE = 88, 122, 49, .01
NAMES = ('incumbent_local67', 'negative_current_annual_lag', 'negative_annual_lag_change')
EXTRA = model.EXTRA
OUT = 'outputs/monthly_annual_lag158'
DESIGN_PATH = 'work/after157_lag_pilot_design.json'
DESIGN_SHA = '215ef056f8ac8fb3ac19e52acc49bd983802bb13cddec0b103826c85640055d5'
PREVIOUS_COMPLETION = 'outputs/monthly_filing_association157/COMPLETION.json'
PREVIOUS_COMPLETION_SHA = 'f4de303fc011a336865519d82b48219833eb0953cc9a8dbed0849f3be2354549'
MODEL_SHA = '07e67f499a9445ae2d2ed53d08721d946eb7a0a360909739c8164b7535ceb04f'
PROTECTED_SHA, LOCAL_SHA = base.PROTECTED_SHA, base.LOCAL_SHA
TOLERANCES = dict(prior.TOLERANCES)
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'protected_manifest',
          'design', 'calendar', 'source_overlay', 'source_review', 'source_prior')
FIXED_INPUTS = {
    'design': (DESIGN_PATH, DESIGN_SHA),
    'meta': ('outputs/monthly_local_linear70/META.json', '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4'),
    'calendar': ('outputs/monthly_paired_target143/CALENDAR.json', '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'),
    'source_overlay': ('outputs/after157_walmart_context_admission_v1/ADMISSION.json', '9e90889a1fb4f057438b6ac16a9e234f97cc9ede08c6cb28a6fe190389f3f9d6'),
    'source_review': ('work/after157_walmart_context_independent.json', '16a666e2d72426a9baa92e693fc003d8c107f9908032762d61232c151dc18814'),
    'source_prior': ('work/after157_annual_pair_source_feasibility.json', '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242')}
HISTORICAL_STATE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
HISTORICAL_SHA = 'f0f6571a3f4de4be06aeb957fe03fd13eed11f86b288abafc91094db38ac32b7'
HISTORICAL_ARCHIVE = 'outputs/monthly_target70_research/PRE157_' + HISTORICAL_SHA + '.json'
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json',
               'FEATURE_AUDIT.json', 'CALENDAR.json', 'RANK_AUDIT.json')
FROZEN_FILES = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json',
                'TRAIN_OPEN_TRIALS.json', 'INDEPENDENT_PREPARE.json')
CHECKS = ('original27_arrays_byte_identical', 'raw_prices_independently_verified',
    'calendar_overlay_independently_verified', 'annual_lags_independently_reconstructed',
    'shared_source_eligible_mask_and_centered_ranks_verified',
    'all_three_score_matrices_and_ordered_choices_independently_verified',
    'all_signal_prefixes_verified', 'initial_partial_and_missing_exact_fallback_verified',
    'one_episode_and_top2_lock_verified', 'full_5978_source_rows_and_cutoffs_verified',
    'source_contradictions_hard_failed')
SOURCE_CHECKS = ('all_5978_exact_rows_independently_reconstructed', 'all_original_fields_and_flags_preserved',
    'all_overlay_input_hashes_verified', 'all_twelve_original_chains_verified', 'complete_eleven_wmt_pairs_verified',
    'exact_publisher_links_verified', 'fy2024_discrepancy_retained_and_citations_corrected',
    'input_hashes_unchanged_after_audit', 'ko_exact_pair_replay_and_inheritance_verified', 'no_producer_imports',
    'only_specific_context_blocker_resolved', 'original_tier3_zero_preserved',
    'other_timing_identity_amendment_selection_gates_preserved',
    'reviewed_originals_match_frozen_metadata_and_raw_source_rows', 'segment_summaries_independently_recomputed',
    'stock3904_etf2074_axes_preserved', 'wmt_may2018_still_blocked')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def measurement_contract():
    return dict(design_sha256=DESIGN_SHA, fixed_issuers=dict(model.ISSUERS),
        current='-(filingDate_current-reportDate_current).days',
        change='previous_lag_days-current_lag_days', source_cutoff='original_XNYS_close_at_t_minus_42',
        availability='second_XNYS_session_strictly_after_filingDate',
        common_mask='conditional_overlay_and_original_eligible_and_valid_exact_selected_pair',
        invalid_latest_blocks_fallback=True, contradictory_admitted_metadata='hard_failure_even_if_ineligible',
        rank='ascending_average_rank0_over_n_minus1_minus_half', min_rank_population=2,
        dose=DOSE, max_absolute_correction=.005, original_arrays=27, extra_arrays=list(EXTRA),
        score_placement='unchanged_local67_pre_plus_dose_rank_then_one_episode',
        first_partial='2016-08-17_exact_incumbent_fallback', additional_breadth_floor=None,
        new_predictive_model_fits=0, nuisance_estimates=0, exact_acceptance_certified=False)


def selection_contract():
    return dict(primary=NAMES[2], controls=list(NAMES[:2]), first_months=88, selection_count=1,
        same_close='strict_cagr_over_both_controls', next_open_5bp='noninferior_incumbent_strict_over_current_lag',
        cagr_epsilon=1e-12, sharpe_tolerance=.05, max_drawdown_tolerance=.03,
        early_cagr_tolerance=.025, recent_cagr_tolerance=.025, prefix_reselection=False,
        later34_diagnostic_only=True, parameter_search=False)


def validate_proposal(proposal):
    require(proposal.get('round') == 158 and proposal.get('arms') == list(NAMES) and
            proposal.get('primary_challenger') == NAMES[2] and
            proposal.get('status') == 'FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION', 'Wrong final proposal')
    require(proposal.get('measurement') == measurement_contract() and
            proposal.get('selection_contract') == selection_contract(), 'Changed annual-lag contracts')
    require(proposal.get('design') == dict(path=DESIGN_PATH, sha256=DESIGN_SHA) and
            proposal.get('previous_completion') == dict(path=PREVIOUS_COMPLETION, sha256=PREVIOUS_COMPLETION_SHA),
            'Changed design/predecessor')
    for key in ('actual_new_lag_values', 'actual_new_features', 'actual_new_model_fits', 'actual_new_orders', 'actual_new_returns'):
        require(type(proposal.get(key)) is int and proposal[key] == 0, 'Premature actual work: ' + key)


def guard(candidate, control, strict=True):
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    values = [row.get(k, {}).get(s) for row in (candidate, control) for k, s in fields]
    if not all(isinstance(v, (int, float, np.integer, np.floating)) and
               not isinstance(v, (bool, np.bool_)) for v in values) or not np.isfinite(values).all():
        return False
    v, b = values[:5], values[5:]
    return bool((v[0] > b[0]+1e-12 if strict else v[0] >= b[0]-1e-12) and
                all(x >= y-t for x, y, t in zip(v[1:], b[1:], (.05, .03, .025, .025))))


def select(rows, open_rows):
    require(all([r['name'] for r in group] == list(NAMES) for group in (rows, open_rows)), 'Three exact arms required')
    close = {NAMES[i]: guard(rows[2], rows[i]) for i in (0, 1)}
    opened = {NAMES[i]: guard(open_rows[2], open_rows[i], strict=i == 1) for i in (0, 1)}
    passed = all(close.values()) and all(opened.values())
    return dict(primary=NAMES[2] if passed else NAMES[0], admitted=passed, guards=passed,
                guards_against=close, next_open_guards_against=opened, mechanism_confirmed=all(opened.values()))


def validate_source_rows(meta, overlay, prior_data, calendar):
    """Metadata-only preflight, including admitted rows outside current eligibility."""
    base.validate_meta(meta)
    rows, originals = overlay['rows'], prior_data['rows']
    require(len(rows) == len(originals) == MONTHS*WIDTH, 'Full5978 source denominator required')
    require(overlay.get('schema') == 'after157_ko_wmt_conditional_annual_source_v1', 'Wrong source overlay schema')
    for key in ('source_admission_changed', 'old_admission_flags_changed', 'full_strategy_admission',
                'numerical_experiment_registered', 'lag_or_feature_values_computed'):
        require(overlay.get(key) is False, 'Unexpected source activity/admission')
    sessions, closes = model._calendar(calendar)
    positions = {day: k for k, day in enumerate(sessions)}
    events = prior_data['selected_original_and_amendment_metadata']
    mask = np.zeros((MONTHS, WIDTH), dtype=bool)
    for position, (item, original) in enumerate(zip(rows, originals)):
        i, j = divmod(position, WIDTH)
        require(canonical_hash(item['original_row']) == canonical_hash(original), 'Original source row changed')
        require(type(original['signal_index']) is int and type(original['candidate_index']) is int and
                (original['signal_index'], original['candidate_index'], original['signal_date'], original['symbol']) ==
                (i, j, meta['dates'][i], meta['symbols'][i][j]), 'Source grid axes changed')
        require(original['tier3'] is False and type(original['is_etf']) is bool and
                original['is_etf'] == (original['symbol'] in meta['etfs']), 'Source original flags changed')
        require(type(item['conditional_annual_pair_admitted']) is bool and item['old_admission_flags_changed'] is False,
                'Conditional source flag type/preservation')
        signal = original['signal_date']; p = positions[signal]
        require(p >= 42, 'Missing cutoff calendar warmup')
        cutoff = sessions[p-42]
        require(original['information_cutoff_date'] == cutoff and type(original['information_cutoff_ns']) is int and
                original['information_cutoff_ns'] == closes[cutoff] == model.timestamp_ns(original['information_cutoff']) and
                closes[signal] == model.timestamp_ns(original['original_decision_cutoff']), 'Original source cutoff changed')
        mask[i,j] = item['conditional_annual_pair_admitted']
        if not mask[i,j]:
            continue
        symbol = original['symbol']
        require(symbol in model.ISSUERS and not original['is_etf'] and original['cik'] == model.ISSUERS[symbol] and
                original['tier2'] is True and item['body_pair_context_verified'] is True and
                item['body_context_source'] == symbol and item['remaining_row_reasons'] == [] and
                original['known_amendment_event_ids'] == [] and
                all(reason == model.MISSING_CONTEXT for reason in original['reasons']), 'Admitted source contradicts retained gates')
        ids = original['selected_event_ids']
        require(type(ids) is list and len(ids) == len(set(ids)) == 2, 'Exact selected pair required')
        pair = []
        for key in ids:
            event = events.get(key, {})
            require(event.get('event_id') == key and event.get('cik') == model.ISSUERS[symbol] and
                    event.get('form') == '10-K' and type(event.get('primaryDocument')) is str and event['primaryDocument'] and
                    type(event.get('accessionNumber')) is str and re.fullmatch(r'\d{10}-\d{2}-\d{6}', event['accessionNumber']),
                    'Admitted event identity/version contradiction')
            report, filed = model.iso_day(event.get('reportDate')), model.iso_day(event.get('filingDate'))
            require(filed >= report and event['filingDate'] >= sessions[0] and
                    bisect_right(sessions, event['filingDate'])+1 <= p-42, 'Admitted event date/availability contradiction')
            pair.append(event)
        require(pair[0]['reportDate'] != pair[1]['reportDate'] and pair[0]['accessionNumber'] != pair[1]['accessionNumber'],
                'Admitted pair collision')
    return mask


def build_features(meta, rows, events, source_mask, eligible, calendar):
    result = model.build(meta, rows, events, source_mask, eligible, calendar)
    allowed = {'ORIGINAL_INELIGIBLE', 'INITIAL_PARTIAL_EXACT_FALLBACK'}
    require(all(not cell['source_conditionally_admitted'] or set(cell['reasons']) <= allowed
                for cell in result['audit']['cells']), 'Admitted source contradiction cannot become missing')
    for n in range(MONTHS+1):
        prefix = model.build(meta, rows, events, source_mask, eligible, calendar, n=n)
        require(all(prefix['arrays'][key].tobytes() == result['arrays'][key][:n].tobytes() for key in EXTRA),
                'Feature prefix dependence')
    coverage = {}
    for label, lo, hi in (('all122', 0, 122), ('first88', 0, 88), ('later34', 88, 122)):
        cells = result['audit']['cells'][lo*WIDTH:hi*WIDTH]
        def pairs(field):
            return {(cell['symbol'], tuple(sorted(rows[cell['signal_index']*WIDTH+cell['candidate_index']]
                    ['original_row']['selected_event_ids']))) for cell in cells if cell[field]}
        coverage[label] = dict(candidate_cells=len(cells),
            source_conditionally_admitted_rows=sum(c['source_conditionally_admitted'] for c in cells),
            usable_rows=sum(c['available'] for c in cells), unique_source_pairs=len(pairs('source_conditionally_admitted')),
            unique_usable_pairs=len(pairs('available')),
            months_with_two_usable=sum(m['common_candidates'] == 2 for m in result['audit']['monthly'][lo:hi]))
    result['audit'].update(coverage=coverage, source_contradictions_hard_failed=True, feature_prefixes_checked=123)
    return result


def ranked_pair(arrays, n):
    shape = np.asarray(arrays['baseline']).shape
    require(len(shape) == 2 and type(n) is int and 0 <= n <= shape[0], 'Valid prefix required')
    def matrix(key, boolean=False):
        a = np.asarray(arrays[key])
        require(a.shape == shape, 'Wrong shape: '+key)
        if boolean:
            require(a.dtype == bool, 'Boolean mask required: '+key)
            return a[:n].copy()
        prefix = a[:n]
        require(all(isinstance(v, (int, float, np.integer, np.floating)) and
                    not isinstance(v, (bool, np.bool_)) for v in prefix.flat), 'Numeric feature required')
        return prefix.astype(float, copy=True)
    use, eligible = matrix('common_available', True), matrix('eligible', True)
    require(not (use & ~eligible).any() and np.all(use.sum(axis=1) <= 2), 'Only common eligible pilot issuers')
    ranks = {}
    for raw_name, rank_name in zip(model.RAW_NAMES, model.RANK_NAMES):
        raw, rank = matrix(raw_name), matrix(rank_name)
        require(not np.isinf(raw).any() and np.array_equal(np.isfinite(raw), use), 'Raw feature common mask mismatch')
        expected = np.zeros((n, shape[1]))
        for i in range(n):
            values = raw[i, use[i]]
            if len(values) > 1:
                expected[i, use[i]] = (pd.Series(values).rank(method='average').to_numpy()-1)/(len(values)-1)-.5
        require(np.isfinite(rank).all() and np.all(np.abs(rank) <= .5) and np.array_equal(rank, expected),
                'Cached rank differs from exact common population')
        ranks[rank_name] = rank
    return ranks, use


def choices_all(arrays, meta, n, backend, policy):
    ranks, use = ranked_pair(arrays, n)
    require(all(all(not use[i,j] or symbol in model.ISSUERS for j, symbol in enumerate(names))
                for i, names in enumerate(meta['symbols'][:n])), 'Feature outside fixed issuer scope')
    initial = np.array([d == model.INITIAL_PARTIAL for d in meta['dates'][:n]], dtype=bool)
    require(not use[initial].any(), 'Initial partial must use exact incumbent fallback')
    original, original_scores = backend.baseline(arrays, meta, n, policy)
    shape = np.asarray(arrays['baseline']).shape
    pre_original = np.asarray(backend.pre(arrays, n))
    require(pre_original.shape == shape, 'Original full local67 pre-score required')
    choices, scores = {NAMES[0]: original}, {NAMES[0]: original_scores}
    for name, key in zip(NAMES[1:], model.RANK_NAMES):
        pre = pre_original.copy(); delta = DOSE*ranks[key]; changed = delta != 0.
        require(np.isfinite(delta).all() and np.all(np.abs(delta) <= .005), 'Fixed .005 pre-score cap')
        view = pre[:n].copy(); view[changed] += delta[changed]; pre[:n] = view
        new, score = backend.adapter.score_and_choose('incumbent66', 0., np.full(shape, np.nan), arrays, meta, n,
            pre_score=pre, effects=arrays['effects'], episode_policy=policy, boundary_selector=backend.selector)
        for i in np.flatnonzero(~changed.any(axis=1)):
            require(new[i] == original[i] and score[i].tobytes() == original_scores[i].tobytes(), 'Exact unchanged-month fallback')
        choices[name], scores[name] = new, score
    audit = [dict(signal_index=i, date=meta['dates'][i], shared_query_population=int(use[i].sum()),
        informative_pair=bool(use[i].sum() == 2), initial_partial_fallback=bool(initial[i]),
        unchanged_current_correction=not bool(np.any(ranks[model.RANK_NAMES[0]][i])),
        unchanged_change_correction=not bool(np.any(ranks[model.RANK_NAMES[1]][i]))) for i in range(n)]
    return choices, scores, audit


def panels(arrays, meta, backend, policy):
    signature = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    choices, scores, audit = choices_all(arrays, meta, MONTHS, backend, policy)
    for n in range(MONTHS+1):
        short, ss, _ = choices_all(arrays, meta, n, backend, policy)
        require(all(short[name] == choices[name][:n] and ss[name].tobytes() == scores[name][:n].tobytes()
                    for name in NAMES), 'All-prefix score/order parity')
    neutral = dict(arrays, common_available=np.zeros_like(arrays['common_available']))
    neutral.update({key: np.full_like(arrays[key], np.nan) for key in model.RAW_NAMES})
    neutral.update({key: np.zeros_like(arrays[key]) for key in model.RANK_NAMES})
    cc, sc, _ = choices_all(neutral, meta, MONTHS, backend, policy)
    require(all(cc[name] == choices[NAMES[0]] and sc[name].tobytes() == scores[NAMES[0]].tobytes()
                for name in NAMES[1:]), 'Missing feature fallback')
    require(signature == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Array mutation')
    return {name: backend.make_orders(choices[name], scores[name], meta) for name in NAMES}, scores, audit


def incremental_audit(orders, rank_audit):
    correction_keys = ('unchanged_current_correction', 'unchanged_change_correction')
    correction_months = {name: sum(not month[key] for month in rank_audit)
                         for name,key in zip(NAMES[1:],correction_keys)}
    changed_months = {name: sum(row['symbols'] != old['symbols']
                              for row,old in zip(orders[name],orders[NAMES[0]])) for name in NAMES[1:]}
    return dict(nonzero_correction_months=correction_months, ordered_holdings_changed_months=changed_months,
                primary_no_incremental_evidence=correction_months[NAMES[2]] == 0 or changed_months[NAMES[2]] == 0)


class Backend(base.Local67Backend):
    price_parity = prior.Backend.price_parity

    def training(self, orders, meta, px, op):
        """Only first88 returns; deliberately no fold loop or selection call."""
        rp = self.local.rp
        blocks, _, bounds, valid = rp.paths(px, meta, TRAIN)
        rows, opened = [], []
        for name in NAMES:
            indices = [[meta['symbols'][i].index(s) for s in row['symbols']] for i, row in enumerate(orders[name][:TRAIN])]
            rr = rp.fast_returns(indices, blocks, valid)
            rows.append(dict(name=name, **self.local.old.session.measure(rr, meta, bounds)))
            _, eq, _, _ = self.local.ex.e.engine.replay(px, op, orders[name][:TRAIN], cost=.0005,
                execution='open', start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][TRAIN-1]))
            item = dict(name=name)
            for key, lo, hi in (('training', 0, 87), ('early', 0, 52), ('recent', 53, 87)):
                start, end = meta['dates'][lo], meta['ends'][hi]; part = eq.loc[start:end]
                require(len(part) >= 2, 'Training interval too short')
                item[key] = rp.stats_rr(part.to_numpy()[1:]/part.to_numpy()[:-1]-1, start, end)
            opened.append(item)
        return rows, opened

    def diagnostics(self, orders, meta, px, op, writer):
        results = {}
        for name in NAMES:
            results[name] = {}
            for mode, cost, key in (('close', 0., 'same_close'), ('open', 0., 'next_open_zero'), ('open', .0005, 'next_open_cost')):
                met, eq, trades, holds = self.local.ex.e.engine.replay(px, op, orders[name], cost=cost,
                    execution=mode, start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][-1]))
                for year in met['annual_returns']:
                    if year['year'] in (2016, 2026): year['partial'] = True
                results[name][key] = dict(metrics=met, slices=self.local.arch.old.prior.old.previous.cuts(eq, meta))
                writer(name+'_'+key+'_DAILY.json', [dict(date=str(t.date()), equity=float(v)) for t,v in eq.items()])
                writer(name+'_'+key+'_TRADES.json', trades); writer(name+'_'+key+'_HOLDINGS.json', holds)
        periods = []
        for i, day in enumerate(meta['dates']):
            held = {name: orders[name][i]['symbols'] for name in NAMES}
            rr = {name: float((px.loc[meta['ends'][i], ss]/px.loc[day, ss]-1).mean()) if ss else 0. for name,ss in held.items()}
            periods.append(dict(date=day, end=meta['ends'][i], split='early53' if i < 53 else 'recent35' if i < TRAIN else 'later34',
                                holdings=held, returns=rr))
        return results, periods


class Runner(base.Runner):
    def path(self, path):
        raw = Path(path); raw = raw if raw.is_absolute() else self.root/raw
        runtime = {Path(sys.executable).resolve(), Path(np.__file__).resolve(), Path(pd.__file__).resolve()}
        require((raw.is_relative_to(self.root) or raw in runtime) and raw.resolve() == raw and
                '..' not in raw.parts and 'CloudStorage' not in raw.parts, 'Nonlocal or symlink dependency')
        return raw

    def verify(self, mapping):
        require(isinstance(mapping, dict) and mapping, 'Nonempty hash closure required')
        mapped = {}
        for path, digest in mapping.items():
            raw = Path(path); raw = raw if raw.is_absolute() else self.root/raw
            if raw == self.root/HISTORICAL_STATE:
                require(digest == HISTORICAL_SHA, 'Unrecognized historical mutable pin')
                path = HISTORICAL_ARCHIVE
            normalized = str(self.path(path))
            require(normalized not in mapped or mapped[normalized] == digest, 'Conflicting normalized pin')
            mapped[normalized] = digest
        return super().verify(mapped)

    def __init__(self, root, plan_path, plan_sha, backend_factory=Backend):
        self.root = Path(root).resolve(); self.plan_path = self.path(plan_path); self.plan_sha = plan_sha
        require(sha(self.plan_path) == plan_sha, 'PLAN hash mismatch')
        self.plan = p = read(self.plan_path)
        require(p.get('round') == 158 and p.get('status') == 'REGISTERED' and p.get('arms') == list(NAMES) and
                p.get('primary_challenger') == NAMES[2] and p.get('dose') == DOSE, 'Registered round158 required')
        require(p.get('measurement') == measurement_contract() and p.get('selection_contract') == selection_contract() and
                p.get('tolerances') == TOLERANCES, 'Frozen mathematical policy')
        require(p.get('runtime') == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), 'Pinned runtime')
        for key in ('actual_feature_calculations_before_registration', 'actual_returns_before_registration',
                    'actual_label_calculations_before_registration', 'actual_lag_calculations_before_registration',
                    'actual_model_fits_before_registration', 'actual_nuisance_estimates_before_registration', 'new_market_downloads'):
            require(type(p.get(key)) is int and p[key] == 0, 'Premature real work')
        require(p.get('no_observer_writes') is True, 'Observer protection required')
        self.pins = self.verify(p['hashes'])
        require(set(p['inputs']) == set(INPUTS), 'Exact named input schema')
        self.inputs = {key: self.bound_path(ref['path']) for key,ref in p['inputs'].items()}
        require(all(sha(self.inputs[key]) == ref['sha256'] for key,ref in p['inputs'].items()), 'Named input digest')
        for key, (path, digest) in FIXED_INPUTS.items():
            require(self.inputs[key] == self.root/path and sha(self.inputs[key]) == digest, 'Fixed input changed: '+key)
        validate_proposal(read(self.inputs['proposal']))
        require(sha(self.inputs['protected_manifest']) == PROTECTED_SHA and
                sha(self.bound_path(PREVIOUS_COMPLETION)) == PREVIOUS_COMPLETION_SHA, 'Protection/predecessor changed')
        require(sha(model.__file__) == MODEL_SHA, 'Frozen pure model changed')
        require(all(self.pins.get(k) == v for k,v in source_files(self.root).items()), 'Implementation not bound')
        for section in ('raw_source_hashes', 'independent_prepare_source_hashes', 'independent_outcome_source_hashes'):
            require(all(self.pins.get(k) == v for k,v in self.verify(p[section]).items()), 'Unbound '+section)
        require(len(p['raw_source_hashes']) == 64, 'Original64 raw price sources required')
        self.out = self.path(p['output_directory'])
        require(self.out == self.root/OUT and self.plan_path == self.out/'PLAN.json', 'Dedicated output path')
        self.backend_factory = backend_factory
        self.require_sources()

    def write(self, name, value):
        self.refresh()
        return super().write(name, value)

    def require_sources(self):
        overlay, review, prior_data = (read(self.inputs[key]) for key in ('source_overlay', 'source_review', 'source_prior'))
        closure = self.verify(overlay['input_hashes']); reviewed = self.verify(review['input_hashes'])
        require(all(self.pins.get(k) == v for k,v in dict(closure, **reviewed).items()), 'Source closure not registered')
        require(review.get('passed') is True and review.get('source_admission_changed') is False and
                all(review.get('checks', {}).get(k) is True for k in SOURCE_CHECKS), 'Independent source review incomplete')
        needed = dict(closure, **{str(self.inputs['source_overlay']): sha(self.inputs['source_overlay']),
                                str(self.inputs['source_prior']): sha(self.inputs['source_prior'])})
        require(all(reviewed.get(k) == v for k,v in needed.items()), 'Independent source review omitted closure')
        require(canonical_hash(review['summary']) == canonical_hash(overlay['summary']), 'Reviewed source summary differs')
        recorded = read(self.inputs['calendar'])
        require(all(self.pins.get(k) == v for k,v in self.verify(recorded['provenance']['source_hashes']).items()),
                'Unbound calendar implementation')
        cal = base.module('annual_lag158_calendar', self.bound_path('outputs/monthly_earnings_repair/research_calendar.py'))
        schedule = cal.market_calendar(recorded['start'], recorded['end']).schedule
        sessions = [str(day.date()) for day in schedule.index]
        require(sessions == recorded['sessions'], 'Original complete XNYS calendar changed')
        calendar = dict(sessions=sessions, closes={day:int(close.value) for day,close in zip(sessions, schedule['close'])})
        meta = read(self.inputs['meta']); mask = validate_source_rows(meta, overlay, prior_data, calendar)
        self.refresh()
        return meta, overlay['rows'], prior_data['selected_original_and_amendment_metadata'], mask, calendar

    def prepare(self):
        self.refresh(); require(not any((self.out/x).exists() for x in (*CACHE_FILES, 'CACHE.json')), 'Preparation already started')
        source_meta, rows, events, mask, calendar = self.require_sources()
        backend = self.backend_factory(self.root, self); meta, original, px, op = backend.load()
        require(meta == source_meta and len(original) == 27, 'Exact original metadata/27 arrays required')
        signature = {k:(v.dtype.str, v.shape, v.tobytes()) for k,v in original.items()}
        parity = backend.price_parity(meta, px, op, self.plan['raw_source_hashes'])
        result = build_features(meta, rows, events, mask, original['eligible'], calendar)
        require(signature == {k:(v.dtype.str, v.shape, v.tobytes()) for k,v in original.items()}, 'Original array mutation')
        require(set(result['arrays']) == set(EXTRA) and not set(original)&set(EXTRA), 'Feature schema/collision')
        arrays = dict(original, **result['arrays'])
        orders, scores, rank_audit = panels(arrays, meta, backend, read(self.inputs['policy']))
        require(orders[NAMES[0]] == read(self.inputs['original_orders']), 'Incumbent orders changed')
        result['audit'].update({key+'_sha256':sha(self.inputs[key]) for key in ('source_overlay','source_review','source_prior')})
        result['audit'].update(incremental_audit(orders,rank_audit))
        self.refresh()
        for name, values in (('DATA.npz', arrays), ('SCORES.npz', scores)):
            with (self.out/name).open('xb') as stream: np.savez_compressed(stream, **values)
        for name,value in (('META.json',meta),('PRETRAIN_ORDERS.json',orders),('PRICE_PARITY.json',parity),
                           ('FEATURE_AUDIT.json',result['audit']),('CALENDAR.json',calendar),
                           ('RANK_AUDIT.json',dict(months=rank_audit,prefixes_checked=123))): self.write(name,value)
        self.write('CACHE.json', {name:sha(self.out/name) for name in CACHE_FILES})

    def cache(self):
        cache = read(self.out/'CACHE.json'); require(set(cache) == set(CACHE_FILES), 'Exact prepared artifacts required')
        return self.verify({str(self.out/name):digest for name,digest in cache.items()})

    def require_prepare_review(self):
        cached = self.cache(); report = read(self.out/'INDEPENDENT_PREPARE.json')
        require(report.get('passed') is True and report.get('stage') == 'prepare' and
                all(report.get(key) is True for key in CHECKS), 'Independent preparation incomplete')
        require(report.get('producer_modules_imported') is False and report.get('returns_or_nav_computed') is False,
                'Independent preparation isolation')
        for key,count in (('orders_checked',366),('candidate_cells_checked',5978),('months_checked',122),
                          ('prefixes_checked',123),('lag_feature_matrices_verified',2),('new_predictive_model_fits',0),('nuisance_estimates',0)):
            require(type(report.get(key)) is int and report[key] == count, 'Independent scope: '+key)
        require(self.verify(report['source_hashes']) == self.verify(self.plan['independent_prepare_source_hashes']), 'Reviewer source changed')
        required = dict(cached, **{str(self.out/x):sha(self.out/x) for x in ('PLAN.json','CACHE.json')})
        got = self.verify(report['artifact_hashes']); allowed = dict(self.pins, **required)
        require(all(got.get(k) == v for k,v in required.items()) and all(allowed.get(k) == v for k,v in got.items()), 'Review artifact closure')
        for key,name in (('plan_sha256','PLAN.json'),('cache_sha256','CACHE.json'),('orders_sha256','PRETRAIN_ORDERS.json')):
            require(report.get(key) == sha(self.out/name), 'Review receipt binding')

    def loaded(self):
        self.cache(); self.require_sources()
        backend = self.backend_factory(self.root, self); meta, old, px, op = backend.load()
        require(meta == read(self.out/'META.json') and len(old) == 27, 'Original metadata/cache changed')
        with np.load(self.out/'DATA.npz', allow_pickle=False) as data:
            require(set(data.files) == set(old)|set(EXTRA), 'Prepared array names changed')
            require(all(data[k].dtype == v.dtype and data[k].shape == v.shape and data[k].tobytes() == v.tobytes()
                        for k,v in old.items()), 'Original27 arrays changed')
        return backend, meta, px, op

    def select_first88(self):
        self.refresh(); require(not any((self.out/x).exists() for x in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','FROZEN.json')), 'Selection already started')
        self.require_prepare_review(); backend, meta, px, op = self.loaded()
        rows, opened = backend.training(read(self.out/'PRETRAIN_ORDERS.json'), meta, px, op)
        selected = select(rows, opened)  # The sole promotion choice; no fold selection.
        self.write('TRAIN_TRIALS.json', rows); self.write('TRAIN_OPEN_TRIALS.json', opened)
        self.write('FROZEN.json', dict(selection=selected, selection_count=1, pristine_oos=False,
            all_diagnostic_orders_frozen=True, hashes={name:sha(self.out/name) for name in FROZEN_FILES}))

    def diagnose_later34(self):
        self.refresh(); self.require_prepare_review()
        outputs = ['DIAGNOSE_STARTED.json','RESULTS.json','MONTH_COMPARISON.json','ATTRIBUTION.json','VALIDATION.json']
        outputs += [name+'_'+mode+'_'+kind+'.json' for name in NAMES for mode in ('same_close','next_open_zero','next_open_cost')
                    for kind in ('DAILY','TRADES','HOLDINGS')]
        require(not any((self.out/x).exists() for x in outputs), 'Diagnostics already started; preserve partial artifacts')
        frozen = read(self.out/'FROZEN.json'); require(set(frozen['hashes']) == set(FROZEN_FILES), 'Frozen artifact schema')
        self.verify({str(self.out/k):v for k,v in frozen['hashes'].items()})
        # Rechecking the same frozen first88 decision is not another selection window.
        require(frozen['selection'] == select(read(self.out/'TRAIN_TRIALS.json'),read(self.out/'TRAIN_OPEN_TRIALS.json')) and
                frozen.get('selection_count') == 1 and frozen.get('pristine_oos') is False and
                frozen.get('all_diagnostic_orders_frozen') is True, 'Frozen first88 selection changed')
        self.write('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,frozen_sha256=sha(self.out/'FROZEN.json'),
                                              started_utc=datetime.now(timezone.utc).isoformat()))
        backend, meta, px, op = self.loaded()
        results, periods = backend.diagnostics(read(self.out/'PRETRAIN_ORDERS.json'),meta,px,op,self.write)
        for name,mode in (('TRAIN_TRIALS.json','same_close'),('TRAIN_OPEN_TRIALS.json','next_open_cost')):
            for row in read(self.out/name):
                require(abs(results[row['name']][mode]['slices']['training2016_2023']['cagr']-row['training']['cagr'])
                        < TOLERANCES['training_cagr'], 'Training replay parity')
        months = read(self.out/'RANK_AUDIT.json')['months']; require(len(months) == len(periods) == MONTHS, 'All122 diagnostic months')
        for i,row in enumerate(periods):
            require(months[i]['signal_index'] == i and months[i]['date'] == row['date'], 'Diagnostic month alignment')
            row['annual_lag'] = months[i]
        comparisons = {label:backend.old130.attribution(periods,results,new=new,old=old) for label,new,old in
            (('change_vs_incumbent',NAMES[2],NAMES[0]),('current_vs_incumbent',NAMES[1],NAMES[0]),('change_vs_current',NAMES[2],NAMES[1]))}
        by_date = {row['date']:row['annual_lag'] for row in periods}
        for comparison in comparisons.values():
            for change in comparison['changes']: change['annual_lag'] = by_date[change['date']]
        self.write('RESULTS.json',results); self.write('MONTH_COMPARISON.json',periods)
        self.write('ATTRIBUTION.json',dict(comparisons=comparisons))
        self.write('VALIDATION.json',dict(primary=frozen['selection']['primary'],selection=frozen['selection'],selection_count=1,
            original_arrays_unchanged=True,independent_outcome_review_pending=True,no_observer_writes=True,
            no_intramonth_trades=True,pristine_oos=False,new_predictive_model_fits=0,nuisance_estimates=0,
            primary_no_incremental_evidence=read(self.out/'FEATURE_AUDIT.json')['primary_no_incremental_evidence']))


def source_files(root):
    root = Path(root)
    paths = [Path(__file__), Path(prior.__file__), Path(model.__file__), root/'work/test_monthly_annual_lag158_runner.py',
             root/'work/test_monthly_annual_lag158_model.py']
    return dict(base.source_files(root), **{str(path.resolve()):sha(path) for path in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','select_first88','diagnose_later34'))
    parser.add_argument('--plan',required=True); parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    getattr(Runner(args.root,args.plan,args.plan_sha),args.phase)()


if __name__ == '__main__':
    main()
