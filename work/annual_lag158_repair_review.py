"""Independent numerical audit route for the registered158 row-schema repair.

Only the independent feature oracle is replaced during an audit. Its arithmetic
is copied unchanged from the frozen independent reviewer. Absent amendment
metadata on two explicitly unadmitted early-return row types remains unknown;
it is neither fabricated as an empty list nor promoted to source eligibility.
All original disk files and their registered source/artifact closures remain
unchanged. Separately registered repair evidence is added by the repair helper.
"""
from contextlib import contextmanager
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import annual_lag158_independent_prepare as original

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_annual_lag158'
SELF = 'work/annual_lag158_repair_review.py'
TEST = 'work/test_annual_lag158_repair_review.py'
PLAN_SHA = '29abc09a1c0a043da286a7d4ac7e1e46f003fcfe9c175d393cf61237b5c64665'
ABSENT_REASON = 'AMENDMENT_METADATA_ABSENT_ON_UNADMITTED_ROW'
ETF_REASON = 'ETF_PRESERVED_NO_ANNUAL_FILING_FEATURE'
UNAVAILABLE_REASON = 'SOURCE_EXTENT_OR_CURRENT_CIK_UNAVAILABLE'
FROZEN_ORACLE = original.features_oracle
require = original.require
RAW, RANKS, ISSUERS, MISSING, INITIAL = original.RAW, original.RANKS, original.ISSUERS, original.MISSING, original.INITIAL
day, calendar_check, cutoff_check = original.day, original.calendar_check, original.cutoff_check
inspected_pair, selected_pair, rank_oracle = original.inspected_pair, original.selected_pair, original.rank_oracle


def independent_imports():
    original.independent_imports()
    forbidden = ('monthly_annual_lag158_repaired_runner', 'annual_lag158_cell_repair')
    require(not any(name in sys.modules for name in forbidden), 'No repaired producer imports')


def helper():
    return importlib.import_module('annual_lag158_repair')


def absent_metadata_permitted(overlay, source):
    """Recognize only the two frozen feasibility early-return shapes."""
    row = overlay['original_row']
    if 'known_amendment_event_ids' in row or source is not False:
        return False
    if any(overlay.get(key) is not False for key in (
            'conditional_annual_pair_admitted', 'body_pair_context_verified', 'old_admission_flags_changed')):
        return False
    if 'body_context_source' not in overlay or overlay['body_context_source'] is not None:
        return False
    if any(key not in row or row[key] is not None for key in ('selected_snapshot', 'cik')):
        return False
    if any(type(row.get(key)) is not list or row[key] for key in ('selected_event_ids', 'period_hint_ids')):
        return False
    if any(row.get(key) is not False for key in ('tier1', 'tier2', 'tier3', 'normal_period_admission')):
        return False
    return ((row.get('is_etf') is True and row.get('reasons') == [ETF_REASON]) or
            (row.get('is_etf') is False and row.get('reasons') == [UNAVAILABLE_REASON]))


def features_oracle(meta, rows, event_metadata, source_mask, eligible, calendar, n=None):
    # Frozen independent oracle, with only the explicit absent-metadata branch.
    count = 122 if n is None else n
    require(type(count) is int and 0 <= count <= 122, 'Valid prefix 0..122')
    require(len(meta['dates']) == len(meta['symbols']) == 122 and len(rows) == 5978, 'Full original grid')
    require(type(meta['etfs']) in (list, tuple) and len(meta['etfs']) == len(set(meta['etfs'])), 'ETF identity list')
    for mask in (source_mask, eligible):
        require(isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (122, 49), 'Boolean original-grid mask')
    positions = calendar_check(calendar)
    arrays = {key: np.full((count, 49), np.nan) for key in RAW}
    arrays.update({key: np.zeros((count, 49)) for key in RANKS})
    arrays['common_available'] = np.zeros((count, 49), bool)
    cells, monthly = [], []
    for i in range(count):
        signal, names = meta['dates'][i], meta['symbols'][i]
        day(signal)
        require(i == 0 or meta['dates'][i-1] < signal, 'Ordered signal dates')
        require(len(names) == len(set(names)) == 49 and all(type(s) is str and s for s in names), '49 original candidates')
        for j, symbol in enumerate(names):
            overlay, source = rows[i*49+j], bool(source_mask[i, j])
            row = overlay['original_row']
            require(type(row['signal_index']) is int and type(row['candidate_index']) is int and
                    (row['signal_index'], row['candidate_index'], row['symbol'], row['signal_date']) == (i, j, symbol, signal),
                    'Original grid alignment')
            require(type(row['is_etf']) is bool and row['is_etf'] == (symbol in meta['etfs']), 'ETF identity alignment')
            require(type(overlay['conditional_annual_pair_admitted']) is bool and
                    overlay['conditional_annual_pair_admitted'] == source, 'Source mask matches exact overlay')
            absent = 'known_amendment_event_ids' not in row
            if absent:
                require(absent_metadata_permitted(overlay, source), 'Unrecognized missing amendment metadata')
            for key in ('reasons',) if absent else ('reasons', 'known_amendment_event_ids'):
                require(type(row[key]) is list and all(type(v) is str and v for v in row[key]), 'Explicit original reason/id list')
            require(type(overlay['remaining_row_reasons']) is list and
                    all(type(v) is str and v for v in overlay['remaining_row_reasons']), 'Explicit remaining reason list')
            cutoff = cutoff_check(row, signal, calendar, positions)
            reasons = []
            if not source:
                reasons.append('SOURCE_NOT_CONDITIONALLY_ADMITTED')
            if overlay.get('body_pair_context_verified') is not True or row.get('tier2') is not True:
                reasons.append('SOURCE_OR_ANNUAL_CONTEXT_NOT_VERIFIED')
            blockers = overlay['remaining_row_reasons'] + [reason for reason in row['reasons'] if not
                (reason == MISSING and overlay.get('body_pair_context_verified') is True)]
            reasons.extend('SOURCE_BLOCKER:' + reason for reason in dict.fromkeys(blockers))
            if absent:
                reasons.append(ABSENT_REASON)
            elif row['known_amendment_event_ids']:
                reasons.append('KNOWN_SELECTED_PERIOD_AMENDMENT')
            if overlay.get('old_admission_flags_changed') is not False:
                reasons.append('ORIGINAL_ADMISSION_FLAGS_NOT_PRESERVED')
            if symbol not in ISSUERS or row['is_etf']:
                reasons.append('OUTSIDE_FIXED_KO_WMT_SCOPE')
            elif row.get('cik') != ISSUERS[symbol]:
                reasons.append('ROW_ISSUER_MISMATCH')
            if overlay.get('body_context_source') not in (None, symbol) or (
                    overlay.get('body_pair_context_verified') is True and overlay.get('body_context_source') != symbol):
                reasons.append('BODY_CONTEXT_ISSUER_MISMATCH')
            require(not source or not reasons, 'Contradictory admitted original metadata: ' + ','.join(reasons))
            if source:
                inspected_pair(row, event_metadata, calendar, cutoff)
            if not eligible[i, j]:
                reasons.append('ORIGINAL_INELIGIBLE')
            if signal == INITIAL:
                reasons.append('INITIAL_PARTIAL_EXACT_FALLBACK')
            details = dict(information_cutoff_date=cutoff)
            if not reasons:
                values, selected = selected_pair(row, event_metadata, calendar, cutoff)
                details.update(selected)
                arrays['common_available'][i, j] = True
                for key, value in zip(RAW, values):
                    arrays[key][i, j] = value
            cells.append(dict(signal_index=i, candidate_index=j, symbol=symbol, signal_date=signal,
                source_conditionally_admitted=source, original_eligible=bool(eligible[i, j]),
                available=not reasons, reasons=list(dict.fromkeys(reasons)), **details))
        mask = arrays['common_available'][i]
        for raw, rank in zip(RAW, RANKS):
            arrays[rank][i, mask] = rank_oracle(arrays[raw][i, mask])
        monthly.append(dict(signal_index=i, signal_date=signal, common_candidates=int(mask.sum()),
            current_rank_nonzero=int(np.count_nonzero(arrays[RANKS[0]][i])),
            change_rank_nonzero=int(np.count_nonzero(arrays[RANKS[1]][i])),
            ranks_identical=bool(np.array_equal(arrays[RANKS[0]][i], arrays[RANKS[1]][i]))))
    require(all(np.array_equal(np.isfinite(arrays[key]), arrays['common_available']) for key in RAW), 'Common raw mask')
    return dict(arrays=arrays, cells=cells, monthly=monthly)


@contextmanager
def repaired_oracle():
    independent_imports()
    require(original.features_oracle is FROZEN_ORACLE, 'Independent oracle already replaced')
    original.features_oracle = features_oracle
    try:
        yield
    finally:
        original.features_oracle = FROZEN_ORACLE


def source_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    outcome = importlib.import_module('annual_lag158_independent_outcome')
    return sorted(set(original.source_files(root) + outcome.source_files(root) + [root/SELF, root/TEST]))


def audit(stage, root, plan_sha, amendment_sha):
    require(stage in ('prepare', 'outcome') and plan_sha == PLAN_SHA, 'Exact repaired158 audit stage/PLAN')
    independent_imports()
    root = Path(root).resolve()
    target = root/OUT/('INDEPENDENT_PREPARE.json' if stage == 'prepare' else 'INDEPENDENT_OUTCOME.json')
    require(not target.exists(), 'Independent audit already published')
    module = helper()
    amendment = module.verify_amendment(root, plan_sha, amendment_sha)
    with repaired_oracle():
        if stage == 'prepare':
            report = original.audit(root, root/OUT/'PLAN.json', plan_sha)
        else:
            report = importlib.import_module('annual_lag158_independent_outcome').audit(root, plan_sha)
    independent_imports()
    require(module.verify_amendment(root, plan_sha, amendment_sha) == amendment,
            'Repair amendment changed during independent audit')
    require(report.get('passed') is True and report.get('producer_modules_imported') is False,
            'Original independent audit must pass')
    source_hashes, artifact_hashes = dict(report['source_hashes']), dict(report['artifact_hashes'])
    report = module.augment_report(root, report, plan_sha, amendment_sha)
    require(report['source_hashes'] == source_hashes and report['artifact_hashes'] == artifact_hashes,
            'Repair augmentation must preserve original closed hash maps')
    module.verify_repair_report(root, report, plan_sha, amendment_sha)
    return report


def publish(path, report):
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'outcome'))
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--amendment-sha', required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    report = audit(args.stage, args.root, args.plan_sha, args.amendment_sha)
    target = args.root/OUT/('INDEPENDENT_PREPARE.json' if args.stage == 'prepare' else 'INDEPENDENT_OUTCOME.json')
    publish(target, report)
    print(json.dumps(dict(passed=True, stage=args.stage, output=str(target)), sort_keys=True))


if __name__ == '__main__':
    main()
