"""Independent revenue preparation audit; no new producer imports.

The separately audited source grid supplies admission. This audit reconstructs
bounded financial arithmetic and all four monthly score/holding paths.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import numpy as np
import pandas as pd
import monthly_cash_quality146_prepare_review as legacy
import after155_revenue_factor_independent as arithmetic

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_revenue_persistence156'
SOURCE = 'outputs/monthly_revenue_pilot_source156'
NAMES = ('incumbent_local67', 'matched_latest_revenue_growth',
         'matched_annualized_total_revenue_growth', 'weakest_annual_revenue_growth')
HANDS = ('latest_revenue_growth', 'annualized_total_revenue_growth', 'weakest_annual_revenue_growth')
EXTRA = HANDS + ('common_available',)
CHECKS = legacy.CHECKS + ('revenue_source_grid_verified',
          'all_three_revenue_features_common_mask', 'original_source_cutoffs_verified')
CACHE_FILES = legacy.CACHE_FILES
USABLE = 'checked_reported_total_chain_with_declared_source_scope'
old, require = legacy.old, legacy.require
INPUTS = ('proposal','meta','policy','original_orders','prices','sessions','protected_manifest',
          'source_manifest','source_grid_review','source_plan','source_calendar')
SOURCE_REVIEW_CHECKS = ('exact_original_axis','original_source_catalog_reconstructed',
    'current_values_never_replaced_by_later_comparatives','independent_calendar_closes',
    'latest_known_original_before_validity','original_four_year_membership',
    'all_supporting_evidence_available_by_cutoff','common_coverage_gate_recomputed',
    'complete_manifest_input_and_output_closure')
SOURCE_CONTRACT = {
    'meta': ('outputs/monthly_local_linear70/META.json',
        '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4'),
    'source_plan': ('work/after155_revenue_pilot_source_plan.json',
        '5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10'),
    'source_calendar': ('outputs/monthly_paired_target143/CALENDAR.json',
        '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622')}


def validate_source_review(review):
    require(review.get('passed') is True and review.get('actual_grid_verified') is True,
            'Independent actual source-grid verification required')
    checks = review.get('checks', {})
    require(all(checks.get(key) is True for key in SOURCE_REVIEW_CHECKS),
            'Independent source calendar, original identities and evidence checks required')
    for key, expected in (('candidate_rows',5978),('decision_dates',122),('candidates_per_date',49)):
        require(type(checks.get(key)) is int and checks[key] == expected, 'Independent source scope')


def source_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    return sorted(set(legacy.source_files(root) + [root/'work'/name for name in (
        'monthly_revenue_persistence156_prepare_review.py',
        'test_monthly_revenue_persistence156_prepare_review.py',
        'after155_revenue_factor_independent.py')]))


def source_inputs(reader, plan, pins):
    """Honor the independent source oracle, checking its whole byte closure."""
    refs = plan['inputs']
    manifest_path = reader.path(refs['source_manifest']['path'])
    require(manifest_path == reader.path(SOURCE+'/MANIFEST.json'), 'Canonical source manifest')
    manifest = reader.json(manifest_path)
    require(manifest['source_only'] is True and manifest['strategy_registration'] is False,
            'Source-only admission')
    needed = reader.verify(manifest['input_hashes'])
    needed[str(manifest_path)] = reader.sha(manifest_path)
    expected_files = {'ROWS.json','REPORTS.json','EDGES.json','COVERAGE_GATE.json','SUMMARY.json'}
    require(set(manifest['outputs']) == expected_files, 'Complete source package')
    for name, ref in manifest['outputs'].items():
        path = reader.path(SOURCE+'/'+name)
        require(reader.sha(path) == ref['sha256'] and path.stat().st_size == ref['bytes'], 'Source output binding')
        needed[str(path)] = ref['sha256']
    review = reader.json(refs['source_grid_review']['path'])
    validate_source_review(review)
    reviewed = reader.verify(review['input_hashes'])
    require(all(reviewed.get(p) == h for p,h in needed.items()), 'Independent source coverage incomplete')
    require(all(pins.get(p) == h for p,h in reviewed.items()), 'Unregistered source closure')
    for key, (canonical, expected) in SOURCE_CONTRACT.items():
        path = reader.path(refs[key]['path'])
        require(path == reader.path(canonical) and reader.sha(path) == expected
                and needed.get(str(path)) == expected, 'Original source contract: '+key)
    gate = reader.json(SOURCE+'/COVERAGE_GATE.json')
    require(gate['source_coverage_floor_passed'] is True and gate['rows_verified'] == 5978
            and gate.get('source_reviews_verified_by_preparation_driver') is True,
            'Prospective source floor')
    return reader.json(SOURCE+'/ROWS.json'), reader.json(SOURCE+'/REPORTS.json')


def features_oracle(meta, rows, reports, n=None):
    """Use admitted original levels; independently reject changed temporal rows."""
    total = len(meta['dates']); n = total if n is None else n
    require(type(n) is int and 0 <= n <= total, 'Valid prefix')
    require(len(rows) == total*49 and len(meta['symbols']) == total, 'Full source axis')
    arrays = {name:np.full((n,49),np.nan) for name in HANDS}
    arrays['common_available'] = np.zeros((n,49),bool)
    cells = []
    for i in range(n):
        names = meta['symbols'][i]
        require(len(names) == len(set(names)) == 49, 'Unique original candidate axis')
        for j,symbol in enumerate(names):
            row = rows[49*i+j]
            require((row['signal_index'],row['candidate_index'],row['symbol'],row['decision_date'])
                    == (i,j,symbol,meta['dates'][i]), 'Every original slot in order')
            cutoff = pd.Timestamp(row['decision_cutoff'])
            require(cutoff.tzinfo is not None and type(row['decision_cutoff_ns']) is int
                    and int(cutoff.value)==row['decision_cutoff_ns']
                    and str(cutoff.tz_convert('America/New_York').date())==meta['dates'][i], 'Source clock')
            item = dict(signal_index=i,candidate_index=j,symbol=symbol,decision_date=meta['dates'][i],
                        status=row['status'],reasons=list(row.get('reasons',[])),available=False)
            if row['status'] == USABLE:
                require(symbol not in meta['etfs'] and symbol in ('MSFT','ABBV'), 'Fixed admitted pilot issuers')
                catalog = {r['report_id']:r for r in reports[symbol]}
                require(len(catalog)==len(reports[symbol]), 'Unique original report identities')
                ids = row['report_ids']; require(len(ids)==len(set(ids))==4, 'Four unique originals')
                selected = [catalog[x] for x in ids]
                years = [r['fiscal_year'] for r in selected]
                require(all(type(year) is int for year in years)
                        and years == list(range(years[-1]-3,years[-1]+1)), 'Four consecutive annuals')
                require(all(r['known_ns'] <= row['decision_cutoff_ns'] for r in selected), 'No future originals')
                known = [r for r in reports[symbol] if r['known_ns'] <= row['decision_cutoff_ns']]
                newest = max(r['known_ns'] for r in known)
                latest = [r for r in known if r['known_ns'] == newest]
                require(len(latest)==1 and latest[0]['report_id']==ids[-1]==row['latest_report_id'],
                        'No fallback from latest known report')
                require(all(sum(r['fiscal_year']==year for r in known)==1 for year in years),
                        'One original report per fiscal year')
                scales = [legacy.Decimal(r['scale']) for r in selected]
                require(all(scale.is_finite() and scale > 0 for scale in scales)
                        and len(set(scales))==1,'Same positive original scale')
                values = arithmetic.oracle([r['current'] for r in selected])
                for key in HANDS: arrays[key][i,j]=values[key]
                arrays['common_available'][i,j]=True
                item.update(available=True,report_ids=ids,edge_ids=row['edge_ids'],
                    latest_report_id=ids[-1],decision_cutoff_ns=row['decision_cutoff_ns'],features=values)
            cells.append(item)
    return dict(arrays=arrays,cells=cells)


def score_oracle(arrays,meta,hands,n=None):
    groups = legacy.score_oracle(arrays,meta,*(hands[key] for key in HANDS),n=n)
    return tuple({new:group[prior] for new,prior in zip(NAMES,legacy.NAMES)} for group in groups)


def order_oracle(meta,scores,choices):
    return {name:[dict(date=meta['dates'][i],signal_date=meta['dates'][i],end=meta['ends'][i],
        symbols=[meta['symbols'][i][j] for j in picked],scores=[float(scores[name][i,j]) for j in picked])
        for i,picked in enumerate(choices[name])] for name in NAMES}


def audit(root,plan_path,plan_sha):
    require(type(plan_sha) is str and re.fullmatch('[0-9a-f]{64}',plan_sha), 'Explicit PLAN SHA')
    reader = old.Reader(root); directory=reader.path(OUT)
    require(reader.path(plan_path)==directory/'PLAN.json' and reader.sha(plan_path)==plan_sha,'Exact registered PLAN')
    plan=reader.json(plan_path)
    require(plan.get('round')==156 and plan.get('status')=='REGISTERED' and plan['arms']==list(NAMES)
            and plan['primary_challenger']==NAMES[3] and plan['dose']==.01,'Four fixed prospective arms')
    require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Pinned runtime')
    require(reader.path(plan['output_directory'])==directory and set(plan['inputs'])==set(INPUTS),
            'Exact output and named inputs')
    require(plan['tolerances']['feature_atol']==1e-12 and plan['tolerances']['scores_atol']==1e-12
            and plan['tolerances']['scores_rtol']==0., 'Registered preparation tolerances')
    require(not any((directory/name).exists() for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json',
            'WALKFORWARD.json','FROZEN.json','DIAGNOSE_STARTED.json','RESULTS.json')),
            'Preparation audit precedes results')
    pins=reader.verify(plan['hashes'])
    sources={str(path):reader.sha(path) for path in source_files(root)}
    require(reader.verify(plan['independent_prepare_source_hashes'])==sources,'Independent implementation binding')
    for key in ('actual_model_fits_before_registration','actual_feature_calculations_before_registration','actual_returns_before_registration'):
        require(type(plan.get(key)) is int and plan[key]==0,'Prospective activity count')
    inputs={name:reader.path(ref['path']) for name,ref in plan['inputs'].items()}
    for name,ref in plan['inputs'].items():
        require(reader.sha(inputs[name])==ref['sha256']==pins.get(str(inputs[name])),'Named input binding')
    for key, filename in (('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')):
        require(inputs[key]==reader.path(old.LOCAL+'/'+filename),'Original local67 input: '+key)
    require(inputs['prices']==reader.path(old.PRICES),'Original price cache')
    policy=reader.json(inputs['policy'])
    require(all(policy.get(key)==value for key,value in old.original_helper().POLICY.items()),
            'Frozen incumbent episode policy')
    rows,reports=source_inputs(reader,plan,pins)
    cache=reader.json(directory/'CACHE.json'); require(set(cache)==set(CACHE_FILES),'Full preparation cache')
    reader.verify(cache,directory)
    meta=reader.json(directory/'META.json'); require(meta==reader.json(inputs['meta']),'Original metadata')
    require(len(meta['dates'])==122 and meta['train_months']==88 and meta['early_months']==53,'Fixed segments')
    original=reader.npz(old.LOCAL+'/DATA.npz'); arrays=reader.npz(directory/'DATA.npz')
    require(len(original)==27 and set(arrays)==set(original)|set(EXTRA),'Original27 plus four arrays')
    for key,value in original.items():
        require(value.dtype==arrays[key].dtype and value.shape==arrays[key].shape
                and value.tobytes()==arrays[key].tobytes(),'Original array changed '+key)
    expected=features_oracle(meta,rows,reports)
    for key in EXTRA:
        require(arrays[key].shape==expected['arrays'][key].shape
                and arrays[key].dtype==expected['arrays'][key].dtype,'Feature shape and dtype')
        np.testing.assert_allclose(arrays[key],expected['arrays'][key],rtol=0,atol=1e-12,equal_nan=True)
    feature=reader.json(directory/'FEATURE_AUDIT.json')
    require(feature.get('source_manifest_sha256')==reader.sha(inputs['source_manifest'])
            and feature.get('source_grid_review_sha256')==reader.sha(inputs['source_grid_review']),
            'Feature audit source bindings')
    require(feature['rows']==5978 and feature['usable_rows']==int(expected['arrays']['common_available'].sum())
            and feature['new_model_fits']==0,'Feature audit counts')
    old.compare(feature['cells'],expected['cells'],'revenue-cells')
    for key in ('source_holds_preserved','all_three_revenue_features_common_mask','original_source_cutoffs_verified','revenue_source_grid_verified'):
        require(feature.get(key) is True,'Feature check '+key)
    scores,choices,_=score_oracle(arrays,meta,expected['arrays'])
    saved_scores=reader.npz(directory/'SCORES.npz'); saved_orders=reader.json(directory/'PRETRAIN_ORDERS.json')
    require(set(saved_scores)==set(saved_orders)==set(NAMES),'Four prepared arms')
    wanted=order_oracle(meta,scores,choices)
    for name in NAMES:
        require(saved_scores[name].shape==scores[name].shape and saved_scores[name].dtype==scores[name].dtype,
                'Prepared score shape and dtype')
        np.testing.assert_allclose(saved_scores[name],scores[name],rtol=0,atol=1e-12)
        old.compare(saved_orders[name],wanted[name],name)
    require(saved_orders[NAMES[0]]==reader.json(inputs['original_orders']),'Exact frozen incumbent orders')
    for n in range(123):
        prefix=features_oracle(meta,rows,reports,n)
        for key in EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
        ss,cc,_=score_oracle(arrays,meta,prefix['arrays'],n=n)
        for name in NAMES:
            require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Causal prefix')
    for fill in (0.,np.nan):
        hand=np.full((122,49),np.nan)
        for i,symbols in enumerate(meta['symbols']):
            for j,symbol in enumerate(symbols):
                if symbol not in meta['etfs']:hand[i,j]=fill
        ss,cc,_=score_oracle(arrays,meta,dict.fromkeys(HANDS,hand))
        for name in NAMES[1:]:
            require(cc[name]==choices[NAMES[0]] and ss[name].tobytes()==scores[NAMES[0]].tobytes(),'Exact absent/zero fallback')
    require(reader.sha(old.LOADER)==old.LOADER_SHA,'Frozen raw price loader')
    spec=importlib.util.spec_from_file_location('revenue156_independent_price_loader',reader.path(old.LOADER))
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(reader.path(old.LOADER))]=old.LOADER_SHA
    raw=reader.verify(plan['raw_source_hashes'])
    require({str(reader.path(p)):h for p,h in used.items()}==raw and len(raw)==64,'Exact raw64')
    reader.verify(used)
    parity=old.price_parity(panel,reader.npz(inputs['prices']))
    old.compare(reader.json(directory/'PRICE_PARITY.json'),parity,'price-parity')
    required={str(directory/name):reader.sha(directory/name) for name in (*CACHE_FILES,'PLAN.json','CACHE.json')}
    allowed=dict(pins,**required)
    for p,h in list(reader.seen.items()):
        require(reader.sha(p)==h and allowed.get(p)==h,'Unbound or changed input '+p)
    return dict(stage='prepare',passed=True,plan_sha256=plan_sha,cache_sha256=reader.sha(directory/'CACHE.json'),
        orders_sha256=reader.sha(directory/'PRETRAIN_ORDERS.json'),source_hashes=sources,artifact_hashes=dict(reader.seen),
        **dict.fromkeys(CHECKS,True),producer_modules_imported=False,returns_or_nav_computed=False,
        raw_prices_independently_verified=True,calendar_overlay_independently_verified=True,
        all_signal_prefixes_verified=True,
        source_clock_verification='explicit_checks_in_hash_bound_independent_actual_grid_review',
        orders_checked=488,candidate_cells_checked=5978,months_checked=122,prefixes_checked=123,
        revenue_feature_matrices_verified=3,new_model_fits=0,available_cells=int(expected['arrays']['common_available'].sum()),
        max_score_error=max(float(np.max(np.abs(saved_scores[name]-scores[name]))) for name in NAMES),
        limitations=['Source admission is conditional on the independently reviewed sparse pilot and original-release vintage assumptions.',
                     'Source admission oracle is separately hash bound; arithmetic and score/order reconstruction here are independent of new producers.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--plan-sha',required=True)
    args=p.parse_args();result=audit(ROOT,args.plan,args.plan_sha)
    path=ROOT/OUT/'INDEPENDENT_PREPARE.json'
    with path.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))


if __name__=='__main__':main()
