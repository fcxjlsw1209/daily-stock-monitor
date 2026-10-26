"""203 source-clock/integration review: metadata/AST/hash only, before PLAN."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/volume203_CODE_REVIEW.json'
SIX=('work/monthly_volume_pair203_host.py','work/test_monthly_volume_pair203_host.py','work/monthly_volume_pair203_kernel.py','work/test_monthly_volume_pair203_kernel.py','work/monthly_volume_pair203_independent.py','work/test_monthly_volume_pair203_independent.py')
EXPECTED=('e9dc238e90c4e18be0775d3f8e3414767f54c1a65c506648d6b5a766913cdcec', '71f6b72e9ed104a8b43d7a62f1d193c6ce039a4f6100a51aaa710e966f75f8b8', 'c658b539519766be1f8fe8d284c101bd5e3b6e96918633e15235f8051020c8e6', 'c97e8cd2a173f71487e5592ccfe9bb8c9957e462c5d12e265b5b4e5a1768c72f', 'bcea8256ccf56b04eaf81b56d78dac232d2102c5ba98168054f21645e41d8ddc', 'c552348256cabaa8c68dcadb36cd718fb5435ad3f82c8d3e24faf7ed4c1df037')
TEST_COUNT=62
TEST_METHODS=(28,18,16)
TESTS_SHA='ecdc20b85fb07a16b22162d0e0c88c942c80cd5b7dbf10972cc5cf5cae0cc009'
CONTEXT='work/monthly_volume_pair203_context.json'
CONTEXT_SHA='b448260b4feb624d18c4813a5d0c671056e45ee2f6ffe62b1f15c73aece17806'
SOURCE='work/volume203_SOURCE_REVIEW.json'
SOURCE_SHA='efc6f15cc052ecc66734f275a267e453694ec78e837deae600d5d22d7a0190c2'
COVERAGE='work/after202_direction_coverage.json'
COVERAGE_SHA='7b441663393d86c3b4bf8c7c5350aa34683b4fdc9a0dbaca5e5704be0b3eedf6'
TESTS='work/monthly_volume_pair203_TESTS.json'
PRIOR='outputs/monthly_breadth_pair202/PLAN.json'
PRIOR_SHA='91edf812d0dbb9ded48bf2a40a0d93470c38f23363d650c200f53c39828e07f3'
DEPENDENCIES=('outputs/monthly_path70_research/incumbent_adapter.py', 'outputs/monthly_local_linear70/study.py', 'outputs/monthly_local_linear70/PLAN.json', 'outputs/monthly_local_linear70/DATA.npz', 'outputs/monthly_local_linear70/META.json', 'outputs/monthly_local_linear70/INCUMBENT_POLICY.json', 'outputs/monthly_local_linear70/local_linear_ORDERS.json', 'outputs/monthly_daily_episodes/episodes.py', 'outputs/monthly_model_scope70/study.py', 'outputs/monthly_target_definition70/study.py', 'outputs/monthly_state_rank_research/state_rank.py', 'outputs/monthly_quality_research/explore.py', 'outputs/monthly_stability_monitor/strategy.py', 'outputs/monthly_stability_monitor/inputs.py', 'outputs/monthly_regime_monitor/strategy.py', 'outputs/monthly_strict_relative/strict_study.py', 'outputs/monthly_paired_target143/CALENDAR.json', 'work/monthly_nested_trend199_kernel.py', 'work/ptr169_rebuild_evaluate.py', 'work/ptr169_results_independent.py', 'work/price_distribution171_independent.py', 'work/monthly_sector_transition189_independent.py', 'work/monthly_sector_recurrence192_independent.py', 'work/monthly_recovery_durability197_host.py', 'work/monthly_pairwise_majority196_host.py', 'work/monthly_median_risk201_host.py', 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json', 'run_local.sh', 'local_run.py', 'local.sandbox.sb')

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads((ROOT/path).read_text())
def main():
    require(len(EXPECTED)==6 and len(TEST_METHODS)==3 and TEST_COUNT is not None and TESTS_SHA!='PENDING','Final sources/tests required')
    require(not OUT.exists() and not (ROOT/'outputs/monthly_volume_pair203').exists(),'Writeonce preregistration review')
    pins={}
    def bind(name,expected=None):
        p=Path(name);p=(p if p.is_absolute() else ROOT/p).resolve();require(p.is_relative_to(ROOT),'Nonlocal source')
        key=str(p.relative_to(ROOT));h=sha(p);require(expected is None or expected==h,'Changed source '+key)
        require(key not in pins or pins[key]==h,'Conflicting pin '+key);pins[key]=h;return h
    code={p:bind(p,h) for p,h in zip(SIX,EXPECTED)}
    for p,h in [(CONTEXT,CONTEXT_SHA),(SOURCE,SOURCE_SHA),(COVERAGE,COVERAGE_SHA),(TESTS,TESTS_SHA),(PRIOR,PRIOR_SHA)]:bind(p,h)
    bind(__file__)
    ctx,src,tst,prior=read(CONTEXT),read(SOURCE),read(TESTS),read(PRIOR)
    require(ctx['round']==203 and ctx['arms']==['incumbent_local67','volume_paired_relative','relative_mean_control'],'Context arms')
    require(all(ctx[k]==0 for k in ['pre_registration_actual_new_features','pre_registration_actual_orders','pre_registration_actual_returns']),'Prospective scope')
    f=ctx['formation'];require((f['price_count'],f['earliest_offset'],f['latest_offset'],f['volume_count'],f['volume_earliest_offset'],f['volume_latest_offset'],f['blocks'],f['block_sessions'])==(253,-294,-42,315,-356,-42,12,21),'Windows')
    require(f['endpoint_indices']==list(range(0,253,21)) and ctx['integration']['dose']==.01 and ctx['integration']['max_absolute_delta']==.005,'Endpoints/dose')
    require(src['passed'] is True and src['source_only'] is True and src['unresolved_blockers']==[] and src['context_sha256']==CONTEXT_SHA and src['coverage_sha256']==COVERAGE_SHA,'Source receipt')
    for k in ['actual_features_computed','actual_signal_returns_computed','actual_medians_computed','actual_high_flags_computed','actual_activity_blocks_computed','actual_covariance_computed','actual_ranks_computed','actual_deltas_computed','actual_orders_computed','actual_returns_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer','observers_accessed']:
        require(src[k] is False,'Source boundary '+k)
    require(src['actual_source_coverage_computed'] is True and src['strict_pit'] is False and len(src['source_sha256'])==357 and len(src['raw_source_hashes'])==64,'Source closure')
    cov=src['source_window_coverage'];require((cov['price_count'],cov['volume_count'],cov['skip_sessions'],cov['checked_cells'],cov['common_available_cells'],cov['accepted_source_price_observations'],cov['accepted_source_volume_observations'])==(253,315,42,5978,3705,1874730,1167075),'Coverage evidence')
    for m in [src['source_sha256'],src['raw_source_hashes'],ctx['evidence_sha256']]:
        for p,h in m.items():bind(p,h)
    require(tst['passed'] is True and tst['synthetic_only'] is True and tst['tests']==TEST_COUNT and tst['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal tests')
    for p,h in tst['code_sha256'].items():bind(p,h)
    for p in SIX:require(tst['code_sha256'][str(ROOT/p)]==code[p],'Tested source differs '+p)
    for p in DEPENDENCIES:
        h=prior['input_sha256'].get(str(ROOT/p)) or ctx['evidence_sha256'].get(p)
        require(h is not None,'Missing dependency '+p);bind(p,h)
    require(len(read('work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'))==21444,'Protection scope')
    require(read('outputs/monthly_local_linear70/PLAN.json')['raw_source_hashes']==src['raw_source_hashes'],'Original raw64')
    txt={p:(ROOT/p).read_text() for p in SIX};trees={p:ast.parse(s) for p,s in txt.items()}
    funcs={n.name:ast.get_source_segment(txt[SIX[0]],n) for n in trees[SIX[0]].body if isinstance(n,ast.FunctionDef)}
    require(funcs['prepare'].index('gate(expected)')<funcs['prepare'].index('study.load()'),'Compute beforePLAN')
    require(funcs['register'].index('code_review_binding()')<funcs['register'].index('OUT.mkdir()'),'Review before registration')
    require("pins = dict(read(PRIOR / 'PLAN.json')['input_sha256'])" in funcs['register'] and "'monthly_nested_trend199_kernel.py'" in funcs['code_files'],'Transitive closure')
    require('delta=DOSE*r' in funcs['scale_and_delta'] and 'if n:delta[0]=0.' in funcs['scale_and_delta'] and 'state_calm10' not in funcs['scale_and_delta'],'Delta clock/scope')
    require('changed = deltas[arm] != 0.' in funcs['decisions'] and 'view[changed] += deltas[arm][changed]' in funcs['decisions'] and "study.adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan)" in funcs['decisions'] and "pre_score=pre, effects=arrays['effects'], episode_policy=policy" in funcs['decisions'] and 'boundary_selector=study.ep.select_boundary' in funcs['decisions'],'Frozen pre-score integration')
    require('for n in range(123)' in funcs['prepare'] and 'len(arrays) == 27' in funcs['prepare'] and 'require_source_support(audit,values)' in funcs['prepare'],'Prepare evidence gates')
    require("orders[arm][:TRAIN]" in funcs['select'] and "meta['ends'][87]" in funcs['select'] and funcs['diagnose'].index('frozen_selection(expected, orders)')<funcs['diagnose'].index('ev.run_path('),'First88 freeze')
    require('monthly_volume_pair203_kernel' not in txt[SIX[4]],'Independent oracle imports kernel')
    methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[p])) for p in SIX[1::2]}
    require(tuple(methods.values())==TEST_METHODS and sum(TEST_METHODS)==TEST_COUNT,'Fixed test methods')
    checks=[
      {'area':'Source clocks','passed':True,'finding':'Both targets require all253 stock/own-ETF prices t-294..t-42 plus315 stock stored volumes t-356..t-42, positive finite nonboolean numeric. No fill/compression or unavailable-cell features. Query-date original mapped benchmark held overformation; no further data source.'},
      {'area':'Activity pairing','passed':True,'finding':'252 strict preceding63 median comparisons, currentvolume strictlyabove median, tiesfalse. First flag volumeindex63 aligns firstprice returnindex1. Twelve disjoint21flags aggregate counts/21; twelveexcessloggains use separate endpoint math.log differences. Canonical fsum means/covariance and exact constant branch preserve Q=C; noepsilon.'},
      {'area':'Common support and small adjustment','passed':True,'finding':'Independent ranks across same available stockdomain prior totradingeligibility. Direct .01rank correction initialrow0; calm active. Frozen Local67config4 pre-score+original singleepisode,27arrays and oldorders preserved. No originalcore/eligibility/hold/ETF/position change.'},
      {'area':'Verification and selection','passed':True,'finding':'Independent allcells/detailed audits, source-support mask,315/253clock and counts. Five physicalsourceprefixes and123orderprefixes. Same first88gates versusincumbent andmeancontrol, onlycandidatepromotable; later34diagnostic.9paths45slices18traininggroups,protected21444closure.'},
      {'area':'Overlap and limits','passed':True,'finding':'Prior absolute-return abnormal-volume covariance exists; six older formula records unavailable. This conditional industry-relative pairing study is not certified novel. Stored volume is not buyerflow/native-turnover; retrospective provider/classification/membership limits retained. Reused holdout is not pristineOOS.'}
    ]
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],code_sha256=code,input_sha256=pins,review_code_sha256=sha(__file__),independent_reviewer='root_static_review_of_delegated_numerical_implementations',review_method='Parent static mathematical/integration and fixed metadata/hash review of separately authored kernel/host/oracle. Parent also authored source-coverage audit; this review does not claim independent raw-vendor reconstruction.',context_sha256=CONTEXT_SHA,source_review_sha256=SOURCE_SHA,coverage_sha256=COVERAGE_SHA,formal_test_receipt_sha256=TESTS_SHA,synthetic_tests=TEST_COUNT,synthetic_test_methods=methods,synthetic_test_receipt_verified=True,formal_tests_rerun_by_reviewer=False,review_checks=checks,resolved_findings=[],actual_features_computed=False,actual_signal_returns_computed=False,actual_medians_computed=False,actual_high_flags_computed=False,actual_activity_blocks_computed=False,actual_covariance_computed=False,actual_ranks_computed=False,actual_deltas_computed=False,actual_orders_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,production_modules_imported=False,limitations=ctx['limitations'])
    for p,h in pins.items():require(sha(ROOT/p)==h,'Changed duringreview '+p)
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),input_pins=len(pins),synthetic_tests_verified=TEST_COUNT)))
if __name__=='__main__':main()
