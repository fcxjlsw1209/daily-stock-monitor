"""Round202 independent static review and exact metadata/hash gate.

No production imports, source-price decoding, feature, order or return execution.
Only the write-once review receipt is created, before PLAN registration.
"""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/breadth202_CODE_REVIEW.json'
SIX=('work/monthly_breadth_pair202_host.py','work/test_monthly_breadth_pair202_host.py',
     'work/monthly_breadth_pair202_kernel.py','work/test_monthly_breadth_pair202_kernel.py',
     'work/monthly_breadth_pair202_independent.py','work/test_monthly_breadth_pair202_independent.py')
EXPECTED=('7dd0c0fa76cab0f27d048ad8f43dcda48e328ec612af136e9b6a91bdc8227cbe', '48fcec38769bc15b293f37b057864e5370f3af635287b6a0b5873e97e137e7fb', '7188416e129f84c07abc8bada368af3da1c1b3807e83eaee6a7a95d9b9888862', 'e807313ad032c148c4dc1894369efc50b7fd4c43a293ebebd52054cde8f2ed47', 'da54cf664a1c511210c187c0e57dfb434e66bcbe1be799d27223fd72e6cba1d9', '2aac40f1aeeebe31d55e24484787b5c90e78deade5e9196d1a8ed1911759b138')
TEST_COUNT=57
TEST_METHODS=(28,14,15)
CONTEXT='work/monthly_breadth_pair202_context.json'
CONTEXT_SHA='0e7eb021c65a22b843c285f83525ce9ff0bc5e908e2a5a0a4cf0b1fd66952eaf'
SOURCE='work/breadth202_SOURCE_REVIEW.json'
SOURCE_SHA='7ff43af545a29b76b8becd1f0d65b8b560baf723d484ce52d80921285b24dd5d'
COVERAGE='work/after201_breadth_pair_coverage.json'
COVERAGE_SHA='08009b20a6e9a4664aaad162d381aa61d88634e5be48fddd97f5e9239bf853a9'
TESTS='work/monthly_breadth_pair202_TESTS.json'
TESTS_SHA='08f7afa29cb63d87a7c0301bbe5287f088ff4ee065d9ba117e11dd0c9bd81d7b'
PRIOR='outputs/monthly_median_risk201/PLAN.json'
PRIOR_SHA='4eb3822b092c7c64809fd2989dd18a62ec25ae3e52eed9a6fea06d0ba48306cd'
DEPENDENCIES=(
    'outputs/monthly_path70_research/incumbent_adapter.py',
    'outputs/monthly_local_linear70/study.py','outputs/monthly_local_linear70/PLAN.json',
    'outputs/monthly_local_linear70/DATA.npz','outputs/monthly_local_linear70/META.json',
    'outputs/monthly_local_linear70/INCUMBENT_POLICY.json','outputs/monthly_local_linear70/local_linear_ORDERS.json',
    'outputs/monthly_daily_episodes/episodes.py','outputs/monthly_model_scope70/study.py',
    'outputs/monthly_target_definition70/study.py','outputs/monthly_state_rank_research/state_rank.py',
    'outputs/monthly_quality_research/explore.py','outputs/monthly_stability_monitor/strategy.py',
    'outputs/monthly_stability_monitor/inputs.py','outputs/monthly_regime_monitor/strategy.py',
    'outputs/monthly_strict_relative/strict_study.py','outputs/monthly_paired_target143/CALENDAR.json',
    'work/monthly_nested_trend199_kernel.py','work/ptr169_rebuild_evaluate.py',
    'work/ptr169_results_independent.py','work/price_distribution171_independent.py',
    'work/monthly_sector_transition189_independent.py','work/monthly_sector_recurrence192_independent.py',
    'work/monthly_recovery_durability197_host.py','work/monthly_pairwise_majority196_host.py',
    'work/monthly_median_risk201_host.py','work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',
    'run_local.sh','local_run.py','local.sandbox.sb')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads((ROOT/path).read_text())


def main():
    require(len(EXPECTED)==6 and TEST_COUNT is not None and len(TEST_METHODS)==3 and
            not TESTS_SHA.startswith('PENDING'),'Final six sources and formal test receipt must be fixed')
    require(not OUT.exists(),'Code review is write-once')
    require(not (ROOT/'outputs/monthly_breadth_pair202').exists(),'Review must precede registration')
    pins={}
    def bind(name,expected=None):
        p=Path(name);p=(p if p.is_absolute() else ROOT/p).resolve()
        require(p.is_relative_to(ROOT),'Nonlocal review input')
        key=str(p.relative_to(ROOT));digest=sha(p)
        require(expected is None or digest==expected,'Changed evidence: '+key)
        require(key not in pins or pins[key]==digest,'Conflicting input: '+key)
        pins[key]=digest
        return digest
    code={p:bind(p,h) for p,h in zip(SIX,EXPECTED)}
    for p,h in ((CONTEXT,CONTEXT_SHA),(SOURCE,SOURCE_SHA),(COVERAGE,COVERAGE_SHA),(TESTS,TESTS_SHA),(PRIOR,PRIOR_SHA)):
        bind(p,h)
    bind(__file__)
    context,source,tests,prior=read(CONTEXT),read(SOURCE),read(TESTS),read(PRIOR)
    require(context['round']==202 and context['arms']==['incumbent_local67','breadth_paired_relative','relative_mean_control'] and
        context['candidate']=='breadth_paired_relative' and context['control_nonpromotable']=='relative_mean_control','Prospective arms differ')
    for k in ('pre_registration_actual_new_features','pre_registration_actual_orders','pre_registration_actual_returns'):
        require(type(context[k]) is int and context[k]==0,'Pre-registration scope: '+k)
    f=context['formation'];require(f['price_count']==253 and f['earliest_offset']==-294 and f['latest_offset']==-42 and
        f['blocks']==12 and f['block_sessions']==21 and f['endpoint_indices']==list(range(0,253,21)) and
        context['integration']['dose']==.01 and context['integration']['max_absolute_delta']==.005,'Formation or dose differs')
    require(source['passed'] is True and source['unresolved_blockers']==[] and source['source_only'] is True and
        source['context_sha256']==CONTEXT_SHA and source['coverage_sha256']==COVERAGE_SHA and source['strict_pit'] is False,'Source admission differs')
    for k in ('actual_features_computed','actual_signal_returns_computed','actual_breadth_computed','actual_covariance_computed',
              'actual_ranks_computed','actual_deltas_computed','actual_orders_computed','actual_returns_computed',
              'actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer','observers_accessed'):
        require(source[k] is False,'Source-only boundary crossed: '+k)
    require(source['actual_source_coverage_computed'] is True and source['new_model_fits']==0 and
        len(source['raw_source_hashes'])==64 and len(source['source_sha256'])==334 and
        source['raw_source_parity']['exact_source_map_equal'] is True,'Source closure differs')
    cov=source['source_window_coverage'];require(cov['price_count']==253 and cov['skip_sessions']==42 and
        cov['checked_cells']==5978 and cov['common_available_cells']==3705 and cov['unavailable_cells']==2273 and
        cov['available_cells_by_basket_size']=={'8':3407,'9':298} and
        cov['accepted_source_price_observations']==9449044,'Frozen source coverage differs')
    for mapping in (source['source_sha256'],source['raw_source_hashes'],context['evidence_sha256']):
        for p,h in mapping.items():bind(p,h)
    require(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==TEST_COUNT and
        tests['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal synthetic evidence differs')
    for p,h in tests['code_sha256'].items():bind(p,h)
    for p in SIX:require(tests['code_sha256'][str(ROOT/p)]==code[p],'Tested source differs: '+p)
    for p in DEPENDENCIES:
        expected=prior['input_sha256'].get(str(ROOT/p)) or context['evidence_sha256'].get(p)
        require(expected is not None,'Missing frozen critical dependency: '+p)
        bind(p,expected)
    require(len(read('work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'))==21444,'Protected manifest size')
    require(read('outputs/monthly_local_linear70/PLAN.json')['raw_source_hashes']==source['raw_source_hashes'],'Original raw64 differs')
    texts={p:(ROOT/p).read_text() for p in SIX};trees={p:ast.parse(s) for p,s in texts.items()}
    host=texts[SIX[0]];funcs={n.name:ast.get_source_segment(host,n) for n in trees[SIX[0]].body if isinstance(n,ast.FunctionDef)}
    require(funcs['prepare'].index('gate(expected)')<funcs['prepare'].index('study.load()'),'Actual compute before PLAN gate')
    require(funcs['register'].index('code_review_binding()')<funcs['register'].index('OUT.mkdir()'),'Registration before independent review')
    require("pins = dict(read(PRIOR / 'PLAN.json')['input_sha256'])" in funcs['register'] and
        "'monthly_nested_trend199_kernel.py'" in funcs['code_files'],'Transitive frozen dependency closure absent')
    require("r=np.asarray(rank[:n],float)" in funcs['scale_and_delta'] and 'delta=DOSE*r' in funcs['scale_and_delta'] and
        'if n:delta[0]=0.' in funcs['scale_and_delta'] and 'state_calm10' not in funcs['scale_and_delta'],'Auxiliary state scope differs')
    require('changed = deltas[arm] != 0.' in funcs['decisions'] and 'view[changed] += deltas[arm][changed]' in funcs['decisions'] and
        "study.adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan)" in funcs['decisions'] and
        "pre_score=pre, effects=arrays['effects'], episode_policy=policy" in funcs['decisions'] and
        'boundary_selector=study.ep.select_boundary' in funcs['decisions'] and
        'orders[arm] = study.rp.make_orders(chosen, score, meta)' in funcs['decisions'],'Frozen adapter wiring differs')
    require('for n in range(123)' in funcs['prepare'] and 'len(arrays) == 27' in funcs['prepare'] and
        'require_source_support(audit,values)' in funcs['prepare'],'Original/source/prefix gates absent')
    require("orders[arm][:TRAIN]" in funcs['select'] and "meta['ends'][87]" in funcs['select'] and
        funcs['diagnose'].index('frozen_selection(expected, orders)')<funcs['diagnose'].index('ev.run_path('),'First88 selection sequence differs')
    require('monthly_breadth_pair202_kernel' not in texts[SIX[4]],'Oracle depends on producer math')
    methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[p])) for p in SIX[1::2]}
    require(tuple(methods.values())==TEST_METHODS and sum(TEST_METHODS)==TEST_COUNT,'Final synthetic method counts differ')
    checks=[
      dict(area='Frozen sources and clocks',passed=True,finding='Original raw64, stored adjusted prices, query-date strict_study mapping and official calendar are pinned. Both arms require all253 positive finite nonboolean numeric prices t-294..t-42 for stock, own sector and every consumed peer. Own sector is excluded only from breadth, remaining independently required. Fixed nine gives M8 or M9; XLC/XLRE are own-only. Mapping clock t is held over formation and does not imply certified historical classification.'),
      dict(area='Canonical paired statistic',passed=True,finding='Twelve nonoverlapping21-session gains use separate math.log endpoint differences. Strict end<start counts down and ties do not. C=fsum(e)/12, wbar=fsum(w)/12, population centered covariance uses fsum/12, Q=C+cov. Exact original e or w equality produces cov0 and direct Q=C; no nearconstant epsilon, rescaling, log(end/start), ordering constraint Q<=C or fitted model. Independent scalar hand fixture returns positive and negative covariance at identical marginal mean.'),
      dict(area='Common symmetric ranks',passed=True,finding='Separate Q/C ranks use the same common available mapped-stock population before eligibility. Pair-count formula (less+(equal-1)/2)/(N-1)-.5 matches pandas average rank transform. N<=1/allties return0. Unavailable raw values remainNaN while missing/unmapped/ETF ranks are0. Finite0 is valid; no second rank or postepisode reranking.'),
      dict(area='Allstate auxiliary and original episode',passed=True,finding='Exactly .01*rank with maximumabsolute .005 is added only at nonzero delta cells to Local67config4 modified_pre_score. Initial partial delta0; later calm remains active. Path adapter uses incumbent66,dose0,allNaN hand factor, original baseline/core/cw/effects/policy. One original episode receives modifiedpre; original27 arrays remain byte-identical and returned choices form orders directly.'),
      dict(area='Independent evidence and accounting',passed=True,finding='Independent scalar mapping/source/log/down/covariance/pair-count rank and full audit compare exactly without tolerance. Main-pass peraccepted-cell counts are253*(2+M) prices,12 excessblocks,12 breadthblocks,12*M signs,1 descriptive covariance,2rawscores,2ranks. No new predictive fits; rejected partial reads and prefix repeats excluded. Full5978cells,123orderprefixes and5physical adj+sectors prefixes required; source admission masks and baskets checked percell.'),
      dict(area='Selection and closure',passed=True,finding='Only candidate promotable. Frozen first88 close CAGR gain.005 versus incumbent and costedopen noninferiority; both paths noninferior to matchedmean under unchanged Sharpe/MDD/early53/recent35 guards. At least3changed holding sets and each era versus eachreference. Later34 runs only after frozen selection. Independent192->189->171->169 accounting verifies9paths45slices18train; original fixedunits/rebalance/no terminal liquidation preserved. Closure requires21444protection and advances199/1114 to200/1117 with queueempty.'),
      dict(area='Prospective hashes and tests',passed=True,finding='Final six numerical/test sources and formal synthetic receipt bind exact bytes. PLAN inherits full201 dependency closure, explicitly includes199metadata helper, source334pins, mapping, original adapter and independentaccounting chain. Review executes only metadata/hash/AST checks, imports no production module and precedes registration.'),
    ]
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
        code_sha256=code,input_sha256=pins,review_code_sha256=sha(__file__),independent_reviewer='nested199_code_audit',
        review_method='Independent static mathematical/source-clock/integration/test review plus exact metadata/dependency hash verification; no actual research calculations.',
        context_sha256=CONTEXT_SHA,source_review_sha256=SOURCE_SHA,coverage_sha256=COVERAGE_SHA,
        formal_test_receipt_sha256=TESTS_SHA,synthetic_tests=TEST_COUNT,synthetic_test_methods=methods,
        synthetic_test_receipt_verified=True,formal_tests_rerun_by_reviewer=False,review_checks=checks,resolved_findings=[],
        actual_features_computed=False,actual_signal_returns_computed=False,actual_breadth_computed=False,
        actual_covariance_computed=False,actual_ranks_computed=False,actual_deltas_computed=False,actual_orders_computed=False,
        actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,
        production_modules_imported=False,limitations=[
          'This pre-registration code review is not actual feature preparation, performance evidence or adoption.',
          'Temporal covariance is descriptive and may reflect beta or defensive exposure, not causal issuer skill.',
          'Matched mean shares observations/support and aggregate weight mass, not concentration or effective sample size.',
          'Proxy membership, historical classification assumptions and retrospective data imply strictPIT false; reused later34 is not pristine OOS.',
          'Unreadable earlier sector_gate/leadership_concentration definitions prevent exhaustive novelty claims.',
          'Reviewer binds the protected21444manifest; registration and closure verify actual protected bytes.'])
    for p,h in pins.items():require(sha(ROOT/p)==h,'Evidence changed during review: '+p)
    with OUT.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),review_code_sha256=sha(__file__),
        code_files=6,input_pins=len(pins),synthetic_tests_verified=TEST_COUNT,unresolved_blockers=[])))


if __name__=='__main__':main()
