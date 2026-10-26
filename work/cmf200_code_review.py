"""Independent static round200 code review; metadata/hash validation only.

Imports no production module and reads no market arrays. Does not compute
actual features, scores, orders or returns, rerun formal tests or register.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/cmf200_CODE_REVIEW.json'
SIX=(
 'work/monthly_cmf_state200_host.py', 'work/test_monthly_cmf_state200_host.py',
 'work/monthly_cmf_state200_kernel.py', 'work/test_monthly_cmf_state200_kernel.py',
 'work/monthly_cmf_state200_independent.py', 'work/test_monthly_cmf_state200_independent.py',
)
EXPECTED=(
 'e13f7ebc8efa072682d8ebef0276194825475bdb3fa070df2785ded63632ecb7',
 '849a119a5737aa1df3e4e1fdded2d0f18f53197fd799f33ac09f909f45b90685',
 'f1fabcb74e94277493fa2cb4865c05156b37c70906ac28fa7aaccd2487b843e4',
 '7d41db911fbe1ba38d263ca2f389e3f50ce0e9cf2927355268656fb4cdd6d1df',
 '0da1474f9ee4817524131e7a0b0071bc3fb503a11b7686f45fa0ffbbd6305592',
 'e22dd58dd32aa744be3300e95085e527d366b89c29e73a574854b226b06003f2',
)
CONTEXT='work/monthly_cmf_state200_context.json'
CONTEXT_SHA='bccab811afc6d2a1fa27d26e205b30b9d0b75df8ca83b69e03734177217d80a0'
SOURCE='work/cmf200_SOURCE_REVIEW.json'
SOURCE_SHA='9742ef91bba7617a95fc738911d167e6abb82b94d95a73f59455253bca815ee7'
COVERAGE='work/after199_cmf_state_coverage.json'
COVERAGE_SHA='1e34e1d4c5177a48ccaa60165060d43afcb6f502203f589445e8ba294b34d128'
TESTS='work/monthly_cmf_state200_TESTS.json'
TESTS_SHA='facee28af4b99f42a55059a1c310abb09f2f91bd3d1dd3fb74e943d0d6feb591'


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    require(not OUT.exists(),'Code review is write-once')
    require(not (ROOT/'outputs/monthly_cmf_state200').exists(),'Code review must precede registration')
    pins={}
    def bind(name,expected=None):
        path=Path(name)
        path=(path if path.is_absolute() else ROOT/path).resolve()
        require(path.is_relative_to(ROOT),'Nonlocal review evidence')
        key=str(path.relative_to(ROOT));digest=sha(path)
        require(expected is None or digest==expected,'Changed evidence: '+key)
        require(key not in pins or pins[key]==digest,'Conflicting evidence: '+key)
        pins[key]=digest
        return digest
    code={p:bind(p,h) for p,h in zip(SIX,EXPECTED)}
    for p,h in ((CONTEXT,CONTEXT_SHA),(SOURCE,SOURCE_SHA),(COVERAGE,COVERAGE_SHA),(TESTS,TESTS_SHA)):
        bind(p,h)
    bind(__file__)
    context,source,tests=read(CONTEXT),read(SOURCE),read(TESTS)
    require(context['round']==200 and context['arms']==['incumbent_local67','cmf_empirical_state','cmf_half_sign_control']
        and context['candidate']=='cmf_empirical_state' and context['control']=='cmf_half_sign_control','Fixed arms differ')
    require(context['pre_registration_actual_new_features']==0 and context['pre_registration_actual_orders']==0
        and context['pre_registration_actual_returns']==0 and context['pre_registration_new_predictive_fits']==0,
        'Prospective actual counters differ')
    require(source['passed'] is True and source['unresolved_blockers']==[] and source['context_sha256']==CONTEXT_SHA
        and source['coverage_sha256']==COVERAGE_SHA and source['strict_pit'] is False,'Source review differs')
    for key in ('actual_features_computed','actual_orders_computed','actual_returns_computed','actual_sigma_computed',
        'actual_state_cdf_computed','actual_cmf_computed','actual_cmf_ranks_computed','actual_deltas_computed'):
        require(source[key] is False,'Source-only boundary crossed: '+key)
    require(len(source['raw_source_hashes'])==64 and len(source['source_sha256'])==305,'Source binding counts differ')
    require(source['spy_state_source_coverage']['complete_positive_windows']==122
        and source['spy_state_source_coverage']['window_sessions']==315,'Source315 coverage differs')
    for mapping in (source['source_sha256'],source['raw_source_hashes'],context['evidence_sha256']):
        for p,h in mapping.items(): bind(p,h)
    require(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==59,'Formal59 tests required')
    require(tests['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Pinned runtime differs')
    for p,h in tests['code_sha256'].items(): bind(p,h)
    for p in SIX: require(tests['code_sha256'][str(ROOT/p)]==code[p],'Final code differs from tested bytes')
    dependencies=(
      'outputs/monthly_local_linear70/study.py','outputs/monthly_local_linear70/PLAN.json',
      'outputs/monthly_local_linear70/INCUMBENT_POLICY.json','outputs/monthly_local_linear70/DATA.npz',
      'outputs/monthly_state_rank_research/state_rank.py','outputs/monthly_target_definition70/study.py',
      'outputs/monthly_model_scope70/study.py','outputs/monthly_quality_research/explore.py',
      'outputs/monthly_stability_monitor/strategy.py','outputs/monthly_regime_monitor/strategy.py',
      'outputs/monthly_path70_research/incumbent_adapter.py','outputs/monthly_daily_episodes/episodes.py',
      'outputs/monthly_quality_research/FEATURES.npz','outputs/monthly_stability_research/FEATURES.npz',
      'outputs/monthly_return_priority/return_priority.py','outputs/monthly_paired_target143/CALENDAR.json',
      'work/monthly_nested_trend199_kernel.py','work/ptr169_rebuild_evaluate.py','work/ptr169_results_independent.py',
      'work/price_distribution171_independent.py','work/monthly_sector_transition189_independent.py',
      'work/monthly_sector_recurrence192_independent.py',
      'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',
      'run_local.sh','local_run.py','local.sandbox.sb',
    )
    prior=read('outputs/monthly_nested_trend199/PLAN.json')
    for p in dependencies: bind(p,prior['input_sha256'].get(str(ROOT/p)))
    require(len(read(dependencies[22]))==21444,'Protected manifest count differs')
    require(read(dependencies[1])['raw_source_hashes']==source['raw_source_hashes'],'Original raw64 closure differs')
    texts={p:(ROOT/p).read_text() for p in SIX};trees={p:ast.parse(s) for p,s in texts.items()}
    host=texts[SIX[0]]
    functions={node.name:ast.get_source_segment(host,node) for node in trees[SIX[0]].body if isinstance(node,ast.FunctionDef)}
    require(functions['prepare'].index('gate(expected)')<functions['prepare'].index('study.load()'),'Missing actual compute gate')
    require(functions['register'].index('code_review_binding()')<functions['register'].index('OUT.mkdir()'),'Missing review gate')
    require('delta=(coefficient[:,None]*q*(g-state)[:,None])*scale' in functions['scale_and_delta']
        and 'if n:delta[0]=0.' in functions['scale_and_delta'],'Changed prescribed delta')
    require('result=1-(.025*calm[:,None]+.05*weak[:,None]+.01*active)/(1-cw)' in functions['support_scale'],
        'Changed existing pre-support scale')
    require("adapter.score_and_choose('incumbent66',0.,np.full(arrays['baseline'].shape,np.nan)" in functions['decisions']
        and 'prefix[changed]+=deltas[arm][changed]' in functions['decisions']
        and 'orders[arm]=study.rp.make_orders(chosen,score,meta)' in functions['decisions'],'Original single-episode wiring differs')
    require('for n in range(123)' in functions['prepare'] and 'len(arrays) == 27' in functions['prepare']
        and "read(LOCAL / 'local_linear_ORDERS.json')" in functions['prepare'],'Original/prefix invariants absent')
    require('orders[arm][:TRAIN]' in functions['select'] and "meta['ends'][87]" in functions['select'],'First88 boundary differs')
    require(functions['diagnose'].index('frozen_selection(expected, orders)')<functions['diagnose'].index('ev.run_path('),
        'Later outcomes lack frozen selection')
    require('monthly_cmf_state200_kernel' not in texts[SIX[4]],'Oracle imports producer')
    methods={p:sum(isinstance(x,ast.FunctionDef) and x.name.startswith('test_') for x in ast.walk(trees[p])) for p in SIX[1::2]}
    require(list(methods.values())==[25,16,18] and sum(methods.values())==59,'Formal source test count differs')
    checks=[
      dict(area='Exact original state and source windows',passed=True,evidence=[SIX[2],SIX[4],SOURCE],
        finding='Original SPY pct_change(fill_method=None), rolling63 sample std ddof1, no annualization for relative state. Each query requires exactly315 official source prices t-314..t, finite positive, and252 finite nonnegative canonical sigmas ending query. No historical fill, compressed sessions or late data conversion. Current252 median matches the original strict above-median +1 else-1 state; calm separately uses sigma*sqrt252<.10.'),
      dict(area='Sign-preserving empirical gate and fixed half control',passed=True,evidence=[SIX[0]+':scale_and_delta',SIX[2]+':market_gate',SIX[4]+':gate_oracle',SIX[3],SIX[5]],
        finding='g is exact0 for sigma==median, otherwise (2*less+equal)/252-1; the denominator is252 rather than251. Exactmedian neutrality prevents asymmetric tied blocks reversing the original sign. Constant window gives0, extremes have absolute251/252. Control is.5*r. Scalar less/equal oracle and host reconstruct exactly; no fitted probabilities, interpolation, thresholds or tie tolerance.'),
      dict(area='Original CMF, percentile convention and source semantics',passed=True,evidence=[SIX[2]+':build_features/cmf_ranks',SIX[4]+':independent_features/rank_oracle',dependencies[8],dependencies[9]],
        finding='Preserves original signal close/high/low/volume and zero-spread location0, negative rolling63 CMF, inf-to-NaN original convention. Average percentile among finite supplied49 values precedes eligibility, then missing neutral0. Independent scalar ranks use (less+(equal+1)/2)/m-.5; singleton+.5 and original all-tie offset remain. No rerank, m-1 denominator, extra eligibility mask, OHLCV adjustment or zero-volume filter. Actual source units remain mixed frozen signal conventions.'),
      dict(area='Effective pre-score CMF coefficient and inherited budgets',passed=True,evidence=[SIX[0]+':support_scale/scale_and_delta',SIX[4]+':independent_deltas',SIX[1]+':test_original_cmf_support_coefficient_against_frozen_pure_functions',*dependencies[4:8],COVERAGE],
        finding='Baseline a=.075 calm, otherwise.075*(1-.10*I[r>0]/.48). Source chain independently shows pre-score scale K=1-(.025calm+.05weak+.01active)/(1-cw), active calm & ~weak & extension_available & forecast_available & eligible & finite(rank_log_excess). Original prior budget scale is1 because total<=.075. Target/local corrections change ranks only; scope subtracts.01support on active. Host and oracle use exact fixed K with cw .50/.52; synthetic extracted original pure functions verify the coefficient algebra without loading actual arrays.'),
      dict(area='Delta, first partial, missing neutral and bit-exact zero',passed=True,evidence=[SIX[0]+':scale_and_delta/decisions',SIX[4]+':independent_deltas',SIX[1],SIX[5]],
        finding='Delta=(a*q*(g-r))*K, K in[.85,1], bounded by.0375. Row0 forced0 after raw validation; missing CMF q0 gives zero. Valid zero gate removes the pre-stage direct CMF component, while fixed half removes half. Only nonzero delta locations are added to copied config4 pre-score, retaining exact bytes including negative zero elsewhere. No original baseline/core/cw/state/model array is changed.'),
      dict(area='One original boundary episode and no postrank',passed=True,evidence=[SIX[0]+':decisions',SIX[1]+':WiringTests',dependencies[10],dependencies[11]],
        finding='Both arms inject complete original Local67config4 pre-score plus fixed delta into original incumbent_adapter with budget0 and all-NaN hand factors. The adapter passes copies of ORIGINAL baseline/core/cw/states, original effects and frozen policy to exactlyone episode per arm. Its returned choices form orders directly. Synthetic callback deliberately selects a non-global top3 and captures original support to detect altered support or post-episode reranking. Original episode CMF dependence is intentionally unchanged.'),
      dict(area='Independent scalar checks and causal prefixes',passed=True,evidence=[SIX[0]+':prepare',SIX[4]+':independent_features/verify_features/verify_deltas',*SIX[1::2]],
        finding='Oracle imports no producer. Canonical rolling sigma/CMF retain original pandas tie semantics; each unique consumed sigma also receives a separate scalar63-return std cross-check rtol1e-10/atol1e-12. This tolerance never changes canonical sigma, gate, rank, delta or order. Features, all audit fields/types and scalar deltas compare exactly. Physical source prefixes0/1/53/88/122 precede numeric conversion; synthetic future objects, zero prefix, missing/extra sessions and invalid source cases are covered. All123 order/transform/delta prefixes must match. Original27 arrays compare byte/dtype/shape and incumbent orders compare exactly.'),
      dict(area='Original cached state parity and evidence accounting',passed=True,evidence=[SIX[0]+':original_state_parity/prepare/verify_prepared_evidence',SIX[1]+':GovernanceTests',SIX[4]+':record_check'],
        finding='Prepared high-vol cache must exactly equal source r>0, and original calm/cw must match. Isolated flow cache allows atol1e-12 rtol0 only because its stored isolation is subtraction-derived; rank/state/gate/order remain exact. Main-pass audit counts122*315 prices and122*252 sigma slots,5978 candidate cells split finite/missing CMF, with0 predictive fits. Strict flags, key sets, hashes, dimensions and counts gate later prepared loading.'),
      dict(area='First88 selection and independent portfolio accounting',passed=True,evidence=[SIX[0]+':select_first88/select/frozen_selection/verify_training_against_audit',SIX[4]+':audit_outcomes',SIX[1]+':SelectionTests',*dependencies[17:22]],
        finding='Candidate is the only promotable arm. First88 same-close CAGR requires incumbent+.005 and control noninferiority; next-open5bp requires both refs, with Sharpe/MDD and early53/recent35 guards on both paths. >=3 changed holding sets with each era is required versus each ref; ordering alone does not count. Frozen hashes precede later34. Pinned independent192->189->171->169 accounting verifies9 paths45 slices,122 executions, fees/cash/units/NAV, no terminal liquidation, and18 frozen training metrics. Original EW3/twohalves/<2cash and monthly fixed units remain.'),
      dict(area='Prospective governance and formal final evidence',passed=True,evidence=[CONTEXT,SOURCE,TESTS,SIX[0]+':register/gate/close'],
        finding='Current six exact sources are bound by formal59-test receipt:25host,16kernel,18oracle, pinned Python3.9.6/NumPy2.0.2/pandas2.3.3. Source receipt binds305 pins/raw64 and current context. Independent six-file review is required before PLAN creation; all actual phases require explicit PLAN hash/central registration. Write-once phase evidence, original protected21444 hashes and central CAS backups remain. Expected197/1108 advances to198/1111 only after completion. Reviewer computes no actual market features, orders or returns and does not register.'),
    ]
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        unresolved_blockers=[],code_sha256=code,input_sha256=pins,review_code_sha256=sha(__file__),
        independent_reviewer='nested199_code_audit',
        review_method='Complete static review of final six files and frozen source algebra, then exact metadata/dependency digest verification. No production imports or market-array reads.',
        context_sha256=CONTEXT_SHA,source_review_sha256=SOURCE_SHA,coverage_sha256=COVERAGE_SHA,
        formal_test_receipt_sha256=TESTS_SHA,synthetic_tests=59,synthetic_test_methods=methods,
        synthetic_test_receipt_verified=True,formal_tests_rerun_by_reviewer=False,
        actual_features_computed=False,actual_sigma_computed=False,actual_state_cdf_computed=False,
        actual_cmf_computed=False,actual_cmf_ranks_computed=False,actual_support_scale_computed=False,
        actual_deltas_computed=False,actual_orders_computed=False,actual_portfolio_returns_computed=False,
        production_files_modified=False,strategy_registered_by_reviewer=False,production_modules_imported=False,
        resolved_findings=[dict(issue='Draft host wiring test read effects from the separate original support payload and assumed unbroadcast cw.',
            resolution='Author corrected test to compare separate effects argument and canonical broadcast cw; final formal59 tests pass.',production_change_required=False)],
        review_checks=checks,limitations=[
          'This pre-registration code review certifies source/test consistency, not actual preparation, portfolio performance or strategy admission.',
          'Empirical CDF location is not a fitted or calibrated state probability. Median ties are deliberately neutralized by the prospective contract.',
          'Only direct pre-episode CMF is softened. Original episode support, indirect frozen forecast features and other hard-state coefficients stay unchanged.',
          'Scalar std cross-check and subtraction-derived cached flow each have explicit numerical-only tolerances; all decision quantities remain canonical/exact.',
          'Original source uses retrospective vendor/action/alias conventions and proxyTop32+theme17 membership; strictPIT remains false and later34 is reused diagnostic history.',
          'Prior smooth market-state and auxiliary percentile mechanisms already exist. Narrow exact200 empirical-CMF or half-CMF coverage was not found; exhaustive historical novelty remains UNKNOWN.',
          'Reviewer validates the21444-file protection manifest binding; registration and closure must verify the actual protected bytes.',
        ])
    for p,h in pins.items(): require(sha(ROOT/p)==h,'Source changed during review: '+p)
    with OUT.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),review_code_sha256=sha(__file__),
        code_files=6,input_pins=len(pins),synthetic_tests_verified=59,unresolved_blockers=[])))


if __name__=='__main__': main()
