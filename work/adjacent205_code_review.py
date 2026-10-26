"""Static/hash review of separately authored205 implementation; no actual arrays."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/adjacent205_CODE_REVIEW.json'
SIX=['work/'+n for n in ('monthly_adjacent_risk205_host.py','test_monthly_adjacent_risk205_host.py','monthly_adjacent_risk205_kernel.py','test_monthly_adjacent_risk205_kernel.py','monthly_adjacent_risk205_independent.py','test_monthly_adjacent_risk205_independent.py')]
EXPECTED=('92df94f93b4b9d6b25c9fb58b0de0512d7dd3fdb32afad1249003aff8a07ddd2','6599e5127f7088805aa2252fa2aa405cace95dd731919f31a37eec4457a46f90','cb1bf675481ba96630abbf7e9f7d4a782c2ace3184fca05b280319e7815ec6f4','6069c83c071953704ee264d54bad40d37058043b307290c5133e79f0f4ad39ce','e28b9cc74cc28f47955c3d52419a7780778b3c75e5ce3f34892c2ef393a37c6a','e589d9b94e895e182ac0bffb471bd16e82b73a538c029812c4f001fc6b5c1a70')
CTX='work/monthly_adjacent_risk205_context.json'
CTX_SHA='60f0a5a6e1e8bd81f80272c5150c7a6baba8154a92a2c2b02e5a7281844d6b3c'
SOURCE='work/adjacent205_SOURCE_REVIEW.json'
SOURCE_SHA='0476491e6bed98c7b38c462d02084e1a1752d30737681ef522be8c756f4212fb'
COVER='work/after204_bipower_coverage.json'
COVER_SHA='4eff269796a894a688cb3a5c16cc1d677376bac0956220c89b00e4bc6799f5a0'
TEST='work/monthly_adjacent_risk205_TESTS.json'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def main(test_sha):
    require(not OUT.exists() and not (ROOT/'outputs/monthly_adjacent_risk205').exists(),'Preregisration writeonce review')
    pins={}
    def bind(name,digest=None):
        p=Path(name);p=(p if p.is_absolute() else ROOT/p).resolve()
        require(p.is_relative_to(ROOT) and p!=CENTRAL,'Nonlocal or mutable central input')
        key=str(p.relative_to(ROOT));h=sha(p)
        require(digest is None or h==digest,'Changed '+key)
        require(key not in pins or pins[key]==h,'Conflicting '+key);pins[key]=h;return h
    codes={p:bind(p,h) for p,h in zip(SIX,EXPECTED)}
    for p,h in ((CTX,CTX_SHA),(SOURCE,SOURCE_SHA),(COVER,COVER_SHA),(TEST,test_sha)):bind(p,h)
    bind(__file__)
    ctx,src,cov,tst=read(CTX),read(SOURCE),read(COVER),read(TEST)
    require(ctx['round']==205 and ctx['arms']==['incumbent_local67','adjacent_risk_core','rms_risk_control'],'Arms')
    require(ctx['pre_registration_actual_new_features']==ctx['pre_registration_actual_orders']==ctx['pre_registration_actual_returns']==0,'Prospective state')
    require(tst['passed'] is True and tst['synthetic_only'] is True and tst['tests']==72,'Formal synthetic tests')
    require(tst['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Runtime')
    for p,h in tst['code_sha256'].items():bind(p,h)
    for p in SIX:require(tst['code_sha256'][str(ROOT/p)]==codes[p],'Untested code')
    require(src['passed'] is True and src['source_only'] is True and src['source_coverage_reused'] is True and src['unchanged_source_bytes_reverified'] is True,'Source reuse evidence')
    for key in ('actual_source_coverage_computed','actual_features_computed','actual_signal_returns_computed','actual_adjacent_products_computed','actual_squared_returns_computed','actual_risk_estimates_computed','actual_dispersion_computed','actual_ranks_computed','actual_deltas_computed','actual_orders_computed','actual_returns_computed','actual_portfolio_returns_computed'):
        require(src[key] is False,'Source review computed '+key)
    for mapping in (ctx['evidence_sha256'],src['source_sha256'],cov['source_sha256']):
        for p,h in mapping.items():bind(p,h)
    require(len(src['raw_source_hashes'])==64 and len(src['source_sha256'])==331,'Source closure counts')
    for p,h in src['raw_source_hashes'].items():bind(p,h)
    coverage=src['source_window_coverage']
    require(coverage['complete_positive_price_cells']==5973 and coverage['incomplete_price_cells']==5 and coverage['field_counts']['observations']==1763510,'Inherited source coverage')
    previous=read('outputs/monthly_aux_rank204/PLAN.json')
    local='outputs/monthly_local_linear70/'
    for name in ('PLAN.json','DESIGN.json','GRID.json','DESIGN.sha256','INPUT_HASHES.json','CACHE_HASHES.json','META.json','DATA.npz','INCUMBENT_POLICY.json','local_linear_ORDERS.json','study.py'):
        p=local+name;require(str(ROOT/p) in previous['input_sha256'],'Missing original cache pin '+p);bind(p,previous['input_sha256'][str(ROOT/p)])
    for name,digest in read(local+'CACHE_HASHES.json').items():bind(local+name,digest)
    meta=read(local+'META.json');require(len(meta['dates'])==122 and all(len(x)==49 for x in meta['symbols']),'Original identities')
    require(meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Period')
    protected='work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
    bind(protected,previous['input_sha256'][str(ROOT/protected)]);require(len(read(protected))==21444,'Protected manifest size')
    text={p:(ROOT/p).read_text() for p in SIX};trees={p:ast.parse(v) for p,v in text.items()}
    functions={x.name:ast.get_source_segment(text[SIX[0]],x) for x in trees[SIX[0]].body if isinstance(x,ast.FunctionDef)}
    require(functions['prepare'].index('gate(expected)')<functions['prepare'].index('study.load()'),'PrePLAN computation')
    require(functions['register'].index('code_review_binding()')<functions['register'].index('OUT.mkdir()'),'Review timing')
    require('FEATURE_PREFIXES = (0, 1, 53, 88, 122)' in text[SIX[0]] and 'feature_prefix_panel(panel, meta, n)' in functions['prepare'],'Physicalsourceprefix verification')
    require('for n in range(123)' in functions['prepare'] and 'orders[k][:n]' in functions['prepare'],'Order prefix verification')
    require('require_source_support(audit)' in functions['prepare'] and 'len(arrays) == 27' in functions['prepare'],'Source support/originalarray checks')
    require('target=validated[key].copy();target[calm]=np.nan' in functions['scale_and_delta'] and 'if n:target[0]=np.nan' in functions['scale_and_delta'] and 'BLEND*(cw[mask]*target[mask]-c[mask])' in functions['scale_and_delta'],'Halfcore formula/state')
    require("adapter.score_and_choose('incumbent66',BLEND,full_target" in functions['decisions'] and 'pre_score=pre,effects=arrays[\'effects\']' in functions['decisions'] and 'boundary_selector=study.ep.select_boundary' in functions['decisions'],'Frozen adapter integration')
    require('orders[arm][:TRAIN]' in functions['select'] and "meta['ends'][87]" in functions['select'],'Training clock')
    require(functions['diagnose'].index('frozen_selection(expected, orders)')<functions['diagnose'].index('ev.run_path('),'Freeze before diagnostics')
    require('monthly_adjacent_risk205_kernel' not in text[SIX[4]],'Oracle imports producer')
    require("require_result_manifest(completion['artifacts_sha256'])" in functions['close'],'Exact resultmanifest gate')
    require('math.fsum(products)' in text[SIX[2]] and '(math.pi/2.)*adjacent_sum/62.' in text[SIX[2]] and 'squared_sum/63.' in text[SIX[2]],'Risk arithmetic')
    methods={p:sum(isinstance(x,ast.FunctionDef) and x.name.startswith('test_') for x in ast.walk(trees[p])) for p in SIX[1::2]}
    require(list(methods.values())==[34,20,18] and sum(methods.values())==72,'Test counts')
    checks=[
      dict(area='Source and clocks',passed=True,finding='Original295adjusted closes t-294..t; full positive nonboolean numeric source required. Numerator P252/P0-1 omits latest42sessions. Current63simple returns use64prices ending t. Raw64/calendar9 source bytes reverified; reused201coverage5973/5 explicitly not a new price-cell computation.'),
      dict(area='Exact adjacent versus RMS risk',passed=True,finding='62nonwrapped abs(rk)*abs(rk-1) products versus63squares; fsum then pi/2 multiplication then /62 and sqrt, control fsum/63 and sqrt. Uncentered, symmetric, no floor/annualization/clipping. Both denominator positive prerequisite; eitherzero keeps both unavailable, valid zero numerator kept. Finite-input intermediate overflow hardfails. Daily proxy is not certified jump identification.'),
      dict(area='Rank and scoring',passed=True,finding='Original ascending average percentile minus.5 on same commonvalid49domain before eligibility includingETFs. Exact scalar oracle, singleton+.5/allties+.5/m. Noncalm halfcore delta .5*(cw*T-C), bound.26; initial/calmtargetsNaN. Missing retains original core; finitezero still updates.'),
      dict(area='Integration',passed=True,finding='Pinned coreadapter copies original baseline/core/pre and applies same delta, restores original support metadata to exactlyone frozen episode callback, directly uses returned choices. Original27arrays/eligibility/pool preserved; no intramonth changes or terminal liquidation. Original orders required exact.'),
      dict(area='Independent checks',passed=True,finding='Separate scalar feature/source/rank/audit implementation imports no producer. Handworked product fixtures and both-fixture allprefix audit match; synthetic future poison/overflow/commonzero covered. Actual preparation requires5978cells, fivephysicalsourceprefixes,123orders and seven mainpass descriptive counters; no new predictive fits.'),
      dict(area='Selection and results',passed=True,finding='First88 only, candidateonlypromotable, two-reference gain/Sharpe/MDD/early53recent35 and changedset guards. Freeze beforelater34diagnostic. Independent9paths45slices18traininggroups. Sameclosezero headline; nextopenzero/5bp separate. Exact38artifact closure and21444archive/CAS guards,203entries1126trials expected.'),
      dict(area='Governance',passed=True,finding='72formal synthetic tests bound to exactsixfiles and pinned runtime. No actual features/orders/returns beforePLAN. Source and review persistentpins exclude mutableCENTRAL. Reusedhistory/proxyTop32theme17/retrospectiveversions remain conditional, not pristineOOS/certifiedstrictPIT or authorformula replication.')]
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],code_sha256=codes,input_sha256=pins,review_code_sha256=sha(__file__),independent_reviewer='root_static_review_of_separately_authored_kernel_host_oracle',review_method='Manual source/math/integration review plus hashbound synthetic tests; metadata/static checks only.',context_sha256=CTX_SHA,source_review_sha256=SOURCE_SHA,coverage_sha256=COVER_SHA,formal_test_receipt_sha256=test_sha,synthetic_tests=72,synthetic_test_methods=methods,review_checks=checks,resolved_findings=['Hostsourceflags strengthened to literalFalse for every actualcalculation flag and sourcecoverage reuseTrue; tested beforefreeze.','Inherited204exact38manifest and mutableCENTRAL exclusions retained; sourcequery clocks and missingidentities now checked against admission.'],actual_features_computed=False,actual_signal_returns_computed=False,actual_adjacent_products_computed=False,actual_squared_returns_computed=False,actual_risk_estimates_computed=False,actual_ranks_computed=False,actual_deltas_computed=False,actual_orders_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,production_modules_imported=False,limitations=ctx['limitations'])
    for p,h in pins.items():require(sha(ROOT/p)==h,'Changed during review '+p)
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),input_pins=len(pins),synthetic_tests_verified=72)))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--test-sha256',required=True);args=parser.parse_args();main(args.test_sha256)
