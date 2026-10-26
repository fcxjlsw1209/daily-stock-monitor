"""Pre-PLAN static code/math and saved-source receipt review; never compute features."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
R=Path(__file__).resolve().parents[1]
OUT=R/'work/downvolume207_CODE_REVIEW.json'
SIX=['work/monthly_down_volume207_host.py','work/test_monthly_down_volume207_host.py','work/monthly_down_volume207_kernel.py','work/test_monthly_down_volume207_kernel.py','work/monthly_down_volume207_independent.py','work/test_monthly_down_volume207_independent.py']
KNOWN={SIX[2]:'bdd78d2222afe702923fc40f26acd56139dee4ecf9e43a377477a9d63b288fe3',SIX[3]:'5c36d56d68f5a8eecbad13c0b262e646248adb2225f19e8c688b2a770bb32151',SIX[4]:'558d7a8fa4dc032a14917cdf9c6d8c91050c1574ce96799f13d34092feda31af',SIX[5]:'e724db8e0362856b5c8ae1b2ebcdfaf29b30a20ddd06bc25464f44d72ebbcdb5'}
CTX='work/monthly_down_volume207_context.json';CS='223ac747a847b9d908cea27bf7596f544c783d0247ce56c255b24029e0712516'
SRC='work/downvolume207_SOURCE_REVIEW.json';SS='77218c4d8e4d2b0d9d64a17adb6370ced64b833a90493c02020e74bbaa1e9d04'
COV='work/after206_volume_dispersion_coverage.json';VS='7178cb8b5730a76d7e14d8b6b58f5b1cf7f83e81b6e43cad517aa7866c2055f3'
TEST='work/monthly_down_volume207_TESTS.json'
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def main():
    require(not OUT.exists() and not (R/'outputs/monthly_down_volume207').exists(),'Writeonce preregistration')
    pins={}
    def bind(path,digest=None):
        p=Path(path);p=(p if p.is_absolute() else R/p).resolve();require(p.is_relative_to(R),'Localreviewinput')
        require(p!=R/'outputs/monthly_target70_research/CONTINUATION_STATUS.json','No mutablecentralpin')
        h=sha(p);require(digest is None or h==digest,'Changed '+str(p));key=str(p.relative_to(R));require(key not in pins or pins[key]==h,'Conflictingpin');pins[key]=h;return h
    code={p:bind(p,KNOWN.get(p)) for p in SIX}
    for p,h in ((CTX,CS),(SRC,SS),(COV,VS)):bind(p,h)
    bind(TEST);bind(__file__)
    ctx,src,tst,cov=read(CTX),read(SRC),read(TEST),read(COV)
    require(ctx['round']==207 and ctx['arms']==['incumbent_local67','down_volume_concentration','all_volume_control'],'Contextidentity')
    require(ctx['formation']['price_count']==64 and ctx['formation']['volume_count']==63 and ctx['integration']['dose']==.025 and ctx['integration']['max_absolute_delta']==.0125,'Fixeddefinition')
    require(src['passed'] is True and src['source_only'] is True and src['context_sha256']==CS and src['unresolved_blockers']==[],'Sourcecert')
    for k in ('actual_features_computed','actual_down_flags_computed','actual_dispersion_computed','actual_ranks_computed','actual_deltas_computed','actual_orders_computed','actual_returns_computed','actual_portfolio_returns_computed','observers_accessed'):
        require(src[k] is False,'Sourceboundary '+k)
    w=src['source_window_coverage'];require(w['checked_cells']==5978 and w['common_available_cells']==5966 and w['unavailable_cells']==12 and w['accepted_source_price_observations']==5966*64 and w['accepted_source_volume_observations']==5966*63,'Sourcesupport')
    require(src['strict_pit'] is False and src['actual_source_coverage_computed'] is True and len(src['raw_source_hashes'])==64,'Sourceclaims')
    for mapping in (src['source_sha256'],src['raw_source_hashes'],ctx['evidence_sha256'],cov['evidence_sha256'],tst['code_sha256']):
        for p,h in mapping.items():bind(p,h)
    require(tst['passed'] is True and tst['synthetic_only'] is True and tst['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formaltests')
    require(tst['tests']>=60 and all(tst['code_sha256'][str(R/p)]==code[p] for p in SIX),'Allsix testedbytes')
    txt={p:(R/p).read_text() for p in SIX};tree={p:ast.parse(s) for p,s in txt.items()}
    funcs={n.name:ast.get_source_segment(txt[SIX[0]],n) for n in tree[SIX[0]].body if isinstance(n,ast.FunctionDef)}
    require(funcs['prepare'].index('gate(expected)')<funcs['prepare'].index('study.load()'),'No computationbeforePLAN')
    require(funcs['register'].index('code_review_binding()')<funcs['register'].index('OUT.mkdir()'),'ReviewbeforePLAN')
    require('delta=DOSE*r' in funcs['scale_and_delta'] and 'if n:delta[0]=0.' in funcs['scale_and_delta'],'Fixedadditiveinitialzero')
    require('view[changed] += deltas[arm][changed]' in funcs['decisions'] and "study.adapter.score_and_choose('incumbent66', 0." in funcs['decisions'] and 'boundary_selector=study.ep.select_boundary' in funcs['decisions'],'Oneoriginalepisode')
    require('for n in range(123)' in funcs['prepare'] and 'require_source_support(audit,values)' in funcs['prepare'] and 'len(arrays) == 27' in funcs['prepare'],'Preparationchecks')
    require("orders[arm][:TRAIN]" in funcs['select'] and "meta['ends'][87]" in funcs['select'] and funcs['diagnose'].index('frozen_selection(expected, orders)')<funcs['diagnose'].index('ev.run_path('),'First88frozenbeforelater')
    require('monthly_down_volume207_kernel' not in txt[SIX[4]],'Independentoracle productionimport')
    require('p[k+1]<p[k]' in txt[SIX[2]] and 'centered_sum/denominator' in txt[SIX[2]],'StrictdownandCV')
    require('distances / (total * total)' in txt[SIX[4]] and 'pairwise_identity_atol=1e-12' in txt[SIX[4]] and 'np.array_equal(actual[key], wanted[key], equal_nan=True)' in txt[SIX[4]],'Independentidentity andexactfeatures')
    methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(tree[p])) for p in SIX[1::2]}
    require(sum(methods.values())==tst['tests'],'Testmethodcount')
    require(len(read('work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'))==21444,'Protectedscope')
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],code_sha256=code,input_sha256=pins,review_code_sha256=sha(__file__),context_sha256=CS,source_review_sha256=SS,coverage_sha256=VS,formal_test_receipt_sha256=sha(R/TEST),synthetic_tests=tst['tests'],synthetic_test_methods=methods,independent_reviewer='root static review of separately authored host/producer/scalar oracle',review_method='Static code/date/integration review and frozen metadata hashes. Root authored sourcecoverage; not independentvendorreconstruction. Numerical independence from scalaroracle with separatepairwiseidentity.',review_checks=[dict(area='Current64/63alignedsources',passed=True,finding='Original adjustedcloses and inheritedstoredvolume, no source substitution. Scomplete-source support checked against separate sourceadmission; A>=2down isseparatelyaudited.'),dict(area='Volume distribution',passed=True,finding='Each group independently maxscaled; fsummean/centeredvariance populationCV2, exactconstant0. Down strictprices, flats excluded. Both armscommon support. Oracleexactraw/audit/rank pluspairwisedistanceidentity1e-12.'),dict(area='Ranking and monthly portfolio',passed=True,finding='Full49common domain includesETF beforeeligibility. HigherCV2 preferred. .025centeredrankbudget max.0125;initial0. Original longskip42 core/stateweights/sourcearrays/support/oneepisode unchanged. Nointramonthactions.'),dict(area='Validation',passed=True,finding='Syntheticfuturepoison/constant/scale/missing/prefix tests, actualfivephysicalprefixes and123orders,9executionpaths45slices18traininggroups. First88selection saved beforelater34. Noparametergrid.'),dict(area='Limitations',passed=True,finding='CV2notcountunbiased; chronology/dropmagnitude ignored; withinwindowunitscanmatter. Fixedproxyuniverse/vendorvintage and reusedlater34 notstrictPIT/pristineOOS. Sameclosezero target distinctfromnextopen/cost.')],resolved_findings=['NormalizedHHIn-1 divisor rejected before anyactual207features; populationCV2fixedprospectively.','Synthetic mathematical.25 expectedvalue compared approximately to canonicalbinary64.25000000000000006; actualproducer/oracle feature/rank comparisons remain exact.','Context native termclarified in sources asoriginalstoredmixedraw/normalizedvolume, not certifiedconsistentunits.'],actual_features_computed=False,actual_model_fits_computed=False,actual_orders_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,production_modules_imported=False,limitations=ctx['limitations'])
    for p,h in pins.items():require(sha(R/p)==h,'Changedduringreview')
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),pins=len(pins),tests=tst['tests'])))
if __name__=='__main__':main()
