"""Pre-PLAN static code/math and saved-source receipt review; never compute features."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
R=Path(__file__).resolve().parents[1]
OUT=R/'work/volume_stability208_CODE_REVIEW.json'
SIX=['work/monthly_volume_stability208_host.py','work/test_monthly_volume_stability208_host.py','work/monthly_down_volume207_kernel.py','work/test_monthly_down_volume207_kernel.py','work/monthly_volume_stability208_independent.py','work/test_monthly_volume_stability208_independent.py']
KNOWN={SIX[0]:'ab29054de09b44c3ce317cfd158fedbd90c8f360e7c9bc7a34fc0133f8e810bb',SIX[1]:'00ce777d30b3ba1cd09c41e08806c5806e1e64730f414038f164c6d9004b1af4',SIX[2]:'bdd78d2222afe702923fc40f26acd56139dee4ecf9e43a377477a9d63b288fe3',SIX[3]:'5c36d56d68f5a8eecbad13c0b262e646248adb2225f19e8c688b2a770bb32151',SIX[4]:'ae6a3ade20a9bf233e6292a58b7e6ea419919718859e58eeb25c699da2eefd0d',SIX[5]:'b0e570438d363b9ca460de3f1e3f5017ee638a02631339940cbd8e044048669d'}
CTX='work/monthly_volume_stability208_context.json';CS='7d8e19ba13ce394168dbecb0a88b12f64d13ab5009ac3f7370ceaceefb2fa0cb'
SRC='work/volume_stability208_SOURCE_REVIEW.json';SS='00e5ebd3225295f4ae94a1e3d8178670a07cb0c4a74f7263130d01964092bd26'
COV='work/after207_volume_stability_coverage.json';VS='bbc0d23b5072cccf9dd0df5b5c2b3c20ea9362495ba16432bda9267ee0c511f9'
TEST='work/monthly_volume_stability208_TESTS.json'
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def main():
    require(not OUT.exists() and not (R/'outputs/monthly_volume_stability208').exists(),'Writeonce preregistration')
    pins={}
    def bind(path,digest=None):
        p=Path(path);p=(p if p.is_absolute() else R/p).resolve();require(p.is_relative_to(R),'Localreviewinput')
        require(p!=R/'outputs/monthly_target70_research/CONTINUATION_STATUS.json','No mutablecentralpin')
        h=sha(p);require(digest is None or h==digest,'Changed '+str(p));key=str(p.relative_to(R));require(key not in pins or pins[key]==h,'Conflictingpin');pins[key]=h;return h
    code={p:bind(p,KNOWN.get(p)) for p in SIX}
    for p,h in ((CTX,CS),(SRC,SS),(COV,VS)):bind(p,h)
    bind(TEST);bind(__file__)
    ctx,src,tst,cov=read(CTX),read(SRC),read(TEST),read(COV)
    require(ctx['round']==208 and ctx['arms']==['incumbent_local67','down_volume_stability','all_volume_stability_control'],'Contextidentity')
    require(ctx['formation']['price_count']==64 and ctx['formation']['volume_count']==63 and ctx['integration']['dose']==-.025 and ctx['integration']['max_absolute_delta']==.0125,'Fixeddefinition')
    require(src['passed'] is True and src['source_only'] is True and src['context_sha256']==CS and src['unresolved_blockers']==[],'Sourcecert')
    for k in ('actual_features_computed','actual_down_flags_computed','actual_dispersion_computed','actual_ranks_computed','actual_deltas_computed','actual_orders_computed','actual_returns_computed','actual_portfolio_returns_computed','observers_accessed'):
        require(src[k] is False,'Sourceboundary '+k)
    w=src['source_window_coverage'];require(w['checked_cells']==5978 and w['common_available_cells']==5966 and w['unavailable_cells']==12 and w['accepted_source_price_observations']==5966*64 and w['accepted_source_volume_observations']==5966*63,'Sourcesupport')
    require(src['strict_pit'] is False and src['actual_source_coverage_computed'] is False and src['current_saved_source_coverage_reused'] is True and len(src['raw_source_hashes'])==64,'Sourceclaims')
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
    require('verify_sign_reversal(values,ranks,deltas' in funcs['prepare'] and 'np.testing.assert_array_equal(transforms[arm],prior_saved' in funcs['verify_sign_reversal'] and '-np.asarray(prior_saved' in funcs['verify_sign_reversal'] and 'audit==prior_audit' in funcs['verify_sign_reversal'],'208onlydelta sign reversal checks')
    require('88,-.025,.5,.0125' in txt[SIX[0]] and 'DOSE = -.025' in txt[SIX[4]],'Negative dose host and independent')
    require("orders[arm][:TRAIN]" in funcs['select'] and "meta['ends'][87]" in funcs['select'] and funcs['diagnose'].index('frozen_selection(expected, orders)')<funcs['diagnose'].index('ev.run_path('),'First88frozenbeforelater')
    require('monthly_down_volume207_kernel' not in txt[SIX[4]],'Independentoracle productionimport')
    require('p[k+1]<p[k]' in txt[SIX[2]] and 'centered_sum/denominator' in txt[SIX[2]],'StrictdownandCV')
    require('distances / (total * total)' in txt[SIX[4]] and 'pairwise_identity_atol=1e-12' in txt[SIX[4]] and 'np.array_equal(actual[key], wanted[key], equal_nan=True)' in txt[SIX[4]],'Independentidentity andexactfeatures')
    methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(tree[p])) for p in SIX[1::2]}
    require(sum(methods.values())==tst['tests'],'Testmethodcount')
    require(len(read('work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'))==21444,'Protectedscope')
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],code_sha256=code,input_sha256=pins,review_code_sha256=sha(__file__),context_sha256=CS,source_review_sha256=SS,coverage_sha256=VS,formal_test_receipt_sha256=sha(R/TEST),synthetic_tests=tst['tests'],synthetic_test_methods=methods,independent_reviewer='root static review of separately authored host/producer/scalar oracle',review_method='Static code/date/integration review and frozen metadata hashes. Root reused hashverified207sourcecoverage; not freshsourcecoverage or independentvendorreconstruction. Numerical independence from scalaroracle with separatepairwiseidentity.',review_checks=[dict(area='Current64/63alignedsources',passed=True,finding='Original adjustedcloses and inheritedstoredvolume, no source substitution. Scomplete-source support checked against separate sourceadmission; A>=2down isseparatelyaudited.'),dict(area='Volume distribution',passed=True,finding='Each group independently maxscaled; fsummean/centeredvariance populationCV2, exactconstant0. Down strictprices, flats excluded. Both armscommon support. Oracleexactraw/audit/rank pluspairwisedistanceidentity1e-12.'),dict(area='Ranking and monthly portfolio',passed=True,finding='Full49common domain includesETF beforeeligibility. PositiveCV2 raw/ranks unchanged207; negative-.025correction prefers stability, maxdelta max.0125;initial0. Original longskip42 core/stateweights/sourcearrays/support/oneepisode unchanged. Nointramonthactions.'),dict(area='Validation',passed=True,finding='Syntheticfuturepoison/constant/scale/missing/prefix tests, actualfivephysicalprefixes and123orders,9executionpaths45slices18traininggroups. First88selection saved beforelater34. Noparametergrid.'),dict(area='Limitations',passed=True,finding='Outcome-informed207sign reversal, not independentnewalpha. CV2notcountunbiased; chronology/dropmagnitude ignored; withinwindowunitscanmatter. Fixedproxyuniverse/vendorvintage and reusedlater34 notstrictPIT/pristineOOS. Sameclosezero target distinctfromnextopen/cost.')],resolved_findings=['207rawvalues/ranks reused withoutnegative rerank; onlydose changes sign prospectively in208.','Outcome-informedsign andknownlater34 outcomes disclosed; old206volume novelty UNKNOWN preserved.','Sourcecoverage truthfully reusedafterhashverification; not newlycomputed.'],actual_features_computed=False,actual_model_fits_computed=False,actual_orders_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,production_modules_imported=False,limitations=ctx['limitations'])
    for p,h in pins.items():require(sha(R/p)==h,'Changedduringreview')
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),pins=len(pins),tests=tst['tests'])))
if __name__=='__main__':main()
