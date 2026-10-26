"""Metadata/static preregistration review; no market arrays or model computation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/aux_rank204_CODE_REVIEW.json'
SIX=['work/'+n for n in ('monthly_aux_rank204_host.py','test_monthly_aux_rank204_host.py','monthly_aux_rank204_kernel.py','test_monthly_aux_rank204_kernel.py','monthly_aux_rank204_independent.py','test_monthly_aux_rank204_independent.py')]
EXPECTED=('d4713fe0b1d273b250bbde10947dbc836ef3d277135802811278cfb9fd2a6647','1e4942b79433e5feb7708c23647036442b1689fbb45238c60a938e00f1942db7','c44dd6b24ce060513a92890148675d7e091a5e0d91061da9d64637ecd160ac65','1b233d87ec34cb83a30a1ede793ad6b20c552a9fe5b3249651eab382ee52d562','e9e878267ea4e0e94bfff760813c7df1deb4e017daa6d0a59e80751e5ea13d61','c81c22969f14dd0a4b7bd375c7a726c2b068be74b4f190534ffeef26d1e25244')
CTX='work/monthly_aux_rank204_context.json'
CTX_SHA='602df5d3ec9ae914f147fee5c66167f6c554c4396c4a11d5dd1c24f78e0f8233'
COVER='work/after203_aux_rank_coverage.json'
COVER_SHA='a8f2388530dcfb0834ea0e3d9e98cce1717c6b6021b7e8ec44cf8c7cb7259b96'
TEST='work/monthly_aux_rank204_TESTS.json'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def main(test_sha):
    require(not OUT.exists() and not (ROOT/'outputs/monthly_aux_rank204').exists(),'Preregisration writeonce review')
    pins={}
    def bind(name,digest=None):
        p=Path(name);p=(p if p.is_absolute() else ROOT/p).resolve()
        require(p.is_relative_to(ROOT) and p!=CENTRAL,'Nonlocal or mutable central input')
        key=str(p.relative_to(ROOT));h=sha(p)
        require(digest is None or h==digest,'Changed '+key)
        require(key not in pins or pins[key]==h,'Conflicting '+key);pins[key]=h;return h
    codes={p:bind(p,h) for p,h in zip(SIX,EXPECTED)}
    for p,h in ((CTX,CTX_SHA),(COVER,COVER_SHA),(TEST,test_sha)):bind(p,h)
    bind(__file__)
    ctx,cov,tst=read(CTX),read(COVER),read(TEST)
    require(ctx['round']==204 and ctx['arms']==['incumbent_local67','aux_rank_full','aux_affine_control'],'Arms')
    require(ctx['pre_registration_actual_new_features']==ctx['pre_registration_actual_orders']==ctx['pre_registration_actual_returns']==0,'Prospective state')
    require(tst['passed'] is True and tst['synthetic_only'] is True and tst['tests']==60,'Formal synthetic tests')
    require(tst['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Runtime')
    for p,h in tst['code_sha256'].items():bind(p,h)
    for p in SIX:require(tst['code_sha256'][str(ROOT/p)]==codes[p],'Untested code')
    for p,h in ctx['evidence_sha256'].items():bind(p,h)
    # Coverage is an immutable historical snapshot. Its mutable-central digest
    # is evidence about that earlier state, never a persistent execution pin.
    excluded=[]
    for p,h in cov['source_sha256'].items():
        full=(Path(p) if Path(p).is_absolute() else ROOT/p).resolve()
        if full==CENTRAL:excluded.append(str(p));continue
        bind(p,h)
    previous=read('outputs/monthly_volume_pair203/PLAN.json')
    local='outputs/monthly_local_linear70/'
    for name in ('PLAN.json','DESIGN.json','GRID.json','DESIGN.sha256','INPUT_HASHES.json','CACHE_HASHES.json','META.json','DATA.npz','INCUMBENT_POLICY.json','local_linear_ORDERS.json','study.py'):
        p=local+name;require(str(ROOT/p) in previous['input_sha256'],'Missing inherited local cache pin '+p);bind(p,previous['input_sha256'][str(ROOT/p)])
    for name,digest in read(local+'CACHE_HASHES.json').items():bind(local+name,digest)
    meta=read(local+'META.json');require(len(meta['dates'])==122 and all(len(x)==49 for x in meta['symbols']),'Original identities')
    require(meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Period')
    text={p:(ROOT/p).read_text() for p in SIX};trees={p:ast.parse(v) for p,v in text.items()}
    functions={x.name:ast.get_source_segment(text[SIX[0]],x) for x in trees[SIX[0]].body if isinstance(x,ast.FunctionDef)}
    require(functions['prepare'].index('gate(expected)')<functions['prepare'].index('study.load()'),'PrePLAN computation')
    require(functions['register'].index('code_review_binding()')<functions['register'].index('OUT.mkdir()'),'Review timing')
    require('prefix_arrays=cache_prefix(arrays,n)' in functions['prepare'] and 'kernel.build_features(prefix_arrays)' in functions['prepare'] and 'FEATURE_PREFIXES = tuple(range(123))' in text[SIX[0]],'Physical cached prefix verification')
    require('oo,tt,dd=decisions(study,arrays,meta,values,n)' in functions['prepare'] and 'orders[k][:n]' in functions['prepare'],'Order prefix verification')
    require('changed = deltas[arm] != 0.' in functions['decisions'] and 'view[changed] += deltas[arm][changed]' in functions['decisions'],'Exact delta integration')
    require("pre_score=pre, effects=arrays['effects'], episode_policy=policy" in functions['decisions'] and 'boundary_selector=study.ep.select_boundary' in functions['decisions'],'Single original episode')
    require('orders[arm][:TRAIN]' in functions['select'] and "meta['ends'][87]" in functions['select'],'Training clock')
    require(functions['diagnose'].index('frozen_selection(expected, orders)')<functions['diagnose'].index('ev.run_path('),'Frozen selection before diagnostics')
    require('monthly_aux_rank204_kernel' not in text[SIX[4]],'Oracle imports producer')
    require("require_result_manifest(completion['artifacts_sha256'])" in functions['close'],'Exact result manifest gate')
    methods={p:sum(isinstance(x,ast.FunctionDef) and x.name.startswith('test_') for x in ast.walk(trees[p])) for p in SIX[1::2]}
    require(list(methods.values())==[23,21,16] and sum(methods.values())==60,'Test counts')
    checks=[
      dict(area='Source and timing',passed=True,finding='27 original registered cached arrays and122x49 identities retained. No new raw prices or fits. All123 physically truncated cache prefixes and123 order prefixes, including empty/full; no claim of fresh raw feature reconstruction.'),
      dict(area='Auxiliary derivation',passed=True,finding='S=(B-C)/(1-k); direct W=(1-k)-(.025calm+.05weak+.01active). Frozen extracted-functions tests establish coefficient. Original Local67 copied before applying W*(T-S); core unchanged, current original state/masks retained.'),
      dict(area='Matched targets',passed=True,finding='Full49 preeligibility average exact tie ranks versus exact endpoint affine control. Twolevel target bit identity; constant/initial row delta0. No epsilon ties or clipping; scalar paircount oracle reconstructs independently.'),
      dict(area='Integration and outcomes',passed=True,finding='Only preepisode composite support remapped. Exactly one original episode using old support, eligibility, effects, and policy. No rerank. Original order and27array preservation, fixed monthly units/EW3, no terminal liquidation,9paths45slices18training groups.'),
      dict(area='Selection and closure',passed=True,finding='Candidate only promotable, first88 double-reference gates with early53/recent35 guards and changedholding coverage. Later34 follows freeze.21444 archive protection and CAS central mutation; planned202entries1123trials.'),
      dict(area='Scope limits',passed=True,finding='Structural normalization experiment, not a new information family or author reproduction. Prior overlap unknown for unreadable originals. Reused historical periods and proxy universe remain non-pristine/noncertified strictPIT. Sameclosezero primary; openzero/cost separate.')]
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],code_sha256=codes,input_sha256=pins,review_code_sha256=sha(__file__),independent_reviewer='root_static_review_of_separately_authored_kernel_host_oracle',review_method='Manual source and mathematical review plus hashbound formal synthetic tests; metadata/static checks only.',context_sha256=CTX_SHA,coverage_sha256=COVER_SHA,formal_test_receipt_sha256=test_sha,synthetic_tests=60,synthetic_test_methods=methods,excluded_mutable_historical_coverage_pins=excluded,review_checks=checks,resolved_findings=['Independent host review found incomplete resultmanifest cardinality validation. Final host/test adds exact38set checking and rejects missing/extra/COMPLETION entries before closure reads; no strategy change.'],actual_features_computed=False,actual_orders_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,production_modules_imported=False,limitations=ctx['limitations'])
    for p,h in pins.items():require(sha(ROOT/p)==h,'Changed during review '+p)
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),input_pins=len(pins),synthetic_tests_verified=60)))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--test-sha256',required=True);args=parser.parse_args();main(args.test_sha256)
