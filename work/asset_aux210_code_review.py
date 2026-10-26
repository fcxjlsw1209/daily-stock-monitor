"""Root static/metadata review of separately authored round210 code; no numeric imports."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def need(x,m):
    if not x:raise ValueError(m)
def main():
    out=R/'work/asset_aux210_CODE_REVIEW.json'
    need(not out.exists() and not (R/'outputs/monthly_asset_aux210').exists(),'Review must precede registration')
    names=['work/'+p for p in ('monthly_asset_aux210_host.py','test_monthly_asset_aux210_host.py','monthly_asset_aux210_kernel.py','test_monthly_asset_aux210_kernel.py','monthly_asset_aux210_independent.py','test_monthly_asset_aux210_independent.py')]
    expected=['a3cf136d20f3d7c43503fadb4c40c2927fde1f0f8b68bdf8166d3409651e1e43','897aca6974d2dbf5f9ed7a6c479c94e0b6f8815d18b5c7b218873e4f9fd5e683','932874775872d95e16696653547a44b9b58d893f4b7da1de9ec5cf2ff3dd08d0','550f2ddbe4a484565ea2e3a48f126d13d44e5d17ba14cb8c45916aa3a3dd15d3','ccaa1e91dbb260921bfa2df48650fe0a380d4465317124a05e2c0b200dc5f552','af423f823bfb057339fb32d37d7b13837826f1577df1028581115168e1b68cae']
    codes=dict(zip(names,expected));need(all(sha(p)==h for p,h in codes.items()),'Code changed after manual review')
    test='work/monthly_asset_aux210_TESTS.json';t=read(test)
    need(t['passed'] and t['synthetic_only'] and t['tests']==69,'Formal tests failed or count changed')
    need(t['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Runtime')
    for p,h in codes.items():need(t['code_sha256'][str(R/p)]==h,'Untested code')
    ctx='work/monthly_asset_aux210_context.json';c=read(ctx)
    need(sha(ctx)=='650b34956368e6df3ce1e542975dc7388048d6df044d6eb2a59f5b22ec865832','Context changed')
    text={p:(R/p).read_text() for p in names};trees={p:ast.parse(v) for p,v in text.items()}
    funcs={f.name:ast.get_source_segment(text[names[0]],f) for f in trees[names[0]].body if isinstance(f,ast.FunctionDef)}
    methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[p])) for p in names[1::2]}
    need(list(methods.values())==[30,20,19] and sum(methods.values())==t['tests'],'Test method coverage')
    need(funcs['prepare'].index('gate(expected)')<funcs['prepare'].index('study.load()'),'Feature gate')
    need(funcs['register'].index('code_review_binding()')<funcs['register'].index('OUT.mkdir()'),'Review gate')
    need('kernel.build_features(prefix_arrays,prefix_meta)' in funcs['prepare'] and 'FEATURE_PREFIXES = tuple(range(123))' in text[names[0]],'Physical prefix coverage')
    need('orders[arm][:TRAIN]' in funcs['select'] and "meta['ends'][87]" in funcs['select'],'First88 clock')
    need(funcs['diagnose'].index('frozen_selection(expected, orders)')<funcs['diagnose'].index('ev.run_path('),'Frozen decision before future diagnostics')
    need("pre_score=pre, effects=arrays['effects'], episode_policy=policy" in funcs['decisions'] and 'boundary_selector=study.ep.select_boundary' in funcs['decisions'],'Original episode')
    need('monthly_asset_aux210_kernel' not in text[names[4]],'Oracle must not import producer')
    for flag in ['asset_type_identity_checked','within_type_rank_domains_checked','constant_groups_unchanged']:
        need(flag in text[names[0]] and flag in text[names[4]],'Missing grouped evidence gate')
    pins=dict(c['evidence_sha256']);pins.update(codes);pins.update({ctx:sha(ctx),test:sha(test),'work/asset_aux210_code_review.py':sha('work/asset_aux210_code_review.py')})
    for p,h in t['code_sha256'].items():pins[str(Path(p).relative_to(R))]=h
    for p,h in pins.items():need(sha(p)==h,'Changed reviewed source '+p)
    m=read('outputs/monthly_local_linear70/META.json');e=set(m['etfs'])
    need(len(e)==len(m['etfs'])==17 and len(m['symbols'])==122 and all(len(s)==len(set(s))==49 and len(set(s)&e)==17 for s in m['symbols']),'Original group identity')
    state=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    need(state['current']=='monthly_relative_path209' and not state['registered_queue'] and len(state['completed'])==207 and sum(r['trials'] for r in state['completed'])==1138,'Stale prospective state')
    findings=[
      'Scope: exact original27cached arrays,122x49identities and17explicitETFIDs. No new data/labels/fits, no claim of raw feature reconstruction or novel economic mechanism.',
      'Producer group membership independent of eligibility. Average-tie symmetric rank and scalar paircount oracle exact; affine endpoints and two-level identity. Constant/singleton groups preserve original contribution, empty groups not ranked; firstpartial delta0.',
      'Originalsupport S and effectiveW derived from frozen source. Full W*(T-S) before one originalepisode; original core,ML,effects,policy,tieorder unchanged. No asset quota or intramonth changes.',
      '69 meaningfulsynthetic checks; all123 cache+metadata physical prefixes andorder prefixes gated in actual preparation. Independent mathematics uses no producer calls; outcomes independently reconcile9paths45slices18train groups.',
      'Onlyfirst88 selects; early53/recent35 risk/support guards and affine reference. Full/later34 afterward; CAScentral update and21444 archive verification at register/close. Zero costs primary, timing/fees separate.',
      'Historicalproxyuniverse, retrospectivevintages and repeatedly researchedlater34 remain explicit limits; frozenobserveraccounts untouched.'
    ]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],context_sha256=sha(ctx),code_sha256=codes,input_sha256=pins,review_code_sha256=sha('work/asset_aux210_code_review.py'),independent_reviewer='root_static_review_of_separately_authored_host_kernel_and_scalar_oracle',formal_test_receipt_sha256=sha(test),synthetic_tests=t['tests'],synthetic_test_methods=methods,findings=findings,actual_features_computed=False,actual_model_fits_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,tests=t['tests'],review_sha256=sha('work/asset_aux210_CODE_REVIEW.json'))))
if __name__=='__main__':main()
