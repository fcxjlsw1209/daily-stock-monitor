"""Root static review of211 numerical code after formal synthetic tests; no model imports."""
from pathlib import Path
from datetime import datetime, timezone
import ast,json,hashlib
R=Path(__file__).resolve().parents[1]
CONTEXT='work/monthly_forecast_blend211_context.json'
CONTEXT_SHA='1cc5934776614b5d338a590ba442573c0d1eab07e32a8a623ef4ae6c8bb6c2fb'
COVERAGE='work/after210_forecast_blend_coverage.json'
SOURCE='work/forecast_blend211_SOURCE_REVIEW.json'
SOURCE_SHA='245505c6c2939a5ecd9f5493f6b917862b4f3d13ff0ece5c1e6f3a791a59b855'
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def need(x,m):
    if not x:raise ValueError(m)
def main():
    out=R/'work/forecast_blend211_CODE_REVIEW.json'
    need(not out.exists() and not (R/'outputs/monthly_forecast_blend211').exists(),'Review must precede registration')
    need(sha(CONTEXT)==CONTEXT_SHA and sha(SOURCE)==SOURCE_SHA,'Prospective source/context changed')
    c=read(CONTEXT);source=read(SOURCE)
    need(source['passed'] and source['context_sha256']==CONTEXT_SHA and source['matched_formal_months']==113,'Source identity review')
    for k in ('actual_new_losses','actual_new_coefficients','actual_new_orders','actual_new_returns'):need(source[k]==0,'Source boundary')
    names=['work/'+prefix+'monthly_forecast_blend211_'+kind+'.py' for kind in ('host','model','independent') for prefix in ('','test_')]
    codes={p:sha(p) for p in names};texts={p:(R/p).read_text() for p in names};trees={p:ast.parse(v) for p,v in texts.items()}
    receipt='work/monthly_forecast_blend211_TESTS.json';tests=read(receipt)
    need(tests['passed'] and tests['synthetic_only'] and tests['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal synthetic receipt')
    methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[p])) for p in names[1::2]}
    need(all(n>0 for n in methods.values()) and sum(methods.values())==tests['tests'],'Exact test methods')
    for p,h in codes.items():need(tests['code_sha256'][str(R/p)]==h,'Tested code changed')
    need('monthly_forecast_blend211_model' not in texts[names[4]],'Oracle imports producer')
    pins=dict(c['evidence_sha256']);pins.update(codes)
    for p,h in tests['code_sha256'].items():pins[str(Path(p).relative_to(R))]=h
    pins.update({CONTEXT:sha(CONTEXT),SOURCE:SOURCE_SHA,receipt:sha(receipt),'work/forecast_blend211_code_review.py':sha('work/forecast_blend211_code_review.py')})
    for p,h in pins.items():need(sha(p)==h,'Changed reviewed source '+p)
    state=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    need(state['current']=='monthly_asset_aux210' and not state['registered_queue'] and len(state['completed'])==208 and sum(x['trials'] for x in state['completed'])==1141,'Stale prospective state')
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],context_sha256=CONTEXT_SHA,coverage_sha256=sha(COVERAGE),code_sha256=codes,input_sha256=pins,review_code_sha256=sha('work/forecast_blend211_code_review.py'),formal_test_receipt_sha256=sha(receipt),synthetic_tests=tests['tests'],synthetic_test_methods=methods,independent_reviewer='root static review of separately authored producer/host/scalar oracle',actual_features_computed=False,actual_model_fits_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,findings=['Reuses independently verified206 causal annual replay forecasts, never the in-sample residual audit as historical forecast errors. No pre2016 predictions invented.','Exact signal/security/end identity; priorDec31 strictfiveyear mature label window and same old labels; common monthly rows, not current holdings, determine scalar-loss samples.','Each annual ownweight is a constrained single scalar forecast-loss fit; fixedhalf matches all other source predictions and score budget. No parameter grid, nonfinite hidden fallback, relabeling or new data.','Original quarter ML slot and one episode, dominant skip42 long core and monthly fixedunits preserved. First88 selects and freezes before later34 diagnostics.','Source/proxy vintage and repeated sample limitations retained; no automatic observer promotion.'])
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,tests=tests['tests'],review_sha256=sha('work/forecast_blend211_CODE_REVIEW.json'))))
if __name__=='__main__':main()
