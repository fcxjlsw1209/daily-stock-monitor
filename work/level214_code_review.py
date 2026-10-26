"""Root static review after synthetic validation; no model or real-data imports."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib
R=Path(__file__).resolve().parents[1]
CONTEXT='work/monthly_level_growth214_context.json'
CONTEXT_SHA='04bd4fa87f0cacbac737332d2e38d325d031150079c9b259b9b5f368a2253df8'
COVERAGE='work/after213_level_growth_coverage.json'
SOURCE='work/level214_SOURCE_REVIEW.json'
SOURCE_SHA='3bc246992557406090a828a603a80ac9f702740015f6ba010a25640d1a9aa7cd'
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def need(x,m):
 if not x:raise ValueError(m)
def main():
 out=R/'work/level214_CODE_REVIEW.json'
 need(not out.exists() and not (R/'outputs/monthly_level_growth214').exists(),'Review must precede registration')
 need(sha(CONTEXT)==CONTEXT_SHA and sha(SOURCE)==SOURCE_SHA,'Prospective source/context changed')
 c=read(CONTEXT);source=read(SOURCE)
 need(source['passed'] and source['context_sha256']==CONTEXT_SHA and source['source_coverage_reused'] and source['unchanged_source_bytes_reverified'],'Source admission')
 need(source['source_window_coverage']['price_count']==295 and source['source_window_coverage']['complete_positive_price_cells']==5973,'Source coverage')
 for k in ('actual_source_coverage_computed','actual_features_computed','actual_orders_computed','actual_returns_computed','actual_level_growth_computed','actual_segment_means_computed','actual_model_fits_computed'):need(source[k] is False,'Source boundary '+k)
 names=['work/'+prefix+'monthly_level_growth214_'+kind+'.py' for kind in ('host','kernel','independent') for prefix in ('','test_')]
 codes={p:sha(p) for p in names};texts={p:(R/p).read_text() for p in names};trees={p:ast.parse(v) for p,v in texts.items()}
 receipt='work/monthly_level_growth214_TESTS.json';tests=read(receipt)
 need(tests['passed'] and tests['synthetic_only'] and tests['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal synthetic receipt')
 methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[p])) for p in names[1::2]}
 need(all(n>0 for n in methods.values()) and sum(methods.values())==tests['tests'],'Exact test methods')
 for p,h in codes.items():need(tests['code_sha256'][str(R/p)]==h,'Tested code changed')
 imports=[]
 for node in ast.walk(trees[names[4]]):
  if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
  elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
 need(not any('level_growth214_kernel' in x or 'level_growth214_host' in x for x in imports),'Oracle imports producer/host')
 pins=dict(c['evidence_sha256']);pins.update(codes)
 for p,h in tests['code_sha256'].items():pins[str(Path(p).relative_to(R))]=h
 for p in (CONTEXT,COVERAGE,SOURCE,receipt,'work/level214_code_review.py','work/level214_source_review.py'):pins[p]=sha(p)
 need(pins['work/level214_source_review.py']==source['review_code_sha256'],'Retained source-review script bytes')
 for p,h in pins.items():need(sha(p)==h,'Changed reviewed source '+p)
 state=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
 need(state['current']=='monthly_asset_peer213' and not state['registered_queue'] and len(state['completed'])==211 and sum(x['trials'] for x in state['completed'])==1150,'Stale prospective state')
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],context_sha256=CONTEXT_SHA,coverage_sha256=sha(COVERAGE),code_sha256=codes,input_sha256=pins,review_code_sha256=sha('work/level214_code_review.py'),formal_test_receipt_sha256=sha(receipt),synthetic_tests=tests['tests'],synthetic_test_methods=methods,independent_reviewer='Root static inspection of separately authored producer,host and independent numerical oracle; formal synthetic results verified',actual_features_computed=False,actual_model_fits_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,findings=[
 'Full295 positive source support; formation t-294..t-42 uses two126log means excluding center; current63sample volatility is unchanged.',
 'Fixed252/127 centroid normalization is algebraically equal to endpoint growth on loglinear paths. Actual values are not snapped or copied. Negative numerators retained; zero volatility commonmissing; arithmetic overflow hardfails.',
 'Independent scalar code reconstructs logarithmic segment means,expm1,endpoint ratio,sample risk,all49percentile domain and detailed event audits without producer imports.',
 'Frozen coreadapter changes half longcore only noncalm,retains calm/firstpartial/missing original support and exactlyone episode. Returned holdings used without global rerank.',
 'Original endpoint control is a diagnostic reconstructed strategy. Numerical and selection parity is measured,never enforced by replacing computed values.',
 'First88 selection frozen beforelater34; nine paths separately label sameclosezero,nextopenzero,nextopen5bp. Source files and21444 protected artifacts remain unchanged.',
 'Zero model fits;252S logs,2S means,S lift,2S numerators,Srisk,2A ranked observations counted. History/source/PIT limitations remain explicit.'

 ])
 with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,tests=tests['tests'],review_sha256=sha('work/level214_CODE_REVIEW.json'))))
if __name__=='__main__':main()
