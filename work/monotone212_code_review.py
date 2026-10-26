"""Root static review after synthetic validation; no model or real-data imports."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib
R=Path(__file__).resolve().parents[1]
CONTEXT='work/monthly_monotone_path212_context.json'
CONTEXT_SHA='b468ca227df4e9af5c043e511fabd4ea2bac13b5550bd226d01a533c65dd6a6c'
COVERAGE='work/after211_monotone_path_coverage.json'
SOURCE='work/monotone212_SOURCE_REVIEW.json'
SOURCE_SHA='678040d534e11a140be318963fb2fb4eda78ff02962e9c22f6e122634872a7e8'
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def need(x,m):
 if not x:raise ValueError(m)
def main():
 out=R/'work/monotone212_CODE_REVIEW.json'
 need(not out.exists() and not (R/'outputs/monthly_monotone_path212').exists(),'Review must precede registration')
 need(sha(CONTEXT)==CONTEXT_SHA and sha(SOURCE)==SOURCE_SHA,'Prospective source/context changed')
 c=read(CONTEXT);source=read(SOURCE)
 need(source['passed'] and source['context_sha256']==CONTEXT_SHA and source['source_coverage_reused'] and source['unchanged_source_bytes_reverified'],'Source admission')
 need(source['source_window_coverage']['price_count']==295 and source['source_window_coverage']['complete_positive_price_cells']==5973,'Source coverage')
 for k in ('actual_source_coverage_computed','actual_features_computed','actual_orders_computed','actual_returns_computed','actual_path_quality_computed','actual_pava_fits_computed','actual_affine_fits_computed','actual_model_fits_computed'):need(source[k] is False,'Source boundary '+k)
 names=['work/'+prefix+'monthly_monotone_path212_'+kind+'.py' for kind in ('host','kernel','independent') for prefix in ('','test_')]
 codes={p:sha(p) for p in names};texts={p:(R/p).read_text() for p in names};trees={p:ast.parse(v) for p,v in texts.items()}
 receipt='work/monthly_monotone_path212_TESTS.json';tests=read(receipt)
 need(tests['passed'] and tests['synthetic_only'] and tests['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal synthetic receipt')
 methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[p])) for p in names[1::2]}
 need(all(n>0 for n in methods.values()) and sum(methods.values())==tests['tests'],'Exact test methods')
 for p,h in codes.items():need(tests['code_sha256'][str(R/p)]==h,'Tested code changed')
 imports=[]
 for node in ast.walk(trees[names[4]]):
  if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
  elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
 need(not any('monotone_path212_kernel' in x or 'monotone_path212_host' in x for x in imports),'Oracle imports producer/host')
 pins=dict(c['evidence_sha256']);pins.update(codes)
 for p,h in tests['code_sha256'].items():pins[str(Path(p).relative_to(R))]=h
 for p in (CONTEXT,COVERAGE,SOURCE,receipt,'work/monotone212_code_review.py','work/monotone212_source_review.py'):pins[p]=sha(p)
 need(pins['work/monotone212_source_review.py']==source['review_code_sha256'],'Retained source-review script bytes')
 for p,h in pins.items():need(sha(p)==h,'Changed reviewed source '+p)
 state=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
 need(state['current']=='monthly_forecast_blend211' and not state['registered_queue'] and len(state['completed'])==209 and sum(x['trials'] for x in state['completed'])==1144,'Stale prospective state')
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],context_sha256=CONTEXT_SHA,coverage_sha256=sha(COVERAGE),code_sha256=codes,input_sha256=pins,review_code_sha256=sha('work/monotone212_code_review.py'),formal_test_receipt_sha256=sha(receipt),synthetic_tests=tests['tests'],synthetic_test_methods=methods,independent_reviewer='Root static inspection of separately authored producer,host and independent numerical oracle; formal synthetic results verified',actual_features_computed=False,actual_model_fits_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,findings=[
 'Exact295-source support with253 formation logs ending42sessions before signal and63current simple-return sample risk. Momentum endpoint and denominator remain original noncalm clocks.',
 'Producer PAVA stack and independent whole-partition violation scan agree in synthetic fixtures; independent KKT certificate and exhaustive small minmax tests validate projection, with tolerance confined to certificate rather than features/ranks.',
 'Both arms positive momentum attenuated by Q, nonpositive momentum exactcopy; joint positiveSST/vol support and NaN unavailable. All49sourceavailable percentile domain before eligibility,type and calm suppression; no rerank.',
 'Frozen adapter replaces halfcore only noncalm; firstpartial/calm/missing unchanged. Original support,effects and exactlyone original episode retained; outcomes use returned choices.',
 'First88-only selection frozen before later34; nine execution paths distinguish zero-cost timing from fees. No automatic promotion to observers.',
 'No new predictive fits; each available cell has two descriptive path fits. Byte/source receipt is reused transparently; actual arithmetic and full original-array/prefix integrity verified only after registration.',
 'PAVA flexibility and jump acceptance are documented limits; not certifiedPIT nor pristineOOS after repeated historical reuse.'
 ])
 with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,tests=tests['tests'],review_sha256=sha('work/monotone212_CODE_REVIEW.json'))))
if __name__=='__main__':main()
