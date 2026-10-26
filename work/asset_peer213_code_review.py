"""Root static-source and synthetic-receipt review; no actual model inputs."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib
R=Path(__file__).resolve().parents[1]
CONTEXT='work/monthly_asset_peer213_context.json';CONTEXT_SHA='44b2225c237c6306f0cf531550e2202ce57ae6357af55ef244c3cc87ef6c6f8d'
SOURCE='work/asset_peer213_SOURCE_REVIEW.json';SOURCE_SHA='7af413ea369b4d95a08a929798f65456b1e1a516c27b0ae260295894ca60c236'
COVERAGE='work/after212_asset_peer_coverage.json'
def sha(p):
 p=Path(p);p=p if p.is_absolute() else R/p
 return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def need(x,m):
 if not x:raise ValueError(m)
def main():
 out=R/'work/asset_peer213_CODE_REVIEW.json'
 need(not out.exists() and not (R/'outputs/monthly_asset_peer213').exists(),'Static review must precede registration')
 need(sha(CONTEXT)==CONTEXT_SHA and sha(SOURCE)==SOURCE_SHA,'Prospective context/source changed')
 c=read(CONTEXT);source=read(SOURCE)
 need(source['passed'] and source['context_sha256']==CONTEXT_SHA and source['unchanged_source_bytes_reverified'] and source['evidence_sha256']==c['evidence_sha256'],'Source provenance')
 for k in ('actual_features_computed','actual_model_fits_computed','actual_orders_computed','actual_returns_computed','actual_source_arrays_loaded','new_data_acquired'):need(source[k] is False,'Source scope '+k)
 names=['work/'+prefix+'monthly_asset_peer213_'+kind+'.py' for kind in ('host','model','independent') for prefix in ('','test_')]
 codes={p:sha(p) for p in names};trees={p:ast.parse((R/p).read_text()) for p in names}
 receipt='work/monthly_asset_peer213_TESTS.json';tests=read(receipt)
 need(tests['passed'] and tests['synthetic_only'] and tests['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal synthetic receipt')
 methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[p])) for p in names[1::2]}
 need(all(v>0 for v in methods.values()) and sum(methods.values())==tests['tests'],'Exact test coverage count')
 for p,h in codes.items():need(tests['code_sha256'][str(R/p)]==h,'Source changed after formal test '+p)
 imports=[]
 for node in ast.walk(trees[names[4]]):
  if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
  elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
 need(not any('asset_peer213_model' in x or 'asset_peer213_host' in x or 'identity_transfer206_model' in x for x in imports),'Independent oracle imports producer')
 pins=dict(c['evidence_sha256']);pins.update(codes)
 for p,h in tests['code_sha256'].items():pins[str(Path(p).relative_to(R))]=h
 for p in [CONTEXT,COVERAGE,SOURCE,receipt,'work/asset_peer213_source_review.py','work/asset_peer213_code_review.py']:pins[p]=sha(p)
 need(pins['work/asset_peer213_source_review.py']==source['review_code_sha256'],'Retained source-review bytes')
 for p,h in pins.items():need(sha(p)==h,'Changed review pin '+p)
 state=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
 need(state['current']=='monthly_monotone_path212' and not state['registered_queue'] and len(state['completed'])==210 and sum(x['trials'] for x in state['completed'])==1147,'Prospective state moved')
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],context_sha256=CONTEXT_SHA,coverage_sha256=sha(COVERAGE),code_sha256=codes,input_sha256=pins,review_code_sha256=sha('work/asset_peer213_code_review.py'),formal_test_receipt_sha256=sha(receipt),synthetic_tests=tests['tests'],synthetic_test_methods=methods,independent_reviewer='Root static review of separate producer/host/independent oracle and final pinned-runtime synthetic receipt',actual_features_computed=False,actual_model_fits_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,findings=[
 'Only fixed stock/ETF membership changes historical nonself peer set. Original four-label common cohort, five-year prior-December training window and plain10 feature values retained.',
 'Only annual pooled log Ridge refitted; original source/local-linear/local-constant vectors reused under immutable206 independent source proof. Pooled, own-constant, all-peer and availability checked against cached originals.',
 'Same-type peer means cover every original own-history month with original rank-unit kernel/age weights and mass+12 shrink denominator. Any missing same-type month causes exact whole-query all-peer raw fallback; no month dropping/reweighting.',
 'Independent augmented-SVD and scalar reductions reconstruct model and audit; production rank-slot separately checked. Cached fallback equality is distinguished from cross-sectional rank equality.',
 'Full original available-and-eligible ranking domain, original quarter ML slot and firstpartial handling retained. Exactly one original episode and unchanged long EW3 monthly fixed units.',
 'First88 admission frozen before later34/full diagnostics; no parameter search or post-result amplitude adjustment. Sameclosezero main target, nextopenzero and nextopen5bp distinct execution sensitivities.',
 'Actual regressions and correction estimations counted separately across main/prefix/oracle runs. Source cache is annual causal replay, peer residuals mature in-sample, not archival OOF forecasts.',
 'Original pools/vintages remain conditional research proxies; repeated data use prevents pristine OOS claims. Frozen observation accounts untouched.'
 ])
 with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,tests=tests['tests'],review_sha256=sha('work/asset_peer213_CODE_REVIEW.json'))))
if __name__=='__main__':main()
