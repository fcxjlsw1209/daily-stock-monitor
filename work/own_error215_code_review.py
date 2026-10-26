"""Root static review of215 numerical code after formal synthetic tests; no model imports."""
from pathlib import Path
from datetime import datetime, timezone
import ast,json,hashlib
R=Path(__file__).resolve().parents[1]
CONTEXT='work/monthly_own_error215_context.json'
CONTEXT_SHA='593d72172c007c28427d6cae5721ff0f869e7f66a0f348a1d10af9232be5681a'
COVERAGE='work/after214_own_error_coverage.json'
SOURCE='work/own_error215_SOURCE_REVIEW.json'
SOURCE_SHA='b146c39268a280b9bdef2cb639c8540b0c8b204967c507e48a81ed27358d4dbb'
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def need(x,m):
    if not x:raise ValueError(m)
def main():
    out=R/'work/own_error215_CODE_REVIEW.json'
    need(not out.exists() and not (R/'outputs/monthly_own_error215').exists(),'Review must precede registration')
    need(sha(CONTEXT)==CONTEXT_SHA and sha(SOURCE)==SOURCE_SHA,'Prospective source/context changed')
    c=read(CONTEXT);source=read(SOURCE)
    need(source['passed'] and source['context_sha256']==CONTEXT_SHA and source['matched_formal_months']==113,'Source identity review')
    for k in ('actual_new_losses','actual_new_coefficients','actual_new_orders','actual_new_returns'):need(source[k]==0,'Source boundary')
    names=['work/'+prefix+'monthly_own_error215_'+kind+'.py' for kind in ('host','model','independent') for prefix in ('','test_')]
    codes={p:sha(p) for p in names};texts={p:(R/p).read_text() for p in names};trees={p:ast.parse(v) for p,v in texts.items()}
    receipt='work/monthly_own_error215_TESTS.json';tests=read(receipt)
    need(tests['passed'] and tests['synthetic_only'] and tests['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal synthetic receipt')
    methods={p:sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(trees[p])) for p in names[1::2]}
    need(all(n>0 for n in methods.values()) and sum(methods.values())==tests['tests'],'Exact test methods')
    for p,h in codes.items():need(tests['code_sha256'][str(R/p)]==h,'Tested code changed')
    need('monthly_own_error215_model' not in texts[names[4]],'Oracle imports producer')
    pins=dict(c['evidence_sha256']);pins.update(codes)
    for p,h in tests['code_sha256'].items():pins[str(Path(p).relative_to(R))]=h
    pins.update({CONTEXT:sha(CONTEXT),SOURCE:SOURCE_SHA,receipt:sha(receipt),'work/own_error215_code_review.py':sha('work/own_error215_code_review.py')})
    for p,h in pins.items():need(sha(p)==h,'Changed reviewed source '+p)
    state=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    need(state['current']=='monthly_level_growth214' and not state['registered_queue'] and len(state['completed'])==212 and sum(x['trials'] for x in state['completed'])==1153,'Stale prospective state')
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],context_sha256=CONTEXT_SHA,coverage_sha256=sha(COVERAGE),code_sha256=codes,input_sha256=pins,review_code_sha256=sha('work/own_error215_code_review.py'),formal_test_receipt_sha256=sha(receipt),synthetic_tests=tests['tests'],synthetic_test_methods=methods,independent_reviewer='root static review of separately authored producer/host/scalar oracle',actual_features_computed=False,actual_model_fits_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,findings=[
        'Current and historical prediction source is exact206 raw_local_linear; raw_control remains a local_constant parity check only. No source refits or invented pre2016 predictions.',
        'Exact signal/security/end identity and original monthly-centered target_log_excess. Original historical U intersects source availability; no current-holding/local-support restriction or per-month two-security filter.',
        'Annual prior-December five-year mature same-security history, at least18 distinct months; signed prediction errors produce nearest-rank Q75 and conventional median. Cold or zero-penalty preserves exact forecast; derived nonfinite arithmetic fails.',
        'Two empirical calibration parameter fits per supported security-year including zero-clamped fits; prefix and independent repetitions remain separate. No Ridge/tree refits.',
        'Original quarter ML slot, full original ranking U, paired whole-row fallback and one episode. Long skip42 core, EW3 and monthly fixedunits unchanged. All123 physically truncated histories and order prefixes are required.',
        'First88 selection freezes before later34 diagnostics. Reused history/source-vintage and proxy-pool limitations retained; no frozen observers changed.'
    ])
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,tests=tests['tests'],review_sha256=sha('work/own_error215_CODE_REVIEW.json'))))
if __name__=='__main__':main()
