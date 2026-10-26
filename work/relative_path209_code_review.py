"""Static independent root review and evidence binding, without model imports."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
R=Path(__file__).resolve().parents[1];W=R/'work'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
names=['monthly_relative_path209_host.py','test_monthly_relative_path209_host.py','monthly_relative_path209_model.py','test_monthly_relative_path209_model.py','monthly_relative_path209_independent.py','test_monthly_relative_path209_independent.py']
code={n:(W/n).read_text() for n in names};trees={n:ast.parse(t) for n,t in code.items()}
source=read(W/'relative_path209_SOURCE_REVIEW.json');context=read(W/'monthly_relative_path209_context.json');tests=read(W/'monthly_relative_path209_TESTS.json')
require(source['passed'] and source['source_only'] and source['training_path_coverage']['same_good_as191'],'Source preflight')
require(all(source[k] is False for k in ('actual_features_computed','actual_labels_computed','actual_model_fits_computed','actual_orders_computed','actual_returns_computed')),'Source scope')
require(source['context_sha256']==sha(W/'monthly_relative_path209_context.json')=='0029cb2f60b8afc8585d34789f7b18b827a7de392f04036c42823dcfeacdbe25','Context binding')
require(tests['passed'] is True and tests['synthetic_only'] is True,'Tests failed/scope')
for n in names:require(tests['code_sha256'].get(str(W/n))==sha(W/n),'Test/code drift '+n)
for n in names:
    if n.startswith('test_'):require(any(isinstance(x,ast.FunctionDef) and x.name.startswith('test') for x in ast.walk(trees[n])),'No real tests '+n)
require('monthly_relative_path209_model' not in code['monthly_relative_path209_independent.py'],'Independent imports production')
require('np.linalg.lstsq' in code['monthly_relative_path209_independent.py'],'Independent solve')
h=code[names[0]];m=code[names[2]];i=code[names[4]]
for s in ['SOURCE_REVIEW_SHA=\'976f4b210176e3a4173bddd28e17004a951b0c5c9f0735929df180d0d6da177e\'','prior191_returns_read=False','feature_prefix_training_inputs_truncated=True','len(state[\'completed\']) == 207','== 1138','exact191_candidate_as_control=True']:
    require(s in h,'Host evidence '+s)
for s in ['Fraction.from_float','total / (len(stock) - 1)','months[_next_month(day)]','ends <= cutoff','dates > lower','HALF_LIFE_MONTHS = 12.0','ALPHA = 10.0']:
    require(s in m,'Model contract '+s)
review_checks=[
 {'area':'Complete official path and exact targets','passed':True,'finding':'Source preflight verifies6808samegoodrows. Producer exact rational summation and independent Fraction ratio oracle includeevery official post-signal session through nextnaturalmonthend; each target separatelybounded,onefinalrounding. Missingpath remainsunavailable forboth.'},
 {'area':'Annual matured model clock','passed':True,'finding':'PriorDec31/preceding5years/exclusivelowerbound/18months; originalplain10,monthbalanced12monthrecency,alpha10,training-onlyweightedscaler,targetmeanintercept. PhysicalprefixtruncationplusindependentLS. Querymaskdoesnotinspectfuturelabel.'},
 {'area':'Original portfolio and matched control','passed':True,'finding':'Small+.01negative-predictionrank corrections,max.005,initial0,originalLocal67prescore+oneepisode/core/support/pool/27arrays.209terminalcontrol exact191candidate byte/order check. Monthlyfixedunits preserved.'},
 {'area':'Selection and protected closure','passed':True,'finding':'Predeclared first88gain/risk/subperiod/support guards versusincumbentandnonpromotablecontrol; freeze beforelater34.13prepared/38result artifacts,9paths45slices18traininggroups,21444protectedarchive,206/1135before and207/1138after,CAScentral.'},
 {'area':'Research interpretation','passed':True,'finding':'Repeated-sampledownsidetargetrefinement,notindependentnewalpha. Proxyuniverse/vendorvintagesnotcertifiedstrictPIT. Sameclosezero primary;nextopenzero/costseparated. Noobserverpromotion.'}]
inputs={str(W/n):sha(W/n) for n in ['monthly_relative_path209_context.json','after208_path_shortfall_coverage.json','relative_path209_SOURCE_REVIEW.json','relative_path209_source_review.py','monthly_relative_path209_TESTS.json']}
inputs[str(Path(__file__))]=sha(Path(__file__))
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],review_method='Root static independent read of producer, independentoracle, orchestration and syntheticreceipt; separateagent crosschecks. No actual209 model or marketdata execution bythisreview.',context_sha256=sha(W/'monthly_relative_path209_context.json'),source_review_sha256=sha(W/'relative_path209_SOURCE_REVIEW.json'),code_sha256={str((W/n).relative_to(R)):sha(W/n) for n in names},input_sha256=inputs,review_checks=review_checks,synthetic_tests=tests['tests'],resolved_findings=['Preflight proved fullpathrestriction retains191cohort; no needrelaxcontrolparity.','Syntheticprefix fixtureobjectint-versusfloat assertion repaired before registration; productionalgorithmunchanged.'],actual_features_computed=False,actual_labels_computed=False,actual_model_fits_computed=False,actual_portfolio_returns_computed=False,production_files_modified=False,strategy_registered_by_reviewer=False,limitations=context['limitations'])
p=W/'relative_path209_CODE_REVIEW.json';require(not p.exists(),'Writeonce');p.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'passed':True,'tests':tests['tests'],'output':str(p),'sha256':sha(p)},indent=2))
