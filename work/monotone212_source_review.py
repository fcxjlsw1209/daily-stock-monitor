from pathlib import Path
import hashlib,json,datetime,copy
R=Path('/Users/shiweili/code/monthly-stock-research')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(s): return json.loads((R/s).read_bytes())
def save(s,d):
 p=R/s
 with p.open('x') as f: f.write(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(s,sha(p))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
central='outputs/monthly_target70_research/CONTINUATION_STATUS.json';st=load(central)
assert len(st['completed'])==209 and sum(x['trials'] for x in st['completed'])==1144 and not st['registered_queue']
assert st['completed'][-1]['directory']=='monthly_forecast_blend211'
evidence=[
 'outputs/monthly_forecast_blend211/PLAN.json','outputs/monthly_forecast_blend211/COMPLETION.json','outputs/monthly_forecast_blend211/results/COMPLETION.json','outputs/monthly_forecast_blend211/RESEARCH_REVIEW.json',
 'outputs/monthly_stability_research/stability.py','outputs/monthly_core_expansion/core_features.py','outputs/monthly_path70_research/features.py',
 'work/monthly_breakpoint177_kernel.py','outputs/monthly_breakpoint177/PLAN.json','outputs/monthly_core_timing70_research/core_adapter.py',
 'outputs/monthly_core_timing70_research/features.py','outputs/monthly_local_linear70/PLAN.json','outputs/monthly_local_linear70/study.py','outputs/monthly_local_linear70/model.py',
 'work/adjacent205_SOURCE_REVIEW.json']
pins={p:sha(R/p) for p in evidence}
coverage='work/after211_monotone_path_coverage.json'
save(coverage,dict(created_utc=now,status='BOUNDED_READ_ONLY_COVERAGE_BEFORE_212_FEATURES',previous_turn_classification='progress',central_observed_sha256=sha(R/central),completed_entries=209,trials=1144,findings=[
 '211 completed with independent outcome and saved review; first88 did not promote candidate. Original Local67 retained at same-close-zero67.66617372041686 percent.',
 'General horizon/risk/regime skeletons heavily explored: stability1440 and core expansion1777 configurations; calendar anchors,HAC,OLS/TheilSen,Kalman,breakpoint and path efficiency also covered. These are not new directions.',
 'Exact monotone least-squares PAVA path-quality substitution was not found in bounded readable Python/context/PLAN searches. Earlier missing originals mean exhaustive novelty is UNKNOWN.',
 'PAVA and positive affine descriptive fits share same253 formation logs, same long endpoint return and current63 risk denominator. PAVA flexibility is not new information or proof of predictive advantage.',
 'Deferred, not implemented here: within-asset-type residual transfer and prediction-error covariance portfolio controls. Generic type ranks, raw correlation diversification and annual loss mixing already covered.'
],source_sha256=pins,actual_new_features=0,actual_new_fits=0,actual_new_orders=0,actual_new_returns=0,algorithm_reference='https://scikit-learn.org/stable/modules/isotonic.html',reference_scope='Only monotone weighted least-squares definition; no new dependency or market data.'))
pins[coverage]=sha(R/coverage)
context='work/monthly_monotone_path212_context.json'
limitations=[
 'History repeatedly reused; later34/full diagnostics are not pristine out-of-sample evidence.',
 'Original proxyTop32/theme17 and retrospective source/classification vintages are not certified strictPIT.',
 'Monotone fit is more flexible than affine, so lower training residual is mechanical and does not establish higher future return. A single upward jump may receive Q=1.',
 'This is a structural noncalm half-core replacement, not a 1 percent auxiliary tweak; original risk denominator may still react to recent data although momentum numerator skips42 sessions.',
 'Same-close zero cost is the idealized main70 percent target. Next-open timing and5bp fees are separate sensitivities.',
 'No parameter search or observer modification. Exact novelty of unavailable older definitions remains unknown.'
]
save(context,dict(created_utc=now,round=212,name='monthly_monotone_path212',status='PROSPECTIVE_BEFORE_NEW_FEATURES_OR_FITS',previous_goal_turn_classification='progress',hypothesis='An upward path may accelerate,decelerate or plateau. Upward monotone least-squares path quality may distinguish persistent long winners better than a single nonnegative affine line, at unchanged endpoint momentum and risk clock.',arms=['incumbent_local67','monotone_path_core','affine_path_control'],candidate='monotone_path_core',control_nonpromotable='affine_path_control',formation={
 'source':'Exactly295 finite positive original stored adjusted prices on official sessions t-294 through t. No filling,new adjustments or source splicing. First253 prices t-294 through t-42 are formation; last64 prices t-63 through t yield63 simple risk returns.',
 'momentum':'N=P[252]/P[0]-1; vol is sample standard deviation of63simple returns,ddof1,fsum two-pass. R=N/vol. No annualization or epsilon denominator.',
 'centered_log_path':'x_j=math.log(P[j]), j0..252; z=x-fsum(x)/253; zmean=fsum(z)/253; SST=fsum((z-zmean)^2). Exact positiveSST andvol necessary for both arms.',
 'candidate_fit':'Equal-weight nondecreasing PAVA on z. Merge equal adjacent means into maximal blocks; each mean=fsum(original z in block)/count. SSE=fsum((z-fitted)^2).',
 'control_fit':'Centered time j-126. beta=max(0,fsum((j-126)*(z-zmean))/fsum((j-126)^2)); fitted=zmean+beta*(j-126), SSE=fsum(residual^2).',
 'quality':'Q=1-SSE/SST. Out-of-[0,1] tolerated only within1e-12 arithmetic roundoff then clipped. PAVA singleblock and OLS beta<=0 exactQ0; SSE0 exactQ1.',
 'raw':'If R>0, raw_candidate=R*Q_iso and raw_control=R*Q_line. Otherwise both exactR, preventing negative momentum improving towardzero. Common invalid/unavailable cells remain NaN in all8float matrices.',
 'invalid':'Missing, nonnumeric or nonpositive/nonfinite raw sources -> common unavailable with reason. Complete source but zeroSST/zeroVol -> common unavailable. Valid finite source arithmetic overflow or nonfinite derived intermediate is a hard error.',
 'ranking':'Average ascending percentile rank minus.5 across identical all-original49 available cells, before eligibility or type filtering; singleton+.5, ties use original percentiles. Candidate/control already ranked; no second transform.',
 'counts':'Let S=complete295-sourcecells and A=commonavailable. valid_price_observations=295S,formation_log_observations=253S,simple_return_observations=63S,numerator_evaluations=S,risk_evaluations=S,descriptive_pava_fits=A,descriptive_affine_fits=A,ranked_observations=2A. Prefix and independent repeats separate. new_model_fits,new_predictive_model_fits,predictive_model_fits=0 means zero predictive fits, not zero descriptive fits.',
 'parameter_search':False},integration={
 'slot':'Only !state_calm10; delta=.5*(cw*T-originalweightedcore), using frozen core_adapter.py. Original cw=.50calm/.52noncalm. Calm and firstpartial row targetsNaN so original core unchanged; firstpartial features still computed normally. Maxabsdelta.26.',
 'episode':'Retain all27original arrays, baseline,coreweights,support,original effects and Local67 config4pre-score/policy; replace halfcore then exactlyone original episode. Use episode choices directly, no rerank.',
 'portfolio':'Original pool unchanged, monthly longEW3, twohalves/<2cash; adjusted units fixed intramonth; difference rebalance, no added stoploss or terminal liquidation.'},evaluation={
 'period':['2016-08-17','2026-09-24'],'training_months':88,'later_months':34,
 'paths':['same_close_zero','next_open_zero','next_open_cost_5bp_per_side'],
 'gates':'Onlycandidate promotable. First88 samecloseCAGR>=incumbent+.005 and nextopen5bp noninferior; noninferior both paths vsaffinecontrol. Sharpe>=eachreference-.05,MDD>=reference-.03,early53/recent35CAGR>=reference-.025. >=3changedholding sets and>=1eachera vsEACHreference. Freeze beforelater34, no reselection.',
 'verification':'Independent full-cell numerical reconstruction plus PAVA KKT certificate; synthetic analytic/invalid/negative momentum/rank/maturity tests. Physical sourcefeature prefixes0/1/53/88/122; all123orderprefixes; all27original arrays and incumbentorders exact;9paths45slices18first88groups;21444protected register/close.',
 'kkt':'Oracle separately verifies nondecreasing fit, blockmeans, residual prefix>=0,total0 and complementarity with explicit scale tolerance1e-10; no rank-tie snapping.'},limitations=limitations,evidence_sha256=pins,pre_registration_actual_features=0,pre_registration_actual_fits=0,pre_registration_actual_orders=0,pre_registration_actual_returns=0,execution_gate='Final synthetic validation and static/source review, then immutable PLAN before actual features,descriptive fits,orders or returns.'))
prior='work/adjacent205_SOURCE_REVIEW.json';d=copy.deepcopy(load(prior))
assert d['passed'] and d['source_coverage_reused'] and d['source_window_coverage']['price_count']==295
sourcepins=d['source_sha256']
for p,h in sourcepins.items():
 q=Path(p);q=q if q.is_absolute() else R/q
 assert 'CloudStorage' not in q.parts and q.resolve().is_relative_to(R)
 assert q.resolve()!=(R/central).resolve()
 assert sha(q)==h, 'source mismatch '+p
sourcepins[str(R/prior)]=sha(R/prior)
sourcepins[str(R/coverage)]=sha(R/coverage)
sourcepins[str(R/context)]=sha(R/context)
d.update(created_utc=now,schema_version='monotone212_source_reuse_v1',context_sha256=sha(R/context),coverage_sha256=sha(R/coverage),prior_source_review_sha256=sha(R/prior),source_sha256=sourcepins,source_coverage_reused=True,unchanged_source_bytes_reverified=True,source_only=True,passed=True,unresolved_blockers=[],review_code_sha256=sha(Path(__file__)),observed_central_state={'sha256':sha(R/central),'completed_entries':209,'trials':1144,'last':'monthly_forecast_blend211'},limitations=[
 'Only original295-source coverage receipt reused after fresh byte verification; no212price-cell arithmetic,features or fits computed.',
 'Finite positive source coverage alone does not establish positiveSST/vol or numerical safety; registered preparation checks those.',
 'Original proxy pools,vendor vintages,aliases and normalizations retained; conditionalresearch only,notstrictPIT or pristineOOS.'])
for k in ['actual_source_coverage_computed','actual_signal_returns_computed','actual_features_computed','actual_orders_computed','actual_returns_computed','actual_risk_estimates_computed','actual_ranks_computed','actual_deltas_computed','actual_portfolio_returns_computed','actual_path_quality_computed','actual_pava_fits_computed','actual_affine_fits_computed','actual_model_fits_computed','production_files_modified','strategy_registered_by_reviewer','observers_accessed']:
 d[k]=False
save('work/monotone212_SOURCE_REVIEW.json',d)
