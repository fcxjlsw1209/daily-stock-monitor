from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,copy
R=Path('/Users/shiweili/code/monthly-stock-research')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def save(p,d):
 q=R/p
 with q.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(p,sha(q))
now=datetime.now(timezone.utc).isoformat();central='outputs/monthly_target70_research/CONTINUATION_STATUS.json';s=read(central)
assert len(s['completed'])==211 and sum(x['trials'] for x in s['completed'])==1150 and not s['registered_queue']
assert s['completed'][-1]['directory']=='monthly_asset_peer213'
evidence=['outputs/monthly_asset_peer213/'+p for p in ['PLAN.json','COMPLETION.json','results/COMPLETION.json','RESEARCH_REVIEW.json']]+['work/monthly_monotone_path212_kernel.py','work/monthly_monotone_path212_independent.py','outputs/monthly_stability_research/stability.py','outputs/monthly_core_expansion/core_features.py','outputs/monthly_core_amplitude70/features.py','outputs/monthly_calendar_mean70/report.py','work/price_distribution171_kernel.py','work/monthly_nested_trend199_kernel.py','outputs/monthly_core_timing70_research/core_adapter.py','outputs/monthly_core_timing70_research/features.py','outputs/monthly_local_linear70/PLAN.json','outputs/monthly_local_linear70/study.py','outputs/monthly_local_linear70/model.py','work/monotone212_SOURCE_REVIEW.json']
pins={p:sha(R/p) for p in evidence}
coverage='work/after213_level_growth_coverage.json'
save(coverage,dict(created_utc=now,status='BOUNDED_READ_ONLY_COVERAGE_BEFORE_214_FEATURES',previous_goal_turn_classification='progress',classification_detail='Round213 completed with independent evidence; subsequent cost clarification is communication only. Coverage review changes next action to a declared structural core test.',central_observed_sha256=sha(R/central),completed_entries=211,trials=1150,source_sha256=pins,findings=['213 changed scores but same-type and all-type transfer chose identical holdings; both failedfirst88. Original Local67 retained.','Continuous magnitude replacement already tested in core_amplitude70; calendar_mean compares month-average endpoints;171compares single endpoint with historical distribution;199aggregates several endpoint ranks;212attenuates endpoint gain with path fit quality.','Whole-formation early/late geometric level growth is distinct from readable definitions. leader_area/record_growth and other missing old originals mean exhaustive historical novelty UNKNOWN.','Selected one structural hypothesis with fixed equation and matched original-endpoint control; no parameter or outcome search.','Deferred original-history empirical forecast overprediction Q75 vsmedian and cross-security error co-shortfall; no actual computations for those.'],actual_new_features=0,actual_new_fits=0,actual_new_orders=0,actual_new_returns=0))
pins[coverage]=sha(R/coverage)
context='work/monthly_level_growth214_context.json'
limits=['Repeatedly reused historical research; later34/full results are not pristine out-of-sample evidence.','Original proxyTop32/theme17 and retrospective price/classification vintages are not certified strictPIT.','Same-close zero-cost main70percent target is idealized; nextopen timing and5bp/side are separately measured sensitivities.','Structural halfcore substitution preserves total longmomentum weight; risk denominator still sees current63 returns while momentum numerator excludes latest42sessions.','Geometric level shift can favor a mid-formation jump; smoother input alone does not establish future advantage.','No parameter search or observer modification; exact novelty against missing historic definitions UNKNOWN.']
save(context,dict(created_utc=now,round=214,name='monthly_level_growth214',status='PROSPECTIVE_BEFORE_NEW_FEATURES_OR_FITS',previous_goal_turn_classification='progress',hypothesis='Sustained increase in price level across the long formation period may rank future monthly leaders better than a difference between two isolated endpoints.',arms=['incumbent_local67','level_growth_core','endpoint_growth_control'],candidate='level_growth_core',control_nonpromotable='endpoint_growth_control',formation={
'source':'Exactly295 finite positive original stored adjusted prices on official sessions t-294..t. Formation first253 prices t-294..t-42. Last64prices t-63..t yield63 simple risk returns. No filling or adjustments.',
'momentum':'Control N=P[252]/P[0]-1; current63 simple-return sample std ddof1 via fsum two-pass. No annualization or epsilon.',
'candidate':'Early P[0:126],late P[127:253],center126 excluded from means. L=(252/127)*(fsum(log late)/126-fsum(log early)/126); candidateN=expm1(L). Exactly252 log observations used. Both negative and positive numerators retained without clipping.',
'normalization':'The factor252/127 is the ratio of endpoint time distance252 to two mean time indices distance127. For strictly loglinear paths candidateN equals endpointN algebraically; no numerical tie snapping or control substitution.',
'raw':'candidateN/vol and controlN/vol, commonfinite support. Missing values areNaN in all8float matrices; ninthmatrix available isbool.',
'invalid':'Missing/nonpositive/nonfinite/nonnumeric source means common unavailable with reason; vol0 meanscommon unavailable. Finite valid source derived overflow or nonfinite arithmetic hardfails.',
'ranking':'Original average ascending pct rank minus.5 on same all49available candidates before eligibility. No second rank; singleton+.5.',
'counts':'S complete295sourcecells,A commonavailable:295S prices,252S logs,63S returns,2S numerators,2S segment means,S risk estimates,2A rankedobservations. No predictive/descriptive model fits; prefix/independent repeats separate.',
'parameter_search':False},integration={
'slot':'Only !state_calm10. Frozen core_adapter delta=.5*(cw*T-originalweightedcore),cw=.52noncalm/.50calm. Calm andfirstpartial targetsNaN; maxabsdelta.26. Original core retained for unavailablecells.',
'episode':'All27original arrays,baseline,coreweights,support and Local67config4pre-score retained. Halfcore then exactlyone original episode and use returnedchoices without global rerank.',
'portfolio':'Original pool, long-onlyEW3,2halves/<2cash; fixed adjustedunits intramonth,difference rebalance,no extra stoploss/terminal liquidation.'},evaluation={
'period':['2016-08-17','2026-09-24'],'training_months':88,'later_months':34,'paths':['same_close_zero','next_open_zero','next_open_cost_5bp_per_side'],
'gates':'Onlycandidatepromotable. First88 samecloseCAGR>=incumbent+.005,nextopen5bpnoninferiorinc; noninferiorbothpathsversusendpointcontrol. Sharpe>=eachreference-.05,MDD>=reference-.03,early53/recent35CAGR>=reference-.025. >=3changedholdingmonths and>=1eachera versusEACHreference. Freeze beforelater34; no reselection.',
'verification':'Independentfullcell rebuilding and originalrisk/coreparity; synthetic scale/loglinear/flat/step/center/invalid/future tests. Physicalfeatureprefixes0/1/53/88/122,123orderprefixes,27originalarrays,9paths45slices18traininggroups,21444protectedfiles.'},limitations=limits,evidence_sha256=pins,pre_registration_actual_features=0,pre_registration_actual_fits=0,pre_registration_actual_orders=0,pre_registration_actual_returns=0,execution_gate='Final syntheticvalidation/staticreview thenimmutablePLAN beforeactualfeatures,ordersorreturns.'))
prior='work/monotone212_SOURCE_REVIEW.json';d=copy.deepcopy(read(prior));assert d['passed'] and d['source_coverage_reused'] and d['source_window_coverage']['price_count']==295
for p,h in d['source_sha256'].items():
 q=Path(p);q=q if q.is_absolute() else R/q
 assert q.resolve().is_relative_to(R) and 'CloudStorage' not in q.parts and q.resolve()!=(R/central).resolve()
 assert sha(q)==h,p
for p in (prior,coverage,context):d['source_sha256'][str(R/p)]=sha(R/p)
d.update(prior_source_review_sha256=sha(R/prior),review_code_sha256=sha(Path(__file__)),observed_central_state=dict(sha256=sha(R/central),completed_entries=211,trials=1150,last='monthly_asset_peer213'),created_utc=now,schema_version='level_growth214_source_reuse_v1',context_sha256=sha(R/context),coverage_sha256=sha(R/coverage),limitations=['Original295price coverage reused after byteverification; no214feature/calculation/order/return.','Coverage does not certifypositivevol ormodeladvantage.']+limits)
for k in list(d):
 if k.startswith('actual_') and k.endswith('_computed'):d[k]=False
for k in ['actual_level_growth_computed','actual_segment_means_computed','actual_model_fits_computed','strategy_registered_by_reviewer','observers_accessed']:d[k]=False
save('work/level214_SOURCE_REVIEW.json',d)
