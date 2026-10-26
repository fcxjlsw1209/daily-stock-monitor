"""Root pre-registration metadata/hash review. No numerical source arrays loaded."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy
R=Path(__file__).resolve().parents[1]
def sha(p):
 p=Path(p);p=p if p.is_absolute() else R/p
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads((R/p).read_text())
def save(p,d):
 with (R/p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(p,sha(p))
now=datetime.now(timezone.utc).isoformat();central='outputs/monthly_target70_research/CONTINUATION_STATUS.json';s=load(central)
assert s['current']=='monthly_monotone_path212' and s['current_status']=='completed_independently_verified'
assert len(s['completed'])==210 and sum(x['trials'] for x in s['completed'])==1147 and not s['registered_queue']
assert not (R/'outputs/monthly_asset_peer213').exists()
prior='work/identity206_SOURCE_REVIEW.json';old=load(prior)
assert old['passed'] and old['prior_parity_receipt_reverified'] and old['strict_pit'] is False
for mapping in [old['source_sha256'],old['raw_source_hashes']]:
 for p,h in mapping.items():
  q=Path(p);q=q if q.is_absolute() else R/q
  assert 'CloudStorage' not in q.parts and (q.resolve().is_relative_to(R) or str(q)=='/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9')
  assert sha(q)==h,p
base=load('work/monthly_forecast_blend211_context.json')['evidence_sha256']
evidence={p:h for p,h in base.items() if not p.startswith('outputs/monthly_asset_aux210/') and p!='work/after210_forecast_blend_coverage.json'}
for p in [
 'outputs/monthly_monotone_path212/PLAN.json','outputs/monthly_monotone_path212/COMPLETION.json','outputs/monthly_monotone_path212/results/COMPLETION.json','outputs/monthly_monotone_path212/RESEARCH_REVIEW.json',
 'outputs/monthly_asset_aux210/PLAN.json','work/monthly_asset_aux210_model.py',
 'outputs/monthly_forecast_blend211/PLAN.json','work/monthly_forecast_blend211_model.py',
 'outputs/monthly_architecture_restart/residual_model.py']:
 if (R/p).exists():evidence[p]=sha(p)
for p,h in evidence.items():assert sha(p)==h,p
coverage='work/after212_asset_peer_coverage.json'
save(coverage,dict(created_utc=now,status='BOUNDED_READ_ONLY_COVERAGE_BEFORE_213_NUMERICS',previous_goal_turn_classification='progress',central_observed_sha256=sha(central),completed_entries=210,trials=1147,findings=[
 '212 completed and independently reviewed; candidate failed first88 gates. Incumbent Local67 same-close-zero CAGR remains0.6766617372041686.',
 '206 uses all nonself peers in each matched mature own-history month. 210 changes asset-type ranking of auxiliary aggregate;211 blends saved own/peer forecasts. Neither reconstructs same-type historical peer residual averages.',
 'Bounded source review found no exact previous same-type residual-transfer implementation. Missing older originals prevent exhaustive novelty certification.',
 '206 cache lacks full pooled coefficients and peer residual vectors. Reconstruct only annual pooled log Ridge; reuse original source/local-linear/local-constant forecasts with immutable independent source proof, avoiding3987 own-linear solves.',
 'Same-type restriction alters both identity and peer count. Improved results cannot uniquely be attributed to type. Fewer peers may increase noise; no amplitude tuning after results.'
],evidence_sha256=evidence,actual_new_features=0,actual_new_fits=0,actual_new_orders=0,actual_new_returns=0))
evidence[coverage]=sha(coverage)
context='work/monthly_asset_peer213_context.json'
save(context,dict(created_utc=now,round=213,name='monthly_asset_peer213',status='PROSPECTIVE_BEFORE_NEW_RESIDUALS_OR_FITS',previous_goal_turn_classification='progress',hypothesis='Historical residual opportunity transfer may be more useful within the same asset type: equity queries reference other equities and ETF queries other ETFs, retaining original own-history weights and small monthly ML slot.',arms=['incumbent_local67','same_asset_peer_transfer','all_asset_peer_control'],candidate='same_asset_peer_transfer',control_nonpromotable='all_asset_peer_control',formation={
 'source':'Immutable206 FORECASTS seven keys raw_source_log/raw_local_linear/raw_local_constant/raw_candidate/raw_control/available/local_available; originalTRAIN_INPUTS,MONTHLY_META and27Local67arrays. Originalcached query forecasts are annual causal replays, not realtime archives.',
 'identity':'Original fixed17ETF identity set from MONTHLY_META.monthly.etfs; all other original security IDs equity. No ticker remap, new universe filter or future classification.',
 'cohort':'Original plain10 finite and all4targets finite common cohort, minimum2rows permonth, original labels unchanged. QueryyearY cutoff previousDecember31; signal strictly after cutoff minus5calendar years and label_end<=cutoff. Original min18maturemonths.',
 'pooled':'Recompute only original annual pooled log_excess Ridge alpha10, original month-balanced12month-decay weights/scaling and numeric solver. Original pooled query forecasts must exactly equal cachedsource_log. Cached original local_linear is reused by source proof, not newly fitted.',
 'own':'Exactly206 own-security mature IDs and local weights: age12month half-life times exp(-mean squared ORIGINAL centered-percentile feature distance/(2*.5^2)); no pooled-scale division; min18ownmonths. Original shrinkage denominator mass+12. Recomputed ownconstant must exactly equal cachedlocalconstant.',
 'all_peer_control':'In each matched ownmonth use all other securities with206 similarity kernel and pooled in-sample training residuals. Existing206 weighted residual correction and pooled addition unchanged. Must exactly equal206cached raw_candidate. Original raw_control=local_constant is a separate source proof, not new control.',
 'candidate':'Same formula, but peer set also requires original asset type equal querytype. Keep every matched ownmonth, original ownweights and same mass+12 denominator. If ANY ownmonth has no same-type nonselfpeer, wholequery rawcandidate exactcopiesallpeercontrol. Do not drop/reweight unsupportedmonths. Ownhistory<18: both pooledexact.',
 'arithmetic':'Original finite input and derived arithmetic assertions retained. Invalid/nonfinite arithmetic is a hard error, never a scarcity fallback. Current unavailable remainsNaN and masks unchanged.',
 'ranking':'Ranking U=original forecast_available AND eligible, both assettypes together; NO local_available or same-type support filter. Rawfallback does not imply equalrank because otherrows canmove.',
 'fit_accounting':'Count annual_pooled_log_refits,cached_forecast_rows,own_weight_evaluations,all_peer_query_evaluations,all_peer_month_evaluations,all_peer_row_observations,same_type_supported_queries,same_type_fallback_queries,same_type_month_evaluations,same_type_row_observations,actual_regression_solves. Only actual annual pooledlog regressions count regression_solves (expected11 main). Corrections still count query/month/row estimations. Main,prefix,independent repeats separately; no zero-fit claim.',
 'parameters':dict(min_months=18,min_own_months=18,lookback_calendar_years=5,age_half_life_months=12,bandwidth=.5,shrink_months=12,ridge_alpha=10), 'parameter_search':False},integration={
 'slot':'Exactly206 quarter-slot: delta=.25*(.05*weak_narrow+.01*(calm10 AND NOT weak_narrow)*extension_available)*(new_rank-old_local_linear_rank). Full originalU; paired whole-row readiness, firstpartialdelta0, maxabsdelta.0125.',
 'episode':'Add delta to original Local67config4prescore then exactlyone originalepisode with originalbaseline/core/support/effects/policy. No second rank or larger amplitude.',
 'portfolio':'Unchanged proxyTop32+theme17, dominantlongmomentum numerator skips42sessions, longEW3/twohalves/<2cash, monthly fixedadjustedunits, difference rebalance, no intramonth exits or terminalforcedsale.'},evaluation={
 'period':['2016-08-17','2026-09-24'],'training_months':88,'later_months':34,'paths':['same_close_zero','next_open_zero','next_open_cost_5bp_per_side'],
 'gates':'Onlycandidatepromotable. First88 samecloseCAGR>=incumbent+.005,nextopen5bp noninferior; noninferior bothpaths vsallpeercontrol; Sharpe>=eachreference-.05,MDD>=reference-.03,early53/recent35CAGR>=reference-.025. >=3changedholding sets and>=1eachera vsEACHreference. Freeze beforelater34; no reselection.',
 'verification':'Independent annual cohort,augmented-SVD pooled fit,scalar ownweight/peermeans/type/fallback reconstruction and rank/order/accounting. Physical model/cache/query/label prefixes0/1/53/88/122; all123orderprefixes; all27original arrays/incumbentorders exact and206allpeercontrol raw/rank/delta/orders parity.9paths45slices18first88groups;21444protected.',
 'diagnostics':'Report bytype supported/fallback counts, samepeer/allpeer month/row counts and raw/rank/holding changes. These explain evidence, never select or retune.'},limitations=[
 'Repeatedly reused historical data; later34/full diagnostic is not pristine out-of-sample.',
 'ProxyTop32/theme17 and retrospectivevendor/classification vintages are not certified strictPIT.',
 'Historical peer residuals are mature training-sample residuals from current annual model, not OOF or contemporaneously archived residuals.',
 'Type restriction changespeer samplecount/variance as well as identity; original smallslot mayyieldno holding differences. No postresult amplification.',
 'Cached original local-linear/sourcevector reuse is authenticated by previous independent proof; only pooledlog and residual transfer newly reconstructed.',
 'Sameclosezero is idealized70target; nextopen execution timing and5bp fees separate sensitivities. Frozen observers unchanged.'
],evidence_sha256=evidence,pre_registration_actual_features=0,pre_registration_actual_fits=0,pre_registration_actual_orders=0,pre_registration_actual_returns=0,execution_gate='Synthetic validation and independent source/static review, then immutablePLAN before actual residuals,forecasts,ranks,orders or returns.'))
d=copy.deepcopy(old);pins=d['source_sha256']
for p,h in evidence.items():pins[str(R/p)]=h
pins[str(R/prior)]=sha(prior);pins[str(R/context)]=sha(context);pins[str(R/'work/asset_peer213_source_review.py')]=sha('work/asset_peer213_source_review.py')
d.update(created_utc=now,context_sha256=sha(context),coverage_sha256=sha(coverage),evidence_sha256=evidence,prior_source_review_sha256=sha(prior),source_sha256=pins,immutable_source_files=len(pins),review_code_sha256=sha('work/asset_peer213_source_review.py'),cached_forecasts_are_prior_annual_asof_replays=True,not_realtime_archival_vintages=True,numeric_type_support_not_yet_computed=True,unchanged_source_bytes_reverified=True,source_kind='Original206 immutable training/query/forecast cache and independent proof; only metadata and bytes reviewed before213 registration.',limitation='Actual213cohort,pooled residuals,type peer support,forecasts and parity remain uncomputed until registered prepare. Cached local-linear reuse is authenticated previous proof, not a new refit. Research proxy assumptions unchanged.')
for k in ['actual_features_computed','actual_model_fits_computed','actual_orders_computed','actual_returns_computed','actual_source_arrays_loaded','new_data_acquired']:d[k]=False
save('work/asset_peer213_SOURCE_REVIEW.json',d)
