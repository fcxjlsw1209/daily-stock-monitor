from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=('prepare_review.py','outcome_review.py')
for tail in files:
 for prefix in ('','test_'):
  source=root/'work'/f'{prefix}monthly_cash_quality146_{tail}'
  dest=root/'work'/f'{prefix}monthly_asset_growth147_{tail}'
  if dest.exists():raise RuntimeError('Preserve existing '+str(dest))
  s=source.read_text().replace('monthly_cash_quality146','monthly_asset_growth147').replace('round146','round147').replace('Round146','Round147').replace("==146","==147").replace("== 146","== 147")
  s=s.replace("'earnings_only'","'earnings_control'").replace("'cash_flow_only'","'low_accrual_control'").replace("'low_accrual_quality'","'low_asset_growth'")
  if prefix=='':
   s=s.replace("'earnings_bounded','cash_flow_bounded','low_accrual_bounded'","'earnings_bounded','low_accrual_bounded','low_asset_growth_bounded'")
   s=s.replace("'earnings_bounded', 'cash_flow_bounded', 'low_accrual_bounded'","'earnings_bounded', 'low_accrual_bounded', 'low_asset_growth_bounded'")
   s=s.replace("cash_quality_feature_matrices_verified","asset_growth_feature_matrices_verified")
   s=s.replace('exact_fraction_scaled_amounts_average_assets_ratios_then_float64_bounded_mapping','exact_fraction_scaled_endpoints_negative_asset_change_over_prior_assets_then_bounded_float')
   s=s.replace('caller_verified_new_field_specific_admission_and_source_hashes_required','caller_verified_147_source_use_review_and_frozen146_source_closure_required')
  else:
   s=s.replace('cash_quality_feature_matrices_verified','asset_growth_feature_matrices_verified')
   s=s.replace('exact_fraction_scaled_amounts_average_assets_ratios_then_float64_bounded_mapping','exact_fraction_scaled_endpoints_negative_asset_change_over_prior_assets_then_bounded_float')
   s=s.replace('caller_verified_new_field_specific_admission_and_source_hashes_required','caller_verified_147_source_use_review_and_frozen146_source_closure_required')
  dest.write_text(s)
