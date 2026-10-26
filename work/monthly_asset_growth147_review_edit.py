from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'work/monthly_asset_growth147_outcome_review.py';s=p.read_text()
s=s.replace('import pandas as pd','import pandas as pd\nimport monthly_asset_growth147_review_contract as contract')
s=s.replace("CONCEPT = 'work/after145_cashflow_experiment_concept.json'",'CONCEPT = contract.CONCEPT')
s=s.replace("CONCEPT_SHA = '4fe507528756a3da14b88d24929fb638a88b785c8294d476418bc8768a8ba4b2'",'CONCEPT_SHA = contract.CONCEPT_SHA')
s=s.replace("METHOD = 'current_annual_cash_quality_v1'","METHOD = 'current_annual_cash_quality_v1'\nMEASUREMENT = contract.MEASUREMENT")
s=s.replace("root/'work/monthly_same_report145_outcome_review.py',","root/'work/monthly_asset_growth147_review_contract.py',\n            root/'work/monthly_same_report145_outcome_review.py',")
a=s.index('def verify_proposal(');b=s.index('\n\ndef select(',a)
s=s[:a]+'''def verify_proposal(reader,plan,pins):
    return contract.verify_proposal(reader,plan,pins)
'''+s[b:]
s=s.replace("'currency_key',*HANDS}","'currency_key','measurement',*HANDS}")
s=s.replace("and cell['method']==METHOD, 'Cell status/method')","and cell['method']==METHOD and cell['measurement']==MEASUREMENT, 'Cell status/method')")
s=s.replace("'Three common bounded features')","'Three common bounded features')\n            require(cell['low_asset_growth_bounded']<=.5, 'Negative asset-growth contraction bound')")
s=s.replace("    require(feature.get('original_months')==122", "    for key in ('earnings_and_low_accrual_controls_identical_to146','common_mask_identical_to146','expansion_negative_contraction_positive_zero_change_zero','asset_growth_not_organic_investment_or_CMA_replication'):\n        require(feature.get(key) is True,'Asset-growth claim '+key)\n    require(feature.get('measurement')==MEASUREMENT and feature.get('growth_denominator')=='comparison_total_assets','Asset-growth definition')\n    require(feature.get('exact_feature_range')=='(-1,0.5); float rounding may reach an endpoint' and feature.get('dose_0_01_exact_score_range')=='(-0.01,0.005)','Unscaled asset-growth range')\n    require(feature.get('original_months')==122")
s=s.replace("    return receipt\n\ndef verify_diagnostic_start", "    contract.verify_asset_use(reader,plan,pins,data)\n    return receipt\n\ndef verify_diagnostic_start")
# Existing file has triple newline separation; insert independent of formatting.
if 'contract.verify_asset_use(reader,plan,pins,data)' not in s:
 s=s.replace('    return receipt\n','    contract.verify_asset_use(reader,plan,pins,data)\n    return receipt\n')
s=s.replace("'all_signal_prefixes_verified','exact_integer_closed_form_arithmetic')", "'all_signal_prefixes_verified','exact_integer_closed_form_arithmetic','asset_growth_purpose_review_verified','identical_four_field_source_mask_verified')")
s=s.replace('source_admission_bound=True,common_availability_verified=True,baseline_local67_full_orders_unchanged=True,','source_admission_bound=True,common_availability_verified=True,baseline_local67_full_orders_unchanged=True,\n        asset_growth_purpose_review_verified=True,identical_four_field_source_mask_verified=True,')
p.write_text(s)
