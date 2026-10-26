"""Create the new162 host and metadata-only preregistration scaffolding.

No historical numerical features, orders, or outcomes are computed here.
Existing frozen161 files are read and preserved byte for byte.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'

def write(name, text):
    with (WORK / name).open('x') as f:
        f.write(text)

def rename(text):
    text = text.replace('rank161', 'coskew162')
    return re.sub(r'\b(160|161|995|998)\b', lambda m: {'160':'161','161':'162','995':'998','998':'1001'}[m[0]], text)

host = rename((WORK/'rank161_host.py').read_text())
host = host.replace('Fixed directed monthly rank experiment.', 'Fixed two-sided coskewness experiment.')
host = host.replace('directed rank proposal', 'coskewness proposal')
host = host.replace('Actual rank coverage', 'Actual coskewness coverage')
host = host.replace('directed monthly rank experiment', 'coskewness experiment')
host = host.replace('fixed six-month rank residence/trend experiment', 'fixed210-return coskewness experiment')
host = host.replace('fixed rank residence/trend contrast', 'fixed signed coskewness contrast')
host = host.replace('Exact rank/delta prefixes', 'Exact coskewness/delta prefixes')
host = host.replace('Valid common rank-source support', 'Valid common coskewness support')
host = host.replace("dict(rank_trajectories=[x['rank_trajectories']for x in audit['event_history']],common_support=", "dict(raw_coskewness=[x['raw_coskewness']for x in audit['event_history']],ranks=[x['ranks']for x in audit['event_history']],common_support=")
host = host.replace("summary.get('historical_names_absent_from_price_panel')==[],", "summary.get('historical_names_absent_from_price_panel')==[] and summary.get('market_proxy_present')is True,")
write('coskew162_host.py', host)
write('test_coskew162_host.py', rename((WORK/'test_rank161_host.py').read_text()))
write('coskew162_validate_code.py', rename((WORK/'rank161_validate_code.py').read_text()))
probe = rename((WORK/'rank161_source_probe.py').read_text())
probe = probe.replace("        'prior_residence': 'outputs/monthly_core_expansion/residence.py',\n        'prior_core_plan': 'outputs/monthly_core_expansion/PLAN.json',\n", '')
probe = probe.replace('historical_names_absent_from_price_panel=missing_names)', "historical_names_absent_from_price_panel=missing_names,market_proxy_present='SPY' in symbols)")
probe = probe.replace('Six-of-six support selects established continuously positive leaders; unavailable history must not alter original eligibility.', 'All211 finite positive source prices are required for stock andSPY; missing or degenerate residuals must not alter original eligibility.')
probe = probe.replace('Matched availability and maximum delta do not match actual delta distributions or identify a pure temporal-order causal effect.', 'Both directions have exactly opposite deltas; this is a fixed hypothesis contrast, not an identified causal effect or a literal replication of the cited excess-return study.')
probe = probe.replace('Prior160', 'Prior161')
write('coskew162_source_probe.py', probe)

proposal = {
    'stage':'prospective_design_before_real_features', 'round':162,
    'name':'formation_window_market_residual_coskewness',
    'hypothesis':'A small nonlinear market-exposure correction may distinguish otherwise similar long-momentum leaders. Negative coskewness tests the risk-premium direction; positive coskewness tests convexity/defensive quality. Both are fixed before feature or outcome inspection.',
    'arms':['incumbent_local67','coskew_negative','coskew_positive'],
    'history':'Exactly211 official-session closing prices from t-252 through t-42 inclusive;210 simple returns. No source observations from the most recent42sessions enter this new correction.',
    'formula':'Demean210 SPY returns m and stock returns y. beta=mean(y*m)/mean(m*m); e=y-beta*m; c=mean(e*m*m)/(sqrt(mean(e*e))*mean(m*m)). Raw-return proxy: no risk-free subtraction.',
    'missing':'Any missing/nonfinite/nonpositive source price, marketvariance<=1e-24, residualSD<=1e-12, or nonfinite result is unavailable. Never fill. Original eligibility is unchanged.',
    'rank':'Current original49 candidate pool; ascending average percentile of finite c minus0.5, before original eligibility filtering. Identical rules for stocks and ETFs.',
    'delta':'negative=-0.02*rank; positive=+0.02*rank. Unavailable cells and initial2016-08-17 exactly zero. Maximum absolute correction0.01; no new fitted prediction model.',
    'placement':'Original Local67 config4 pre-score plus one delta, then exactly one original episode adjustment. Preserve original27arrays, pool, eligibility and exit rules.',
    'selection':'Once on first88; same-close CAGR gain>=0.005, Sharpe tolerance0.05, DD tolerance0.03, early53/recent35 CAGR tolerance0.025. Costed next-open CAGR noninferior with same guards. At least3changed training heldsets and>=1eachera. Highest passing first88 CAGR; ties within1e-12 favor negative. Later34 diagnostic only.',
    'evaluation':'Three frozen streams; same-close0, next-open0, next-open0.05% eachside; full2016-08-17 through2026-09-24; originalEW3/EW2/cash, no leverage, fixed intramonth units. Independently verify all9paths and366orders.',
    'novelty_scope':'No explicit coskew implementation found in inspected localPython/PLAN files. Distinct mathematical target from linearbeta, own-stockskew and thresholdtailcooccurrence. Some earlier experiment source directories are absent locally; exhaustive mathematical novelty is NOT established.',
    'source_references':[{'url':'https://people.duke.edu/~charvey/Research/Published_Papers/P157_Conditional_skewness_in.pdf','authors':'Campbell R. Harvey and Akhtar Siddique','year':2023,'title':'Conditional Skewness in Asset Pricing:25Years of Out-of-Sample Evidence','use':'Motivates nonlinear market co-movement and distinguishes residual third moment from beta. Does not validate this210-day raw-return proxy, either dose, or this portfolio.'}],
    'source_limitations':[
        'Existing estimated historicalTop32 plus assumedtheme17, not certified all-US point-in-time membership or confirmed authorETFlist.',
        'Inherited later-retrieved historicalpricevintages; retrospective data and repeated development use cannot establish pristineout-of-sampleperformance.',
        'Same-close execution is idealized; next-open costed comparison is mandatory.',
        'A third-moment estimate can be noisy;210sessionwindow,variancefloors,rankdose anddirections are fixed, not tuned after outcomes.',
        'No riskfree subtraction; this is a researchproxy, not exact reproduction of the cited empiricalmeasure.',
        'Missing older source directories limit exhaustive noveltyclaims; all attempts remain counted.'
    ],
    'actual_features_computed':0,'actual_orders_computed':0,'actual_returns_computed':0,'observer_writes':False
}
write('coskew162_proposal.json', json.dumps(proposal,indent=2,sort_keys=True)+'\n')
print('Created new162host,hosttests,validator,sourceprobe,proposal; zero historical numerical work.')
