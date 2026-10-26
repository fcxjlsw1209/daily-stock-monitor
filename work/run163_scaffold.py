"""Create separate round163 lifecycle files; no historical numerical work."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'

def renamed(text):
    text = text.replace('coskew162', 'run163')
    text = re.sub(r'\b(161|162|998|1001)\b', lambda m: {
        '161':'162', '162':'163', '998':'1001', '1001':'1004'}[m[0]], text)
    text = text.replace('PRE162_', 'PRE163_')
    text = text.replace('round162', 'round163').replace('162rounds', '163rounds')
    text = text.replace('1001trials', '1004trials').replace('998trials', '1001trials')
    text = text.replace('Expected161', 'Expected162').replace('Prior161', 'Prior162')
    text = text.replace('coskewness', 'terminal-run').replace('Coskewness', 'Terminal-run')
    text = text.replace('raw_coskewness', 'raw_run').replace('raw_terminal-run', 'raw_run')
    text = text.replace('fixed210-return', 'fixed ten-block')
    return text

def save(name, value):
    with (WORK/name).open('x') as f:
        f.write(value)

for suffix in ('host.py', 'validate_code.py', 'source_probe.py'):
    value = renamed((WORK/('coskew162_'+suffix)).read_text())
    if suffix == 'source_probe.py':
        value = value.replace('All211 finite positive source prices are required for stock andSPY; missing or degenerate residuals must not alter original eligibility.',
            'All211 finite positive source prices are required for each security; missing observations must not alter original eligibility.')
        value = value.replace('Both directions have exactly opposite deltas; this is a fixed hypothesis contrast, not an identified causal effect or a literal replication of the cited excess-return study.',
            'Both directions have exactly opposite deltas; this is a fixed continuation-versus-exhaustion hypothesis contrast, not an identified causal effect.')
    save('run163_'+suffix, value)
tests = renamed((WORK/'test_coskew162_host.py').read_text())
tests = tests.replace('coskew_positive','run_positive').replace('coskew_negative','run_negative')
# 161 six-trial rows + one35-trial row = 162rounds/1001trials.
tests = tests.replace('range(160)', 'range(161)').replace("[{'trials':38}]", "[{'trials':35}]")
save('test_run163_host.py', tests)

proposal = {
    'stage':'prospective_design_before_real_features','round':163,
    'name':'skip42_signed_terminal_block_run',
    'hypothesis':'Within established long-momentum candidates, the terminal sequence of monthly-scale return signs may distinguish continued directional leadership from an exhausted run. Both opposite fixed adjustments are preregistered.',
    'arms':['incumbent_local67','run_negative','run_positive'],
    'history':'Exactly211 official-session closing prices from t-252 through t-42 inclusive, forming ten disjoint21-session blocks. The most recent42sessions never enter the new feature.',
    'formula':'For k=1..10, s[k]=sign(P[t-252+21*k]-P[t-252+21*(k-1)]), equivalent to the log-return sign. L is the number of consecutive terminal blocks with the same nonzero sign as s[10]; if s[10]=0 then L=0. raw=s[10]*L/10.',
    'missing':'Any of the211 source prices missing, nonfinite or nonpositive makes the cell unavailable. Allzero returns are available with raw0. No filling, market proxy, fitted model or outcome labels.',
    'rank':'Current original49 candidate pool; ascending average percentile of finite raw values minus0.5, before original eligibility filtering. Identical policy for stocks and ETFs.',
    'delta':'negative=-0.02*rank; positive=+0.02*rank. Unavailable cells and initial2016-08-17 exactly zero. Maximum absolute correction0.01.',
    'placement':'Original Local67 config4 pre-score plus one delta, then exactly one original episode adjustment. Preserve original27arrays, pool, eligibility and exit rules.',
    'selection':'Once on first88; same-close CAGR gain>=0.005, Sharpe tolerance0.05, DD tolerance0.03, early53/recent35 CAGR tolerance0.025. Costed next-open CAGR noninferior with same guards. At least3changed training heldsets and>=1eachera. Highest passing first88 CAGR; ties within1e-12 favor negative. Later34 diagnostic only.',
    'evaluation':'Three frozen streams; same-close0, next-open0, next-open0.05% eachside; full2016-08-17 through2026-09-24; originalEW3/EW2/cash, no leverage, fixed intramonth units. Independently verify all9paths and366orders.',
    'novelty_scope':'New terminal signed-run descriptor proposed for local audit against prior frequency, serial correlation, rank-residence and path descriptors. No claim of exhaustive literature novelty.',
    'source_limitations':[
        'Existing estimated historicalTop32 plus assumedtheme17, not certified all-US point-in-time membership or confirmed authorETFlist.',
        'Inherited later-retrieved historicalpricevintages and repeated development use cannot establish pristineout-of-sampleperformance.',
        'Same-close execution is idealized; next-open costed comparison is mandatory.',
        'Ten coarse return signs discard amplitude and within-block path. Window, blocksize, rankdose and directions are fixed, not tuned after outcomes.',
        'Some earlier source directories are absent locally, limiting exhaustive mathematical novelty claims; all attempts remain counted.'
    ],
    'actual_features_computed':0,'actual_orders_computed':0,'actual_returns_computed':0,'observer_writes':False
}
save('run163_proposal.json',json.dumps(proposal,indent=2,sort_keys=True)+'\n')
print('Created new163 host/tests/validator/probe/proposal; no historical numerical work.')
