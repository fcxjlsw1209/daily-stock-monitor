"""Write-once limited source-definition review; run only via run_local.sh."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/after220_tail_moment_coverage.json'
PINS = {}

def bind(name):
    path = ROOT / name
    if not path.is_relative_to(ROOT) or 'CloudStorage' in str(path):
        raise ValueError('Nonlocal source')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    result = digest.hexdigest()
    PINS[str(path)] = result
    return dict(path=name, sha256=result)

def evidence(name, line=None):
    out = bind(name)
    if line is not None:
        out['line'] = line
    return out

def definition(name, formula, distinction, sources):
    return dict(name=name, exact_definition=formula, distinction=distinction,
                evidence=[evidence(path, line) for path, line in sources])

if OUT.exists():
    raise FileExistsError('Write-once coverage already exists')
confirmed = [
    definition('Original current63 downside-RMS core alternatives',
        'long=P[t-42]/P[t-294]-1; downside63=sqrt(mean(min(r,0)^2)) over all current63 simple returns. core_downside=long/downside63. Original code also uses aligned252 sample volatility ending t-42. core alternatives half/aligned/downside are blended at10% and25%.',
        'Generic downside normalization and changed risk windows are already tested. It is not a matched same252 formation-window third-versus-second lower-partial-moment test.',
        [('outputs/monthly_industry_refinement/refine.py',27),('outputs/monthly_2016_extension/extend.py',98)]),
    definition('path70 worst segment downside',
        '211 prices t-252..t-42 provide210 returns in three70-return segments. g_j is the sum of log returns in segment j; D_j=sqrt(mean(min(log_return,0)^2)) over all70 days. worst_segment_downside=sum(g_j)/max(D_j). The companion centered_mad_momentum divides endpoint simple momentum by mean absolute deviation of210 simple returns around their mean.',
        'Different210-return period, log-return numerator, segment aggregation and auxiliary insertion; only second lower partial moments.',
        [('outputs/monthly_path70_research/features.py',43)]),
    definition('179 downside share change',
        'Full support is295 prices t-294..t. Old252 returns end t-42; current63 returns end t. In each window downside_share=sum(r^2 for r<0)/sum(r^2 over all days), with a strict1e-24 total-squares floor. Candidate=old252_share-current63_share; control=0.5-current63_share.',
        'A contrast of downside shares, not cumulative momentum divided by downside-risk magnitude. No third-order own-loss moment.',
        [('work/monthly_downside_mix179_kernel.py',1),('work/monthly_downside_mix179_context.json',None)]),
    definition('162 residual coskewness',
        '211 prices t-252..t-42 give210 simple returns. Demean market m and stock y; beta=sum(m*y)/sum(m^2); e=y-beta*m. coskew=mean(e*m^2)/(sqrt(mean(e^2))*mean(m^2)), with market/residual variance guards. Negative and positive coskew arms.',
        'This is a signed joint market third moment, not mean(max(-r,0)^3), and is not own-stock downside-risk normalization.',
        [('work/coskew162_kernel.py',10),('work/coskew162_kernel.py',90),('work/coskew162_proposal.json',None)]),
    definition('219 momentum and same-window Sharpe consensus',
        '253 prices t-294..t-42 give252 simple returns. M=P[-1]/P[0]-1. mean=fsum(r)/252; sample_vol=sqrt(fsum((r-mean)*(r-mean))/251); Sharpe=mean/sample_vol. Compare minimum versus arithmetic mean of centered percentile ranks of M and Sharpe.',
        'Same-window total-risk normalization and rank consensus already tested, but not loss-only p3 versus p2 risk with fixed cumulative numerator.',
        [('work/monthly_risk_consensus219_kernel.py',45),('work/monthly_risk_consensus219_context.json',None)]),
    definition('Broad volatility and risk-core alternatives',
        'stability.features defines252/294 endpoint momentum with current63, current126, aligned formation252 ending t-42 and square-root-current63 risk denominators, plus path efficiency/raw-risk mixtures.',
        'A new lookback grid or another generic risk-adjusted core is not a new family. Exact lower-partial-order comparison is the limited proposed distinction.',
        [('outputs/monthly_stability_research/stability.py',48)]),
    definition('Market beta and residual mechanisms relevant to deferred alternative',
        'Core-expansion uses lagged daily market beta from preceding252 returns and mean/sample-SD of210 residual returns. Beta-prior report/tests describe preceding126 joint market/sector coefficients and252 residual formation returns, beta shrinkage, OLS and sector-prior variants. Beta-asymmetry uses separate up/down-market betas over252 log returns and their difference.',
        'A denominator combining OLS residual variance with downside-market beta risk is not identical to those residual numerators or beta-spread features, but substantial family overlap and missing old source definitions remain.',
        [('outputs/monthly_core_expansion/core_features.py',1),('outputs/monthly_beta_prior70/report.py',14),('outputs/monthly_beta_prior70/test_features.py',1),('outputs/monthly_beta_asymmetry70/features.py',1)])
]
missing=[]
for directory in ('monthly_ewvol70','monthly_upside_trim70','monthly_tail_core70','monthly_downrisk_horizon70','monthly_robust_slope70','monthly_slope_risk70','monthly_huber70','monthly_huber_beta70_v2'):
    states=[]
    for name in ('PLAN.json','features.py','study.py'):
        relative='outputs/'+directory+'/'+name
        path=ROOT/relative
        states.append(dict(path=relative,exists=path.is_file(),sha256=bind(relative)['sha256'] if path.is_file() else None))
    missing.append(dict(experiment=directory,status='UNKNOWN_EXACT_DEFINITION',checked_members=states))
background=[evidence('work/after200_median_risk_coverage.json'),evidence('work/after150_downside_target_archive_followup.json')]
selected=dict(name='cumulative momentum divided by third versus second lower partial risk',
    status='SELECTED_PROSPECTIVE_DESIGN_NOT_REGISTERED_BY_THIS_REVIEW',
    source='Existing original adjusted-price lineage; no new fields or vendors.',
    support='Exactly253 prices t-294..t-42 and252 simple returns in the same formation window; latest42 sessions excluded from both momentum and risk.',
    numerator='M=P[t-42]/P[t-294]-1',
    candidate='M / (sum(max(-r_j,0)^3)/252)^(1/3)',
    matched_control='M / sqrt(sum(max(-r_j,0)^2)/252)',
    denominator_convention='All252 sessions are included in each moment; nonnegative-return days contribute zero. No centering, negative-days-only average or riskfree subtraction.',
    mechanism='Holding cumulative momentum, source support and risk clock fixed, the third-order norm responds more strongly to concentrated large negative returns than the second-order norm. This may distinguish gradual downside variation from occasional large losses among similarly strong trends.',
    sign_caveat='D3>=D2; candidate<=control only for nonnegative M. The inequality reverses for negative M. This is a mathematical definition, not an empirical finding.',
    boundary='With no strictly negative return both denominators are zero and both arms must be unavailable; zero M with positive downside risk is a valid zero. No inferred floor, fill or adjusted eligibility is introduced by this review.',
    design_history='An initial discussion used mean(r) as numerator. Before any new numeric computation root fixed the numerator to original cumulative M to isolate the denominator. No mean-numerator trial was run.',
    integration='Final host dose, rank domain, common support rules and incumbent insertion are reserved to the final221 context. This metadata review does not choose numerical weights.',
    falsifiability='The prospective matched replay can fail to improve the declared training/validation criteria or can produce identical holdings. No factor or return result was consulted to choose the moment order.')
deferred=dict(name='market-down-beta risk decomposition',status='DEFERRED_NOT_SELECTED_NOT_REGISTERED',
    candidate='M / sqrt(full-OLS residual variance + beta_down^2 * market variance)',
    matched_control='Same M and source window, replacing beta_down by full-window beta_all.',
    distinction='Separates systematic bad-market exposure in the risk denominator while retaining the same cumulative trend; unlike residual momentum it does not subtract market returns from the numerator.',
    limitations='Requires descriptive regressions and sufficient down-market observations, has more overlap with prior beta families and less complete available-source coverage. No exact support thresholds, blend or implementation have been selected; no computation performed.')
helper=evidence(str(Path(__file__).relative_to(ROOT)))
report=dict(schema='after220-tail-moment-limited-coverage-v1',created_utc=datetime.now(timezone.utc).isoformat(),
    status='SOURCE_DEFINITION_METADATA_ONLY_NOT_REGISTRATION',
    scope='Bounded reading of local Python, PLAN/context and prior metadata coverage records only. No outcomes selected and no numeric data arrays decoded.',
    search_scope=dict(paths=['work/monthly*.py','work/*context*.json','work/after*.json','outputs/monthly*/*.py','outputs/monthly*/PLAN.json'],
        concepts=['Kappa3','third lower partial moment','downside cube','third-order partial moment','p3/p2 risk denominator'],
        conclusion='No exact cumulative-M/same252 D3-versus-D2 matched implementation was identified in the accessible definitions. This is limited negative evidence; unavailable old definitions prevent an archive-wide never-tested claim.'),
    family_novelty=False,confirmed_overlap=confirmed,coverage_unknown=missing,prior_coverage=background,
    mechanism_comparison=[selected,deferred],selected_mechanism=selected['name'],
    implementation=helper,input_sha256=dict(sorted(PINS.items())),
    assumptions=['Available local historical source definitions are not the full original archive.', 'Third-order loss emphasis is not a new generic downside-risk family.', 'This review does not certify price coverage, membership PIT or actual common feature support.'],
    new_features_computed=0,new_orders_computed=0,new_returns_computed=0,new_predictive_models_fit=0,new_trials_registered=0,
    actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,
    npz_decoded=False,network_requests=0,central_modified=False,observers_modified=False)
with OUT.open('x') as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),pinned_files=len(PINS),source_only=True)))
