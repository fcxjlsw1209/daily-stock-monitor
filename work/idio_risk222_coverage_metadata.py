"""Write-once finite definition coverage; metadata only, via run_local.sh."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/after221_idio_risk_coverage.json"
PINS = {}

def bind(name):
    p = ROOT / name
    if not p.resolve().is_relative_to(ROOT) or "CloudStorage" in str(p):
        raise ValueError("Nonlocal source")
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    value = h.hexdigest()
    PINS[str(p)] = value
    return {"path": name, "sha256": value}

def evidence(name, line=None):
    item = bind(name)
    if line is not None:
        item["line"] = line
    return item

def definition(name, formula, distinction, sources):
    return dict(name=name, exact_definition=formula, distinction=distinction,
                evidence=[evidence(path, line) for path, line in sources])

if OUT.exists():
    raise FileExistsError("Write-once coverage exists")
confirmed = [
    definition("Same-window current63 idiosyncratic-risk descriptor already exists",
        "monthly_factor_search_v2.features uses 63 simple returns through signal t. beta63=cov63(stock,SPY)/sample_var63(SPY). idio63=sqrt(clip(sample_var63(stock)-beta63**2*sample_var63(SPY), lower=0)). Its risk descriptor equals sqrt(RSS/62) in exact real arithmetic for an intercept-including, centered single-factor OLS fit on the same63 observations, when all variances are defined. Positive and negative idio63 are side factors; recipes combine a separate core with one or two normalized side factors. adaptive_core risk_path carries signed idio63 as a volatility-side feature; open_inverse carries idio63 as a signed risk feature in inverse-ranking fits.",
        "This is strong existing overlap: the proposed current63 residual-risk measurement is not a new factor family. Those inspected scripts do not define cumulative skip42 momentum divided by idio63 as a core. The proposed narrow comparison changes integration from an additive side factor to a fixed cumulative numerator divided by residual versus total risk, with matched support. Old subtractive variance arithmetic clips negatives; proposed explicit RSS has no clipping or epsilon.",
        [("outputs/monthly_ordered_listwise/experiment.py",15),("outputs/monthly_strict_relative/strict_study.py",17),("outputs/monthly_conditional_retention/policy_study.py",18),("outputs/monthly_style_inverse/style_study.py",18),("outputs/monthly_factor_search_v2/search.py",23),("outputs/monthly_factor_search_v2/search.py",36),("outputs/monthly_factor_search_v2/search.py",116),("outputs/monthly_adaptive_core/adaptive.py",34),("outputs/monthly_open_hypotheses/open_inverse.py",37)]),
    definition("Positive high residual126-volatility auxiliary",
        "industry_refinement uses rolling126 sample covariance/variance beta shifted one session, e_u=r_i,u-beta_u*r_SPY,u without subtracting an intercept, followed by current126 sample SD of e. residual126 replaces 25% or50% of the original positively ranked vol126 support via .219375*vol_blend*(rank(residual126)-rank(vol126)). Core-risk alternatives there are sqrt-current63, aligned252 ending t-42, and current63 downside RMS.",
        "Daily beta varies across the126 residual returns and is estimated on prior126 days. Residual volatility is preferred as a positive auxiliary. New222 instead estimates one centered coefficient on current63 returns and divides the unchanged cumulative M by its residual dispersion; it does not positively add high residual volatility or use the residual drift as numerator.",
        [("outputs/monthly_industry_refinement/refine.py",27),("outputs/monthly_industry_refinement/refine.py",43),("outputs/monthly_industry_refinement/refine.py",62),("outputs/monthly_2016_extension/extend.py",98)]),
    definition("Long residual mean/sample-SD core",
        "core_expansion market_residual_sharpe uses210 simple formation returns ending t-42, from211 prices t-252..t-42. Each return u has a beta estimated on prior252 returns u-252..u-1. Residual is r_i,u-beta_i,u*r_SPY,u; no fitted intercept is subtracted. Raw score is mean(residual)/sampleSD(residual). The ordinary matched control is mean(original returns)/sampleSD(original returns). Standard-deviation tolerance1e-12 and residual-zero tolerance1e-12 produce documented zero or fallback paths, including SPY self-exposure.",
        "Both numerator and denominator are changed to residual returns. Formation duration, risk clock, lagged coefficient estimation and zero/fallback handling differ. New222 retains original252-session cumulative M and only replaces current63 total-risk denominator; it uses common missing support at exact-zero residual risk, including SPY self.",
        [("outputs/monthly_core_expansion/core_features.py",15),("outputs/monthly_core_expansion/core_features.py",35),("outputs/monthly_core_expansion/core_features.py",73),("outputs/monthly_core_expansion/study.py",43)]),
    definition("Prior-day market plus sector residual-momentum support",
        "quality_research.residual_returns estimates market-only beta or a joint market/own-sector two-factor beta using rolling126 or252 covariance moments, all shifted one session before applying daily returns. The joint fit uses det>var_market*var_sector*1e-8 and falls back to market-only where joint data/mapping fail. It does not remove fitted drift. Features are sums of210 or252 residual returns ending t-42, optionally divided by sampleSD of those residuals; ETF features are masked. These are 2.5/5/10% support-only corrections under global/calm/highvol policies.",
        "Old residual momentum removes systematic-return exposure from the numerator, uses old formation-period residual risk and sector mappings. New222 has no sector fit, no residual numerator, no predictive return labels, and replaces half the noncalm core on a common current63 support.",
        [("outputs/monthly_quality_research/explore.py",27),("outputs/monthly_quality_research/explore.py",52),("outputs/monthly_quality_research/explore.py",100)]),
    definition("Joint-beta prior and shrinkage estimator comparison",
        "Available beta_prior70 report and synthetic tests define379 prices t-420..t-42:126 warmup plus252 residual-formation returns. Each residual uses a joint stock-on-SPY/sector fit over prior126 returns with intercept; fitted intercept is retained in residual drift. Three coefficient paths are OLS, sector-only beta=(0,1), and noise-dependent shrinkage toward that prior with prior covariance .25I. Raw score is sum252(residual)/sampleSD252(residual), positive-SD threshold1e-12 and common support. ETFs and unknown sector maps are unavailable; score replaces2.5% or5% of support. Primary features.py is unavailable locally, so exact implementation completeness remains UNKNOWN despite explicit report/test formula.",
        "Coefficient regularization and residual-momentum numerator differ from fixed single-factor current63 denominator-only normalization. This is not evidence that the general residual-risk idea is untested.",
        [("outputs/monthly_beta_prior70/report.py",14),("outputs/monthly_beta_prior70/test_features.py",1)]),
    definition("Market up/down-beta asymmetry",
        "beta_asymmetry70 uses253 prices t-294..t-42 and252 log returns. It computes separate intercept-including slopes on positive versus negative SPY-return days, each requiring at least40 days, excluding zero-market returns; feature is beta_up-beta_down.",
        "Conditional market exposure spread is an auxiliary, not a full-window residual-dispersion denominator or cumulative-return risk ratio.",
        [("outputs/monthly_beta_asymmetry70/features.py",1)]),
    definition("Original total-risk denominator and same-window risk alternatives",
        "stability.features defines M=P[t-42]/P[t-294]-1 divided by current63 or126 sample SD, aligned252 SD ending t-42, or sqrt(current63 SD). Its prepare verifies .52*rank(294_v63)==original core. Calm uses a separate raw-momentum branch. Round219 subsequently compares rank-consensus of M and mean252/simple-return sampleSD252 on the same excluded42-day formation window.",
        "Total-risk normalization and matching risk horizons are already covered. Proposed222 total-volatility control uses the same current63 SD formula on newly matched residual-valid support; its ranks can differ from incumbent where the shared mask excludes degenerate assets, so it must not be declared byte-identical to incumbent before verification.",
        [("outputs/monthly_stability_research/stability.py",48),("outputs/monthly_stability_research/stability.py",82),("work/monthly_risk_consensus219_kernel.py",45),("work/monthly_risk_consensus219_context.json",None)])
]
missing = []
for directory in ("monthly_huber_beta70_v2", "monthly_huber70", "monthly_ewvol70", "monthly_slope_risk70", "monthly_sector_residual70"):
    members = []
    for name in ("PLAN.json", "features.py", "study.py"):
        relative = "outputs/" + directory + "/" + name
        exists = (ROOT / relative).is_file()
        members.append(dict(path=relative, exists=exists, sha256=bind(relative)["sha256"] if exists else None))
    missing.append(dict(experiment=directory, status="UNKNOWN_EXACT_DEFINITION", checked_members=members))
relative = "outputs/monthly_beta_prior70/features.py"
missing.append(dict(experiment="monthly_beta_prior70", status="PRIMARY_FEATURE_IMPLEMENTATION_UNAVAILABLE_REPORT_AND_TEST_DEFINE_FORMULA", checked_members=[dict(path=relative, exists=(ROOT/relative).is_file(), sha256=bind(relative)["sha256"] if (ROOT/relative).is_file() else None)]))
selected = dict(name="cumulative long momentum divided by current idiosyncratic versus total risk",
    status="ROOT_FIXED_PROSPECTIVE_DESIGN_NOT_REGISTERED_BY_THIS_METADATA_REVIEW",
    numerator="M=P[t-42]/P[t-294]-1;253 positive finite formation prices,252 intervals; no market component is removed from M.",
    risk_clock="Current64 prices t-63..t provide63 simple returns for each security and SPY. No return after signal close is used. Current-risk information does not change the skip42 price-momentum numerator and is consistent with the inherited current63-risk clock.",
    regression="Center stock and SPY returns over exactly63 days. beta=sum((x-xbar)*(y-ybar))/sum((x-xbar)^2). Residual e=(y-ybar)-beta*(x-xbar). One single-factor descriptive fit per valid security-window; no future-return targets, predictive Ridge/tree fit or parameter search.",
    candidate="M/sqrt(sum(e^2)/62)", matched_control="M/sqrt(sum((y-ybar)^2)/62)",
    denominator_convention="Use62 for both arms to compare sample dispersion on63 observations. This is intentionally not an unbiased regression-noise estimator with61 residual degrees of freedom. No beta shrinkage or alpha subtraction from M.",
    common_support="Missing/invalid source, exactlyzero marketSS, total-risk zero, or residual-risk zero makes both arms unavailable. No epsilon, clipping, inferred fill or fallback-to-total-risk candidate. SPY self-input degenerates. Final floating-point and overflow behavior belongs to the final prospective context/kernel tests.",
    support_caveat="The matched control is the original M/current63sampleSD formula only on shared support. Removing SPY or another degenerate asset from both percentile rank domains can affect all ranks, so distinguish paired-denominator effect from the shared-mask change relative to original Local67.",
    integration="Root-fixed names incumbent_local67/idiosyncratic_core/total_volatility_control. Both paired arms replace half of noncalm original core; keep calm and initial partial event unchanged, original pool/eligibility/auxiliary budgets/EW3/monthly fixed shares and one episode. Final executable details await context.",
    mechanism="Among positive long trends with equal total risk, a smaller stock-specific residual component produces a higher candidate ratio when common market variation explains more total movement. This tests whether treating shared market variation differently from security-specific noise helps boundary ranking. It is not a claim that the numerator is stock-specific alpha or that market risk disappears from the portfolio.",
    sign_caveat="Residual dispersion cannot exceed total centered dispersion in exact OLS arithmetic, but lower denominator magnifies negative M downward rather than improving it. No negative orzero M filter is added by this metadata review.",
    novelty="Narrow matched integration test inside an already explored idiosyncratic/residual-risk family. The exact current63 residual-risk descriptor was already implemented as idio63. No exact M/idio63 versus M/total63 half-core comparison was located in this bounded readable set; missing historical implementations prohibit an archive-wide novelty claim.")
implementation = evidence(str(Path(__file__).relative_to(ROOT)))
report = dict(schema="after221-idio-risk-limited-coverage-v1", created_utc=datetime.now(timezone.utc).isoformat(),
    status="SOURCE_DEFINITION_METADATA_ONLY_NOT_REGISTRATION", family_novelty=False,
    scope="Read-only local Python and PLAN/context definitions plus streaming SHA. No raw price rows, array decoding, candidate rankings, author holdings, later-month diagnostics or performance-driven parameter selection.",
    search_scope=dict(paths=["outputs/monthly*/*.py", "outputs/monthly*/PLAN.json", "work/*context*.json"], concepts=["idio63", "residual risk", "residual volatility", "market beta risk denominator", "residual momentum numerator"], exhaustive=False),
    confirmed_overlap=confirmed, coverage_unknown=missing, selected_mechanism=selected,
    implementation=implementation, input_sha256=dict(sorted(PINS.items())),
    source_admission_status="PENDING_FINAL_CONTEXT_AND_INDEPENDENT_RAW64_METADATA_REVIEW", actual_source_coverage_recomputed=False,
    new_features_computed=0, new_orders_computed=0, new_returns_computed=0, new_predictive_models_fit=0, new_trials_registered=0,
    actual_new_features_computed=False, actual_new_orders_computed=False, actual_new_returns_computed=False,
    npz_decoded=False, network_requests=0, central_modified=False, observers_modified=False, strict_pit=False)
with OUT.open("x") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(dict(path=str(OUT), sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(), pinned_files=len(PINS), source_only=True)))
