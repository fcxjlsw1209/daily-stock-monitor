"""Post-freeze monthly rank-IC diagnostic; never train or change selection."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def spearman(x, y):
    """Average exact-tie ranks, no zero substitution for undefined IC."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Spearman arrays must be one-dimensional and aligned")
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    a = pd.Series(x).rank(method="average").to_numpy()
    b = pd.Series(y).rank(method="average").to_numpy()
    a -= a.mean()
    b -= b.mean()
    norm = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if norm == 0:
        return None
    return float(np.sum(a * b) / norm)


def summarize(rows):
    retained = [r for r in rows if r["status"] == "matched_ic"]
    out = dict(candidate_months=len(rows), matched_months=len(retained),
               excluded_months=len(rows) - len(retained))
    for arm in ["return", "ordinal"]:
        values = np.array([r[arm + "_ic"] for r in retained], dtype=float)
        out[arm] = dict(count=len(values), mean_ic=float(values.mean()) if len(values) else None,
                       positive_ic_fraction=float(np.mean(values > 0)) if len(values) else None)
    diff = np.array([r["ordinal_minus_return_ic"] for r in retained], dtype=float)
    out["paired_ordinal_minus_return"] = dict(
        count=len(diff), mean_ic_difference=float(diff.mean()) if len(diff) else None,
        positive_difference_fraction=float(np.mean(diff > 0)) if len(diff) else None,
        identical_ic_months=int(np.sum(diff == 0)))
    return out


def main():
    # All models and model-selection choices already exist. Repeated evaluation
    # has made these historical periods unsuitable for a pristine-OOS claim.
    names = ["FROZEN.json", "FROZEN.sha256", "FORECASTS.npz", "DATA.npz",
             "META.json", "ordinal_model.py", "INPUT_HASHES.json", "FORECAST_INPUT_HASHES.json"]
    input_hashes = {str(P / name): sha(P / name) for name in names}
    frozen = json.loads((P / "FROZEN.json").read_text())
    assert sha(P / "FROZEN.json") == (P / "FROZEN.sha256").read_text().strip()
    for name, value in frozen["hashes"].items():
        assert sha(P / name) == value, name
    inherited = json.loads((P / "INPUT_HASHES.json").read_text())
    for name, value in inherited.items():
        assert sha(name) == value, name
    for name, value in json.loads((P / "FORECAST_INPUT_HASHES.json").read_text()).items():
        assert sha(P / name) == value, name

    m = json.loads((P / "META.json").read_text())["monthly"]
    a, forecasts = dict(np.load(P / "DATA.npz")), dict(np.load(P / "FORECASTS.npz"))
    n = len(m["dates"])
    assert n == 122 and a["eligible"].shape == (122, 49)
    assert all(forecasts[k].shape == a["eligible"].shape for k in ["return", "ordinal", "available"])
    assert forecasts["available"].dtype.kind == "b"
    assert not (forecasts["available"] & ~a["eligible"]).any()
    for arm in ["return", "ordinal"]:
        assert np.isfinite(forecasts[arm]).all()

    # This is the exact loader called by arch.old.panel_load, without importing
    # unrelated historical optimizer modules (and their optional SciPy stack).
    loader_path = P.parent / "monthly_stability_monitor/inputs.py"
    assert inherited[str(loader_path)] == sha(loader_path)
    inputs = load_module("_ordinal_diagnostic_inputs", loader_path)
    model = load_module("_ordinal_diagnostic_model", P / "ordinal_model.py")
    panel, panel_info = inputs.load_panel(m["etfs"])
    for name, value in panel_info["source_hashes"].items():
        assert inherited[name] == value, name
    event_indices = list(range(1, n - 1))
    dates = np.concatenate([np.repeat(np.datetime64(m["dates"][i], "D"), len(m["symbols"][i]))
                            for i in event_indices])
    ends = np.concatenate([np.repeat(np.datetime64(m["ends"][i], "D"), len(m["symbols"][i]))
                           for i in event_indices])
    symbols = [s for i in event_indices for s in m["symbols"][i]]
    # Build on the original, complete past-eligible universe, not the subset
    # whose future path or model forecast happens to be available.
    targets, target_audit = model.build_targets(
        panel["adj"], dates, ends, symbols, a["eligible"][event_indices].reshape(-1))
    ordinal_targets = targets["ordinal"].reshape(len(event_indices), 49)
    rows = []
    for j, i in enumerate(event_indices):
        original = a["eligible"][i]
        target = ordinal_targets[j]
        matched = original & forecasts["available"][i]
        record = dict(event_index=i, signal_date=m["dates"][i], label_end=m["ends"][i],
                      split="first88" if i < 88 else "later34",
                      eligible_count=int(original.sum()), matched_forecast_count=int(matched.sum()),
                      status="incomplete_label_month")
        if original.any() and np.isfinite(target[original]).all():
            ic = {arm: spearman(forecasts[arm][i][matched], target[matched])
                  for arm in ["return", "ordinal"]}
            if all(value is not None for value in ic.values()):
                record.update(status="matched_ic", return_ic=ic["return"], ordinal_ic=ic["ordinal"],
                              ordinal_minus_return_ic=ic["ordinal"] - ic["return"])
            else:
                record.update(status="insufficient_or_constant_matched_cross_section",
                              undefined_arms=[arm for arm, value in ic.items() if value is None])
        rows.append(record)

    # Refuse to persist a result if another process altered any frozen input.
    for name, value in {**input_hashes, **inherited}.items():
        assert sha(name) == value, name
    output = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        purpose="Post-freeze descriptive monthly Spearman IC; no training, grid change or model selection",
        definition="Cross-sectional Spearman correlation of each existing forecast with unclipped next-calendar-month return ranks; exact ties use average ranks",
        matched_sample="Original past-eligible universe supplies full-valid-month labels; both arms use identical available rows and both must have defined IC to retain the month",
        month_weighting="Each retained monthly IC has equal weight in summaries",
        date_cuts=dict(first88="event_index < 88; initial event_index 0 excluded",
                       later34="event_index >= 88; final event_index 121 excluded"),
        excluded_partial_periods=[dict(event_index=i, signal_date=m["dates"][i], label_end=m["ends"][i])
                                  for i in [0, n - 1]],
        pristine_out_of_sample=False, inferential_pvalues_computed=False,
        summaries={split: summarize([r for r in rows if r["split"] == split])
                   for split in ["first88", "later34"]},
        target_audit=target_audit, observations=rows, input_hashes=input_hashes,
        diagnostic_code_sha256=sha(Path(__file__)),
        protected_inputs_unchanged=True,
    )
    (P / "MODEL_DIAGNOSTICS.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
