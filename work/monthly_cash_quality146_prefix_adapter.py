"""Recorded interface repair: pass only the requested pre-score prefix.

Original round146 PLAN, algorithm, sources and runner remain byte-identical.
Only preparation uses this adapter; full-horizon scores are unchanged.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from datetime import datetime, timezone
import monthly_cash_quality146_runner as original

ROOT=Path(__file__).resolve().parents[1]
PLAN_SHA="23807fe79e5cb9bd6a21273ec7ade26d8a6cb36aab18715e2e052fe1437e81f2"


def prefix_view(value, n):
    original.require(type(n) is int and 0<=n<=original.MONTHS, "Exact prefix range")
    value=np.asarray(value)
    original.require(value.shape==(original.MONTHS,original.WIDTH)
                     and value.dtype.kind=="f", "Original full pre-score contract")
    return value[:n].copy()


class PrefixBackend(original.Backend):
    def pre(self, arrays, n):
        return prefix_view(super().pre(arrays,n), n)


def run(amendment_path, amendment_sha):
    original.require(original.sha(amendment_path)==amendment_sha,"Changed execution amendment")
    amendment=original.read(amendment_path)
    original.require(amendment["plan_sha256"]==PLAN_SHA
        and amendment["repair"]=="full_pre_score_to_requested_prefix"
        and amendment["selection_or_factor_changes"] is False,
        "Exact interface-only repair")
    for path,digest in amendment["bindings"].items():
        original.require(original.sha(ROOT/path)==digest,"Changed amendment dependency")
    original.require(amendment["bindings"].get(str(Path(__file__).relative_to(ROOT)))==
                     original.sha(Path(__file__)), "Adapter not bound")
    runner=original.Runner(ROOT,ROOT/original.OUT/"PLAN.json",PLAN_SHA,
                           backend_factory=PrefixBackend)
    with (ROOT/original.OUT/"PREFIX_ADAPTER_STARTED.json").open("x") as f:
        json.dump(dict(plan_sha256=PLAN_SHA,amendment_sha256=amendment_sha,
                       started_utc=datetime.now(timezone.utc).isoformat()),f,indent=2)
        f.write("\n")
    runner.prepare()
    target=ROOT/original.OUT/"PREFIX_ADAPTER_EXECUTION.json"
    with target.open("x") as f:
        json.dump(dict(plan_sha256=PLAN_SHA,amendment_sha256=amendment_sha,
            cache_sha256=original.sha(ROOT/original.OUT/"CACHE.json"),
            full_horizon_pre_score_unchanged=True,selection_or_factor_changes=False),f,indent=2)
        f.write("\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amendment",type=Path,required=True)
    p.add_argument("--amendment-sha",required=True)
    a=p.parse_args()
    run(a.amendment,a.amendment_sha)
