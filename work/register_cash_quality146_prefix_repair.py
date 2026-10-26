"""Record failed preparation and freeze a non-economic prefix interface repair."""
from pathlib import Path
from datetime import datetime, timezone
import json
import monthly_cash_quality146_lifecycle as life

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/monthly_cash_quality146"


def main():
    review_path=ROOT/"work/monthly_cash_quality146_prefix_adapter_review.json"
    review=life.r.read(review_path)
    life.require(review.get("passed") is True,"Independent repair review required")
    log=(OUT/"PREFIX_ADAPTER_TESTS.log").read_text()
    life.require("Ran 4 tests in " in log and log.rstrip().endswith("OK"),"Repair tests failed")
    failed=(OUT/"PREPARE.log").read_text()
    life.require("ValueError: Original pre-score shape" in failed,"Exact failure not recorded")
    life.require(not (OUT/"CACHE.json").exists() and not (OUT/"RESULTS.json").exists(),
                 "Repair must precede completed preparation and return computation")
    paths=["work/monthly_cash_quality146_prefix_adapter.py",
           "work/test_monthly_cash_quality146_prefix_adapter.py",
           "work/monthly_cash_quality146_prefix_adapter_review.json",
           "work/register_cash_quality146_prefix_repair.py",
           "outputs/monthly_cash_quality146/PREFIX_ADAPTER_TESTS.log",
           "outputs/monthly_cash_quality146/PREPARE.log"]
    for path,digest in review["input_hashes"].items():
        life.require(life.r.sha(ROOT/path)==digest,"Changed reviewed repair dependency")
    bindings={p:life.r.sha(ROOT/p) for p in paths}
    bindings.update(review["input_hashes"])
    doc=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=life.r.sha(OUT/"PLAN.json"),
        repair="full_pre_score_to_requested_prefix",
        selection_or_factor_changes=False,bindings=bindings,
        failure=dict(phase="prepare",exit_code=1,cache_written=False,
            real_factor_computation_started=True,ephemeral_orders_computation_started=True,
            returns_computed=False,reason="Old backend returns full122 pre-score; caller expects n rows."),
        effect="Only PrefixBackend.pre crops original returned array to[:n]. "
               "At n=122 bytes are unchanged. Original choices, features and all original hashes remain fixed.",
        subsequent_checks="Original independent oracle must verify all123 prefixes and all488 orders; "
                          "original outcome verifier must independently recompute all12 accounting paths.",
        registration_count_change=0)
    amendment=OUT/"PREFIX_EXECUTION_AMENDMENT.json"
    life.write(amendment,doc)
    with life.state_lock(ROOT):
        state,digest=life.state_snapshot(ROOT/life.STATE)
        life.predecessor(state)
        life.require(state["current"]==life.STUDY and len(state["registered_queue"])==1,
                     "Active registered146 required")
        ref=dict(path=str(amendment.relative_to(ROOT)),sha256=life.r.sha(amendment))
        state.setdefault("failed_numerical_attempts",[]).append(dict(
            round=146,phase="prepare",failure=doc["failure"],repair=ref))
        state["current_status"]="registered_prefix_interface_repair_before_reexecution"
        state["updated_utc"]=doc["created_utc"]
        life.update_state(ROOT/life.STATE,digest,state)
    print(json.dumps(dict(amendment_sha256=life.r.sha(amendment))))


if __name__=="__main__":
    main()
