"""Publish only independently verified, formally completed round153 results."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_financial_coverage153"
NAMES = ("incumbent_local67", "original_earnings", "batch1_earnings",
         "batch2_earnings", "batch3_earnings", "expanded_earnings")
LABELS = dict(zip(NAMES, ("原最佳", "原四家公司修正", "六家公司修正",
                          "八家公司修正", "十家公司修正", "十二家公司修正")))


def read(path):
    return json.loads(path.read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def ref(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def check(ok, message):
    if not ok:
        raise ValueError(message)


def preserve(path, raw):
    if path.exists():
        check(path.read_bytes() == raw, "Conflicting saved output: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(raw)


def main():
    plan, done, independent = (read(OUT / name) for name in (
        "PLAN.json", "COMPLETION.json", "INDEPENDENT_OUTCOME.json"))
    check(done["round"] == 153 and done["plan_sha256"] == ref(OUT / "PLAN.json")["sha256"],
          "Formal153 completion required")
    check(plan["arms"] == list(NAMES) and done["trials"] == 6
          and done["independent_execution_paths"] == 18 and done["protected_files"] == 21444,
          "Complete six-arm research scope required")
    check(independent["passed"] is True and independent["plan_sha256"] == done["plan_sha256"]
          and independent["selected_name"] == done["primary"]
          and ref(OUT / "INDEPENDENT_OUTCOME.json")["sha256"] == done["independent_outcome_sha256"],
          "Independent outcome binding failed")
    state = read(ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    check(len(state["completed"]) == 153 and not state["registered_queue"]
          and sum(x["trials"] for x in state["completed"]) == 968
          and state["completed"][-1]["completion"] == ref(OUT / "COMPLETION.json"),
          "Central completion accounting differs")
    for name in ("RESULTS.json", "PRETRAIN_ORDERS.json", "FEATURE_AUDIT.json", "META.json"):
        check(independent["artifact_hashes"][str(OUT / name)] == ref(OUT / name)["sha256"],
              "Unverified report input: " + name)
    protection = read(OUT / "PROTECTED_AFTER.json")
    check(done["protected_sha256"] == ref(OUT / "PROTECTED_AFTER.json")["sha256"]
          and protection["passed"] is True and protection["verified"] == 21444,
          "Protected archive was not verified")
    results, orders, feature, meta = (read(OUT / name) for name in (
        "RESULTS.json", "PRETRAIN_ORDERS.json", "FEATURE_AUDIT.json", "META.json"))
    check(results == done["all_arm_results"] and set(orders) == set(NAMES),
          "Completion/result mismatch")
    for name in NAMES:
        check(len(orders[name]) == 122
              and [r["date"] for r in orders[name]] == meta["dates"]
              and [r["signal_date"] for r in orders[name]] == meta["dates"]
              and [r["end"] for r in orders[name]] == meta["ends"],
              "Monthly decision axis changed")
    changes = {
        control: [dict(date=a["date"], control=a["symbols"], expanded=b["symbols"])
                  for a, b in zip(orders[control], orders[NAMES[-1]])
                  if a["symbols"] != b["symbols"]]
        for control in NAMES[:-1]}
    saved = read(OUT / "SUMMARY.json") if (OUT / "SUMMARY.json").exists() else None
    created = saved["created_utc"] if saved else datetime.now(timezone.utc).isoformat()
    summary = dict(
        created_utc=created, round=153, completed_rounds=153, total_trials=968,
        primary=done["primary"], target_reached=done["target_reached"],
        selection=done["selection"], source_admission_conditional=True,
        period=dict(start="2016-08-17", end="2026-09-24", months=122,
                    selection_months=88, later_diagnostic_months=34),
        results={name: dict(same_close_zero_cost=result["same_close"]["metrics"],
                            next_open_5bp_each_side=result["next_open_cost"]["metrics"],
                            slices=result["same_close"]["slices"])
                 for name, result in results.items()},
        availability={key: feature[key] for key in (
            "original_cells", "batch1_cells", "batch2_cells", "batch3_cells",
            "expanded_cells", "added_cells", "reasons")},
        expanded_order_changes=changes,
        plan=ref(OUT / "PLAN.json"), completion=ref(OUT / "COMPLETION.json"),
        independent=ref(OUT / "INDEPENDENT_OUTCOME.json"),
        orders=ref(OUT / "PRETRAIN_ORDERS.json"), protected=ref(OUT / "PROTECTED_AFTER.json"),
        limitations=[
            "Historical research on repeatedly explored periods, not observer or live returns.",
            "Later34 months are diagnostic and cannot rescue a first88 rejection.",
            "MU/PEP annual boundaries and source vintage retain declared assumptions.",
            "Early calendar holds, PEP2011 unresolved accounting and all older holds remain.",
            "Fixed1% correction; unavailable financial evidence does not exclude a stock.",
            "All frozen daily and monthly observers remain unchanged.",
        ])
    lines = [
        "# 月策略财报覆盖对照：第153轮", "",
        "2016-08-17至2026-09-24；历史同收盘零成本。保持长期动量、1%财报修正、月末选三只及月内固定持仓。本轮只增加MU、PEP的已核验财报覆盖。", "",
        "|策略|全期年化|Sharpe|最大回撤|下一开盘每边5bp年化|",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        met, op = results[name]["same_close"]["metrics"], results[name]["next_open_cost"]["metrics"]
        lines.append(f"|{LABELS[name]}|{met['cagr']:.2%}|{met['sharpe']:.3f}|"
                     f"{met['max_drawdown']:.2%}|{op['cagr']:.2%}|")
    sequence = " → ".join(str(feature[k]) for k in (
        "original_cells", "batch1_cells", "batch2_cells", "batch3_cells", "expanded_cells"))
    lines += [
        "", f"首88个月选择：{LABELS[done['primary']]}；扩展候选"
        + ("通过" if done["selection"]["admitted"] else "未通过")
        + "预先声明的收益、Sharpe、回撤及执行成本门槛。",
        f"财报有效位置：{sequence}；本批增加{feature['added_cells']}个股票—月份。",
        f"十二家公司修正相对十家公司修正改变{len(changes['batch3_earnings'])}个月的有序持仓。",
        "", "这是反复研究过的历史样本。未通过准入的报告及旧缺值继续保留；没有修改冻结观察账户。",
    ]
    if changes["batch3_earnings"]:
        lines += ["", "本批数据导致的持仓变化：", ""]
        for row in changes["batch3_earnings"]:
            before, after = "、".join(row["control"]), "、".join(row["expanded"])
            lines.append(f"- {row['date']}：{before or '现金'} → {after or '现金'}。")
    preserve(OUT / "SUMMARY.json", encoded(summary))
    preserve(OUT / "SUMMARY.md", ("\n".join(lines) + "\n").encode())
    resume_path = ROOT / "migration/RESEARCH_RESUME.json"
    raw = resume_path.read_bytes()
    resume = json.loads(raw)
    if resume.get("formal_round153_completed") is True:
        check(resume["formal_round153_completion"] == ref(OUT / "COMPLETION.json")
              and resume["current_experiment_artifacts"]["SUMMARY.json"] == ref(OUT / "SUMMARY.json"),
              "Existing resume differs")
        print(json.dumps(dict(summary=ref(OUT / "SUMMARY.json"), already_applied=True)))
        return
    check(resume["round"] == 152, "Refuse to replace another checkpoint")
    backup = ROOT / ("migration/round153_resume_" + hashlib.sha256(raw).hexdigest() + ".json")
    preserve(backup, raw)
    artifacts = {name: ref(OUT / name) for name in (
        "PLAN.json", "INDEPENDENT_PREPARE.json", "INDEPENDENT_OUTCOME.json",
        "PROTECTED_AFTER.json", "COMPLETION.json", "SUMMARY.json")}
    resume.update(
        round=153, completed_research_rounds=153, strategy_evaluations=968,
        current_experiment="monthly_financial_coverage153",
        current_experiment_artifacts=artifacts, formal_round153_completed=True,
        formal_round153_completion=artifacts["COMPLETION.json"],
        plan_sha256=done["plan_sha256"], frozen_primary=done["primary"],
        latest_protected_check=artifacts["PROTECTED_AFTER.json"],
        full_protection_check_passed=True, target_70_reached=done["target_reached"],
        next_mechanism_registered=False,
        next_research_preparation=dict(status="round153_completed_no_next_registration",
                                       source_symbols=["PLTR", "PYPL", "TMO"], new_trials=0),
        next_research_status="round153_complete_no_next_mechanism_registered",
        next_step="Follow the predeclared source-only feasibility order for PLTR/PYPL/TMO. Register any further experiment before actual factors or returns; do not tune completed outcomes.",
        previous_goal_turn_classification=resume.get("last_goal_turn_classification"),
        last_goal_turn_classification="progress",
        remaining_formal_archive_check="None for153;21444 protected files verified",
        resume_update_checkpoint=str(backup.relative_to(ROOT)), updated_utc=created)
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        check(resume_path.read_bytes() == raw, "Concurrent resume update")
        os.replace(temporary, resume_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(dict(summary=ref(OUT / "SUMMARY.json"), primary=done["primary"],
                          target=done["target_reached"])))


if __name__ == "__main__":
    main()
