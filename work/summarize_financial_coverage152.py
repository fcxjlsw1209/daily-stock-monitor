"""Summarize the formally completed five-arm coverage experiment."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_financial_coverage152"
NAMES = ("incumbent_local67", "original_earnings", "batch1_earnings",
         "batch2_earnings", "expanded_earnings")
LABELS = dict(zip(NAMES, ("原最佳", "原四家公司修正", "六家公司修正",
                          "八家公司修正", "十家公司修正")))


def read(path):
    return json.loads(path.read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def ref(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def preserve(path, raw):
    if path.exists():
        assert path.read_bytes() == raw, "Conflicting existing output: " + str(path)
    else:
        with path.open("xb") as stream:
            stream.write(raw)


def main():
    plan, done, independent = (read(OUT / name) for name in (
        "PLAN.json", "COMPLETION.json", "INDEPENDENT_OUTCOME.json"))
    assert done["round"] == 152 and done["plan_sha256"] == ref(OUT / "PLAN.json")["sha256"]
    assert plan["arms"] == list(NAMES) and done["trials"] == 5
    assert done["independent_execution_paths"] == 15 and done["protected_files"] == 21444
    assert independent["passed"] is True and independent["plan_sha256"] == done["plan_sha256"]
    assert independent["selected_name"] == done["primary"]
    assert ref(OUT / "INDEPENDENT_OUTCOME.json")["sha256"] == done["independent_outcome_sha256"]
    state = read(ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    assert len(state["completed"]) == 152 and not state["registered_queue"]
    assert sum(x["trials"] for x in state["completed"]) == 962
    assert state["completed"][-1]["completion"] == ref(OUT / "COMPLETION.json")
    for name in ("RESULTS.json", "PRETRAIN_ORDERS.json", "FEATURE_AUDIT.json", "META.json"):
        assert independent["artifact_hashes"][str(OUT / name)] == ref(OUT / name)["sha256"]
    assert done["protected_sha256"] == ref(OUT / "PROTECTED_AFTER.json")["sha256"]
    protection = read(OUT / "PROTECTED_AFTER.json")
    assert protection["passed"] is True and protection["verified"] == 21444
    results, orders, feature = (read(OUT / name) for name in (
        "RESULTS.json", "PRETRAIN_ORDERS.json", "FEATURE_AUDIT.json"))
    meta = read(OUT / "META.json")
    assert results == done["all_arm_results"] and set(orders) == set(NAMES)
    for name in NAMES:
        assert len(orders[name]) == 122
        assert [r["date"] for r in orders[name]] == meta["dates"]
        assert [r["signal_date"] for r in orders[name]] == meta["dates"]
        assert [r["end"] for r in orders[name]] == meta["ends"]
    changes = {
        control: [dict(date=a["date"], control=a["symbols"], expanded=b["symbols"])
                  for a, b in zip(orders[control], orders[NAMES[-1]])
                  if a["symbols"] != b["symbols"]]
        for control in NAMES[:-1]}
    previous = read(OUT / "SUMMARY.json") if (OUT / "SUMMARY.json").exists() else None
    created = previous["created_utc"] if previous else datetime.now(timezone.utc).isoformat()
    summary = dict(
        created_utc=created, round=152, completed_rounds=152, total_trials=962,
        primary=done["primary"], target_reached=done["target_reached"],
        selection=done["selection"], source_admission_conditional=True,
        period=dict(start="2016-08-17", end="2026-09-24", months=122,
                    selection_months=88, later_diagnostic_months=34),
        results={name: dict(same_close_zero_cost=result["same_close"]["metrics"],
                            next_open_5bp_each_side=result["next_open_cost"]["metrics"],
                            slices=result["same_close"]["slices"])
                 for name, result in results.items()},
        availability={key: feature[key] for key in (
            "original_cells", "batch1_cells", "batch2_cells", "expanded_cells",
            "added_cells", "reasons")},
        expanded_order_changes=changes,
        plan=ref(OUT / "PLAN.json"), completion=ref(OUT / "COMPLETION.json"),
        independent=ref(OUT / "INDEPENDENT_OUTCOME.json"),
        orders=ref(OUT / "PRETRAIN_ORDERS.json"), protected=ref(OUT / "PROTECTED_AFTER.json"),
        limitations=[
            "Historical research, not observer or real returns; all periods have been repeatedly explored.",
            "Later34 is diagnostic only and cannot rescue a first88 rejection.",
            "KO/LRCX source admission retains annual-boundary assumptions and early calendar holds.",
            "KO FY2012 remains a producer gap despite independently readable adjusted-GAAP source.",
            "Only source coverage changes at the fixed1% dose; no financial missingness stock exclusion.",
            "The four frozen monthly observers and six daily accounts remain unchanged.",
        ])
    lines = [
        "# 月策略财报覆盖对照：第152轮", "",
        "2016-08-17至2026-09-24；历史同收盘零成本。固定长期动量核心、1%财报修正、月末选三只和月内固定持仓，逐级比较四家、六家、八家、十家公司财报覆盖。", "",
        "|策略|全期年化|Sharpe|最大回撤|下一开盘每边5bp年化|",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        met = results[name]["same_close"]["metrics"]
        op = results[name]["next_open_cost"]["metrics"]
        lines.append(f"|{LABELS[name]}|{met['cagr']:.2%}|{met['sharpe']:.3f}|"
                     f"{met['max_drawdown']:.2%}|{op['cagr']:.2%}|")
    lines += [
        "", f"首88个月选择：{LABELS[done['primary']]}；扩展候选"
        + ("通过" if done["selection"]["admitted"] else "未通过")
        + "预先声明的收益、Sharpe、回撤及执行成本门槛。",
        f"财报有效位置：{feature['original_cells']} → {feature['batch1_cells']} → {feature['batch2_cells']} → {feature['expanded_cells']}；本批增加{feature['added_cells']}个股票—月份。",
        f"十家公司候选相对八家公司候选改变{len(changes['batch2_earnings'])}个月的有序持仓。",
        "", "未达准入要求的报告和缺值保留；早期交易日历、公司身份及数据版本限制仍在。以上为反复研究过的历史样本，未修改任何冻结观察账户。",
    ]
    if changes["batch2_earnings"]:
        lines += ["", "新增数据导致的持仓变化：", ""]
        for row in changes["batch2_earnings"]:
            before, after = "、".join(row["control"]), "、".join(row["expanded"])
            lines.append(f"- {row['date']}：{before or '现金'} → {after or '现金'}。")
    preserve(OUT / "SUMMARY.json", encoded(summary))
    preserve(OUT / "SUMMARY.md", ("\n".join(lines) + "\n").encode())
    resume_path = ROOT / "migration/RESEARCH_RESUME.json"
    raw = resume_path.read_bytes()
    resume = json.loads(raw)
    if resume.get("formal_round152_completed") is True:
        assert resume["formal_round152_completion"] == ref(OUT / "COMPLETION.json")
        assert resume["current_experiment_artifacts"]["SUMMARY.json"] == ref(OUT / "SUMMARY.json")
        assert resume["round"] == 152 and resume["target_70_reached"] == done["target_reached"]
        print(json.dumps(dict(summary=ref(OUT / "SUMMARY.json"), already_applied=True)))
        return
    assert resume["round"] == 151, "Refuse to overwrite a different research checkpoint"
    previous_sha = hashlib.sha256(raw).hexdigest()
    backup = ROOT / ("migration/round152_resume_" + previous_sha + ".json")
    preserve(backup, raw)
    artifacts = {name: ref(OUT / name) for name in (
        "PLAN.json", "INDEPENDENT_PREPARE.json", "INDEPENDENT_OUTCOME.json",
        "PROTECTED_AFTER.json", "COMPLETION.json", "SUMMARY.json")}
    next_feasibility = ref(ROOT / "work/after152_next_source_feasibility.json")
    next_context = ref(ROOT / "work/after152_next_source_context.json")
    assert next_feasibility["sha256"] == "c79d523697ecbb570ac99a2ecddeebc998db6ab77c92cd23697cf70596c5b468"
    assert next_context["sha256"] == "f48acb7e7e57d1212b03bcf435b510dd75670f334a3bc228ceacb22ce65bfea7"
    resume.update(
        round=152, completed_research_rounds=152, strategy_evaluations=962,
        current_experiment="monthly_financial_coverage152",
        current_experiment_artifacts=artifacts, formal_round152_completed=True,
        formal_round152_completion=artifacts["COMPLETION.json"],
        plan_sha256=done["plan_sha256"], frozen_primary=done["primary"],
        latest_protected_check=artifacts["PROTECTED_AFTER.json"],
        full_protection_check_passed=True, target_70_reached=done["target_reached"],
        next_mechanism_registered=False,
        next_research_preparation=dict(status="round152_completed_no_next_registration",
                                       source_symbols=["MU", "PEP"], new_trials=0,
                                       source_feasibility=next_feasibility,
                                       source_context=next_context, values_extracted=0),
        next_research_status="round152_complete_no_next_mechanism_registered",
        next_step="Review MU/PEP source metadata under the fixed source-availability order. Any further extraction, audit and admission precede another registered fixed-dose contrast; do not tune on completed outcomes.",
        previous_goal_turn_classification=resume.get("last_goal_turn_classification"),
        last_goal_turn_classification="progress",
        remaining_formal_archive_check="None for152;21444 protected files verified",
        resume_update_checkpoint=str(backup.relative_to(ROOT)), updated_utc=created)
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        assert hashlib.sha256(resume_path.read_bytes()).hexdigest() == previous_sha
        os.replace(temporary, resume_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(dict(summary=ref(OUT / "SUMMARY.json"), primary=done["primary"],
                          target=done["target_reached"])))


if __name__ == "__main__":
    main()
