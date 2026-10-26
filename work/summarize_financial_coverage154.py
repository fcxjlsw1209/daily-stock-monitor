"""Report only formally closed, independently verified round154 evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_financial_coverage154"
NAMES = ("incumbent_local67", "original_earnings", "batch1_earnings",
         "batch2_earnings", "batch3_earnings", "batch4_earnings", "expanded_earnings")
LABELS = dict(zip(NAMES, ("原最佳", "原四家公司修正", "六家公司修正",
                          "八家公司修正", "十家公司修正", "十二家公司修正",
                          "十五家公司修正")))


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
    plan, done, audit = (read(OUT / n) for n in
                         ("PLAN.json", "COMPLETION.json", "INDEPENDENT_OUTCOME.json"))
    check(done["round"] == 154 and done["plan_sha256"] == ref(OUT / "PLAN.json")["sha256"],
          "Formal154 completion required")
    check(plan["arms"] == list(NAMES) and done["trials"] == 7
          and done["independent_execution_paths"] == 21 and done["protected_files"] == 21444,
          "Complete seven-arm scope required")
    check(audit["passed"] is True and audit["plan_sha256"] == done["plan_sha256"]
          and audit["selected_name"] == done["primary"]
          and ref(OUT / "INDEPENDENT_OUTCOME.json")["sha256"] == done["independent_outcome_sha256"],
          "Independent outcome binding failed")
    state = read(ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    check(len(state["completed"]) == 154 and not state["registered_queue"]
          and sum(x["trials"] for x in state["completed"]) == 975
          and state["completed"][-1]["completion"] == ref(OUT / "COMPLETION.json"),
          "Central completion accounting differs")
    for n in ("RESULTS.json", "PRETRAIN_ORDERS.json", "FEATURE_AUDIT.json", "META.json"):
        check(audit["artifact_hashes"][str(OUT / n)] == ref(OUT / n)["sha256"],
              "Unverified report input: " + n)
    protection = read(OUT / "PROTECTED_AFTER.json")
    check(done["protected_sha256"] == ref(OUT / "PROTECTED_AFTER.json")["sha256"]
          and protection["passed"] is True and protection["verified"] == 21444,
          "Protected archive not verified")
    results, orders, feature, meta = (read(OUT / n) for n in
                                      ("RESULTS.json", "PRETRAIN_ORDERS.json", "FEATURE_AUDIT.json", "META.json"))
    check(results == done["all_arm_results"] and set(orders) == set(NAMES), "Result mismatch")
    for name in NAMES:
        check(len(orders[name]) == 122 and [r["date"] for r in orders[name]] == meta["dates"]
              and [r["signal_date"] for r in orders[name]] == meta["dates"]
              and [r["end"] for r in orders[name]] == meta["ends"], "Decision axis changed")
    changes = {
        control: [dict(date=a["date"], control=a["symbols"], expanded=b["symbols"])
                  for a, b in zip(orders[control], orders[NAMES[-1]])
                  if a["symbols"] != b["symbols"]]
        for control in NAMES[:-1]}
    saved = read(OUT / "SUMMARY.json") if (OUT / "SUMMARY.json").exists() else None
    created = saved["created_utc"] if saved else datetime.now(timezone.utc).isoformat()
    keys = ("original_cells", "batch1_cells", "batch2_cells", "batch3_cells",
            "batch4_cells", "expanded_cells")
    summary = dict(
        created_utc=created, round=154, completed_rounds=154, total_trials=975,
        primary=done["primary"], target_reached=done["target_reached"], selection=done["selection"],
        source_admission_conditional=True,
        period=dict(start="2016-08-17", end="2026-09-24", months=122,
                    selection_months=88, later_diagnostic_months=34),
        results={name: dict(same_close_zero_cost=x["same_close"]["metrics"],
                            next_open_5bp_each_side=x["next_open_cost"]["metrics"],
                            slices=x["same_close"]["slices"]) for name, x in results.items()},
        availability={k: feature[k] for k in (*keys, "added_cells", "reasons")},
        expanded_order_changes=changes, plan=ref(OUT / "PLAN.json"),
        completion=ref(OUT / "COMPLETION.json"), independent=ref(OUT / "INDEPENDENT_OUTCOME.json"),
        orders=ref(OUT / "PRETRAIN_ORDERS.json"), protected=ref(OUT / "PROTECTED_AFTER.json"),
        limitations=[
            "Repeatedly explored historical data, not observer returns or pristine out-of-sample evidence.",
            "Later34 is diagnostic only and cannot rescue a first88 rejection.",
            "PLTR/PYPL/TMO annual boundaries, common-dollar units and source vintage retain declared assumptions.",
            "PYPL2015 combined/consolidated scope and TMO2018 comparative restatement are retained as reported.",
            "Early calendar holds and latest-unprocessed catalog blocking remain.",
            "Fixed1% correction; missing financial evidence never removes a stock from the pool.",
            "Initial alphabetic source-lead batches are complete; this does not certify all73 issuers.",
            "Frozen observer accounts were not modified.",
        ])
    lines = ["# 月策略财报覆盖对照：第154轮", "",
             "2016-08-17至2026-09-24；历史同收盘零成本。长期动量主导、跳过最近42个交易日、月末选三只及月内固定持仓。本轮仅增加PLTR、PYPL、TMO的已核验财报覆盖，修正权重仍为1%。", "",
             "|策略|全期年化|Sharpe|最大回撤|下一开盘每边5bp年化|",
             "|---|---:|---:|---:|---:|"]
    for name in NAMES:
        m, o = results[name]["same_close"]["metrics"], results[name]["next_open_cost"]["metrics"]
        lines.append(f"|{LABELS[name]}|{m['cagr']:.2%}|{m['sharpe']:.3f}|"
                     f"{m['max_drawdown']:.2%}|{o['cagr']:.2%}|")
    lines += ["", f"首88个月选择：{LABELS[done['primary']]}；扩展候选"
              + ("通过" if done["selection"]["admitted"] else "未通过")
              + "预先声明的收益、Sharpe、回撤及执行成本门槛。",
              "财报有效位置：" + " → ".join(str(feature[k]) for k in keys)
              + f"；本批增加{feature['added_cells']}个股票—月份。",
              f"新增覆盖相对上一轮改变{len(changes['batch4_earnings'])}个月的有序持仓。",
              "", "|阶段|原最佳年化 / Sharpe|扩展覆盖年化 / Sharpe|",
              "|---|---:|---:|"]
    for label, key in (("2016–2020", "early2016_2020"), ("首88个月", "training2016_2023"),
                       ("后34个月（诊断）", "later34")):
        a, b = (results[n]["same_close"]["slices"][key] for n in (NAMES[0], NAMES[-1]))
        lines.append(f"|{label}|{a['cagr']:.2%} / {a['sharpe']:.3f}|{b['cagr']:.2%} / {b['sharpe']:.3f}|")
    lines += ["", "以上均为反复研究过的历史样本，后34个月不是独立样本外验证。已保留缺失来源和时点假设，不将财报覆盖完成等同于股票池全面认证。冻结观察账户不变。"]
    if changes["batch4_earnings"]:
        lines += ["", "本批数据导致的持仓变化：", ""]
        for row in changes["batch4_earnings"]:
            lines.append(f"- {row['date']}：{'、'.join(row['control']) or '现金'} → "
                         f"{'、'.join(row['expanded']) or '现金'}。")
    preserve(OUT / "SUMMARY.json", encoded(summary))
    preserve(OUT / "SUMMARY.md", ("\n".join(lines) + "\n").encode())
    path = ROOT / "migration/RESEARCH_RESUME.json"
    raw = path.read_bytes()
    resume = json.loads(raw)
    if resume.get("formal_round154_completed") is True:
        check(resume["formal_round154_completion"] == ref(OUT / "COMPLETION.json")
              and resume["current_experiment_artifacts"]["SUMMARY.json"] == ref(OUT / "SUMMARY.json"),
              "Existing resume differs")
        print(json.dumps(dict(summary=ref(OUT / "SUMMARY.json"), already_applied=True)))
        return
    check(resume["round"] == 153, "Refuse to replace another checkpoint")
    backup = ROOT / ("migration/round154_resume_" + hashlib.sha256(raw).hexdigest() + ".json")
    preserve(backup, raw)
    artifacts = {n: ref(OUT / n) for n in ("PLAN.json", "INDEPENDENT_PREPARE.json",
                 "INDEPENDENT_OUTCOME.json", "PROTECTED_AFTER.json", "COMPLETION.json", "SUMMARY.json")}
    resume.update(
        round=154, completed_research_rounds=154, strategy_evaluations=975,
        current_experiment="monthly_financial_coverage154", current_experiment_artifacts=artifacts,
        formal_round154_completed=True, formal_round154_completion=artifacts["COMPLETION.json"],
        plan_sha256=done["plan_sha256"], frozen_primary=done["primary"],
        latest_protected_check=artifacts["PROTECTED_AFTER.json"], full_protection_check_passed=True,
        target_70_reached=done["target_reached"], next_mechanism_registered=False,
        next_research_preparation=dict(status="round154_complete_initial_source_batches_tested", new_trials=0),
        next_research_status="round154_complete_no_next_mechanism_registered",
        next_step="Review remaining original guidance mechanisms and source coverage without retuning completed outcomes; predeclare any new hypothesis before scores or returns.",
        previous_goal_turn_classification=resume.get("last_goal_turn_classification"),
        last_goal_turn_classification="progress", remaining_formal_archive_check="None for154;21444 files verified",
        resume_update_checkpoint=str(backup.relative_to(ROOT)), updated_utc=created)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        check(path.read_bytes() == raw, "Concurrent resume update")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(dict(summary=ref(OUT / "SUMMARY.json"), primary=done["primary"],
                          target=done["target_reached"])))


if __name__ == "__main__":
    main()
