"""Publish the completed coverage experiment and its exact research checkpoint."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_financial_coverage150"


def read(path):
    return json.loads(path.read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def ref(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    plan, done, independent = (read(OUT / name) for name in (
        "PLAN.json", "COMPLETION.json", "INDEPENDENT_OUTCOME.json"))
    assert done["round"] == 150 and done["plan_sha256"] == ref(OUT / "PLAN.json")["sha256"]
    assert done["independent_execution_paths"] == 9 and done["protected_files"] == 21444
    assert independent["passed"] and independent["plan_sha256"] == done["plan_sha256"]
    assert independent["selected_name"] == done["primary"]
    assert ref(OUT / "INDEPENDENT_OUTCOME.json")["sha256"] == done["independent_outcome_sha256"]
    state = read(ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    assert len(state["completed"]) == 150 and not state["registered_queue"]
    assert sum(x["trials"] for x in state["completed"]) == 953
    results, orders, feature = (read(OUT / name) for name in (
        "RESULTS.json", "PRETRAIN_ORDERS.json", "FEATURE_AUDIT.json"))
    assert results == done["all_arm_results"]
    changes = {}
    for control in ("incumbent_local67", "original_earnings"):
        changes[control] = [
            dict(date=a["date"], control=a["symbols"], expanded=b["symbols"])
            for a, b in zip(orders[control], orders["expanded_earnings"])
            if a["symbols"] != b["symbols"]]
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        round=150, completed_rounds=150, total_trials=953,
        primary=done["primary"], target_reached=done["target_reached"],
        selection=done["selection"], source_admission_conditional=True,
        period=dict(start="2016-08-17", end="2026-09-24", months=122,
                    selection_months=88, later_diagnostic_months=34),
        results={name: dict(same_close_zero_cost=result["same_close"]["metrics"],
                            next_open_5bp_each_side=result["next_open_cost"]["metrics"],
                            slices=result["same_close"]["slices"])
                 for name, result in results.items()},
        availability={key: feature[key] for key in (
            "original_cells", "expanded_cells", "added_cells", "reasons")},
        expanded_order_changes=changes,
        plan=ref(OUT / "PLAN.json"), completion=ref(OUT / "COMPLETION.json"),
        independent=ref(OUT / "INDEPENDENT_OUTCOME.json"),
        orders=ref(OUT / "PRETRAIN_ORDERS.json"), protected=ref(OUT / "PROTECTED_AFTER.json"),
        limitations=[
            "All reported performance is historical research, not observer or real returns.",
            "The full history has repeatedly been explored; later34 is a diagnostic, not pristine OOS.",
            "New financial sources remain conditional, with original extraction gaps and accounting holds.",
            "The experiment changes only company-information coverage at the fixed1% dose.",
            "Frozen observers and their plans remain unchanged.",
        ])
    with (OUT / "SUMMARY.json").open("xb") as stream:
        stream.write(encoded(summary))
    labels = {"incumbent_local67": "原最佳", "original_earnings": "原四家公司财报修正",
              "expanded_earnings": "扩展AMZN/CSCO财报修正"}
    lines = [
        "# 月策略财报覆盖对照：第150轮",
        "",
        "2016-08-17至2026-09-24；历史同收盘零成本。仅扩大财报覆盖，动量核心、1%修正、月末选三只及月内固定持仓保持不变。",
        "",
        "|策略|全期年化|Sharpe|最大回撤|下一开盘每边5bp年化|",
        "|---|---:|---:|---:|---:|",
    ]
    for name in plan["arms"]:
        met, op = results[name]["same_close"]["metrics"], results[name]["next_open_cost"]["metrics"]
        lines.append(f"|{labels[name]}|{met['cagr']:.2%}|{met['sharpe']:.3f}|"
                     f"{met['max_drawdown']:.2%}|{op['cagr']:.2%}|")
    lines += ["", f"首88个月规则选择：{labels[done['primary']]}；扩展候选"
              + ("通过" if done["selection"]["admitted"] else "未通过")
              + "预先声明的收益、Sharpe、回撤和执行成本门槛。",
              f"财报有效位置：{feature['original_cells']} → {feature['expanded_cells']}，"
              f"增加{feature['added_cells']}个股票—月份。",
              f"扩展候选相对原财报修正改变{len(changes['original_earnings'])}个月的有序持仓。",
              "", "所有数字属于反复研究过的历史样本；未改动任何冻结观察账户。"]
    with (OUT / "SUMMARY.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    resume_path = ROOT / "migration/RESEARCH_RESUME.json"
    raw = resume_path.read_bytes()
    previous_sha = hashlib.sha256(raw).hexdigest()
    backup = ROOT / ("migration/round150_resume_" + previous_sha + ".json")
    with backup.open("xb") as stream:
        stream.write(raw)
    resume = json.loads(raw)
    artifacts = {name: ref(OUT / name) for name in (
        "PLAN.json", "INDEPENDENT_PREPARE.json", "INDEPENDENT_OUTCOME.json",
        "PROTECTED_AFTER.json", "COMPLETION.json", "SUMMARY.json")}
    resume.update(round=150, completed_research_rounds=150, strategy_evaluations=953,
                  current_experiment="monthly_financial_coverage150",
                  current_experiment_artifacts=artifacts,
                  formal_round150_completed=True,
                  formal_round150_completion=artifacts["COMPLETION.json"],
                  plan_sha256=done["plan_sha256"], frozen_primary=done["primary"],
                  latest_protected_check=artifacts["PROTECTED_AFTER.json"],
                  full_protection_check_passed=True, target_70_reached=done["target_reached"],
                  next_mechanism_registered=False,
                  next_research_status="round150_complete_no_next_mechanism_registered",
                  next_step="Review this coverage experiment before preregistering further guidance-consistent research; preserve all source gaps and do not tune the fixed dose to these outcomes.",
                  previous_goal_turn_classification="progress: completed150 with source expansion and independent9-path accounting",
                  remaining_formal_archive_check="None for150;21444 protected files verified",
                  resume_update_checkpoint=str(backup.relative_to(ROOT)),
                  updated_utc=summary["created_utc"])
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume))
    assert hashlib.sha256(resume_path.read_bytes()).hexdigest() == previous_sha
    os.replace(temporary, resume_path)
    print(json.dumps(dict(summary=ref(OUT / "SUMMARY.json"),
                          primary=done["primary"], target=done["target_reached"])))


if __name__ == "__main__":
    main()
