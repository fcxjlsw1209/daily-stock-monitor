"""Publish audited round146 results and repair provenance without modifying freezes."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import monthly_cash_quality146_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_cash_quality146"
PLAN_SHA = "23807fe79e5cb9bd6a21273ec7ade26d8a6cb36aab18715e2e052fe1437e81f2"
AMENDMENT_SHA = "4a4f270bac7c7db020ba8357badbd4b3d4de3eb956dc35547eca114e21af5d79"


def load(path):
    return json.loads(path.read_text())


def ref(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    review_path = OUT / "EXECUTION_REPAIR_COMPLETION_REVIEW.json"
    review = load(review_path)
    done = load(OUT / "COMPLETION.json")
    results = load(OUT / "RESULTS.json")
    outcome = load(OUT / "INDEPENDENT_OUTCOME.json")
    assert review["passed"] is True and review["phase"] == "complete"
    assert review["plan_sha256"] == done["plan_sha256"] == PLAN_SHA
    assert review["amendment_sha256"] == AMENDMENT_SHA
    assert all(review["checks"].values())
    review_pins = {str((ROOT / path).resolve()): digest
                   for path, digest in review["input_hashes"].items()}
    assert ref(OUT / "COMPLETION.json")["sha256"] == review_pins[str(OUT / "COMPLETION.json")]
    assert ref(OUT / "INDEPENDENT_OUTCOME.json")["sha256"] == done["independent_outcome_sha256"]
    assert done["all_arm_results"] == results
    assert done["primary"] == "incumbent_local67" and done["target_reached"] is False
    assert done["selection"]["admitted"] is False
    monthly = outcome["independent_monthly_comparison"]
    changed = [row for row in monthly
               if row["holdings"]["incumbent_local67"] != row["holdings"]["low_accrual_quality"]]
    assert len(changed) == 2
    assert all(row["split"] != "later34" for row in changed)
    assert all(row["holdings"]["earnings_only"] == row["holdings"]["cash_flow_only"]
               for row in monthly)
    now = datetime.now(timezone.utc).isoformat()
    repair_ref = ref(review_path)
    completion_ref = ref(OUT / "COMPLETION.json")
    with lifecycle.state_lock(ROOT):
        state_path = ROOT / lifecycle.STATE
        state, expected = lifecycle.state_snapshot(state_path)
        assert len(state["completed"]) == 146
        assert sum(x["trials"] for x in state["completed"]) == 939
        assert state["completed"][-1]["completion"] == completion_ref
        state["completed"][-1]["execution_repair_review"] = repair_ref
        state["latest_execution_repair_review"] = repair_ref
        state["updated_utc"] = now
        lifecycle.update_state(state_path, expected, state)

        resume_path = ROOT / "migration/RESEARCH_RESUME.json"
        resume, resume_sha = lifecycle.state_snapshot(resume_path)
        checkpoint = ROOT / "migration" / ("round146_resume_" + resume_sha + ".json")
        if not checkpoint.exists():
            checkpoint.write_bytes(resume_path.read_bytes())
        resume.update(
            completed_research_rounds=146, strategy_evaluations=939, round=146,
            current_experiment="monthly_cash_quality146",
            plan_sha256=PLAN_SHA, formal_round146_completed=True,
            formal_round146_completion=completion_ref,
            frozen_primary=done["primary"], target_70_reached=False,
            latest_execution_repair_review=repair_ref,
            latest_protected_check=ref(OUT / "PROTECTED_AFTER.json"),
            latest_source_admission=ref(ROOT / "outputs/monthly_cash_quality_source_admission_v1/ADMISSION.json"),
            current_experiment_artifacts={name: ref(OUT / (name + ".json")) for name in (
                "PLAN", "CACHE", "FROZEN", "RESULTS", "INDEPENDENT_PREPARE",
                "INDEPENDENT_OUTCOME", "COMPLETION", "PROTECTED_AFTER",
                "PREFIX_EXECUTION_AMENDMENT", "EXECUTION_REPAIR_COMPLETION_REVIEW")},
            next_mechanism_registered=False,
            next_research_status="round146_complete_next_distinct_hypothesis_pending",
            next_step="Keep the incumbent. Cash-quality experiment failed the preregistered first88 selection. Do not rescue it using later34 or change its frozen dose.",
            next_candidate_review={
                "candidate": "annual_total_asset_expansion",
                "status": "concept_only_requires_full_novelty_review_and_registration",
                "rationale": "Recent rounds tested profitability and accrual quality, not a standalone asset-growth signal.",
                "caveats": [
                    "Existing data, not a new information source.",
                    "Need same-mask profitability control; growth is not directly capital expenditure.",
                    "No new factor, ranking, orders, returns or model fitting performed."
                ]},
            remaining_formal_archive_check="None for round146; 21444 protected files verified.",
            previous_goal_turn_classification="progress: completed cash-quality experiment with independent twelve-path verification; target unmet.",
            resume_update_checkpoint=str(checkpoint.relative_to(ROOT)), updated_utc=now)
        lifecycle.update_state(resume_path, resume_sha, resume)

    labels = {"incumbent_local67": "现有最佳", "earnings_only": "盈利修正",
              "cash_flow_only": "经营现金流修正", "low_accrual_quality": "现金质量修正"}
    lines = [
        "# 第146轮：现金质量因子",
        "",
        "已完成四组预登记比较、独立数值复核及归档。现金质量修正未通过前88个月的筛选，保留现有最佳策略，70%全期年化目标尚未达到。",
        "",
        "区间为2016-08-17至2026-09-24。主口径为同收盘价、零交易成本的研究回放；次日开盘口径每边成本0.05%。",
        "",
        "| 策略 | 全期年化 | Sharpe | 最大回撤 | 次日开盘含成本年化 |",
        "|---|---:|---:|---:|---:|",
    ]
    compact = {}
    for name, label in labels.items():
        m = results[name]["same_close"]["metrics"]
        opened = results[name]["next_open_cost"]["metrics"]
        compact[name] = {key: m[key] for key in ("cagr", "sharpe", "max_drawdown")}
        compact[name]["next_open_cost_cagr"] = opened["cagr"]
        lines.append(f"| {label} | {m['cagr']:.2%} | {m['sharpe']:.3f} | {m['max_drawdown']:.2%} | {opened['cagr']:.2%} |")
    lines += [
        "",
        "前88个月中，现金质量组年化56.26%，现有最佳56.70%；因此没有通过预先确定的优选条件。后34个月仅为已重复使用的诊断区间，不是独立样本外验证。",
        "",
        "现金质量组只改变两个月的持仓或排序；后34个月的持仓没有改变。盈利与现金流两组得到相同交易路径，不能视为两份独立支持证据。",
        "",
        "| 信号日 | 原持仓顺序 | 现金质量组顺序 | 随后持有期收益差 |",
        "|---|---|---|---:|",
    ]
    for row in changed:
        before = "、".join(row["holdings"]["incumbent_local67"])
        after = "、".join(row["holdings"]["low_accrual_quality"])
        delta = row["returns"]["low_accrual_quality"] - row["returns"]["incumbent_local67"]
        lines.append(f"| {row['date']} | {before} | {after} | {delta * 100:+.3f}个百分点 |")
    lines += [
        "",
        "本轮财报仅涵盖AAPL、AMD、MSFT、NVDA的固定44份年度材料，31份有条件通过、13份保留缺口；可用特征318个股票—月份。无法据此宣称完整股票池或完整历史时点财报已经覆盖。",
        "",
        "首次准备在前缀数组长度接口处失败。原代码和计划保留，另行登记仅截取已请求前缀的执行修复；全长评分不变。修复后的缓存、全部123个前缀、488条月度选股记录和12条交易执行路径经独立验证。21,444个受保护历史文件哈希一致。",
        "",
        "现有最佳67.67%仍为多轮研究筛选后的历史结果，不保证未来收益；观察账户未更新。",
        "",
        "- [预登记计划](PLAN.json)",
        "- [完整结果](RESULTS.json)",
        "- [独立数值审核](INDEPENDENT_OUTCOME.json)",
        "- [执行修复最终核验](EXECUTION_REPAIR_COMPLETION_REVIEW.json)",
        "- [正式完成记录](COMPLETION.json)",
        "",
    ]
    text = "\n".join(lines)
    report_path = OUT / "SUMMARY.md"
    if report_path.exists():
        assert report_path.read_text() == text
    else:
        with report_path.open("x") as stream:
            stream.write(text)
    print(json.dumps({"completed_rounds": 146, "trials": 939,
                      "summary": ref(report_path), "repair": repair_ref,
                      "results": compact}, ensure_ascii=False))


if __name__ == "__main__":
    main()
