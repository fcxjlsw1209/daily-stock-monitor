"""Publish a completed experiment's results, without new strategy evaluation."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/monthly_asset_growth147"
PLAN_SHA = "61f94781edb5c4f6f98f9c4753db8a4d99fe887b795fb39e120270402c7549c1"


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def save_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    assert digest(OUT/"PLAN.json") == PLAN_SHA
    done = load(OUT/"COMPLETION.json")
    independent = load(OUT/"INDEPENDENT_OUTCOME.json")
    protected = load(OUT/"PROTECTED_AFTER.json")
    assert done["plan_sha256"] == PLAN_SHA
    assert independent["passed"] and independent["execution_paths_verified"] == 12
    assert independent["plan_sha256"] == PLAN_SHA
    assert protected["passed"] and protected["verified"] == 21444
    state = load(ROOT/"outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    assert state["completed"][-1]["directory"] == OUT.name
    assert state["completed"][-1]["completion"]["sha256"] == digest(OUT/"COMPLETION.json")
    results = load(OUT/"RESULTS.json")
    orders = load(OUT/"PRETRAIN_ORDERS.json")
    names = ("incumbent_local67", "earnings_control", "low_accrual_control", "low_asset_growth")
    labels = ("原最佳策略", "盈利控制组", "低应计控制组", "低资产增长")
    changed = []
    nav = {
        name: {row["date"]: row["equity"] for row in load(OUT/(name+"_same_close_DAILY.json"))}
        for name in ("incumbent_local67", "low_asset_growth")
    }
    for old, new in zip(orders["incumbent_local67"], orders["low_asset_growth"]):
        assert old["date"] == new["date"] and old["end"] == new["end"]
        if old["symbols"] == new["symbols"]:
            continue
        start, end = old["date"], old["end"]
        baseline_return = nav["incumbent_local67"][end]/nav["incumbent_local67"][start]-1
        challenger_return = nav["low_asset_growth"][end]/nav["low_asset_growth"][start]-1
        changed.append(dict(
            signal_date=start, end=end, original=old["symbols"], challenger=new["symbols"],
            holding_set_changed=set(old["symbols"]) != set(new["symbols"]),
            baseline_return=baseline_return, challenger_return=challenger_return,
            return_difference_pp=100*(challenger_return-baseline_return)))
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), round=147,
        plan=ref(OUT/"PLAN.json"), completion=ref(OUT/"COMPLETION.json"),
        independent=ref(OUT/"INDEPENDENT_OUTCOME.json"),
        results=ref(OUT/"RESULTS.json"), orders=ref(OUT/"PRETRAIN_ORDERS.json"),
        primary=done["primary"], target_reached=done["target_reached"],
        changed_order_months=changed, completed_rounds=len(state["completed"]),
        total_trials=sum(row["trials"] for row in state["completed"]),
        inherited_limitations=["Repeatedly reused history is not pristine out of sample.",
                              "Conditional four-issuer reporting data; historical price and universe proxy assumptions remain.",
                              "Same-close zero-cost research differs from executable next-open cost results."])
    save_new(OUT/"SUMMARY.json", summary)
    lines = [
        "# 第147轮：资产增长修正",
        "",
        "区间：2016-08-17—2026-09-24。主表为同收盘价、零成本历史回测。",
        "固定三只等权、月内固定持仓；长期动量主导并跳过最近两个月。",
        "",
        "|候选|全期年化|Sharpe|最大回撤|次日开盘含成本年化|",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in zip(names, labels):
        metrics = results[name]["same_close"]["metrics"]
        live = results[name]["next_open_cost"]["metrics"]
        lines.append(f"|{label}|{metrics['cagr']:.2%}|{metrics['sharpe']:.3f}|"
                     f"{metrics['max_drawdown']:.2%}|{live['cagr']:.2%}|")
    lines += [
        "",
        f"预声明筛选结果：{done['primary']}；资产增长候选获准替换：{done['selection']['admitted']}。",
        "资产增长候选未通过前88个月筛选；后34个月不得用于挽救这个选择。"
        if not done["selection"]["admitted"] else "候选通过预声明筛选，仍保留历史反复使用的局限。",
        "",
        "|选股日|原组合|资产增长组合|持仓集合改变|随后月收益差|",
        "|---|---|---|---|---:|",
    ]
    for change in changed:
        lines.append(f"|{change['signal_date']}|{', '.join(change['original'])}|"
                     f"{', '.join(change['challenger'])}|"
                     f"{'是' if change['holding_set_changed'] else '仅顺序'}|"
                     f"{change['return_difference_pp']:+.3f}个百分点|")
    lines += [
        "",
        "所有因子使用同一财报可用性掩码；44个事件中31个条件准入、13个继续排除。"
        "有效数据覆盖318个股票候选月单元格，不能当成318次独立财报。",
        "",
        "核验：183项合成测试；独立核对488条月度选股与12条执行路径；"
        "21,444个保护文件一致。没有修改冻结观察账户。",
        f"累计完成{summary['completed_rounds']}轮、{summary['total_trials']}次尝试。70%目标仍以完整结果为准。",
        "",
        "这是一项条件数据准入、反复使用历史区间的研究结果；不是纯样本外、"
        "不是实盘收益。同收盘价假设不能保证可执行，另列次日开盘每边0.05%成本结果。",
    ]
    with (OUT/"SUMMARY.md").open("x") as stream:
        stream.write("\n".join(lines)+"\n")
    path = ROOT/"migration/RESEARCH_RESUME.json"
    raw = path.read_bytes()
    prior_sha = hashlib.sha256(raw).hexdigest()
    checkpoint = ROOT/"migration"/("round147_resume_"+prior_sha+".json")
    with checkpoint.open("xb") as stream:
        stream.write(raw)
    resume = json.loads(raw)
    resume.update(
        round=147, current_experiment=OUT.name, plan_sha256=PLAN_SHA,
        current_experiment_artifacts={name: ref(OUT/name) for name in
                                     ("PLAN.json", "COMPLETION.json", "INDEPENDENT_PREPARE.json",
                                      "INDEPENDENT_OUTCOME.json", "PROTECTED_AFTER.json", "SUMMARY.json")},
        formal_round147_completed=True, formal_round147_completion=ref(OUT/"COMPLETION.json"),
        completed_research_rounds=summary["completed_rounds"], strategy_evaluations=summary["total_trials"],
        target_70_reached=done["target_reached"], latest_protected_check=ref(OUT/"PROTECTED_AFTER.json"),
        next_research_status="round147_complete_session_cluster148_synthetic_preparation",
        next_step="Keep the incumbent. Prepare and preregister the adjusted overnight/intraday event-clustering hypothesis; no actual148 calculations before registration.",
        next_candidate_review=dict(
            candidate="adjusted_overnight_positive_intraday_negative_event_clustering",
            status="prospective_synthetic_preparation_not_registered",
            concept=ref(ROOT/"work/monthly_session_cluster148_concept.json"),
            factor=ref(ROOT/"work/monthly_session_cluster148_factor.py"),
            factor_tests=ref(ROOT/"work/test_monthly_session_cluster148_factor.py"),
            no_actual_features_or_returns=True,
            caveats=["Already proposed after140; not a newly discovered mechanism.",
                     "Missing historical experiment files limit full novelty certification.",
                     "Adjusted overnight returns include dividend adjustments; not pure quotation gaps."]),
        next_mechanism_registered=False,
        previous_goal_turn_classification="progress: completed147 with independent12-path verification; target unmet",
        remaining_formal_archive_check="None for147;21444 protected files verified",
        resume_update_checkpoint=str(checkpoint.relative_to(ROOT)),
        updated_utc=summary["created_utc"])
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staging = Path(stream.name)
        stream.write((json.dumps(resume, ensure_ascii=False, indent=2, sort_keys=True)+"\n").encode())
    try:
        assert digest(path) == prior_sha, "Concurrent resume update"
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()
    print(json.dumps(dict(summary=ref(OUT/"SUMMARY.json"), report=ref(OUT/"SUMMARY.md"),
                          changed_months=len(changed), target_reached=done["target_reached"])))


if __name__ == "__main__":
    main()
