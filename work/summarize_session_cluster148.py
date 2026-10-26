"""Publish independently completed round148; never calculate new strategies."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_session_cluster148"
NAMES = ("incumbent_local67", "event_frequency_control", "low_event_clustering")
LABELS = ("原最佳策略", "低事件频率对照", "低连续聚集修正")


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


def main(plan_sha):
    assert digest(OUT / "PLAN.json") == plan_sha
    done = load(OUT / "COMPLETION.json")
    independent = load(OUT / "INDEPENDENT_OUTCOME.json")
    protected = load(OUT / "PROTECTED_AFTER.json")
    assert done["plan_sha256"] == plan_sha
    assert independent["passed"] and independent["execution_paths_verified"] == 9
    assert independent["total_monthly_orders_verified"] == 366
    assert independent["plan_sha256"] == plan_sha
    assert protected["passed"] and protected["verified"] == 21444
    state = load(ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    assert state["completed"][-1]["directory"] == OUT.name
    assert state["completed"][-1]["completion"]["sha256"] == digest(OUT / "COMPLETION.json")
    results = load(OUT / "RESULTS.json")
    orders = load(OUT / "PRETRAIN_ORDERS.json")
    audit = load(OUT / "FEATURE_AUDIT.json")
    tests = load(OUT / "TESTS.json")
    nav = {
        name: {row["date"]: row["equity"]
               for row in load(OUT / (name + "_same_close_DAILY.json"))}
        for name in NAMES
    }
    changed = {}
    for name in NAMES[1:]:
        changed[name] = []
        for old, new in zip(orders[NAMES[0]], orders[name]):
            assert old["date"] == new["date"] and old["end"] == new["end"]
            if old["symbols"] == new["symbols"]:
                continue
            start, end = old["date"], old["end"]
            old_return = nav[NAMES[0]][end] / nav[NAMES[0]][start] - 1
            new_return = nav[name][end] / nav[name][start] - 1
            changed[name].append(dict(
                signal_date=start, end=end, original=old["symbols"],
                challenger=new["symbols"],
                holding_set_changed=set(old["symbols"]) != set(new["symbols"]),
                baseline_return=old_return, challenger_return=new_return,
                return_difference_pp=100 * (new_return - old_return),
            ))
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), round=148,
        plan=ref(OUT / "PLAN.json"), completion=ref(OUT / "COMPLETION.json"),
        independent=ref(OUT / "INDEPENDENT_OUTCOME.json"),
        results=ref(OUT / "RESULTS.json"), orders=ref(OUT / "PRETRAIN_ORDERS.json"),
        primary=done["primary"], target_reached=done["target_reached"],
        changed_order_months=changed, completed_rounds=len(state["completed"]),
        total_trials=sum(row["trials"] for row in state["completed"]),
        feature_availability=audit["reasons"],
        inherited_limitations=[
            "Repeatedly reused history is not pristine out of sample.",
            "Historical estimated universe and retrieved adjusted-price vintage retained.",
            "Adjusted overnight includes dividend effects; no investor identity inference.",
            "Frequency control does not fully isolate covariance.",
            "Same-close zero-cost differs from executable next-open cost results.",
        ],
    )
    save_new(OUT / "SUMMARY.json", summary)
    lines = [
        "# 第148轮：隔夜上涨后日内回落的连续聚集",
        "",
        "区间：2016-08-17—2026-09-24。主表为同收盘价、零成本历史回测。",
        "长期动量、原始股票池及月内固定持仓保持不变。新特征使用跳过最近42日的210个交易日。",
        "",
        "|候选|全期年化|Sharpe|最大回撤|次日开盘含成本年化|",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in zip(NAMES, LABELS):
        met = results[name]["same_close"]["metrics"]
        opened = results[name]["next_open_cost"]["metrics"]
        lines.append(f"|{label}|{met['cagr']:.2%}|{met['sharpe']:.3f}|"
                     f"{met['max_drawdown']:.2%}|{opened['cagr']:.2%}|")
    lines += [
        "",
        f"预声明筛选保留：{done['primary']}；聚集修正获准替换：{done['selection']['admitted']}。",
        "只有前88个月用于选择，后34个月仅作诊断；不会因后段结果改变选定规则。",
        "",
        "|修正|顺序变化月数|实际持仓集合变化月数|",
        "|---|---:|---:|",
    ]
    for name, label in zip(NAMES[1:], LABELS[1:]):
        rows = changed[name]
        lines.append(f"|{label}|{len(rows)}|{sum(x['holding_set_changed'] for x in rows)}|")
    lines += [
        "",
        "以下为聚集修正相对原策略的逐月变化；同一持仓的排列变化不代表调仓。",
        "",
        "|选股日|原组合|聚集修正组合|集合改变|随后月收益差|",
        "|---|---|---|---|---:|",
    ]
    for row in changed[NAMES[2]]:
        lines.append(f"|{row['signal_date']}|{', '.join(row['original'])}|"
                     f"{', '.join(row['challenger'])}|"
                     f"{'是' if row['holding_set_changed'] else '仅顺序'}|"
                     f"{row['return_difference_pp']:+.3f}个百分点|")
    lines += [
        "",
        f"核验：{tests['test_count']}项合成测试；独立核对366条月度选股、9条执行路径；"
        "21,444个保护文件一致。未修改冻结观察账户。",
        f"累计完成{summary['completed_rounds']}轮、{summary['total_trials']}次尝试。",
        "",
        "这是反复使用历史区间的研究结果，不是纯样本外或实盘收益。"
        "复权隔夜变化包含股息调整，不能解释为特定投资者买卖；事件频率对照也不能完全隔离聚集效应。"
        "次日开盘结果按每边0.05%成本。",
    ]
    with (OUT / "SUMMARY.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    path = ROOT / "migration/RESEARCH_RESUME.json"
    raw = path.read_bytes()
    prior_sha = hashlib.sha256(raw).hexdigest()
    checkpoint = ROOT / "migration" / ("round148_resume_" + prior_sha + ".json")
    with checkpoint.open("xb") as stream:
        stream.write(raw)
    resume = json.loads(raw)
    resume.update(
        round=148, current_experiment=OUT.name, plan_sha256=plan_sha,
        current_experiment_artifacts={name: ref(OUT / name) for name in (
            "PLAN.json", "COMPLETION.json", "INDEPENDENT_PREPARE.json",
            "INDEPENDENT_OUTCOME.json", "PROTECTED_AFTER.json", "SUMMARY.json")},
        formal_round148_completed=True,
        formal_round148_completion=ref(OUT / "COMPLETION.json"),
        completed_research_rounds=summary["completed_rounds"],
        strategy_evaluations=summary["total_trials"],
        target_70_reached=done["target_reached"],
        latest_protected_check=ref(OUT / "PROTECTED_AFTER.json"),
        next_research_status="round148_completed_independently_verified",
        next_step="Review completed evidence before defining a distinct prospective experiment; preserve frozen observers.",
        next_candidate_review=None, next_mechanism_registered=False,
        previous_goal_turn_classification="progress: completed148 with independent9-path verification",
        remaining_formal_archive_check="None for148;21444 protected files verified",
        resume_update_checkpoint=str(checkpoint.relative_to(ROOT)),
        updated_utc=summary["created_utc"],
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staging = Path(stream.name)
        stream.write((json.dumps(resume, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
    try:
        assert digest(path) == prior_sha, "Concurrent resume update"
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()
    print(json.dumps(dict(summary=ref(OUT / "SUMMARY.json"),
                          report=ref(OUT / "SUMMARY.md"),
                          target_reached=done["target_reached"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha", required=True)
    main(parser.parse_args().plan_sha)
