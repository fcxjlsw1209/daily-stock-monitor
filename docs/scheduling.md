# 当前三账户调度

America/Los_Angeles 工作日13:20收盘复盘并冻结下一交易日计划，20:20补跑。固定在同一仓库和运行目录执行：

```sh
/path/to/venv/bin/python /absolute/repo/outputs/paired_stock_monitor/run_pair.py --phase auto
```

当前入口使用Unix文件锁，适用于macOS/Linux。程序串行运行三账户、只下载一次行情，检查快照一致性。报告读取 `outputs/paired_stock_monitor/comparison.json` 和 `COMPARISON.md`；失败状态为 `ERROR_NO_NEW_RECOMMENDATIONS`。晚间无变化保持安静，收盘交易日汇报三个独立账户及次日计划。定时器不随代码安装。

下文只适用于保留的旧单账户CLI，不适用于三账户入口。

---

# 调度与通知

建议 America/Los_Angeles 时区的工作日06:20执行盘前任务，13:20执行收盘任务。程序根据NYSE日历判断休市和已完成交易日，并处理夏令时；提前收盘日在下午任务时统一结算。

```sh
/path/to/venv/bin/stock-monitor --home /absolute/paper-account run --phase morning
/path/to/venv/bin/stock-monitor --home /absolute/paper-account run --phase close
```

调度器需要串行执行同一账户，避免同时运行多个实例。保留足够下载时间；盘前计划在开盘前不足一分钟时会被拒绝保存。建议监控退出码及输出的ERROR_NO_NEW_PLAN，失败时不要把旧报告当成新信号。

本地调度依赖电脑开机、进程运行和网络。这个仓库不配置系统cron、GitHub定时任务或任何推送渠道。GitHub Actions定时任务可能延迟，也不会天然持久保存本地账本，不建议直接把它当作准时成交提醒。

## 可交给Codex定时任务的提示词

> 在指定账户的运行目录，执行上述命令，读取JSON输出。盘前报告股票、每只预算、分数、卖出原因和持有项；无信号明确不开新仓。收盘报告模拟平仓结果、胜率、累计收益、未实现盈亏和回撤。把回测、事前模拟和真实成交分开。休市日不发交易建议。失败或迟到时说明原因，不捏造行情、不补记交易、不修改固定规则或清空账本。

初始化已有账户只做一次；后续调用run或status。不要在每次调度中重新初始化账户。
