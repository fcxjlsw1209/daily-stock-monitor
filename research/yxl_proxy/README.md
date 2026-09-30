# YXL 近似复刻与动能因子增量研究

2026-09-29 的离线研究。不是 YXL 原始信号回测，不连接券商，不修改三个冻结模拟账户，也不安装自动化。

## 内容

- `backtest.py` / `test_backtest.py`：14种预先限定的YXL近似规则、日线单标的模拟、前缀因果性及成交核算测试。
- `PROTOCOL.md`：公开指标参数、未知公式、所有近似假设。
- `report.py`：根据本地计算结果生成HTML报告。
- `momentum-increment/study.py`：把MACD柱两日改善量÷既有ATR14，以10%权重加入原策略排序，复用冻结的纯计算模块。
- `momentum-increment/weight_control.py`：事后追加常数排名对照，区分权重重分配与新信号的贡献。
- `momentum-increment/test_study.py`：排名权重、未来标签边界、持仓MAE和分块抽样测试。

## 历史结果摘要

来源为本地存量Yahoo数据；当前大市值股票名单有幸存者偏差，历史已经多次探索，不是独立样本外。

近似复刻主规格：QQQ 2000-01-03至2026-09-25，年化1.06%、最大日终回撤46.19%；买入持有8.10%/82.96%。SPY 2011-01-03至2026-09-18，主规格3.87%/16.72%，买入持有12.82%/30.36%。单边5bp，分红留现金，现金零利息。不能据此评价原指标的真实效果。

增量研究2019-01-02至2026-09-18，双边10bp，原基准年化11.39%/回撤13.30%；动能版12.55%/11.55%；压力影子13.07%/12.04%；抗跌影子12.17%/12.78%。这些都是同区间离线模拟，不是前向账本。常数权重对照年化11.88%；新增动能的直接候选内预测力接近零，分块抽样区间跨零，暂不推广。

原基准10/20bp结果准确重现；8项近似模型测试、4项增量专项测试及真实数据前缀验证通过。未上传行情、账本、逐笔记录或带本地路径的数据清单。

## 环境与合成测试

使用仓库 `pyproject.toml` 的固定依赖，或已有匹配环境。在仓库根目录运行：

```sh
python -B -m unittest discover -s research/yxl_proxy -p 'test_*.py' -v
python -B -m unittest discover -s research/yxl_proxy/momentum-increment -p 'test_*.py' -v
```

不设置环境变量时，增量测试使用本仓库 `outputs/` 下的纯计算模块，不需要下载行情。

## 完整研究重跑

`STOCK_RESEARCH_OUTPUTS` 指向**已有研究输出树的 outputs 目录**，不是输出结果存放位置，也不是单个数据库路径。默认是本仓库 `outputs/`。该目录需要保留匹配的纯计算模块和未发布的数据：

- `stock_database/stocks.sqlite`：原100股数据库及OHLC、复权收盘、分红。
- `qqq_ma200_research/data/QQQ.csv`、`overnight_research/spy_source.csv`、`swing_research/qqq_source.csv`。
- `overnight_research/intraday_validation/download_manifest.json` 及 `raw/*_daily.csv`（行业ETF等）。
- 原 `daily_stock_monitor`、`pressure_shadow_monitor`、`resilience_shadow_monitor`、`factor_research`、`swing_research` 的匹配纯计算代码与基准配置。
- 增量复现检查需要 `strategy_selection/metrics.csv`，用于核对旧 `broad_guarded10` 结果。

全新克隆仅含代码，不具备这些数据，不能直接重现完整历史绩效。脚本不会联网下载或初始化模拟账户；缺失数据会报错。请使用可信、匹配的现有研究树，因为增量脚本会从指定目录导入计算模块。

```sh
export STOCK_RESEARCH_OUTPUTS=/absolute/path/to/research/outputs
python -B research/yxl_proxy/backtest.py
python -B research/yxl_proxy/report.py
python -B research/yxl_proxy/momentum-increment/study.py
python -B research/yxl_proxy/momentum-increment/weight_control.py
python -B research/yxl_proxy/momentum-increment/report.py
```

运行结果写到脚本所在研究目录，已通过 `.gitignore` 排除。不会写入 `STOCK_RESEARCH_OUTPUTS`，不会调用 `monitor.py` 或 `run_pair.py`。重复运行会覆盖本地研究产物；保留旧结果时先另存研究目录。

详细假设见两个 `PROTOCOL.md`；常数对照属于结果后的归因诊断，见 `momentum-increment/ATTRIBUTION_CONTROL.md`。
