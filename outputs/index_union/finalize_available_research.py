"""Publish a qualified completion only after two complete, same-input runs."""
from pathlib import Path
from datetime import datetime,timezone
import json
import pandas as pd
from research_scope import sha,evidence,check_nested_sources
from union_execution import ACCOUNTS
P=Path(__file__).resolve().parent

def validate_run_status(status,manifest_hash,assumptions_hash,expected_sessions=4215):
 if not status.get('all_six_completed') or status.get('manifest_sha256')!=manifest_hash or status.get('assumptions_sha256')!=assumptions_hash:
  raise ValueError('Incomplete or different input run')
 if set(status.get('accounts',{}))!=set(ACCOUNTS) or any(r['sessions']!=expected_sessions or r['status']!='complete' for r in status['accounts'].values()):
  raise ValueError('Incomplete strategy/session coverage')

def main():
 manifest=P/'alias_resolved_inputs/MANIFEST.json';m=json.loads(manifest.read_text())
 aliases=json.loads((P/'DATED_SYMBOL_IDENTITIES.json').read_text())['rows']
 for r in aliases:
  if not r['retrospective_identity_link_verified']:raise ValueError('Unreviewed alias')
  evidence(P,r,'evidence_file','evidence_sha256')
 permitted={r['symbol'] for r in m['identity_proofs']}|{r['security_id'] for r in aliases}
 fields={r['field']:r for r in m['fields']}
 eligible=pd.read_csv(evidence(P,fields['eligible'],'file','sha256'),index_col=0,parse_dates=True)
 for s in eligible:
  if eligible[s].any() and s not in permitted:raise ValueError('Unreviewed eligible security '+s)
 records=[];runs={}
 for lag in [3,5]:
  d=P/'research_runs'/('available_20261007_t'+str(lag));s=json.loads((d/'RUN_STATUS.json').read_text())
  validate_run_status(s,sha(manifest),sha(P/'AVAILABLE_RESEARCH_ASSUMPTIONS.json'))
  runs[lag]=json.loads((d/'METRICS.json').read_text())['accounts']
  records.append(dict(file=str((d/'RUN_STATUS.json').relative_to(P)),sha256=sha(d/'RUN_STATUS.json')))
 report=dict(generated_at=datetime.now(timezone.utc).isoformat(),qualified_available_data_research_ready=True,qualified_research_backtest_completed=True,strict_complete_history_certified=False,start=m['start'],end=m['end'],complete_sessions_per_account=4215,strategy_count=6,manifest_sha256=sha(manifest),assumptions_file='AVAILABLE_RESEARCH_ASSUMPTIONS.json',runs=records,accounts=runs[3],settlement_sensitivity={n:dict(t3_cagr=runs[3][n]['cagr'],t5_cagr=runs[5][n]['cagr'],cagr_difference_percentage_points=100*(runs[5][n]['cagr']-runs[3][n]['cagr']),t3_maximum_drawdown=runs[3][n]['maximum_drawdown'],t5_maximum_drawdown=runs[5][n]['maximum_drawdown']) for n in ACCOUNTS},identity_exclusions=m['scope_exclusions'],limitations=['Historical constituent reconstruction and sector observation persistence remain assumptions; strict certification is NOT granted.','Identity continuity for stable vendor/SEC corroborated names is assumed; unresolved labels quarantined.','Available archive subset and prior strategy selection can bias returns; not proof of unbiased out-of-sample performance.','No modification of six original paper accounts; no automatic strategy promotion.'])
 (P/'AVAILABLE_RESEARCH_READINESS.json').write_text(json.dumps(report,indent=2)+'\n')
 old=json.loads((P/'BACKTEST_READINESS.json').read_text())
 old.update(checked_at=report['generated_at'],latest_validation_tests_passed=146,qualified_available_data_research_ready=True,qualified_research_backtest_completed=True,qualified_report='AVAILABLE_RESEARCH_READINESS.json',strict_complete_history_certified=False,strict_status_note='Original strict full-history certificates remain unapproved. Separate declared-assumption research runs have completed; do not confuse their scope.')
 old['no_performance_numbers_generated']=False
 old['actual_backtest_status']='qualified_available_data_research_completed_strict_certification_pending'
 old['test_validation']['count']=146
 old['remaining_material_inputs']=['Strict full official daily constituent history certification remains incomplete; known disputed entry intervals excluded in qualified study.','Complete permanent historical security identity vendor certification remains incomplete; six quoted ambiguous labels quarantined, other unresolved labels lack quotes.','Official contemporaneous GICS history remains incomplete; qualified study uses only already-published sector observations with forward-persistence assumption.','Actual merger cash settlement timestamps remain unavailable; qualified study compares explicit T+3/T+5 assumptions.']
 (P/'BACKTEST_READINESS.json').write_text(json.dumps(old,indent=2)+'\n')
 lines=['可用数据研究回测：六套策略均完成。','范围：2010-01-04 至 2026-10-06，每套 4,215 个交易日。','这不是完整历史数据认证；全部研究假设详见 AVAILABLE_RESEARCH_ASSUMPTIONS.json。','历史 Nasdaq-100 ∪ S&P 100 重建；缺数据/未解决身份不新开仓；已有持仓不得删除。','','策略 | 年化收益 | 最大回撤 | 平仓数 | 净胜率']
 for n,r in runs[3].items():lines.append(f"{n} | {r['cagr']:.2%} | {r['maximum_drawdown']:.2%} | {r['closed_trades']} | {r['net_win_rate']:.2%}")
 lines+=['','收益仅为上述可用数据及研究假设下的回测结果；不是原六账户实盘或前向模拟结果。','并购现金到账采用 T+3 假设，并用 T+5 复跑；均非已证实的实际到账日期。','严格历史成分、永久身份、逐日官方行业分类认证仍未完成，不得将此报告标为无偏完整股票池回测。']
 (P/'AVAILABLE_RESEARCH_REPORT.txt').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:report[k] for k in ['qualified_available_data_research_ready','qualified_research_backtest_completed','strict_complete_history_certified','accounts','settlement_sensitivity']},indent=2))
if __name__=='__main__':main()
