"""Record source-only expansion without registering or executing a strategy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_revenue_expansion156_checkpoint'
REVIEW_PDF_PACKAGES=tuple(Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')/name
                          for name in ('pypdf','pypdfium2','pypdfium2_raw'))


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def check(ok,why):
    if not ok: raise ValueError(why)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--aapl-review-sha',required=True)
    parser.add_argument('--abbv-review-sha',required=True)
    args=parser.parse_args()
    observed={}
    def read(path,expected=None):
        p=Path(path)
        p=p if p.is_absolute() else ROOT/p
        check(p.is_relative_to(ROOT) and p.resolve()==p,'Nonlocal source '+str(p))
        raw=p.read_bytes(); h=digest(raw)
        check(expected is None or h==expected,'Changed source '+str(p))
        rel=str(p.relative_to(ROOT)); observed[rel]=h
        return json.loads(raw)
    fixed={
        'outputs/monthly_endpoint155/COMPLETION.json':'a0ebeb12d7bca362b2995c377382cdd40449c13f4b2b4bbbb0a98b5a4717dc06',
        'outputs/monthly_source_readiness156/SUMMARY.json':'cb4066ee20c5a94803b1e7137783eb1d133e45d444bcd48d55c93dcb577571e8',
        'outputs/monthly_aapl_revenue156/v4/PLAN.json':'3c841c5d80659e5f1911035e027518e51a788f7b7e5afba529d15a0324e3be6b',
        'outputs/monthly_aapl_revenue156/v4/RECEIPT.json':'c6d4d1578b02fac01bec8f566dec0e7795c6b29baa718a745d5bf4b61fa3733c',
        'outputs/monthly_aapl_revenue156/v4/DATASET.json':'24eb0618ea6718e074ae8c5eafd7ee1d3abcfe970b895ee256731ff527936e20',
        'outputs/monthly_abbv_revenue156/EXTRACTION_PROFILE_V2.json':'1262433e45dfb805631b0b24bfd47915536dcd57aec7443caab3ec4ad2e51723',
        'outputs/monthly_abbv_revenue156/EXTRACTION_V2.json':'5c81e1e23eefda6118739b938cd06074119ad9083f827e499cb5567ffe8f93fd',
        'work/after155_multi_issuer_chain_review.json':'1aabe00960b8c09c3794ad2d0bc9de2d03159fadaddb7acb211a07ea32232e8c',
        'work/after155_multi_issuer_chain_policy.json':'25820e3cdb6273a9bedadfc8253a415bc2f6515b3d61e9d59ab7a465cdc0acc0',
        'work/after155_revenue_pilot_source_plan.json':'5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10',
        'work/after155_revenue_experiment_path.json':'a84440879cf3db80782190de8eb7b43c402c25bdb88fa07e1d9b3401bf092806',
        'work/after155_aapl_accounting_edge_plan.json':'c198f97a24770d339419dd4daf491bbda86de07ffb5b990d78081478fb55153d',
        'work/after155_aapl_revenue_independent.json':args.aapl_review_sha,
        'work/after155_abbv_revenue_independent.json':args.abbv_review_sha,
    }
    check(not OUT.exists(),'Write-once checkpoint exists')
    docs={p:read(p,h) for p,h in fixed.items()}
    reviews={p:docs[p] for p in fixed if p.endswith('_independent.json') or p.endswith('chain_review.json')}
    for path,review in reviews.items():
        check(review.get('passed') is True,'Review not passed '+path)
        check(bool(review.get('input_hashes')),'Review lacks source bindings '+path)
        for p,h in review['input_hashes'].items():
            local=Path(p); local=local if local.is_absolute() else ROOT/local
            # The independent PDF reader binds its installed source files as well
            # as research inputs. Verify those exact local bytes; never relocate
            # or substitute the runtime dependency or permit other external data.
            installed_reader=(path=='work/after155_aapl_revenue_independent.json'
                              and any(local.is_relative_to(root) for root in REVIEW_PDF_PACKAGES)
                              and local.suffix in ('.py','.json','.dylib'))
            check((local.is_relative_to(ROOT) or installed_reader)
                  and local.resolve()==local,'Nonlocal review binding: '+str(local))
            actual=digest(local.read_bytes())
            check(actual==h,'Review source changed '+p)
            observed[str(local.relative_to(ROOT)) if local.is_relative_to(ROOT) else str(local)]=h
    a=docs['outputs/monthly_aapl_revenue156/v4/RECEIPT.json']
    b=docs['outputs/monthly_abbv_revenue156/EXTRACTION_V2.json']
    check(a['original_reports_extracted']==10 and a['monetary_cells_extracted']==20 and a['held_reports']==0,'AAPL source count differs')
    check(b['source_count']==13 and b['exact_report_pairs']==13 and b['exact_amount_cells']==26 and b['failed_sources']==0,'ABBV source count differs')
    for key in ('dataset','tests','source_checks'):
        r=a[key];read(r['path'],r['sha256'])
    state=read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    check(state['current']=='monthly_endpoint155' and len(state['completed'])==155
          and sum(x['trials'] for x in state['completed'])==978 and state['registered_queue']==[], 'Research state advanced')
    resume_path=ROOT/'migration/RESEARCH_RESUME.json'
    old=resume_path.read_bytes(); resume=json.loads(old)
    check(resume['round']==155 and resume['strategy_evaluations']==978,'Resume advanced')
    prior='migration/after155_revenue_expansion_before_'+digest(old)+'.json'
    record={
        'stage':'source_expansion_after155_not_strategy_round156',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'goal_turn_classification':'progress',
        'new_strategy_trials':0,'new_factors_models_orders_returns':0,
        'completed_rounds':155,'total_trials':978,
        'aapl':{'original_reports':10,'current_comparative_cells':20,'source_parser_holds':0,
                'source_review_passed':True,'fiscal_53_week_years':[2017,2023],
                'unknown_comparative_start_fiscal_years':[2016],
                'iso_currency_certified':False,'accounting_chains_admitted':False},
        'abbv':{'original_reports':13,'current_comparative_cells':26,'source_parser_holds':0,
                'source_review_passed':True,'iso_currency_certified':False,
                'accounting_chains_admitted':False,
                'early_clock_note':'FY2013 original-release date verified; producer narrow calendar leaves computed availability null. Use separately bound existing2008-2035 calendar in future chain preparation; preserve original output.'},
        'new_eligibility_implementation':{'independent_review_passed':True,'actual_new_issuer_chains_evaluated':0},
        'prior_source_checkpoint':{'path':'outputs/monthly_source_readiness156/SUMMARY.json','sha256':fixed['outputs/monthly_source_readiness156/SUMMARY.json']},
        'unchanged_best':state['best_training_selected_candidate'],
        'target_70_reached':False,'observer_changes':False,
        'next':[
            'Independently review AAPL/ABBV adjacent-year accounting contexts and bind explicit reported-dollar assumption; do not promote source-cell verification into accounting-chain approval.',
            'Build complete original122x49 chain grid with latest-before-validity, timed edges and actual frozen decision cutoffs; retain all missing/held rows.',
            'Evaluate preregistered source-coverage gate. If passed, register fixed four-arm revenue-persistence pilot before financial factors, ranks, orders or returns; first88 selection and later34 diagnostics remain separate.'
        ],
        'prior_resume':{'path':prior,'sha256':digest(old)},
        'input_hashes':observed,
    }
    for p,h in observed.items():check(digest((ROOT/p).read_bytes())==h,'Concurrent source edit '+p)
    check(resume_path.read_bytes()==old,'Concurrent resume edit')
    OUT.mkdir()
    raw=encoded(record)
    with (OUT/'SUMMARY.json').open('xb') as f:f.write(raw)
    note=('本轮新增AAPL十份、ABBV十三份原始年报的收入提取，共46个当期及上年对照值，已通过独立原稿核对。'
          '早期失败的解析版本保留。Apple的52/53周财年、缺失的FY2015起始日期、美元符号与ISO币种的区别均显式记录。\n\n'
          '跨公司数据链检查实现已通过独立测试；这不等于实际四年收入链准入。下一步核对逐年会计口径、补绑定早期发布时间，'
          '然后在完整候选网格上运行已预声明的覆盖门槛。通过后才登记并测试收入持续性及对应增长对照。\n\n'
          '本轮无新增策略回测。仍为155轮、978次测试；2016-08-17至2026-09-24同收盘零成本最佳年化67.67%，'
          'Sharpe约1.735，最大回撤31.26%，未达到70%。冻结观察账户未改动。\n')
    with (OUT/'SUMMARY.md').open('x') as f:f.write(note)
    archive=ROOT/prior
    if archive.exists():check(archive.read_bytes()==old,'Prior snapshot conflict')
    else:
        with archive.open('xb') as f:f.write(old)
    ref={'path':str((OUT/'SUMMARY.json').relative_to(ROOT)),'sha256':digest(raw)}
    resume.update(last_goal_turn_classification='progress',after155_revenue_expansion_checkpoint=ref,
                  next_source_preparation={'status':'aapl_abbv_cells_reviewed_accounting_chains_pending',
                                           'checkpoint':ref,'new_trials':0,
                                           'next_steps':record['next']})
    temp=resume_path.with_name('RESEARCH_RESUME.revenue_expansion.tmp')
    with temp.open('xb') as f:f.write(encoded(resume))
    os.replace(temp,resume_path)
    print(json.dumps({'checkpoint':ref,'new_reports':23,'new_source_cells':46,'new_strategy_trials':0}))


if __name__=='__main__':main()
