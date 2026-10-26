"""Checkpoint source-only progress after completed round155; never a new trial."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_source_readiness156'
PINS = {
    'outputs/monthly_endpoint155/COMPLETION.json': 'a0ebeb12d7bca362b2995c377382cdd40449c13f4b2b4bbbb0a98b5a4717dc06',
    'outputs/monthly_endpoint155/SUMMARY.json': '26dd846191d464ff9a84415f26e858bbeb2a031f6ec9c0ec49912055c5a59f79',
    'outputs/monthly_event_patch156/ADDED_EVENTS.json': '1d107c783d93a78ced29572383c55f55c50d24108014aee417dc3772fef2a5e3',
    'outputs/monthly_event_overlay_scope156/AUDIT.json': 'd834d8e6a29c5bba4574eb92015d5da992cd12f7f54b73c1d8b0ed1d37ec88cb',
    'outputs/monthly_consumed_clock156/SPEC.json': '7c07670c8d264748e46366c5e0781b14c89f6ee88c8704ad0a9f0bc55d2d5bb6',
    'outputs/monthly_msft_event_evidence156/SPEC.json': 'a24ec71fa8a5773350cab49b65cc14dcfca03f6137f8f9802caa6d9c708dfc98',
    'outputs/monthly_revenue_coverage156/SUMMARY.json': '72480ffe552c4c44cfb1ee3e0786f60623cbc5597f44fc6e85b42f5dfb5f6c8b',
    'outputs/monthly_revenue_coverage156/VALIDATION.json': 'e46f3557c6d5a482ee31578ac499808dd3de1faf81c6fefccf5ad04860bd56e4',
    'outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json': '0f5aa563ac7eac950b9594740d14111dfa8aac1608f5e73558d39baee6066aef',
    'work/after155_revenue_extended_review.json': 'f18d530b9cea67d0fce5d82f8910b8a7a20264813e058e296058c8e783a687bd',
    'work/after155_revenue_chain_policy_review.json': '8564d0d052e4a7cfb89bd73c553b46c6ea831afacd384a5d69fe63c52732ce36',
    'outputs/monthly_event_gap156_cat_hsbc/PROPOSAL.json': 'f3819ca884fb7a35fcd8cb7265e2e6224a93e4c20bd3ada1ee917bf3a6f15137',
    'outputs/monthly_event_gap156_crm_sap_tsla/SOURCE_PROPOSAL.json': '071d0ae9fff44191859b9ba8d06659bc03382c6add89748f289b3ffbd0916978',
    'outputs/monthly_event_patch156_batch2/INTEGRATION.json': 'e071785472846455da33773d9d114d012e2ed2876040b03dffb0b49a64a5d53e',
    'outputs/monthly_event_patch156_batch3/INTEGRATION.json': 'fe552f15b4fd710d56345bb41e461588ec8d96dc5b4721976cb84aeb8fac579a',
    'outputs/monthly_msft_revenue_chain156/SPEC.json': 'ffa2849c2f3a4a2ded6bc4e3004a64843b45e80cda286d3ca0f74a3e773ea098',
    'outputs/monthly_msft_revenue_chain156/MANIFEST.json': '4d84d4c9957551f931158789218ea44d817bbd961323156f1a28aa0249e92008',
    'outputs/monthly_event_tsla_wire156/SOURCE_PROPOSAL.json': '976d24b389f1b43a8e307edd30a272ea39cf45f023b0585a9a5a26fc1525e408',
}
REVIEW_FILES = {
    'work/after155_event_source_independent_review.json': '4cf546abd7e9be42c2bd7e54114c6140c7c422dc8b9d388f09ae8bcf4e177d00',
    'work/after155_crm_sap_independent_review.json': '02c138c6d3f81f701c40109b1f85fdbe74d8558ad93449a47eca18a5ac104fc4',
    'work/after155_hsbc_independent_review.json': '5e92978a2d3a5165eb1152b84397a9c99f9a4c53660b345dc9f141b1efe20316',
    'work/after155_msft_basis_independent.json': 'd848815a6d7093fed8c796f81e41015a92f104308300113a35ec0082e99b1f4f',
    'work/after155_revenue_chain_implementation_review_v2.json': 'baf5522432470e852ba3ac335b65d55f3974ae8f5b841bc7cd22444f8c1d5374',
    'work/after155_cat_wire_independent_review.json': '0dbbfb8357716ef6b142224fd82d529a4685a31e84e159ad623c7e39445a9b30',
}
NEXT = ('Extend original annual-revenue extraction and comparable four-year-chain verification beyond MSFT; '
        'finish bounded event-interval enumeration and resolve the TSLA2019Q3 original-response gap. Source metadata '
        'coverage and MSFT-only eligibility are not broad factor admission. No new factor/model/return calculations until a distinct '
        'hypothesis and source/missing-data contract are registered. Preserve completed155 and observers.')


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_once(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--chain-review-sha',required=True)
    args=parser.parse_args()
    observed = {}
    def read(rel, expected=None):
        path = ROOT / rel
        check(path.resolve() == path and path.is_file(), 'Missing/nonlocal input '+rel)
        raw = path.read_bytes(); actual = digest(raw)
        check(expected is None or expected == actual, 'Changed input '+rel)
        observed[rel] = actual
        return json.loads(raw)
    def ref(rel):
        return {'path': rel, 'sha256': observed[rel]}
    def unchanged():
        for rel, h in observed.items():
            check(digest((ROOT/rel).read_bytes()) == h, 'Concurrent change '+rel)

    check(not OUT.exists(), 'Source checkpoint already exists; do not overwrite')
    values = {p: read(p, h) for p, h in PINS.items()}
    reviews = {p: read(p, h) for p, h in REVIEW_FILES.items()}
    chain_review=read('work/after155_msft_chain_output_independent.json',args.chain_review_sha)
    check(chain_review.get('passed') is True, 'Actual chain output independent review did not pass')
    for p,h in chain_review['input_hashes'].items():
        check(digest((ROOT/p).read_bytes())==h, 'Independent output-review input changed: '+p)
    # Each review is preserved as evidence, not reduced to a guessed success flag.
    for directory in ('monthly_msft_event_evidence156', 'monthly_consumed_clock156','monthly_msft_revenue_chain156'):
        base = 'outputs/'+directory+'/'
        manifest = read(base+'MANIFEST.json')
        for filename, h in manifest['outputs'].items():
            read(base+filename, h)
    clock = read('outputs/monthly_consumed_clock156/SUMMARY.json')
    check(clock['passed'] and clock['checked'] == 1681 and not clock['mismatches'], 'Clock audit incomplete')
    chain=read('outputs/monthly_msft_revenue_chain156/SUMMARY.json')
    for name in ('SPEC.json','MANIFEST.json','ROWS.json','REPORTS.json','EDGES.json'):
        p='outputs/monthly_msft_revenue_chain156/'+name
        check(chain_review['input_hashes'].get(p)==digest((ROOT/p).read_bytes()),
              'Independent review missing output binding: '+p)
    check(chain['rows']==5978 and chain['msft_status_counts']=={
        'missing_required_source':24,'blocked_known_basis_break':36,
        'checked_reported_total_chain_with_declared_source_scope':62}, 'Unexpected source-chain result')
    increments=[]
    for directory in ('monthly_event_patch156_batch2','monthly_event_patch156_batch3'):
        integration=values['outputs/'+directory+'/INTEGRATION.json']
        for field in ('added_events','manifest','consumption_impact'):
            b=integration[field];read(b['path'],b['sha256'])
        increments.append({'integration':ref('outputs/'+directory+'/INTEGRATION.json'),
                           'combined_events':integration['combined_count'],
                           'impact':read(integration['consumption_impact']['path'])})
    check(increments[-1]['combined_events']==3781 and
          increments[-1]['impact']['consumed_unique_after']==1687, 'Unexpected event increment')
    state_path = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    state = read(state_path)
    check(state['current'] == 'monthly_endpoint155' and len(state['completed']) == 155
          and sum(x['trials'] for x in state['completed']) == 978
          and state['registered_queue'] == [], 'Research state changed')
    resume_path = ROOT/'migration/RESEARCH_RESUME.json'
    old = resume_path.read_bytes(); resume = json.loads(old)
    check(resume['round'] == 155 and resume['strategy_evaluations'] == 978, 'Resume changed')
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'source_preparation_after155_not_registered_round156',
        'goal_turn_classification': 'progress', 'completed_rounds':155, 'total_trials':978,
        'new_strategy_trials':0, 'new_labels_fits_orders_returns':0,
        'new_mechanism_registered':False, 'target_70_reached':False,
        'calendar_completeness_admitted':False, 'revenue_chain_admitted_broadly':False,
        'msft_only_chain_independently_verified':True, 'msft_source_chain_counts':chain['msft_status_counts'],
        'msft_source_chain_review':ref('work/after155_msft_chain_output_independent.json'),
        'factor_admitted':False, 'source_refs':{p:ref(p) for p in observed},
        'checkpoint_code':{'path':str(Path(__file__).relative_to(ROOT)), 'sha256':digest(Path(__file__).read_bytes())},
        'source_review_records':{p:ref(p) for p in reviews},
        'event_overlay_counts':values['outputs/monthly_event_overlay_scope156/AUDIT.json']['counts'],
        'consumed_clock_counts':{'events':clock['checked'],'mismatches':len(clock['mismatches']),
                                 'synthetic_checks':clock['synthetic_checks']},
        'revenue_source_counts':values['outputs/monthly_revenue_coverage156/SUMMARY.json']['counts'],
        'newest_event_increments':increments,
        'pending':[
            'Reconcile newest incremental event metadata with bounded whole-interval enumeration; single-source admission is not complete event-calendar coverage.',
            'TSLA2019Q3 publication evidence remains unresolved; direct issuer/SEC access returned403. Do not retry blocked sources today. Preserve receipts.',
            'All99 event intervals still require bounded enumeration/completeness checks; label continuity is not complete-news proof.',
            'MSFT-only source chains are independently reviewed:62 usable,36 blocked by accounting transition,24 missing original reports. Other issuers still require original-cell and comparable-chain validation.',
            'Annual report presence and companyfacts filing metadata do not prove comparable revenue values.',
            'Historical128/133 complete source/PLAN artifacts remain missing locally, so mechanism novelty is only bounded, not globally established.'
        ],
        'next_step':NEXT,
        'prior_resume':{'path':'migration/after155_source_resume_'+digest(old)+'.json','sha256':digest(old)},
    }
    summary = '''# 第155轮后的来源准备进度

本次只处理来源和时间信息，没有新增策略试验；累计仍为155轮、978个测试。最佳已完成候选仍为2016-08-17至2026-09-24同收盘零成本年化67.67%、Sharpe约1.735，未达到70%。

- 三批独立来源增量共补入10条：ABT三条、AVGO一条、CRM一条、SAP一条、HSBC两条、CAT两条；原始目录不变。逻辑事件目录共3,781条，形成期实际消费事件从1,677增至1,687。
- 第一批后的1,681个消费事件已独立重算可知时间；后两批6条逐条与独立审核的时间比较一致。这不证明事件日历完整。HSBC两条保留半年报/年报性质，SAP保留初步业绩性质。
- TSLA2019Q3实际发布公告已找到GlobeNewswire公开线索，但唯一原稿请求返回403；既有PDF能核对期间，仍不足以通过当前原始响应证据要求。停止请求并保留缺口，未加入事件。
- 收入来源盘点覆盖73家公司、3,904个历史时点。56家公司/3,033个时点有至少四年度报告线索，这不等于四年同口径收入可用。
- MSFT已完成12份原稿、24个年度收入单元格和11个相邻年度口径审查；全5,978个原候选行保留。122个MSFT时点中62个具备四年链、36个跨越FY2018会计准则变化被阻断、24个缺原报告。26个生产测试、30个独立合成测试通过，实际逐月结果另经独立复核。尚未生成收入增长因子或测算任何新策略收益。

下一步：将原始收入单元格与可比四年链验证扩展至其他发行人，并继续有界窗口的事件枚举。任何新因子或回测另行预登记；不将62个MSFT时点的通过说成全部公司数据已齐备。冻结观察账户未改动。详尽来源哈希、覆盖数据和剩余事项见同目录SUMMARY.json。
'''
    unchanged()
    write_once(ROOT/record['prior_resume']['path'], old)
    write_once(OUT/'SUMMARY.json', encode(record))
    write_once(OUT/'SUMMARY.md', summary.encode())
    checkpoint = {'path':'outputs/monthly_source_readiness156/SUMMARY.json',
                  'sha256':digest(encode(record))}
    resume.update(after155_source_checkpoint=checkpoint,
        next_source_preparation={'status':'msft_chain_verified_other_issuers_and_calendar_pending',
                                 'checkpoint':checkpoint,'new_trials':0},
        next_step=NEXT, last_goal_turn_classification='progress',updated_utc=record['created_utc'])
    unchanged(); check(resume_path.read_bytes() == old, 'Concurrent resume update')
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        temp = Path(stream.name); stream.write(encode(resume)); stream.flush(); os.fsync(stream.fileno())
    try:
        unchanged(); check(resume_path.read_bytes() == old, 'Concurrent resume update')
        os.replace(temp, resume_path)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({'checkpoint':checkpoint,'completed_rounds':155,'total_trials':978,
                      'new_trials':0,'target_reached':False}))


if __name__ == '__main__':
    main()
