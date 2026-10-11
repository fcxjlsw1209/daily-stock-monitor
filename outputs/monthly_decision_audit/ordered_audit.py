"""Treat supplied list order as user-approved working rank hypothesis."""
from pathlib import Path
from collections import Counter
import hashlib,itertools,json
P=Path(__file__).resolve().parent
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    inputs=[P/'MONTHS.json',P/'SUMMARY.json',P.parent/'monthly_supplied_ledger_audit/PERIODS.json']
    before={str(f):sha(f) for f in inputs}
    original=read(inputs[2]);ms=read(inputs[0]);records=[];counts=Counter();drivers=Counter()
    for p,m in zip(original,ms):
        assert p['start']==m['date']
        author=p['holdings'];ours=m['ours'];rows={r['symbol']:r for r in m['rows']}
        assert len(author)==3 and len(set(author))==3
        assert set(author)==set(m['author'])
        counts['months']+=1;counts['set_exact']+=set(author)==set(ours);counts['order_exact']+=author==ours
        counts['top1_exact']+=bool(ours) and author[0]==ours[0]
        counts['position_exact']+=sum(a==b for a,b in zip(author,ours))
        pairs=[]
        for i,j in itertools.combinations(range(3),2):
            a,b=author[i],author[j];ra,rb=rows[a],rows[b]
            comparable=ra['eligible'] and rb['eligible'];counts['pairs_total']+=1
            z={'expected_higher':a,'expected_lower':b,'author_positions':[i+1,j+1],
               'model_ranks':[ra['rank'],rb['rank']],'comparable':comparable}
            if comparable:
                counts['pairs_comparable']+=1
                gap=ra['score']-rb['score'];delta={k:ra['contributions'][k]-rb['contributions'][k] for k in ra['contributions']}
                assert abs(sum(delta.values())-gap)<1e-10
                same=ra['rank']<rb['rank'];counts['pairs_order_correct']+=same
                counts['pairs_inverted']+=not same
                driver=min(delta,key=delta.get) if not same else None
                if driver:drivers[driver]+=1
                z.update(model_order_agrees=same,score_gap_points=gap,factor_gaps=delta,
                         largest_opposing_factor=driver,score_tie=abs(gap)<1e-10)
            else:counts['pairs_not_comparable']+=1
            pairs.append(z)
        full=all(z['comparable'] for z in pairs)
        internal=full and all(z['model_order_agrees'] for z in pairs)
        counts['all_three_comparable_months']+=full;counts['all_three_relative_order_correct']+=internal
        records.append({'date':m['date'],'holding_month':m['holding_month'],'author_rank_hypothesis':author,
            'our_selected_order':ours,'author_symbols_model_ranks':[rows[a]['rank'] for a in author],
            'same_set':set(author)==set(ours),'same_order':author==ours,
            'author_internal_order_matches':internal if full else None,'pairs':pairs})
    assert len(records)==69 and counts['pairs_total']==207
    assert counts['pairs_order_correct']+counts['pairs_inverted']==counts['pairs_comparable']
    assert counts['order_exact']<=counts['set_exact']
    assert before=={str(f):sha(f) for f in inputs}
    summary={'assumption':'用户要求将原始名单顺序当作作者第1/2/3名；作为待验证假设，不称来源已经确认。',
      'ranking_constraints':'score(author1) > score(author2) > score(author3); all three selected above nonselected eligible pool members. Ties remain unknown.',
      'no_ml_or_refit':True,'no_return_calculation':True,'counts':dict(counts),'inversion_largest_opposing_factors':dict(drivers),
      'input_sha256':before,'source_order_preserved':True,'prior_unordered_analysis_unchanged':True}
    dump('ORDERED_MONTHS.json',records);dump('ORDERED_SUMMARY.json',summary)
    lines=['作者名单顺序假设：逐月有序排名审计','',summary['assumption'],
      '作者原始顺序完整保留，不用字母排序或我们的分数重排作者名单。原69期集合审计不改写；本文件增加有序约束，不改变持仓、收益或模拟账户。',
      '',json.dumps(dict(counts),ensure_ascii=False,indent=2),'']
    def rank(v):return '池外/不合格' if v is None else '#'+str(v)
    for m in records:
        lines += [f"{m['holding_month']}（信号 {m['date']}）",
          '作者假设排名：'+' > '.join(m['author_rank_hypothesis']),
          '这三只在我们全池的排名：'+' > '.join(rank(r) for r in m['author_symbols_model_ranks']),
          '我们实际选股顺序：'+' > '.join(m['our_selected_order']),
          f"集合一致：{m['same_set']}；前三次序也一致：{m['same_order']}。"]
        for z in m['pairs']:
            if not z['comparable']:lines.append(f"  {z['expected_higher']} > {z['expected_lower']}：有股票不在当前合格池，不能比较。");continue
            if not z['model_order_agrees']:
                d=z['factor_gaps'];lines.append(f"  顺序冲突 {z['expected_higher']} > {z['expected_lower']}：我们的前者减后者分差 {z['score_gap_points']:+.2f}；动量 {d['momentum']:+.2f}、波动 {d['volatility']:+.2f}、加速 {d['acceleration']:+.2f}、状态 {d['state_cmf']:+.2f}。")
        lines.append('')
    (P/'ORDERED_REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'counts':dict(counts),'drivers':dict(drivers),'report':str(P/'ORDERED_REPORT.txt')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
