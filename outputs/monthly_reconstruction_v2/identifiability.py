"""Agreement of hindsight explanations; never confirmation of author trades."""
import itertools
import json
from research import P, dump


def main():
    tags = ['core17_q0', 'theme17_q0', 'theme17_q0.5']
    paths = {k: json.loads((P / f'INVERSE_{k}.json').read_text()) for k in tags}
    months = [r['month'] for r in paths[tags[0]]['orders']]
    assert all([r['month'] for r in v['orders']] == months for v in paths.values())
    rows = []
    for i, month in enumerate(months):
        holdings = {k: paths[k]['orders'][i]['holdings'] for k in tags}
        sets = [set(h) for h in holdings.values()]
        rows.append(dict(month=month, rebalance_date=paths[tags[0]]['orders'][i]['date'],
                         candidate_holdings=holdings,
                         intersection=sorted(set.intersection(*sets)),
                         union=sorted(set.union(*sets)),
                         same_complete_combination=all(s == sets[0] for s in sets),
                         verified_author_holdings=False))
    pairwise = {}
    for a, b in itertools.combinations(tags, 2):
        counts = {str(k): 0 for k in range(4)}
        for row in rows:
            n = len(set(row['candidate_holdings'][a]) & set(row['candidate_holdings'][b]))
            counts[str(n)] += 1
        pairwise[f'{a} vs {b}'] = counts
    result = dict(months=len(rows), all_three_agree=sum(r['same_complete_combination'] for r in rows),
                  zero_common_names=sum(not r['intersection'] for r in rows),
                  pairwise_intersection_sizes=pairwise,
                  interpretation='Agreement across these three assumptions is a clue, not a probability or verified trade. All paths use future returns and the full image. They are neither independent estimates nor exhaustive alternatives.',
                  author_holdings_identified=False, rows=rows)
    dump(P / 'HOLDING_AMBIGUITY.json', result)
    lines = ['月度候选持仓表：事后反推，不是作者交易记录', '',
             f"121个月中三种假设完全同组合：{result['all_three_agree']}个月；没有三方共同股票：{result['zero_common_names']}个月。",
             '各路径都使用未来价格；相同持仓也不能视为已验证。表中日期是候选调仓日，非作者真实成交记录。', '',
             '月份 | 调仓日 | 原ETF假设 | 主题ETF假设 | 主题ETF+强势门槛 | 共同股票']
    for r in rows:
        h = r['candidate_holdings']
        lines.append(' | '.join([r['month'], r['rebalance_date']] + [', '.join(h[k]) for k in tags] + [', '.join(r['intersection']) or '无']))
    (P / 'CANDIDATE_HOLDINGS.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
