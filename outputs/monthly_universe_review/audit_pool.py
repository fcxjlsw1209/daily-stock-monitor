"""Inventory of the existing monthly proxy universe; no fitting, trading or pool edits."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import sys,json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_factor_search_v2'))
from data_inputs import load_inputs,sha,read,actual,display

def dump(name,x):(P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    px,raw,members,periods,sources=load_inputs()
    extra=[O/'daily_rebuild_research/price_scaled_caps.csv.gz',O/'index_union/alias_resolved_inputs/eligible.csv.gz',O/'index_union/alias_resolved_inputs/MANIFEST.json',O/'monthly_factor_search_v2/INPUTS.json',O/'monthly_multilayer_reconstruction/FEATURES.json',O/'monthly_supplied_ledger_audit/SOURCE.txt']
    sources.update({str(f):sha(f) for f in extra})
    caps=pd.read_csv(extra[0],index_col=0,parse_dates=True).reindex(px.index)
    eligible=pd.read_csv(extra[1],index_col=0,parse_dates=True).reindex(px.index).fillna(False).astype(bool)
    issuers=read(extra[2])['issuers'];foreign=['TSM','BABA','NVO','SAP','TM']
    for s in foreign:
        f=O/'monthly_reconstruction_v2/foreign'/f'{s}_cap.csv';sources[str(f)]=sha(f)
        z=pd.read_csv(f,parse_dates=['date']);z['available']=z.date+pd.Timedelta(days=1)
        joined=pd.merge_asof(pd.DataFrame({'session':px.index}),z,left_on='session',right_on='available',direction='backward')
        close=raw['close'][s];known=pd.DataFrame({'price_date':px.index,'price':close}).dropna().reset_index(drop=True)
        den=pd.merge_asof(joined.dropna().sort_values('date'),known,left_on='date',right_on='price_date',direction='backward').set_index('session').reindex(px.index)
        good=(pd.Series(px.index,index=px.index)-den.date).dt.days.le(45)&(den.date-den.price_date).dt.days.le(7)
        caps[s]=(den.market_cap_usd*close/den.price).where(good);eligible[s]=px[s].notna()
    core=sorted(members['core17'][1]);theme=sorted(members['theme17'][1]);assert len(core)==len(theme)==17
    warm=px.notna().rolling(295,min_periods=295).sum().eq(295)&px.gt(0);momentum=px.shift(42)/px.shift(252)-1
    modelmeta=read(O/'monthly_factor_search_v2/INPUTS.json')['theme17__percentile'];packed=np.load(modelmeta['file']);ev=np.flatnonzero(packed['label_mask'])
    sources[modelmeta['file']]=sha(modelmeta['file']);assert sources[modelmeta['file']]==modelmeta['sha256']
    protected_files=read(O/'monthly_multilayer_reconstruction/PROTECTED_BEFORE.json').keys()
    protected={f:sha(f) for f in protected_files}
    for folder in ['monthly_multilayer_reconstruction','monthly_factor_search_v2']:
        for f in (O/folder).glob('*'):
            if f.is_file():protected[str(f)]=sha(f)
    dump('PROTECTED_BEFORE.json',protected)
    snapshots=[];outside=[];failed=[];allstocks=set();pricefail=[];duplicate_groups=[]
    for i,row in enumerate(periods):
        d=pd.Timestamp(row['start']);stockcols=[s for s in eligible if s in caps]
        values=caps.loc[d,stockcols].where(eligible.loc[d,stockcols]).dropna().sort_values(ascending=False,kind='stable')
        seen=set();ranked=[];dups=[]
        for s in values.index:
            identity=issuers.get(s,'FOREIGN_'+s)
            if identity in seen:dups.append(display(s));continue
            seen.add(identity);ranked.append(s)
        proxy=memb=members['theme17'][0];cand=sorted(px.columns[proxy.loc[d]]);stocks=[s for s in cand if s not in theme]
        assert set(stocks)==set(ranked[:32]) and len(cand)==49
        assert cand==modelmeta['symbols'][ev[i]]
        good=[s for s in cand if warm.at[d,s] and momentum.at[d,s]>0]
        assert good==[s for s,ok in zip(cand,packed['eligible'][ev[i]]) if ok]
        allstocks.update(stocks)
        targets=[]
        for name in row['holdings']:
            s=actual(name);etf=s in theme;rank=None if etf else (ranked.index(s)+1 if s in ranked else None)
            inpool=s in cand;gate=(s in good)
            reason=[]
            if not inpool:reason.append('outside_proxy_Top32' if s in ranked else 'not_in_available_ranked_mother_pool')
            elif not warm.at[d,s]:reason.append('insufficient_295_price_history')
            elif not momentum.at[d,s]>0:reason.append('nonpositive_skip42_momentum')
            prices=px.loc[d:row['end'],s];priceok=bool(prices.notna().all() and prices.gt(0).all())
            t=dict(symbol=name,asset_type='ETF' if etf else 'stock',proxy_rank=rank,in_pool=inpool,passes_our_entry_gate=gate,exclusion_reason=reason,holding_price_complete=priceok)
            targets.append(t)
            if not inpool:
                e=dict(date=row['start'],symbol=name,proxy_rank=rank,cap_usd=float(caps.at[d,s]),rank32=display(ranked[31]),rank32_cap_usd=float(caps.at[d,ranked[31]]))
                e['below_rank32_pct']=1-e['cap_usd']/e['rank32_cap_usd'];outside.append(e)
            if inpool and not gate:failed.append(dict(date=row['start'],**t))
            if not priceok:pricefail.append(dict(date=row['start'],symbol=name))
        borders=[dict(rank=j+1,symbol=display(s),cap_usd=float(caps.at[d,s]),identity=issuers.get(s,'FOREIGN_'+s),selected=j<32) for j,s in enumerate(ranked[:40])]
        snapshots.append(dict(date=row['start'],end=row['end'],stock32=[display(s) for s in ranked[:32]],ETF17=theme,candidate49=[display(s) for s in cand],eligible=[display(s) for s in good],ineligible=[display(s) for s in cand if s not in good],ranked_available_issuers=len(ranked),eligible_candidates=len(good),skipped_duplicate_class_tickers=dups,rank1_to40=borders,author_selected=targets))
    authorcounts=Counter(s for z in periods for s in z['holdings']);authorETFs={s:n for s,n in authorcounts.items() if s in theme}
    fundcounts={}
    paths={'previous_return_first':O/'monthly_factor_search_v2/return_leader_open_cost_HOLDINGS.json','current_reconstruction':O/'monthly_multilayer_reconstruction/LEADERS.json'}
    for name,f in paths.items():
        sources[str(f)]=sha(f);rawrows=read(f)
        chosen=[z['symbols'] for z in rawrows] if name=='previous_return_first' else [z['selected'] for z in rawrows['A']['details']]
        fundcounts[name]=dict(Counter(s for rr in chosen for s in rr if s in theme))
    allstocklist=sorted(display(s) for s in allstocks)
    summary=dict(period_start=periods[0]['start'],period_end=periods[-1]['end'],periods=69,candidate_size=49,stock_slots=32,ETF_slots=17,unique_proxy_stocks=len(allstocks),unique_author_assets=len(authorcounts),unique_author_stocks=len(authorcounts)-len(authorETFs),author_ETF_counts=authorETFs,author_ETF_unknown_count=17-len(authorETFs),author_all_selected_covered_periods=69-len({x['date'] for x in outside}),outside_positions=len(outside),outside_unique_tickers=sorted({x['symbol'] for x in outside}),failed_gate_after_pool=len(failed),author_price_gaps=len(pricefail),eligible_range=[min(x['eligible_candidates'] for x in snapshots),max(x['eligible_candidates'] for x in snapshots)],ranked_mother_range=[min(x['ranked_available_issuers'] for x in snapshots),max(x['ranked_available_issuers'] for x in snapshots)],theme17=theme,core17=core,ETF_changes_from_core=dict(added=sorted(set(theme)-set(core)),removed=sorted(set(core)-set(theme))),model_ETF_counts=fundcounts)
    dump('SUMMARY.json',summary);dump('MONTHLY_POOLS.json',snapshots);dump('OUTSIDE_TARGETS.json',outside)
    dump('AUTHOR_OBSERVED_ASSETS.json',dict(stocks={k:v for k,v in sorted(authorcounts.items()) if k not in theme},ETFs=authorETFs,meaning='These assets were demonstrably selected in the source ledger, not a complete contemporaneous candidate universe.'))
    dump('SOURCE_HASHES.json',sources)
    dump('STATUS.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),algorithm_search_paused=True,pause_requested_by_user=True,active_focus='Review universe and evidence before further factor fitting',inventory_complete=True,author_full_universe_identified=False,observer_changes=False,comparison_cost_policy='Zero costs for future reconstruction performance comparisons; do not rewrite frozen paper ledgers.'))
    dump('UNRESOLVED.json',[
      dict(priority=1,item='US-listed mother universe',known='Author describes floating Top32 including US-listed foreign issuers; our eligible universe is inherited historical NDX100/SP100 union plus five explicit foreign additions.',needed='Dated full candidate list or independent coverage of other potentially large US-listed issuers. Do not inject only known winners.'),
      dict(priority=1,item='ETF17',known=f'Only {sorted(authorETFs)} appear in supplied selections; the other15 remain unidentified. Core17/theme17 are our hypotheses.',needed='Original full ETF list or explicit versioned hypotheses; absence from holdings does not establish exclusion.'),
      dict(priority=1,item='Market-cap definition and time',known='Our vendor snapshots use observation+1calendar day, <=45days stale, price-scaled forward. Historical publication vintage is not certified.',needed='Align company-level cap versus class/ADR representation, snapshot dates and membership evaluation time; audit ranks28-40 independently of target winners.'),
      dict(priority=2,item='Share class and ticker identity',known='Our ranks use issuer dedup; META mapped to historical FB price; class choice follows inherited security mappings.',needed='Author class choice/issuer aggregation and whether foreign listings are issuer-wide or ADR-line capitalization.'),
      dict(priority=2,item='Membership versus signal eligibility',known='Our49-name pool is ranked before295-day history and positive P[t-42]/P[t-252] gate; 253-day history used in older research.',needed='Author exact minimum history, eligibility and normalization order. No supplied target is additionally rejected by current gate when inside pool.'),
      dict(priority=3,item='Zero-cost reference',known='User reaffirmed zero trading costs.',needed='Maintain one continuous common period and show performance only after pool version is explicit; do not compare different universes as algorithm-only differences.')])
    lines=['Monthly stock/ETF universe inventory \u2014 69 signal dates','', '|Signal date|Stock32 (descending estimated issuer cap)|Eligible after our gate|Author assets outside proxy|','|---|---|---:|---|']
    for x in snapshots:lines.append(f"|{x['date']}|{', '.join(x['stock32'])}|{x['eligible_candidates']}|{', '.join(t['symbol'] for t in x['author_selected'] if not t['in_pool']) or '-'}|")
    (P/'MONTHLY_POOLS.txt').write_text('\n'.join(lines)+'\n')
    lines=['\u6708\u7b56\u7565\u80a1\u7968\u6c60\u6574\u7406\uff1a\u6682\u505c\u8c03\u53c2\uff0c\u5148\u6838\u5bf9\u5019\u9009\u8303\u56f4','', '\u672c\u8f6e\u4ec5\u6574\u7406\u4e0e\u6838\u5bf9\uff0c\u6ca1\u6709\u65b0\u62df\u5408\u3001\u65b0\u56de\u6d4b\u3001\u6539\u6210\u5458\u6216\u6539\u89c2\u5bdf\u8d26\u672c\u3002\u6bd4\u8f83\u8303\u56f42020-12-31\u8c03\u4ed3\u81f32026-09-24\uff0c\u517169\u671f\u3002','', '\u4e00\u3001\u4f5c\u8005\u63cf\u8ff0\u4e0e\u5f53\u524d\u5b9e\u73b0','\u4f5c\u8005\uff1a\u7f8e\u80a1\u4e0a\u5e02\u516c\u53f8\u7684\u6d6e\u52a8\u5e02\u503cTop32\uff0b17\u53eaETF\uff0c\u4e4b\u540e\u9009\u4e09\u53ea\u3002\u6ca1\u6709\u5b8c\u6574Top32\u9010\u65e5\u540d\u5355\uff0c\u4e5f\u6ca1\u6709ETF17\u5b8c\u6574\u6e05\u5355\u3002','\u6211\u4eec\uff1a\u5386\u53f2Nasdaq-100\u222aS&P100\u53ef\u7528\u6570\u636e\u6bcd\u6c60\uff0bTSM/BABA/NVO/SAP/TM\u4e94\u4e2a\u663e\u5f0f\u8865\u5145\u53d1\u884c\u4eba\uff0c\u6309\u4f30\u7b97\u5e02\u503c\u3001\u53d1\u884c\u4eba\u53bb\u91cd\u9009Top32\uff0c\u518d\u52a0theme17\u3002ASML\u7b49\u5176\u4ed6\u5916\u56fd\u516c\u53f8\u53ef\u4ee5\u901a\u8fc7\u6307\u6570\u6bcd\u6c60\u8fdb\u5165\uff0c\u4e0d\u662f\u4ec5\u6709\u8fd9\u4e94\u5bb6\u5916\u56fd\u516c\u53f8\u3002','\u6b64\u524d78.31%\u5e74\u5316\u5019\u9009\u548c\u672c\u8f6e48.90%\u5019\u9009\u4f7f\u7528\u540c\u4e00theme17\u53caTop32\u4ee3\u7406\u6c60\u3001\u540c\u4e00295\u65e5\u9884\u70ed\u4e0e\u6b63\u957f\u671f\u52a8\u91cf\u8d44\u683c\u53e3\u5f84\uff1b\u4e24\u8005\u4e0e\u4f5c\u8005\u5b8c\u6574\u6c60\u4ecd\u672a\u5bf9\u9f50\u3002','\u5f53\u524d\u516c\u53f8\u5e02\u503c\u53d6\u5386\u53f2\u4f9b\u5e94\u5546\u5feb\u7167\uff0c\u5feb\u7167\u65e5\u671f\uff0b1\u81ea\u7136\u65e5\u4f7f\u7528\uff0c\u6700\u957f45\u5929\uff0c\u6309\u4e4b\u540e\u4ef7\u683c\u53d8\u5316\u5916\u63a8\u3002\u6570\u636e\u5386\u53f2\u7248\u672c\u548c\u5f53\u65f6\u771f\u5b9e\u53d1\u5e03\u65e5\u671f\u672a\u77e5\uff0c\u8fd9\u79cd\u65f6\u95f4\u6ede\u540e\u4e0d\u80fd\u81ea\u52a8\u6d88\u9664\u6570\u636e\u4fee\u8ba2\u504f\u5dee\u3002','', '\u4e8c\u300117\u53eaETF','\u5f53\u524dtheme17\uff1a'+', '.join(theme),'\u8f83\u65e9core17\uff1a'+', '.join(core),'core\u2192theme\u589e\u52a0\uff1a'+', '.join(summary['ETF_changes_from_core']['added'])+'\uff1b\u79fb\u9664\uff1a'+', '.join(summary['ETF_changes_from_core']['removed'])+'\u3002','\u4f5c\u8005\u8bb0\u5f55\u53ea\u786e\u8ba4\u9009\u8fc7\uff1a'+json.dumps(authorETFs,ensure_ascii=False)+'\u3002\u5176\u4f5915\u53ea\u4e0d\u660e\uff1b\u6ca1\u6709\u9009\u8fc7\u4e0d\u7b49\u4e8e\u4e0d\u5728\u6c60\u5185\u3002','\u6211\u4eecETF\u6301\u4ed3\u6b21\u6570\uff1a'+json.dumps(fundcounts,ensure_ascii=False),'', '\u4e09\u3001\u5df2\u786e\u8ba4\u7684\u516d\u5904\u6210\u5458\u51b2\u7a81','|\u8c03\u4ed3\u65e5|\u4f5c\u8005\u80a1\u7968|\u6211\u65b9\u4f30\u7b97\u540d\u6b21|\u7b2c32\u540d|\u8ddd\u7b2c32\u540d\u5e02\u503c\u5dee|','|---|---|---:|---|---:|']
    for x in outside:lines.append(f"|{x['date']}|{x['symbol']}|{x['proxy_rank']}|{x['rank32']}|\u4f4e{x['below_rank32_pct']:.2%}|")
    lines+=['',f"69\u671f\u4e2d63\u671f\u8986\u76d6\u4f5c\u8005\u4e09\u53ea\u5168\u90e8\u6301\u4ed3\uff0c6\u671f\u5404\u7f3a\u4e00\u53ea\uff1b\u5171{len(authorcounts)}\u79cd\u4f5c\u8005\u6301\u4ed3\u6807\u7684\uff0c\u5176\u4e2d{len(authorcounts)-len(authorETFs)}\u53ea\u80a1\u7968\u3001{len(authorETFs)}\u53eaETF\u3002",'\u8fd9\u4e0d\u662f\u201c63\u671f\u80a1\u7968\u6c60\u5df2\u590d\u73b0\u201d\uff1a\u672a\u9009\u4e2d\u7684\u5019\u9009\u672a\u77e5\uff0c\u4e5f\u4f1a\u6539\u53d8\u5168\u6c60\u767e\u5206\u4f4d\u548c\u7ade\u4e89\u6392\u5e8f\u3002\u6b64\u524d54\u671f\u300139\u671f\u8fc7\u6ee4\u4ec5\u4e3a\u8bca\u65ad\uff0c\u4e0d\u662f\u540c\u6c60\u8bc1\u660e\u3002',f"\u73b0\u670969\u6b21\u80a1\u7968\u6c60\u5171\u51fa\u73b0{len(allstocks)}\u53ea\u4e0d\u540c\u80a1\u7968\uff1b\u6bcf\u671f\u5019\u900949\u53ea\uff0c\u7ecf\u8fc7\u5f53\u524d\u8d44\u683c\u6761\u4ef6\u540e\u4e3a{summary['eligible_range'][0]}\u2013{summary['eligible_range'][1]}\u53ea\u3002\u4f5c\u8005\u76ee\u6807\u5728\u6c60\u5185\u65f6\uff0c\u65e0\u989d\u5916\u56e0295\u65e5\u9884\u70ed\u6216\u6b63\u52a8\u91cf\u95e8\u69db\u88ab\u6392\u9664\uff1b\u4f5c\u8005\u5df2\u9009\u6301\u4ed3\u4ef7\u683c\u8def\u5f84\u6ca1\u6709\u7f3a\u5931\u3002",'\u516d\u5904\u51b2\u7a81\u76ee\u524d\u662f\u4f30\u7b97\u6392\u540d33\u201335\u540d\uff0c\u4e0d\u80fd\u76f4\u63a5\u628a\u5b83\u4eec\u6c38\u4e45\u7eb3\u5165Top32\uff0c\u4e5f\u4e0d\u80fd\u901a\u8fc7\u5220\u6389\u5176\u4ed6\u7ade\u4e89\u8005\u6765\u5f3a\u884c\u590d\u73b0\u3002','', '\u56db\u3001\u7248\u672c\u4e0e\u540e\u7eed\u987a\u5e8f','1. \u4fdd\u7559\u5f53\u524d\u4ee3\u7406\u6c60\u4e3a\u7814\u7a76\u57fa\u7ebf\uff0c\u6682\u505c\u65b0\u7684\u56e0\u5b50/\u6743\u91cd/ML\u641c\u7d22\u3002','2. \u5148\u6838\u5bf9\u5168\u7f8e\u4e0a\u5e02\u6bcd\u6c60\u8986\u76d6\u548cETF17\u5b9a\u4e49\uff1b\u6309\u201c\u5df2\u786e\u8ba4\uff0f\u7814\u7a76\u5047\u8bbe\uff0f\u672a\u77e5\u201d\u7ef4\u62a4\uff0c\u4e0d\u628a\u4f5c\u8005\u9009\u8fc7\u768431\u79cd\u8bc1\u5238\u5f53\u5b8c\u6574\u5019\u9009\u6c60\u3002','3. \u9010\u6b21\u6708\u672b\u68c0\u67e528\u201340\u540d\u7684\u5e02\u503c\u5feb\u7167\u3001\u65f6\u70b9\u3001\u80a1\u7c7b\u548cADR\u53e3\u5f84\u3002\u5168\u90e869\u671f\u539f32\u53ea\u53ca1\u201340\u540d\u5ba1\u8ba1\u5df2\u5bfc\u51fa\uff0c\u4e0d\u53ea\u68c0\u67e5\u5df2\u77e5\u8d62\u5bb6\u3002','4. \u6c60\u7248\u672c\u660e\u786e\u540e\u56fa\u5b9a\u5019\u9009\u6c60\uff0c\u91cd\u65b0\u8ba1\u7b97\u56e0\u5b50\u767e\u5206\u4f4d\uff0c\u518d\u6bd4\u8f83\u540c\u533a\u95f4\u3001\u96f6\u6210\u672c\u6536\u76ca\u4e0eSharpe\uff1b\u4e0d\u80fd\u53ea\u628a\u65b0\u80a1\u7968\u585e\u8fdb\u539f\u6392\u540d\u3002','5. \u540e\u7eed\u7814\u7a76\u5982\u91c7\u7528\u65e0\u6cd5\u8bc1\u5b9e\u7684ETF\u5907\u9009\u540d\u5355\uff0c\u8981\u660e\u786e\u6807\u4e3a\u4e0d\u540c\u6c60\u7684\u654f\u611f\u6027\uff0c\u4e0d\u518d\u636e\u5176\u6536\u76ca\u53cd\u5411\u5ba3\u79f0\u627e\u5230\u4f5c\u8005\u539f\u6c60\u3002','', '\u8d44\u6599\uff1aSUMMARY.json\u3001AUTHOR_OBSERVED_ASSETS.json\u3001MONTHLY_POOLS.txt/JSON\u3001OUTSIDE_TARGETS.json\u3001UNRESOLVED.json\u3001SOURCE_HASHES.json\u3002']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    assert len(outside)==6 and not failed and not pricefail
    assert all(sha(f)==h for f,h in sources.items()) and all(sha(f)==h for f,h in protected.items())
    validation=dict(months_rebuilt=69,cached_Top32_matches=69,cached_49_candidates_match=69,cached_entry_gate_matches=69,issuer_dedup_checked=True,author_price_paths_complete=True,protected_files_unchanged=len(protected),source_files_unchanged=len(sources),scope='Inventory checks only; does not certify author pool or publishable PIT data.')
    dump('VALIDATION.json',validation);dump('CODE_HASHES.json',{str(Path(__file__)):sha(__file__)})
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('VALIDATION',validation)

if __name__=='__main__':main()
