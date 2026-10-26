"""Render the completed frozen evaluation; this is not a research input."""
from pathlib import Path
import json
P = Path(__file__).resolve().parent
read = lambda n: json.loads((P/n).read_text())
r, f, v = [read(n) for n in ('RESULTS.json', 'FROZEN.json', 'VALIDATION.json')]
lines = ['Three fixed cached factors with support-preserving composites inside the original local boundary.',
    'Ordinal complexity, security/SPY lower-tail overlap, and own-sector lead-lag retain their exact audited ranks and separate availability masks. No factor recomputation or reranking.',
    'For available nonzero F inside original local rank3..5/gap.02 only: H=clip(original_support+F,-.5,.5), score_new=original_episode_score+.025*(H-original_support). Zero/missing gives no cell delta; no nonzero cohort delta preserves original score bytes and choices exactly.',
    'Original local top2 and candidate cohort locked; exactly one unchanged episode adjustment. No global reranking or original-support feedback.',
    'Longtrend50/52% core, original models/eligibility and monthly fixed units unchanged. The fixed2.5% hand slot is funded from original support, not an extra unbounded budget.',
    'Source identities:27 inherited arrays and META byte/value exact,64 raw hashes matched; target contains33 arrays. Independent actual audit binds all six prepared artifacts beforetrain.',
    'First88 selection with early53/recent35 guards; all three diagnostics frozen beforelater34. These controlled placement/composite hypotheses are not new predictive features.',
    '2016-08-17..2026-09-24 sameclosezero is idealized. Prior component and funding outcomes informed the design; this is reused development history, not pristine OOS.',
    'Name | Full CAGR | Sharpe | MaxDD | First88 CAGR | Later34 CAGR | Nextopen5bp/side CAGR']
for name,z in r.items():
    m,c=z['metrics'],z['slices']
    lines.append(f"{name} | {m['cagr']:.2%} | {m['sharpe']:.3f} | {m['max_drawdown']:.2%} | {c['training2016_2023']['cagr']:.2%} | {c['later34']['cagr']:.2%} | {z['next_open_cost']['metrics']['cagr']:.2%}")
lines += ['',f"Selected: {f['selected']['primary']['spec']}; newly admitted: {f['new_admitted']}.",
    f"Target70 reached: {v['primary_target70_reached']}.",
    'Estimated historicalTop32/assumedETF17 and inherited adjustment limitations remain; no guarantee of future performance.',
    f"Protected files checked: {v['protected_files']}; observer accounts unchanged."]
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
