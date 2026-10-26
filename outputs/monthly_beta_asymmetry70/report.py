from pathlib import Path
import json
P = Path(__file__).resolve().parent
read = lambda n: json.loads((P/n).read_text())
r, f, v = [read(n) for n in ('RESULTS.json', 'FROZEN.json', 'VALIDATION.json')]
lines = ['Separate-centered conditional up-market minus down-market beta, fixed252logreturns through t-42.',
    'F=beta_up-beta_down with SPY sign groups, each>=40 and positive within-group market variance. Positive gap need not imply low downside beta. Missing falls back; validzero retained.',
    'One fixed auxiliary support replacement budget2.5%; current50/52% longtrend core and existing model/episode structure unchanged.',
    'First88 selection, early53/recent35 guards, frozen before later34. Failed diagnostics retained.',
    '2016-08-17..2026-09-24 same complete-close signal/fill zero-cost idealization. Reused development history, not pristine OOS.',
    'Name | Full CAGR | Sharpe | MaxDD | First88 CAGR | Later34 CAGR | Nextopen 5bp/side CAGR']
for name, z in r.items():
    m, c = z['metrics'], z['slices']
    lines.append(f"{name} | {m['cagr']:.2%} | {m['sharpe']:.3f} | {m['max_drawdown']:.2%} | {c['training2016_2023']['cagr']:.2%} | {c['later34']['cagr']:.2%} | {z['next_open_cost']['metrics']['cagr']:.2%}")
lines += ['', f"Selected: {f['selected']['primary']['spec']}; newly admitted: {f['new_admitted']}.",
    f"Target70 reached: {v['primary_target70_reached']}.",
    'Historical Top32 estimates and assumed theme17 limitations remain; no claim of exact author replication or guaranteed future performance.',
    f"Protected files checked: {v['protected_files']}; observer accounts unchanged."]
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
