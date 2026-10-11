"""Find broken primary-event chains. Diagnostic output never grants admission."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
from datetime import date, timedelta


def valid_day(value):
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("Invalid effective date: " + repr(value))
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value or not 1900 <= parsed.year <= 2100:
        raise ValueError("Invalid effective date: " + repr(value))
    return parsed

P = Path(__file__).resolve().parent


def audit(seed, rows, start='2010-01-04', end='2026-10-06'):
    valid_day(start)
    end_day = valid_day(end)
    if start > end:
        raise ValueError("Reversed audit window")
    current = set(seed)
    opened = {s: start for s in current}
    groups = defaultdict(list)
    for row in rows:
        valid_day(row['effective_session'])
        if start <= row['effective_session'] <= end:
            groups[row['effective_session']].append(row)
    errors, counts, intervals = [], [], []
    for day, changes in sorted(groups.items()):
        for row in sorted(changes, key=lambda r: r['action'] != 'deletion'):
            symbol = row.get('ticker_as_announced_or_effective') or row['ticker']
            if row['action'] == 'deletion':
                if symbol not in current:
                    errors.append(dict(date=day, symbol=symbol, error='delete_absent'))
                else:
                    intervals.append(dict(symbol=symbol, start=opened.pop(symbol), end_exclusive=day))
                    current.remove(symbol)
            elif row['action'] == 'addition':
                if symbol in current:
                    errors.append(dict(date=day, symbol=symbol, error='add_present'))
                else:
                    current.add(symbol)
                    opened[symbol] = day
            else:
                raise ValueError('Unknown action')
        counts.append(dict(date=day, security_count=len(current)))
    intervals += [dict(symbol=s, start=opened[s], end_exclusive=(end_day + timedelta(days=1)).isoformat()) for s in sorted(current)]
    return dict(errors=errors, counts=counts, provisional_intervals=intervals,
                ending_symbols=sorted(current), admission=False,
                note='Missing events can cause apparently continuous intervals; not certified daily membership.')


def main():
    rows = json.loads((P/'ndx_primary_reviewed_rows.json').read_text())
    for row in rows:
        if hashlib.sha256((P/row['source_file']).read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError('Source hash mismatch: '+row['source_file'])
    seed = json.loads((P/'NDX_INITIAL_RECONSTRUCTION.json').read_text())['constructed_members']
    result = audit(seed, rows)
    (P/'NDX_PRIMARY_CHAIN_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    sp_rows = []
    for name in ['sp100_modern_event_rows.json', 'sp100_legacy_reviewed_rows.json', 'sp100_supplemental_reviewed_rows.json']:
        sp_rows.extend(json.loads((P/name).read_text()))
    for row in sp_rows:
        if hashlib.sha256((P/row['source_file']).read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError('Source hash mismatch: '+row['source_file'])
    sp_seed = json.loads((P/'SP100_INITIAL_RECONSTRUCTION.json').read_text())['constructed_members']
    sp_result = audit(sp_seed, sp_rows)
    (P/'SP100_PRIMARY_CHAIN_AUDIT.json').write_text(json.dumps(sp_result, indent=2)+'\n')
    print(json.dumps(dict(sp_errors=sp_result['errors'], sp_ending_count=len(sp_result['ending_symbols']))))
    print(json.dumps(dict(errors=result['errors'], ending_count=len(result['ending_symbols'])), indent=2))


if __name__ == '__main__':
    main()
