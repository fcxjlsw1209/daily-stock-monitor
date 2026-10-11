"""Exchange-origin 2010 listing evidence; not a daily eligible-universe certificate."""
from pathlib import Path
import datetime, hashlib, json, re

P = Path(__file__).resolve().parent / 'additional_sources'
ROW = re.compile(r'^(.*?)\s+(\S+)\s+(NYSE(?: Amex| Arca)?|Amex)\s*(.*?)\s+(\d{1,2}/\d{1,2}/\d{2}|5/505)(?:\s+(GS|[AOGNU]))?(?:\s+(IPO.*))?$')


def parse(text):
    rows, unresolved = [], []
    country = None
    for line in text.splitlines():
        line = line.strip()
        if re.match(r'^[A-Z &]+ \(.*Issuer', line):
            country = line.split(' (')[0]
        match = ROW.match(line)
        if not match:
            if ('NYSE' in line or re.search(r'\sAmex\s', line)) and re.search(r'\d/\d', line):
                if 'reorganized' not in line:
                    unresolved.append(line)
            continue
        name, symbol, market, industry, listed, share_type, ipo = match.groups()
        # Preserve the exchange's malformed TGP date rather than invent a day.
        admission = None
        if listed != '5/505':
            month, day, year = map(int, listed.split('/'))
            year += 1900 if year > 10 else 2000
            admission = datetime.date(year, month, day).isoformat()
        excluded = '(PFD)' in name or industry in {'Equity Investment Instruments', 'Nonequity Investment Instruments'} or market == 'NYSE Arca'
        rows.append(dict(issuer_as_printed=name, symbol=symbol, country_heading=country, exchange_as_printed=market, industry_as_printed=industry, listing_date_as_printed=listed, admission_date=admission, share_type_as_printed=share_type, ipo_marker=ipo, source_line=line, common_equity_candidate=not excluded, daily_listing_continuity_verified=False))
    return rows, unresolved


if __name__ == '__main__':
    path = P / 'nyse_foreign_2010.txt'
    rows, unresolved = parse(path.read_text())
    result = dict(source='https://www.nyse.com/publicdocs/nyse/data/All_Non_US_Stocks_12-31-10.pdf', snapshot_date='2010-12-31', source_sha256=hashlib.sha256((P / 'nyse_foreign_2010.pdf').read_bytes()).hexdigest(), extracted_text_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), security_rows=len(rows), unresolved_rows=unresolved, rows=rows, scope='Security rows include preferred/multiple classes, so not equal to 519 issuer headline. Admission dates may reference predecessors; footnotes and intervening suspensions need checking. Never backfill complete daily coverage from this year-end roster.')
    (P / 'nyse_foreign_2010_parsed.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
