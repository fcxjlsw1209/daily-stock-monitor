"""Strict, offline parsing of the registered Alphabet consolidated statements."""
import calendar
from datetime import date
import re


class ParseError(ValueError):
    pass


def check(ok, reason):
    if not ok:
        raise ParseError(reason)


def clean(text):
    return ' '.join(text.split())


MONTHS = {name: i for i, name in enumerate(calendar.month_name) if name}
DATE = r'([A-Z][a-z]+\s+\d{1,2},\s*\d{4})'
TITLE = 'CONSOLIDATED STATEMENTS OF INCOME'


def printed_date(value):
    match = re.fullmatch(r'([A-Z][a-z]+) (\d{1,2}),\s*(\d{4})', clean(value))
    check(match is not None and match[1] in MONTHS, 'invalid_printed_date')
    try:
        return date(int(match[3]), MONTHS[match[1]], int(match[2])).isoformat()
    except ValueError as exc:
        raise ParseError('invalid_printed_date') from exc


def identity(text):
    body = clean(text)
    head = re.search(r'Alphabet Announces (First|Second|Third|Fourth) Quarter(?: and Fiscal Year)? (\d{4}) Results', body)
    check(head is not None and head.start() < 40, 'missing_original_headline')
    pattern = (r'MOUNTAIN VIEW, Calif\. [–—-] ' + DATE +
               r' [–—-] Alphabet Inc\. \(NASDAQ: GOOG, GOOGL\) today announced financial '
               r'results for the quarter(?: and fiscal year)? ended ' + DATE + r'\s*\.')
    matches = list(re.finditer(pattern, body))
    check(len(matches) == 1, 'missing_or_ambiguous_dated_identity')
    subtitle = 'Announces Plans for Chief Financial Officer Transition and Appointment of President and Chief Investment Officer'
    prefix = body[:matches[0].start()].strip()
    check(prefix in (head[0], head[0] + ' ' + subtitle), 'unexpected_text_before_dated_identity')
    match = matches[0]
    released, ended = printed_date(match[1]), printed_date(match[2])
    quarter = ('First', 'Second', 'Third', 'Fourth').index(head[1]) + 1
    year = int(head[2]); month = quarter * 3
    check(ended == date(year, month, calendar.monthrange(year, month)[1]).isoformat(), 'headline_period_mismatch')
    check(ended < released, 'release_precedes_period_end')
    return dict(issuer='Alphabet Inc.', exchange='NASDAQ', printed_symbols=['GOOG', 'GOOGL'],
                release_date=released, report_date=ended, fiscal_year=year, fiscal_quarter=quarter,
                headline=head[0], excerpt=match[0], precision='date_only',
                exact_publication_time_verified=False, continuous_identity_verified=False)


def statement(text):
    # Select the named consolidated statement, never a financial highlights table.
    body = clean(text)
    check(body.count(TITLE) == 1, 'missing_or_ambiguous_income_statement')
    prefix, tail = body.split(TITLE)
    check(prefix.endswith('Alphabet Inc. '), 'statement_issuer_missing')
    match = re.match(r'\s*(\(In millions, except [^)]*\))\s*(.*?)\bRevenues\s+(.*?)\s+Costs and expenses:', tail)
    check(match is not None, 'revenue_row_or_units_missing')
    units, header, values = (v.strip() for v in match.groups())
    check(units in (
        '(In millions, except per share amounts; unaudited)',
        '(In millions, except per share amounts)',
        '(In millions, except per share amounts, unaudited)',
        '(In millions, except share amounts which are reflected in thousands and per share amounts)',
        '(In millions, except share amounts which are reflected in thousands and per share amounts; unaudited)',
    ), 'unsupported_unit_statement')
    year_match = re.fullmatch(r'(.*?)((?:20\d\d\s+){1,3}20\d\d)(?:\s+\(unaudited\))*', header)
    check(year_match is not None, 'invalid_year_columns')
    period_text = year_match[1].strip()
    years = [int(v) for v in year_match[2].split()]
    check(len(years) in (2, 4), 'unsupported_column_count')
    n = len(years) // 2
    check(all(years[2*i+1] == years[2*i]+1 for i in range(n)) and years == years[:2] * n,
          'noncomparable_year_columns')
    old = re.fullmatch(r'Three Months Ended(?: (Six|Nine|Twelve) Months Ended)? '
                       r'([A-Z][a-z]+) (\d{1,2}),(?: ([A-Z][a-z]+) (\d{1,2}),)?', period_text)
    modern = re.fullmatch(r'Quarter Ended ([A-Z][a-z]+) (\d{1,2}),(?: (Year Ended|Year [Tt]o Date) '
                          r'([A-Z][a-z]+) (\d{1,2}),)?', period_text)
    check(old is not None or modern is not None, 'unsupported_period_header')
    if old:
        month, day = old[2], int(old[3]); second = {'Six': 6, 'Nine': 9, 'Twelve': 12}.get(old[1])
        check((n == 1 and old[1] is None and old[4] is None) or
              (n == 2 and old[4] == month and int(old[5]) == day), 'period_column_mismatch')
    else:
        month, day = modern[1], int(modern[2]); second = None
        if modern[3]:
            second = 12 if modern[3] == 'Year Ended' else MONTHS.get(month)
        check((n == 1 and modern[3] is None) or
              (n == 2 and modern[4] == month and int(modern[5]) == day), 'period_column_mismatch')
    check(month in MONTHS and MONTHS[month] in (3, 6, 9, 12), 'nonquarter_end')
    m = MONTHS[month]
    check(day == calendar.monthrange(years[1], m)[1], 'noncalendar_quarter_end')
    check((m == 3 and n == 1) or (n == 2 and second == m), 'missing_or_wrong_cumulative_period')
    amount_pattern = r'\$\s*((?:[1-9]\d{0,2}(?:,\d{3})+)|0|[1-9]\d*)'
    tokens = re.findall(amount_pattern, values)
    check(len(tokens) == len(years) and re.sub(amount_pattern, '', values).strip() == '',
          'invalid_revenue_cells')
    amounts = [int(v.replace(',', '')) * 1_000_000 for v in tokens]
    periods = []
    for i, duration in enumerate([3] if n == 1 else [3, second]):
        periods.append(dict(duration_months=duration, **{
            role: dict(end=date(years[2*i+j], m, day).isoformat(), revenue_usd=amounts[2*i+j])
            for role, j in (('prior', 0), ('current', 1))}))
    return dict(title=TITLE, units=units, header=header, revenue_row='Revenues '+values,
                column_order='prior_then_current', concept='consolidated_revenues', periods=periods,
                period_starts_directly_printed=False, units_multiplier=1_000_000,
                currency='USD', currency_basis='Fixed US issuer consolidated dollar table; dollar symbol printed in every revenue cell, not a printed ISO currency code')


def parse_release(text):
    a, b = identity(text), statement(text)
    check(all(p['current']['end'] == a['report_date'] for p in b['periods']), 'identity_table_period_mismatch')
    return dict(identity=a, statement=b)
