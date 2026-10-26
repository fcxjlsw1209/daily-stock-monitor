"""SEC source collection and metadata-only accounting coverage; no factors/trades.

Collection is a separately authorized network step, never an offline replay.
The filed-date filter is NOT an exchange-session availability/admission rule.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.request
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 40 * 1024 * 1024
TIMEOUT = 15
PURPOSE = 'profitability_source_preparation_only'
TAGS = ('GrossProfit', 'Assets', 'NetIncomeLoss', 'OperatingIncomeLoss',
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations')
HEADERS = ('Content-Type', 'Date', 'Retry-After', 'Last-Modified', 'ETag')
TEMPORAL_NOTE = ('filed-through filters filing calendar dates only; first usable official '
                 'exchange session, amendments, historical identity and source semantics '
                 'require independent audit. No factor admission or historical snapshot '
                 'completeness is asserted.')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    def bad(_):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad,
                      parse_float=Decimal)


def iso_day(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('ISO date required')
    return date.fromisoformat(value)


def check_cik(cik):
    if not isinstance(cik, str) or not re.fullmatch(r'[0-9]{10}', cik) or int(cik) == 0:
        raise ValueError('Positive ten-digit CIK required')
    return cik


def endpoint(cik):
    return 'https://data.sec.gov/api/xbrl/companyfacts/CIK' + check_cik(cik) + '.json'


def validate_facts(raw, cik):
    check_cik(cik)
    obj = strict_json(raw)
    if not isinstance(obj, dict):
        raise ValueError('Companyfacts must be an object')
    actual = obj.get('cik')
    if isinstance(actual, bool) or not isinstance(actual, (str, int)):
        raise ValueError('Invalid companyfacts CIK')
    if not re.fullmatch(r'[0-9]{1,10}', str(actual)) or str(actual).zfill(10) != cik:
        raise ValueError('Companyfacts CIK mismatch')
    if not isinstance(obj.get('entityName'), str) or not obj['entityName'].strip():
        raise ValueError('Companyfacts entityName missing')
    if not isinstance(obj.get('facts'), dict) or any(not isinstance(taxonomy, dict) for taxonomy in obj['facts'].values()):
        raise ValueError('Companyfacts taxonomy objects missing')
    return obj


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def exclusive_bytes(path, raw):
    """Install complete bytes with an atomic no-replace hard link."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def load_plan(path, expected_sha, root=ROOT):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha or ''):
        raise ValueError('Frozen plan SHA-256 required')
    raw = Path(path).read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError('Frozen plan digest mismatch')
    plan = strict_json(raw)
    if not isinstance(plan, dict) or type(plan.get('schema_version')) is not int or plan['schema_version'] != 1 or plan.get('purpose') != PURPOSE:
        raise ValueError('Unsupported data-only plan')
    ciks = plan.get('allowed_ciks')
    if not isinstance(ciks, list) or not ciks or len(set(map(str, ciks))) != len(ciks):
        raise ValueError('Unique planned CIK list required')
    for cik in ciks:
        check_cik(cik)
    relative = plan.get('output_dir')
    if relative != 'outputs/monthly_profitability141_sources':
        raise ValueError('Dedicated source output directory required')
    output = Path(root).resolve() / relative
    if output.resolve() != output:
        raise ValueError('Source output may not redirect through symlinks')
    if plan.get('source_stop_timezone') != 'America/Los_Angeles':
        raise ValueError('Plan must preserve America/Los_Angeles source stop day')
    return plan, output


def retry_instant(value, now):
    if value is None:
        return None
    try:
        if re.fullmatch(r'[0-9]+', value.strip()):
            return now + timedelta(seconds=int(value.strip()))
        result = parsedate_to_datetime(value)
        if result.tzinfo is None:
            return None
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError, TypeError):
        return None


def check_stops(output, now, zone):
    today = now.astimezone(ZoneInfo(zone)).date().isoformat()
    for path in sorted(output.glob('STOPPED_*.json')):
        stop = strict_json(path.read_bytes())
        if stop.get('source_stop_timezone') != zone:
            raise ValueError('Unrecognized source stop timezone')
        retry = stop.get('retry_not_before_utc')
        bound = datetime.fromisoformat(retry) if retry else None
        if bound is not None and bound.utcoffset() is None:
            raise ValueError('Naive source retry bound')
        if stop.get('local_date') == today or (bound is not None and now < bound):
            raise ValueError('SEC source stopped for local day or Retry-After bound')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Even same-host redirects are not the registered exact endpoint.


def _read_response(response):
    raw = response.read(MAX_BYTES + 1)
    return raw[:MAX_BYTES], len(raw) <= MAX_BYTES


def collect(plan_path, plan_sha, cik, user_agent, *, root=ROOT, opener=None, now=None):
    """Collect one planned CIK. Tests inject an opener; no automatic retries."""
    plan, output = load_plan(plan_path, plan_sha, root)
    check_cik(cik)
    if cik not in plan['allowed_ciks']:
        raise ValueError('CIK absent from frozen plan')
    if not isinstance(user_agent, str) or not user_agent.strip() or '\r' in user_agent or '\n' in user_agent:
        raise ValueError('Explicit single-line User-Agent required')
    if 'user_agent' in plan and user_agent != plan['user_agent']:
        raise ValueError('User-Agent does not match frozen plan')
    now = now or datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise ValueError('Aware collection timestamp required')
    now = now.astimezone(timezone.utc)
    url = endpoint(cik)
    raw_path = output / 'raw' / ('CIK' + cik + '.json')
    receipt_path = output / 'receipts' / ('CIK' + cik + '.json')
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.collection.lock'
    fd = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    try:
        if raw_path.exists() or receipt_path.exists():
            if not raw_path.is_file() or not receipt_path.is_file():
                raise ValueError('Incomplete success cache; manual source review required')
            raw = raw_path.read_bytes()
            receipt = strict_json(receipt_path.read_bytes())
            if len(raw) > MAX_BYTES or receipt.get('sha256') != sha(raw) or receipt.get('http_status') != 200 or receipt.get('url') != url or receipt.get('cik') != cik or receipt.get('body_complete') is not True:
                raise ValueError('Success cache provenance mismatch')
            if type(receipt.get('schema_version')) is not int or receipt['schema_version'] != 1 or receipt.get('error') is not None or receipt.get('plan_sha256') != plan_sha:
                raise ValueError('Success cache receipt schema mismatch')
            try:
                timestamp = datetime.fromisoformat(receipt['retrieved_at_utc'])
                if timestamp.utcoffset() is None:
                    raise ValueError('Naive timestamp')
            except (KeyError, ValueError, TypeError):
                raise ValueError('Success cache timestamp invalid')
            validate_facts(raw, cik)
            return dict(status='reused', cik=cik, sha256=sha(raw), raw_path=str(raw_path))
        check_stops(output, now, plan['source_stop_timezone'])
        request = urllib.request.Request(url, headers={'User-Agent': user_agent, 'Accept': 'application/json'})
        opener = opener or urllib.request.build_opener(NoRedirect())
        status, headers, body, complete, error = None, {}, b'', False, None
        try:
            try:
                response = opener.open(request, timeout=TIMEOUT)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                status = response.code
                headers = {key.lower(): response.headers.get(key) for key in HEADERS if response.headers.get(key) is not None}
                if response.geturl() != url:
                    raise ValueError('Unregistered response URL')
                body, complete = _read_response(response)
            if not complete:
                error = 'response_exceeds_size_limit'
            elif status != 200:
                error = 'http_status_not_200'
            elif headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'application/json':
                error = 'response_content_type_not_json'
            else:
                try:
                    validate_facts(body, cik)
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    error = 'invalid_companyfacts_response'
        except Exception as exc:
            # Do not include request/exception text, which may expose headers.
            error = 'transport_' + type(exc).__name__
        receipt = dict(schema_version=1, cik=cik, url=url, retrieved_at_utc=now.isoformat(),
                       http_status=status, sha256=sha(body), bytes=len(body), body_complete=complete,
                       headers=headers, plan_sha256=plan_sha, error=error)
        if status in (403, 429):
            retry = retry_instant(headers.get('retry-after'), now)
            day = now.astimezone(ZoneInfo(plan['source_stop_timezone'])).date().isoformat()
            stop = dict(schema_version=1, host='data.sec.gov', http_status=status, local_date=day,
                        source_stop_timezone=plan['source_stop_timezone'], recorded_at_utc=now.isoformat(),
                        retry_after=headers.get('retry-after'), retry_not_before_utc=retry.isoformat() if retry else None,
                        plan_sha256=plan_sha)
            exclusive_bytes(output / ('STOPPED_' + day + '.json'), json_bytes(stop))
        if error is not None:
            name = now.strftime('%Y%m%dT%H%M%S%fZ') + '-CIK' + cik + '-' + uuid4().hex[:8]
            exclusive_bytes(output / 'failures' / (name + '.body'), body)
            exclusive_bytes(output / 'failures' / (name + '.json'), json_bytes(receipt))
            return dict(status='failed', cik=cik, error=error, http_status=status,
                        receipt_path=str(output / 'failures' / (name + '.json')))
        exclusive_bytes(raw_path, body)
        exclusive_bytes(receipt_path, json_bytes(receipt))
        return dict(status='saved', cik=cik, sha256=sha(body), raw_path=str(raw_path))
    finally:
        lock.unlink()


def filed_rows(rows, filed_through=None):
    """Calendar-date truncation only; does not assert next-session availability."""
    cutoff = iso_day(filed_through) if filed_through is not None else None
    kept, future, invalid = [], 0, 0
    for row in rows:
        try:
            filed = iso_day(row.get('filed'))
        except (AttributeError, ValueError, TypeError):
            invalid += 1
            continue
        if cutoff is not None and filed > cutoff:
            future += 1
        else:
            kept.append(row)
    return kept, future, invalid


def annual_period(row):
    if row.get('form') != '10-K':
        return None
    try:
        start, end = iso_day(row.get('start')), iso_day(row.get('end'))
        if end > iso_day(row.get('filed')):
            return None
        days = (end - start).days + 1
        return (start.isoformat(), end.isoformat()) if 330 <= days <= 400 else None
    except (ValueError, TypeError):
        return None


def _valid_instant(row):
    try:
        return iso_day(row.get('end')) <= iso_day(row.get('filed'))
    except (ValueError, TypeError):
        return False


def _amount(row):
    value = row.get('val')
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError('invalid_amount')
    value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError('invalid_amount')
    return value


def inspect_facts(raw, cik, filed_through=None):
    obj = validate_facts(raw, cik)
    gaap = obj['facts'].get('us-gaap', {})
    if not isinstance(gaap, dict):
        raise ValueError('Invalid us-gaap object')
    summaries, selected = {}, {}
    for tag in TAGS:
        concept = gaap.get(tag)
        if concept is None:
            summaries[tag] = dict(present=False, units={})
            continue
        if not isinstance(concept, dict) or not isinstance(concept.get('units'), dict):
            raise ValueError('Invalid concept units')
        units = {}
        for unit, rows in sorted(concept['units'].items()):
            if not isinstance(rows, list):
                raise ValueError('Invalid unit rows')
            kept, future, invalid = filed_rows(rows, filed_through)
            selected[tag, unit] = kept
            periods = Counter(annual_period(row) for row in kept)
            periods.pop(None, None)
            units[unit] = dict(row_count=len(rows), retained_row_count=len(kept),
                              after_filed_cutoff=future, invalid_filed_rows=invalid,
                              annual_periods=[dict(start=a, end=b, row_count=n) for (a, b), n in sorted(periods.items())],
                              instant_ends=sorted({row['end'] for row in kept if tag == 'Assets' and row.get('form') == '10-K' and 'start' not in row and _valid_instant(row)}),
                              forms=dict(sorted(Counter(str(row.get('form')) for row in kept).items())))
        summaries[tag] = dict(present=True, units=units)
    groups = defaultdict(list)
    for (tag, unit), rows in selected.items():
        if tag != 'GrossProfit' or not re.fullmatch(r'[A-Z]{3}', unit):
            continue
        for row in rows:
            period = annual_period(row)
            if period:
                groups[unit, row.get('accn'), row['filed'], period[1]].append(row)
    pairs = []
    for (unit, accn, filed, end), gross in sorted(groups.items(), key=lambda item: repr(item[0])):
        reasons = []
        if not isinstance(accn, str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accn):
            reasons.append('invalid_accession')
        starts = sorted({row['start'] for row in gross})
        if len(starts) != 1:
            reasons.append('ambiguous_annual_duration')
        assets = [row for row in selected.get(('Assets', unit), []) if row.get('accn') == accn and row.get('filed') == filed and row.get('end') == end and row.get('form') == '10-K' and 'start' not in row]
        if not assets:
            reasons.append('missing_same_filing_instant_assets')
        supplied_years = set()
        for row in gross + assets:
            if 'fp' in row and row['fp'] != 'FY':
                reasons.append('nonannual_fiscal_period_metadata')
            if 'fy' in row:
                if type(row['fy']) is not int or not 1900 <= row['fy'] <= 2200:
                    reasons.append('invalid_fiscal_year_metadata')
                else:
                    supplied_years.add(row['fy'])
        if len(supplied_years) > 1:
            reasons.append('conflicting_filing_fiscal_years')
        for name, rows in (('gross_profit', gross), ('assets', assets)):
            try:
                values = {_amount(row) for row in rows}
                if len(values) > 1:
                    reasons.append('conflicting_' + name + '_values')
                if name == 'assets' and any(v <= 0 for v in values):
                    reasons.append('nonpositive_assets')
            except ValueError:
                reasons.append('invalid_' + name + '_amount')
        pairs.append(dict(cik=cik, accession=accn, filed=filed, currency=unit, end=end,
                          gross_profit_starts=starts, gross_profit_rows=len(gross), assets_rows=len(assets),
                          status='ambiguous_or_unusable' if reasons else 'structurally_paired_not_admitted', reasons=sorted(set(reasons))))
    return dict(schema_version=1, purpose=PURPOSE, cik=cik, source_sha256=sha(raw),
                filed_through=filed_through, temporal_semantics=TEMPORAL_NOTE,
                tag_summaries=summaries, annual_gp_assets_pairs=pairs,
                pair_status_counts=dict(sorted(Counter(p['status'] for p in pairs).items())),
                factor_computed=False, factor_admitted=False, returns_computed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collection = commands.add_parser('collect')
    for name in ('plan', 'plan-sha256', 'cik', 'user-agent'):
        collection.add_argument('--' + name, required=True)
    inspection = commands.add_parser('inspect')
    inspection.add_argument('--facts', required=True)
    inspection.add_argument('--cik', required=True)
    inspection.add_argument('--filed-through')
    args = parser.parse_args(argv)
    if args.command == 'collect':
        result = collect(args.plan, args.plan_sha256, args.cik, args.user_agent)
    else:
        with Path(args.facts).open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('Facts file exceeds size limit')
        result = inspect_facts(raw, args.cik, args.filed_through)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 1 if result.get('status') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
