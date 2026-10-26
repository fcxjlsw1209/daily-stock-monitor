"""One bounded public quote-table request using the issuer-embedded form."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_source168/external_followup'
PAGE = OUT / 'euroland_old_structure.body'
SCRIPT = OUT / 'euroland_form.body'
URL = 'https://tools.eurolandir.com/tools/splookup/Download/SearchResult?companycode=uk-rdsa&v=old&lang=en-gb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')


def main():
    assert sha(PAGE) == '074b6c92617139d846fd482ed631992600a8390993278bcb61337f2c91e6bb6b'
    assert sha(SCRIPT) == 'c3ec74e39c5f82aa199acba139974825464a83e21f083ff29eaa3c1556a660e2'
    text = PAGE.read_text()
    for value in ['instrument="20642" isin="US7802592060" market="12" currency="USD"',
                  'instrument="20643" isin="US7802591070" market="12" currency="USD"',
                  '/tools/splookup/Download/SearchResult']:
        assert value in text
    for path in OUT.glob('*.receipt.json'):
        old = json.loads(path.read_text())
        if old.get('status') in [403, 429] and old.get('requested_utc', '')[:10] == now()[:10]:
            assert urlparse(old['url']).hostname != 'tools.eurolandir.com', 'Source stopped today'
    form = dict(shareList='20642;US7802592060;12;USD,20643;US7802591070;12;USD',
                currency='USD', fromDate='2016-01-01T00:00:00.000Z', toDate='2020-11-30T00:00:00.000Z')
    plan = dict(registered_utc=now(), endpoint=URL, method='POST', form=form,
                source_page_sha256=sha(PAGE), source_script_sha256=sha(SCRIPT),
                collector_sha256=sha(Path(__file__)), maximum_requests=1, no_retry=True,
                timeout_seconds=30, maximum_bytes=12000000, strategy_trials=0,
                scope='Collect issuer-embedded historical quote response only; no price transformation, cap, rank, factor or return calculation.')
    write_once(OUT / 'euroland_price_preview.plan.json', plan)
    receipt = dict(requested_utc=now(), url=URL, method='POST', form=form)
    try:
        request = Request(URL, data=urlencode(form).encode(), headers={
            'User-Agent': 'StockResearchSourceAudit/1.0 (public historical source research)',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': 'https://tools.eurolandir.com/tools/splookup/?companycode=uk-rdsa&v=old&lang=en-gb'})
        try:
            response = urlopen(request, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            receipt.update(status=response.status, final_url=response.geturl(),
                           content_type=response.headers.get('Content-Type'),
                           retry_after=response.headers.get('Retry-After'))
            body = response.read(12000001)
        assert len(body) <= 12000000
        path = OUT / 'euroland_price_preview.body'
        with path.open('xb') as handle:
            handle.write(body)
        receipt.update(bytes=len(body), path=str(path.relative_to(ROOT)), sha256=sha(path),
                       stop_host_for_today=receipt['status'] in [403, 429])
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    receipt['completed_utc'] = now()
    write_once(OUT / 'euroland_price_preview.receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
