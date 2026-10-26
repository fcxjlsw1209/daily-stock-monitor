"""One observed public source per invocation; separate from offline research."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_source168/external_followup'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('url')
    parser.add_argument('--discovered-in', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9_]+', args.name)
    parsed = urlparse(args.url)
    assert parsed.scheme == 'https' and not parsed.username and not parsed.password
    assert parsed.hostname not in {'www.sec.gov', 'sec.gov', 'data.sec.gov'}, 'SEC stopped after 403'
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob('*.receipt.json'):
        prior = json.loads(path.read_text())
        if prior.get('status') in [403, 429] and prior.get('requested_utc', '')[:10] == now()[:10]:
            assert urlparse(prior['url']).hostname != parsed.hostname, 'Host stopped today'
    plan = dict(name=args.name, url=args.url, discovered_in=args.discovered_in,
                registered_utc=now(), maximum_requests=1, timeout_seconds=25,
                maximum_bytes=12000000, no_retry=True, purpose='Source evidence only',
                no_market_cap_rank_factor_or_return_computation=True,
                collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_once(OUT / (args.name + '.plan.json'), plan)
    result = dict(name=args.name, url=args.url, requested_utc=now())
    try:
        request = Request(args.url, headers={'User-Agent': 'StockResearchSourceAudit/1.0 (public historical source research)'})
        try:
            response = urlopen(request, timeout=25)
        except HTTPError as error:
            response = error
        with response:
            result.update(status=response.status, final_url=response.geturl(),
                          content_type=response.headers.get('Content-Type'),
                          retry_after=response.headers.get('Retry-After'))
            body = response.read(12000001)
        assert len(body) <= 12000000, 'Source exceeds declared byte limit'
        target = OUT / (args.name + '.body')
        with target.open('xb') as handle:
            handle.write(body)
        result.update(path=str(target.relative_to(ROOT)), bytes=len(body),
                      sha256=hashlib.sha256(body).hexdigest(), stop_host_for_today=result['status'] in [403, 429])
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    result['completed_utc'] = now()
    write_once(OUT / (args.name + '.receipt.json'), result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
