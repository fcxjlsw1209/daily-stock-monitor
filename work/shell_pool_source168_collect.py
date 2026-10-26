"""Separate, bounded public-source collection; never used by a replay."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_source168/external'
SOURCES = {
    'current_vendor_history': 'https://companiesmarketcap.com/shell/marketcap/',
    'issuer_2016q2_ads_ratio': 'https://www.shell.com/investors/dividend-information/dividend-announcements/second-quarter-2016-interim-dividend.html',
    'sec_20160919_issuance': 'https://www.sec.gov/Archives/edgar/data/1306965/000130901416001314/htm_10456.htm',
    'issuer_wire_20160729_capital': 'https://www.prnewswire.com/news-releases/royal-dutch-shell-plc---voting-rights-and-capital-588638281.html',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_once(OUT / 'COLLECTION_PLAN.json', dict(
        created_utc=now(), sources=SOURCES, maximum_requests=4, timeout_seconds=25,
        no_retry=True, stop_host_on=[403, 429],
        purpose='Save current retrieved source bodies for class-cap provenance; not the missing historical vendor snapshot and not proof of exact PIT data vintage.',
        strategy_trials=0, replay_allowed=False))
    results = []
    stopped_hosts = set()
    for name, url in SOURCES.items():
        host = urlparse(url).hostname
        if host in stopped_hosts:
            results.append(dict(name=name, url=url, skipped='host stopped'))
            continue
        record = dict(name=name, url=url, requested_utc=now(), method='GET')
        try:
            request = Request(url, headers={'User-Agent': 'StockResearchSourceAudit/1.0 (public historical filing research)'})
            try:
                response = urlopen(request, timeout=25)
            except HTTPError as error:
                response = error
            with response:
                body = response.read(8 * 1024 * 1024 + 1)
                if len(body) > 8 * 1024 * 1024:
                    raise ValueError('Response exceeds fixed source limit')
                record.update(status=response.status, final_url=response.geturl(),
                              content_type=response.headers.get('Content-Type'),
                              retry_after=response.headers.get('Retry-After'))
            if record['status'] in (403, 429):
                stopped_hosts.add(host)
                record['stop_host_for_today'] = True
            destination = OUT / (name + '.html')
            with destination.open('xb') as handle:
                handle.write(body)
            record.update(path=str(destination.relative_to(ROOT)), bytes=len(body),
                          sha256=hashlib.sha256(body).hexdigest(), completed_utc=now())
        except (OSError, URLError, ValueError) as error:
            record.update(error_type=type(error).__name__, error=str(error), completed_utc=now())
        write_once(OUT / (name + '.receipt.json'), record)
        results.append(record)
    write_once(OUT / 'COLLECTION.json', dict(completed_utc=now(), records=results,
                                            source_admission=False, strategy_trials=0))
    print(json.dumps([{'name': r['name'], 'status': r.get('status'), 'bytes': r.get('bytes'),
                       'error': r.get('error_type')} for r in results]))


if __name__ == '__main__':
    main()
