"""Enumerate public archive links, preserving every fetched source page."""
from pathlib import Path
from urllib.parse import urljoin
import json
import re
import shutil
from bs4 import BeautifulSoup
from collector import P, fetch, read, write, digest, now


def reuse_old_sources():
    old = P.parents[1] / 'monthly_cost_portfolio_research/events'
    records = read('manifest.json', {})
    for meta in sorted((old / 'raw').glob('NVDA_*.meta.json')):
        obj = json.loads(meta.read_text())
        if obj.get('http_status') != 200:
            continue
        body = meta.with_name(meta.name.replace('.meta.json', '.html')).read_bytes()
        assert digest(body) == obj['sha256']
        source = obj['source']
        dest = 'raw/' + digest(source.encode())[:24] + '.html'
        (P / dest).write_bytes(body)
        records[source] = dict(url=source, final_url=obj['final_url'],
            retrieved_at=obj['retrieved_at'], http_status=200, content_type=obj['content_type'],
            evidence_path=dest, sha256=digest(body), bytes=len(body),
            public_unauthenticated=True, reused_from=str(meta))
    # Prior source observations distinguish exact newswire times from dates.
    for file in old.glob('NVDA_*wire_observation.json'):
        shutil.copyfile(file, P / file.name)
    write('manifest.json', records)


def main():
    reuse_old_sources()
    base = 'https://nvidianews.nvidia.com/news?q=financial+results'
    record, text = fetch(base)
    if record['http_status'] != 200:
        raise RuntimeError('Official archive unavailable')
    soup = BeautifulSoup(text, 'html.parser')
    pages = [int(m.group(1)) for a in soup.select('a[href]')
             for m in [re.search(r'[?&]page=(\d+)', a['href'])] if m]
    last = max(pages)
    links = {}
    archives = []
    for page in range(1, last + 1):
        url = base if page == 1 else base + '&page=' + str(page)
        record, text = fetch(url)
        archives.append(record)
        if record['http_status'] != 200:
            continue
        soup = BeautifulSoup(text, 'html.parser')
        found = 0
        for anchor in soup.select('a[href]'):
            href = urljoin(url, anchor['href'])
            if '/news/' not in href or not re.search(r'/nvidia-(?:announces-)?financial-results-for-', href):
                continue
            match = re.search(r'fiscal-(\d{4})', href)
            if not match or not 2015 <= int(match.group(1)) <= 2027:
                continue
            item = links.setdefault(href, dict(url=href, archive_sources=[], title_candidates=[]))
            if url not in item['archive_sources']:
                item['archive_sources'].append(url)
            title = anchor.get_text(' ', strip=True)
            if 'Financial Results' in title and title not in item['title_candidates']:
                item['title_candidates'].append(title)
            found += 1
        write('release_catalog.json', dict(created_at=now(), discovery='All pages of official financial-results news archive',
            archive_last_page=last, completed_pages=len(archives), archives=archives, releases=list(links.values())))
        print('archive', page, 'quarter links', found, 'unique total', len(links), flush=True)


if __name__ == '__main__':
    main()
