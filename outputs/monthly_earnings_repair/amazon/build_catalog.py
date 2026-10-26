"""Enumerate quarter-release archive matches; never download other articles."""
from urllib.parse import urljoin, urlsplit
import re
from bs4 import BeautifulSoup
from collector import P, fetch, read, write, now

RELEASE_TITLE = re.compile(r'^Amazon\.com Announces (?:(First|Second|Third|Fourth) Quarter (?:Results|Sales)|Financial Results and CEO Transition)', re.I)


def main():
    links, pages, external = {}, [], []
    # The archive language taxonomy omits migrated US releases; do not set f0.
    for year in range(2015, 2027):
        url = 'https://press.aboutamazon.com/press-release-archive?q=quarter&f2=' + str(year)
        seen = set()
        while url and url not in seen:
            seen.add(url)
            record, text = fetch(url)
            pages.append(record)
            if record['http_status'] != 200:
                break
            soup = BeautifulSoup(text, 'html.parser')
            for heading in soup.select('h2'):
                anchor = heading.find('a', href=True)
                if anchor is None:
                    continue
                title = heading.get_text(' ', strip=True)
                if not RELEASE_TITLE.search(title):
                    continue
                href = urljoin(url, anchor['href'])
                item = links.setdefault(href, dict(url=href, title=title, calendar_release_year=year, archive_sources=[]))
                if url not in item['archive_sources']:
                    item['archive_sources'].append(url)
                if '/https:/ir.aboutamazon.com/' in href:
                    item['malformed_external_ir_reference'] = True
                    item['referenced_ir_url'] = href.replace('https://press.aboutamazon.com/https:/', 'https://')
            next_links = [anchor['href'] for anchor in soup.select('a[href]') if anchor.get_text(' ', strip=True) == 'Load more']
            url = urljoin(url, next_links[0]) if next_links else None
            write('release_catalog.json', dict(created_at=now(), method='All yearly quarter search result pages, 2015 through2026, strict earnings-title filter',
                language_filter_omitted_due_to_incomplete_migrated_taxonomy=True,
                archives=pages, releases=list(links.values())))
        print('year', year, 'pages', len(seen), 'total release URLs', len(links), flush=True)


if __name__ == '__main__':
    main()
