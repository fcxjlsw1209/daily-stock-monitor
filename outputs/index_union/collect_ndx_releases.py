"""Cache discovered primary releases, without certifying index completeness."""
import datetime
import hashlib
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent


def main():
    directory = ROOT / 'sources/ndx_releases'
    directory.mkdir(exist_ok=True)
    inventory = []
    for url in dict.fromkeys(json.loads((ROOT / 'ndx_globewire_queue.json').read_text())):
        path = directory / (hashlib.sha256(url.encode()).hexdigest()[:16] + '.html')
        provenance = path.with_suffix('.provenance.json')
        if path.exists() and provenance.exists():
            metadata = json.loads(provenance.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError('Cached source checksum mismatch: ' + str(path))
        else:
            response = requests.get(url, timeout=40)
            if response.status_code != 200:
                print('Stopped source:', response.status_code, 'Retry-After:', response.headers.get('Retry-After'), flush=True)
                break
            path.write_bytes(response.content)
            metadata = dict(url=url, sha256=hashlib.sha256(response.content).hexdigest(),
                            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            provenance.write_text(json.dumps(metadata, indent=2) + '\n')
            time.sleep(.5)
        soup = BeautifulSoup(path.read_text(), 'html.parser')
        body = soup.select_one('#main-body-container')
        if body is None:
            raise ValueError('Missing article body; do not treat navigation as evidence')
        path.with_suffix('.txt').write_text(body.get_text(' ', strip=True))
        inventory.append(dict(url=url, file=str(path.relative_to(ROOT)), reviewed=False))
        (ROOT / 'ndx_release_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        print('cached', len(inventory), flush=True)


if __name__ == '__main__':
    main()
