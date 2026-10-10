"""Download and verify only the three assays named in the frozen manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main(a):
    root=Path(a.dataset).resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    archive=Path(a.archive).resolve();archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        partial=archive.with_suffix(archive.suffix+'.partial')
        urllib.request.urlretrieve(manifest['sources']['data'],partial)
        if sha256(partial)!=manifest['archive_sha256']:raise ValueError('Downloaded archive differs from frozen input')
        partial.replace(archive)
    if sha256(archive)!=manifest['archive_sha256']:raise ValueError('Archive SHA-256 differs from reference')
    with zipfile.ZipFile(archive) as z:
        for assay in manifest['quality_assays']:
            dest=root/Path(assay['path']).name
            value=z.read(assay['member'])
            if hashlib.sha256(value).hexdigest()!=assay['sha256']:raise ValueError('Assay SHA-256 mismatch: '+assay['id'])
            if dest.exists() and sha256(dest)!=assay['sha256']:raise FileExistsError('Different existing assay: '+str(dest))
            if not dest.exists():dest.write_bytes(value)
            print('Verified',assay['id'],assay['rows'],'rows')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='inputs/proteingym_v1.3_20261009')
    p.add_argument('--archive',default='cache/DMS_ProteinGym_substitutions_v1.3.zip')
    main(p.parse_args())
