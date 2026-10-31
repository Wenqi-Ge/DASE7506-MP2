"""Check release file hashes and deployed source/model integrity."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_text())
    for name, record in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid release path')
        path = root / relative
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Release hash mismatch: ' + name)
    run = root / 'code/artifacts/v73'
    freeze = json.loads((run / 'freeze.json').read_text())
    for name, digest in freeze['source_sha256'].items():
        if sha(root / 'code' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for name, digest in freeze['asset_sha256'].items():
        if sha(run / 'checkpoint' / name) != digest:
            raise ValueError('Frozen inference asset changed: ' + name)
    print(json.dumps(dict(passed=True, release_files=len(manifest['files']),
                          source_hashes_match=True, asset_hashes_match=True,
                          inference_asset_bytes=freeze['inference_asset_bytes']), indent=2))


if __name__ == '__main__':
    main()
