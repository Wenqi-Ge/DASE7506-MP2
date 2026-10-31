"""Inspect saved-asset size and a fresh validation predictor using stdlib only.

This is a diagnostic, not a replacement for the complete official TEST timing.
No labels are opened and no checkpoint/model source is modified.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_metadata(data):
    if data[:6] != b'\x93NUMPY':
        raise ValueError('NPY entry expected')
    version = data[6:8]
    size_field = 2 if version[0] == 1 else 4
    header_length = int.from_bytes(data[8:8 + size_field], 'little')
    offset = 8 + size_field
    header = ast.literal_eval(data[offset:offset + header_length].decode('latin1'))
    array_bytes = len(data) - offset - header_length
    return header['descr'], array_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('artifacts/v73'))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    run = args.run.resolve()
    freeze = json.loads((run / 'freeze.json').read_text())
    for name, digest in freeze['source_sha256'].items():
        assert sha(code / name) == digest, name
    for name, digest in freeze['asset_sha256'].items():
        assert sha(run / 'checkpoint' / name) == digest, name
    assets = []
    for path in sorted((run / 'checkpoint').iterdir()):
        row = dict(file=path.name, stored_bytes=path.stat().st_size, arrays=[])
        if path.suffix == '.npz':
            with zipfile.ZipFile(path) as archive:
                row['all_entries_uncompressed'] = all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
                for item in archive.infolist():
                    dtype, size = array_metadata(archive.read(item))
                    expanded = size * 2 if dtype[-2:] == 'f2' else size
                    row['arrays'].append(dict(name=item.filename, dtype=dtype, bytes=size, expanded_fp32_bytes=expanded))
        assets.append(row)
    stored_bytes = sum(row['stored_bytes'] for row in assets)
    expanded_bytes = sum(array['expanded_fp32_bytes'] for row in assets for array in row['arrays'])
    assert stored_bytes == freeze['inference_asset_bytes'] < 16 * 2**20
    assert expanded_bytes < 16 * 2**20
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    env['OPENBLAS_CORETYPE'] = 'Haswell'
    env['NPY_DISABLE_CPU_FEATURES'] = 'X86_V4,AVX512_ICL,AVX512_SPR'
    peak_threads, child_pids, gpu_libraries = 0, set(), set()
    with tempfile.TemporaryDirectory(prefix='mp2-resource-audit-') as directory:
        predictions = Path(directory) / 'validation-predictions.json'
        command = [args.python, str(code / 'predict.py'), '--method', str(code / 'submission_v73.py'),
                   '--checkpoint', str(run / 'checkpoint'), '--split', 'validation', '--output', str(predictions)]
        process = subprocess.Popen(command, cwd=code, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while process.poll() is None:
            try:
                status = Path(f'/proc/{process.pid}/status').read_text()
                count = next(int(line.split()[1]) for line in status.splitlines() if line.startswith('Threads:'))
                peak_threads = max(peak_threads, count)
                maps = Path(f'/proc/{process.pid}/maps').read_text()
                gpu_libraries.update(line.split()[-1] for line in maps.splitlines()
                                     if any(name in line.lower() for name in ('libcuda', 'libcublas', 'libtorch', 'libnccl')))
                for task in Path(f'/proc/{process.pid}/task').iterdir():
                    child_pids.update((task / 'children').read_text().split())
            except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
                pass
            time.sleep(.025)
        stdout, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(stdout + stderr)
        prediction_count = len(json.loads(predictions.read_text())['ids'])
    assert peak_threads <= 4 and not child_pids and not gpu_libraries
    result = dict(passed=True, at_utc=datetime.now(timezone.utc).isoformat(), split='validation',
                  references_read=False, test_accessed=False, frozen_core_and_assets_unchanged=True,
                  inference_files_bytes=stored_bytes, all_fp16_expanded_array_bytes=expanded_bytes,
                  inference_asset_limit_bytes=16 * 2**20, assets=assets,
                  inspected_predictor=dict(peak_os_threads=peak_threads, spawned_child_pids=sorted(child_pids),
                                           loaded_cuda_or_torch_libraries=sorted(gpu_libraries), questions=prediction_count),
                  note='Fresh-process validation resource diagnostic only; complete official TEST timing remains in the benchmark records.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: value for k, value in result.items() if k != 'assets'}, indent=2))


if __name__ == '__main__':
    main()
