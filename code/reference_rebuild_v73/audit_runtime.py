"""NumPy-only validation deployment checks, with reference I/O blocked."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['OPENBLAS_CORETYPE'] = 'Haswell'
os.environ['NPY_DISABLE_CPU_FEATURES'] = 'X86_V4,AVX512_ICL,AVX512_SPR'

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common import load_inputs, parse_answer
from .runtime import components, read_npz, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bank, arrays = load_inputs('validation')
    config = json.loads((args.run / 'checkpoint/config.json').read_text())
    stored = read_npz(args.run / 'validation-components.npz')
    with patch('common.read_jsonl', side_effect=AssertionError('reference read during inference')):
        replay = components(bank, arrays, args.run / 'checkpoint', list(config['weights']))
        errors = {k: float(np.max(np.abs(v - stored[k]))) for k, v in replay.items()}
        assert max(errors.values()) < 5e-6
        original = predict(bank, arrays, args.run / 'checkpoint')
        rng = np.random.default_rng(731)
        selected = rng.choice(len(bank), 40, replace=False)
        candidate_orders = np.stack([rng.permutation(4) for _ in selected])
        reordered = []
        vectors = []
        for i, order in zip(selected, candidate_orders):
            reordered.append(dict(bank[i], id='unseen-audit-' + str(i), candidates=[dict(bank[i]['candidates'][int(j)], index=k) for k, j in enumerate(order)]))
            vectors.append(arrays['embeddings'][i, order])
        changed = predict(reordered, {'embeddings': np.asarray(vectors)}, args.run / 'checkpoint')
        for i, order, row, choice in zip(selected, candidate_orders, reordered, changed):
            assert parse_answer(row['candidates'][choice]['text']) == parse_answer(bank[i]['candidates'][original[i]]['text'])
        singles = []
        for i in selected[:8]:
            choice = predict([dict(bank[i], id='fresh-single')], {'embeddings': arrays['embeddings'][i:i+1]}, args.run / 'checkpoint')[0]
            assert choice == original[i]
            singles.append(int(i))
        assert predict([], {}, None) == []
    result = dict(passed=True, at_utc=datetime.now(timezone.utc).isoformat(), test_accessed=False,
                  numpy=np.__version__, validation_component_errors=errors, reference_reads_blocked=True,
                  reordered_questions=40, unseen_ids=40, candidate_permutations=40,
                  single_calls=len(singles), empty_bank=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
