"""Validation-only report diagnostics for the already frozen v73 submission.

No fitting, checkpoint writes, TEST input/reference reads or variant selection.
Component deletion renormalizes the remaining frozen weights. These analyses
were added after final TEST scoring and do not change the submitted method.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['OPENBLAS_CORETYPE'] = 'Haswell'
os.environ['NPY_DISABLE_CPU_FEATURES'] = 'X86_V4,AVX512_ICL,AVX512_SPR'

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from common import DATA, load_bank, parse_answer, read_jsonl
from reference_rebuild_v73.features import groups
from reference_rebuild_v73.runtime import pool_choices, read_npz


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels(bank, split):
    if split not in ('train', 'validation'):
        raise ValueError('Report diagnostics may read TRAIN/validation only')
    references = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    result = []
    for row, ref in zip(bank, references, strict=True):
        assert row['id'] == ref['id'] and row['question'] == ref['question']
        answer = parse_answer(ref['answer'])
        result.append([parse_answer(c['text']) == answer for c in row['candidates']])
    return np.array(result, bool), references


def summarize_split(bank, y):
    parsed, same, valid = groups(bank)
    hard = np.array([len({v for v in row if v is not None}) >= 2 for row in parsed])
    majority = same.sum(-1).argmax(-1)
    trap = (y.sum(-1) == 1) & (same.sum(-1).max(-1) >= 2) & ~y[np.arange(len(y)), majority]
    return dict(questions=len(bank), candidates=4 * len(bank),
                informative_questions=int(hard.sum()), agreement_questions=int((~hard).sum()),
                wrong_majority_single_correct_questions=int(trap.sum()),
                invalid_candidates=int((~valid).sum()),
                correct_candidate_count_histogram={str(i): int((y.sum(-1) == i).sum()) for i in range(1, 5)}), hard, trap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('artifacts/v73'))
    parser.add_argument('--output-dir', type=Path, default=Path('../report/data'))
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    run = args.run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((run / 'freeze.json').read_text())
    for name, digest in freeze['source_sha256'].items():
        assert sha(code / name) == digest, name
    for name, digest in freeze['asset_sha256'].items():
        assert sha(run / 'checkpoint' / name) == digest, name
    config = json.loads((run / 'checkpoint/config.json').read_text())
    train = load_bank('train')
    train_y, _ = labels(train, 'train')
    train_info, _, _ = summarize_split(train, train_y)
    bank = load_bank('validation')
    y, references = labels(bank, 'validation')
    valid_info, hard, trap = summarize_split(bank, y)
    parsed, same, valid = groups(bank)
    stored = read_npz(run / 'validation-components.npz')
    weights = config['weights']
    full_probability = sum(w * stored[name] for name, w in weights.items())
    full_choices = np.asarray(pool_choices(full_probability, bank, config['pooling']))
    full_correct = y[np.arange(len(y)), full_choices]
    assert int(full_correct.sum()) == config['selected']['correct']

    def result(name, choices, kind, description):
        choices = np.asarray(choices)
        correct = y[np.arange(len(y)), choices]
        return dict(name=name, kind=kind, description=description,
                    correct=int(correct.sum()), n=len(y), accuracy=100 * float(correct.mean()),
                    informative_correct=int(correct[hard].sum()), informative_n=int(hard.sum()),
                    informative_accuracy=100 * float(correct[hard].mean()),
                    trap_correct=int(correct[trap].sum()), trap_n=int(trap.sum()),
                    delta_correct_vs_full=int(correct.sum() - full_correct.sum()),
                    full_fixes=int((full_correct & ~correct).sum()),
                    full_regressions=int((~full_correct & correct).sum()))

    majority = same.sum(-1).argmax(-1)
    baseline = [result('Majority answer', majority, 'baseline', 'Most frequent canonical valid answer; first candidate breaks ties.'),
                result('First candidate', np.zeros(len(y), int), 'baseline', 'Select candidate zero without learned evidence.')]
    components = [result(name, pool_choices(probability, bank, 1.), 'component', 'Frozen standalone component with same-answer mean pooling.')
                  for name, probability in stored.items() if name in weights]
    full = result('Full frozen ensemble', full_choices, 'primary', 'Frozen tree 1/2, balanced compact attention 1/3, rich attention 1/6; pooling 1.')
    ablations = []
    for removed in weights:
        kept = {name: value for name, value in weights.items() if name != removed}
        denominator = sum(kept.values())
        probability = sum(value * stored[name] for name, value in kept.items()) / denominator
        ablations.append(result('Without ' + removed, pool_choices(probability, bank, 1.), 'fixed_weight_deletion',
                                'Delete one component; renormalize remaining frozen weights; no reoptimization.'))
    ablations.append(result('Without answer pooling', pool_choices(full_probability, bank, 0.), 'decision_ablation',
                            'Keep all component probabilities and weights; select maximum candidate probability.'))
    strata = []
    for distinct in range(1, 5):
        selected = np.array([len({v for v in row if v is not None}) == distinct for row in parsed])
        if not selected.any():
            continue
        baseline_correct = y[np.arange(len(y)), majority]
        strata.append(dict(distinct_answers=distinct, n=int(selected.sum()),
                           full_correct=int(full_correct[selected].sum()), majority_correct=int(baseline_correct[selected].sum())))
    pooled = (same @ full_probability[..., None])[..., 0] / np.maximum(same.sum(-1), 1)
    margins = []
    for i, row in enumerate(parsed):
        unique = {answer: float(pooled[i, j]) for j, answer in enumerate(row) if answer is not None}
        scores = sorted(unique.values(), reverse=True)
        margins.append(scores[0] - scores[1] if len(scores) > 1 else np.nan)
    margins = np.asarray(margins)
    margin_bins = []
    for left, right in ((0., .1), (.1, .25), (.25, 1.01)):
        selected = hard & (margins >= left) & (margins < right)
        margin_bins.append(dict(lower=left, upper=right, n=int(selected.sum()),
                                correct=int(full_correct[selected].sum()),
                                accuracy=None if not selected.any() else 100 * float(full_correct[selected].mean())))
    majority_correct = y[np.arange(len(y)), majority]
    example_pools = {
        'full_correct_majority_wrong': np.flatnonzero(full_correct & ~majority_correct),
        'full_wrong_majority_correct': np.flatnonzero(~full_correct & majority_correct),
        'both_wrong': np.flatnonzero(~full_correct & ~majority_correct),
    }
    examples = []
    for category, indices in example_pools.items():
        if not len(indices):
            continue
        ordered = sorted(indices, key=lambda i: (float(margins[i]) if np.isfinite(margins[i]) else 0., int(i)))
        for i in ordered[:3]:
            examples.append(dict(category=category, split='validation', id=bank[i]['id'], question=bank[i]['question'],
                                 gold_answer=parse_answer(references[i]['answer']),
                                 full_candidate=int(full_choices[i]), majority_candidate=int(majority[i]),
                                 answer_score_gap=float(margins[i]), candidates=[dict(index=j, answer=parsed[i][j],
                                     text=candidate['text'], correct=bool(y[i, j]), probability=float(full_probability[i, j]),
                                     answer_mean_probability=float(pooled[i, j])) for j, candidate in enumerate(bank[i]['candidates'])]))
    metrics = dict(at_utc=datetime.now(timezone.utc).isoformat(), test_inputs_or_references_read=False,
                   diagnostics_written_after_final_test=True, final_method_changed=False,
                   plan='Fixed component deletions and no answer pooling. Validation diagnostics only; no new selection.',
                   freeze_sha256=sha(run / 'freeze.json'),
                   train=train_info, validation=valid_info, baseline=baseline, components=components,
                   full=full, ablations=ablations, distinct_answer_strata=strata, informative_answer_margin_bins=margin_bins,
                   limitations=['Validation was adaptively used during development.',
                                'These fixed-weight component and pooling diagnostics were added after final scoring and did not change the submitted model.',
                                'No component-level TEST scores or TEST error examples are computed.'])
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    (output / 'validation-cases.json').write_text(json.dumps(examples, indent=2, allow_nan=False) + '\n')
    rows = baseline + components + [full] + ablations
    with (output / 'validation-results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: metrics[k] for k in ('train', 'validation', 'baseline', 'components', 'full', 'ablations', 'informative_answer_margin_bins')}, indent=2))


if __name__ == '__main__':
    main()
