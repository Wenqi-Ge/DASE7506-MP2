"""Replay TRAIN-derived statistics and held-question predictions; never open TEST."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['OPENBLAS_CORETYPE'] = 'Haswell'
os.environ['NPY_DISABLE_CPU_FEATURES'] = 'X86_V4,AVX512_ICL,AVX512_SPR'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import load_inputs, read_jsonl, parse_answer, DATA
from .features import rich_and_compact, quantitative, text_scores
from .stack import VIEWS, balanced_copies, view_matrix, margin_scores, text_matrix
from .runtime import read_npz, neural_inputs, NETWORKS, components, pool_choices


def digest(indices):
    return hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest()


def targets(bank, split):
    if split not in ('train', 'validation'):
        raise ValueError('TRAIN/validation only')
    references = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    return np.array([[parse_answer(c['text']) == parse_answer(ref['answer'])
                      for c in row['candidates']]
                     for row, ref in zip(bank, references, strict=True)], bool)


def check_equal(left, right, tolerance=0.):
    error = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'Replay error {error} > {tolerance}')
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    record = json.loads((run / 'experiments.json').read_text())
    bank, arrays = load_inputs('train')
    vb, va = load_inputs('validation')
    y, vy = targets(bank, 'train'), targets(vb, 'validation')
    cached = read_npz(run / 'search/features.npz')
    rich, compact, repair = rich_and_compact(bank)
    quantity, schemas = quantitative(bank)
    vr, vc, vp = rich_and_compact(vb)
    vq, vs = quantitative(vb)
    feature_error = max(check_equal(actual, cached[key]) for key, actual in (
        ('rich', rich), ('compact', compact), ('repair', repair),
        ('quantity', quantity), ('schemas', schemas), ('validation_rich', vr),
        ('validation_compact', vc), ('validation_repair', vp),
        ('validation_quantity', vq), ('validation_schemas', vs)))
    evidence = read_npz(run / 'search/base-evidence.npz')
    stack = read_npz(run / 'search/stack.npz')
    fold_records = []
    partitions = StratifiedKFold(5, shuffle=True, random_state=7526).split(np.arange(len(bank)), y.sum(-1))
    for fold, (fit, held) in enumerate(partitions):
        expected = record['base_oof'][fold]
        assert digest(fit) == expected['fit_hash'] and digest(held) == expected['held_hash']
        assert not np.intersect1d(fit, held).size
        fb, hb = [bank[i] for i in fit], [bank[i] for i in held]
        fa, ha = {'embeddings': arrays['embeddings'][fit]}, {'embeddings': arrays['embeddings'][held]}
        ab, aa, _, origins = balanced_copies(fb, fa, y[fit])
        # Every augmented row maps to a fit question; held originals cannot enter.
        assert np.isin(fit[origins], fit).all() and not np.isin(fit[origins], held).any()
        ar, _, _ = rich_and_compact(ab)
        saved = read_npz(run / 'search' / f'fold{fold}-bases.npz')
        normalization_error = 0.
        prediction_error = 0.
        for j, (mode, balanced, _) in enumerate(VIEWS):
            b, a, r = (ab, aa, ar) if balanced else (fb, fa, rich[fit])
            matrix = view_matrix(b, a, r, mode)
            mean = matrix.mean(0, dtype=np.float64).astype(np.float32)
            scale = matrix.std(0, dtype=np.float64)
            scale = np.where(scale < 1e-10, 1., scale).astype(np.float32)
            model = dict(mode=mode, **{key: saved[f'base{j}_{key}'] for key in ('mean', 'scale', 'weights')})
            normalization_error = max(normalization_error, check_equal(mean, model['mean']), check_equal(scale, model['scale']))
            prediction_error = max(prediction_error, check_equal(margin_scores(model, hb, ha, rich[held]), evidence['oof'][held, :, j], 1e-7))
            del matrix
        row = dict(fold=fold, fit_questions=len(fit), held_questions=len(held),
                   normalization_error=normalization_error, oof_prediction_error=prediction_error, augmentation_leakage=False)
        fold_records.append(row)
        print(json.dumps(row), flush=True)
    ab, aa, ay, origins = balanced_copies(bank, arrays, y)
    er, ec, ep = rich_and_compact(ab[len(bank):])
    ar, ac, ap = np.concatenate((rich, er)), np.concatenate((compact, ec)), np.concatenate((repair, ep))
    full_base_error = 0.
    for j, (mode, balanced, _) in enumerate(VIEWS):
        b, a, r = (ab, aa, ar) if balanced else (bank, arrays, rich)
        matrix = view_matrix(b, a, r, mode)
        mean = matrix.mean(0, dtype=np.float64).astype(np.float32)
        scale = matrix.std(0, dtype=np.float64)
        scale = np.where(scale < 1e-10, 1., scale).astype(np.float32)
        full_base_error = max(full_base_error, check_equal(mean, stack[f'base{j}_mean']), check_equal(scale, stack[f'base{j}_scale']))
        del matrix
    neural_records = []
    held_folds = np.split(np.random.default_rng(13).permutation(len(bank)), 5)
    for family, config in NETWORKS.items():
        raw = neural_inputs(aa, ar, ac, ap, family) if config['balanced'] else neural_inputs(arrays, rich, compact, repair, family)
        saved = read_npz(run / 'search' / (family + '.npz'))
        members = [r for r in record['neural'] if r['family'] == family]
        for row in members:
            member = row['member']
            fit = np.arange(len(bank)) if family == 'rich' else np.setdiff1d(np.arange(len(bank)), held_folds[member])
            assert digest(fit) == row['fit_hash']
            selected = np.flatnonzero(np.isin(origins, fit)) if config['balanced'] else fit
            flat = (selected[:, None] * 4 + np.arange(4)).ravel()
            # Select original questions, then flatten all four candidate rows.
            selected_rows = raw.reshape(-1, raw.shape[-1])[flat]
            error = max(check_equal(selected_rows.mean(0), saved['mean'][member]),
                        check_equal(selected_rows.std(0) + 1e-6, saved['scale'][member]))
            if family != 'rich' and config['balanced']:
                assert not np.isin(origins[selected], held_folds[member]).any()
            neural_records.append(dict(family=family, member=member, fit_questions=len(fit), fit_candidate_rows=len(flat), normalization_error=error))
    counts = text_matrix(bank)
    df = np.asarray((counts != 0).sum(0)).ravel()
    idf = (np.log((len(bank) * 4 + 1) / (df + 1)) + 1).astype(np.float32)
    lexical_error = check_equal(idf, stack['text_idf'])
    lexical_prediction_error = check_equal(text_scores(vb, idf, stack['text_coef']), evidence['text_validation'], 1e-7)
    pca_mean_error = check_equal(arrays['embeddings'].reshape(-1, 1024).mean(0), stack['projection_mean'])
    replay = components(vb, va, run / 'search', list(read_npz(run / 'validation-components.npz')))
    saved_components = read_npz(run / 'validation-components.npz')
    component_errors = {name: check_equal(value, saved_components[name], 1e-7) for name, value in replay.items()}
    selected = record['selected']
    probability = sum(w * replay[name] for name, w in selected['weights'].items())
    choices = pool_choices(probability, vb, selected['pooling'])
    correct = int(vy[np.arange(len(vy)), choices].sum())
    assert correct == selected['correct']
    result = dict(passed=True, at_utc=datetime.now(timezone.utc).isoformat(), test_accessed=False,
                  feature_replay_error=feature_error, base_folds=fold_records, full_base_normalization_error=full_base_error,
                  neural=neural_records, lexical_idf_error=lexical_error, lexical_validation_error=lexical_prediction_error,
                  pca_train_mean_error=pca_mean_error, validation_components=component_errors, validation_correct=correct,
                  limitations=['Does not refit the optimizers or claim full nested pipeline cross-validation.',
                               'Validation is an adaptively used development set.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
