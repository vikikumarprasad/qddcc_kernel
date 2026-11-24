"""Evaluate saved blind-test amplitudes in the CCSD correlation-energy expression.

Energies are evaluated with Coulomb integrals and amplitudes stored in the same
orbital basis. MP2 and ML energies use zero singles without CCSD iterations.
The reference uses the stored CCSD doubles and singles. For the canonical
RHF orbitals in these datasets, the occupied-virtual Fock block vanishes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_pickle(path):
    if int(np.__version__.split('.')[0]) < 2:
        import numpy.core.multiarray as multiarray
        import numpy.core.numeric as numeric
        sys.modules.setdefault('numpy._core.multiarray', multiarray)
        sys.modules.setdefault('numpy._core.numeric', numeric)
    return pd.read_pickle(path)


def load_reference(data_root, molecule):
    folder = data_root / 'blind' / molecule / 'pickle'
    data = read_pickle(folder / f'{molecule}_alldata.pkl.gz')
    singles = read_pickle(folder / f't1_{molecule}_alldata.pkl.gz')['t1'].to_numpy()
    required = ['Eocc1', 'Eocc2', 'Evir1', 'Evir2', 'doublecheck', 'orbdiff', 't2start', 't2']
    if not np.isfinite(data[required].to_numpy()).all() or not np.isfinite(singles).all():
        raise ValueError(f'{molecule}: nonfinite reference data')
    # Validate C-order tensor dimensions using all four orbital-energy columns.
    shapes = []
    for occupied in range(1, len(singles) + 1):
        if len(singles) % occupied:
            continue
        virtual = len(singles) // occupied
        shape = (occupied, occupied, virtual, virtual)
        if np.prod(shape) != len(data):
            continue
        valid = True
        # The feature generator names the b-orbital energy Evir1 and a as Evir2.
        for axis, column in enumerate(['Eocc1', 'Eocc2', 'Evir2', 'Evir1']):
            tensor = data[column].to_numpy().reshape(shape)
            index = [0, 0, 0, 0]
            index[axis] = slice(None)
            dimensions = [1, 1, 1, 1]
            dimensions[axis] = shape[axis]
            valid &= np.allclose(tensor, tensor[tuple(index)].reshape(dimensions), atol=1e-12, rtol=0)
        if valid:
            shapes.append(shape)
    if len(shapes) != 1:
        raise ValueError(f'{molecule}: cannot uniquely determine orbital dimensions: {shapes}')
    shape = shapes[0]
    integral = data['doublecheck'].to_numpy().reshape(shape)
    mp2 = data['t2start'].to_numpy()
    if not np.allclose(mp2 * data['orbdiff'].to_numpy(), integral.ravel(), atol=1e-10, rtol=1e-8):
        raise ValueError(f'{molecule}: MP2 amplitudes and stored integrals disagree')
    weights = (2 * integral - integral.swapaxes(2, 3)).ravel()
    t1 = singles.reshape(shape[0], shape[2])
    target = data['t2'].to_numpy()
    tau = target + np.einsum('ia,jb->ijab', t1, t1).ravel()
    return {'weights': weights, 'target': target, 'mp2': mp2, 'ccsd': float(tau @ weights)}


def collect_predictions(path, molecules=None):
    groups = defaultdict(list)
    counters = defaultdict(int)
    for chunk in pd.read_csv(path, chunksize=250000):
        required = {'Molecule', 'Circuit', 'Expected', 'Predicted'}
        if not required.issubset(chunk.columns):
            raise ValueError(f'Missing prediction columns: {required - set(chunk.columns)}')
        if chunk[['Molecule', 'Circuit']].isna().any().any():
            raise ValueError('Prediction records have missing molecule/circuit labels')
        if molecules:
            chunk = chunk[chunk.Molecule.isin(molecules)]
        for key, frame in chunk.groupby(['Molecule', 'Circuit'], sort=False):
            frame = frame.copy()
            if 'Position' not in frame:
                frame['Position'] = np.arange(counters[key], counters[key] + len(frame))
                counters[key] += len(frame)
            values = frame[['Position', 'Expected', 'Predicted']].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f'{key}: nonfinite predictions, targets or positions')
            if (values[:, 0] < 0).any() or (values[:, 0] != np.floor(values[:, 0])).any():
                raise ValueError(f'{key}: positions must be nonnegative integers')
            groups[key].append(values)
    if not groups:
        raise ValueError('No prediction records selected')
    if molecules and set(molecules) - {key[0] for key in groups}:
        raise ValueError('One or more selected molecules have no prediction records')
    return groups


def aggregate(values, reference, incomplete, label):
    n = len(reference['target'])
    positions = values[:, 0].astype(int)
    if (positions >= n).any():
        raise ValueError(f'{label}: positions exceed the {n} stored amplitudes; repeated exports need explicit Position values')
    # Split at non-increasing positions. Only an exact 0..n-1 sequence is
    # eligible for complete-export averaging; duplicate fragments are excluded.
    resets = np.diff(positions) <= 0
    starts = np.r_[0, np.flatnonzero(resets) + 1, len(positions)]
    complete, missing = [], 0
    for start, end in zip(starts[:-1], starts[1:]):
        part = values[start:end]
        pos = part[:, 0].astype(int)
        if len(pos) == n and np.array_equal(pos, np.arange(n)):
            if not np.allclose(part[:, 1], reference['target'], atol=1e-10, rtol=1e-8):
                raise ValueError(f'{label}: Expected values disagree with the stored CCSD amplitudes')
            complete.append(part[:, 2])
        else:
            missing += 1
    if missing and incomplete == 'error':
        raise ValueError(f'{label}: {missing} incomplete exports; choose an explicit --incomplete policy')
    if incomplete == 'position-mean':
        if not np.allclose(values[:, 1], reference['target'][positions], atol=1e-10, rtol=1e-8):
            raise ValueError(f'{label}: partial exports have misaligned Expected values; use --incomplete exclude')
        counts = np.bincount(positions, minlength=n)
        if (counts == 0).any():
            raise ValueError(f'{label}: some amplitudes have no prediction')
        predicted = np.bincount(positions, weights=values[:, 2], minlength=n) / counts
    else:
        if not complete:
            raise ValueError(f'{label}: no complete exports')
        predicted = np.mean(complete, axis=0)
        counts = np.full(n, len(complete))
    if missing:
        print(f'{label}: {len(complete)} complete, {missing} incomplete exports; policy={incomplete}', file=sys.stderr)
    return predicted, len(complete), missing, int(counts.min()), int(counts.max())


def energy_record(molecule, circuit, predicted, reference):
    energy = float(predicted @ reference['weights'])
    return {'Molecule': molecule, 'Circuit': circuit,
            't2 MA': float(np.abs(predicted - reference['target']).mean()),
            'Energy Devience': abs(energy - reference['ccsd']),
            'Correlation Energy': energy, 'CCSD Correlation Energy': reference['ccsd']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--molecules', nargs='+')
    parser.add_argument('--incomplete', choices=['error', 'exclude', 'position-mean'], default='error',
                        help='Reject incomplete exports (default), exclude them, or average available rows at each position; the latter can use unequal counts.')
    parser.add_argument('--output', type=Path, default=ROOT / 'paper_results/blind/recomputed_energy.csv')
    args = parser.parse_args()
    if args.output.resolve() in {args.predictions.resolve(), (ROOT / 'Experiments/results/blind/energy_mae.csv').resolve()}:
        parser.error('Use a separate output path for calculated energies')
    groups = collect_predictions(args.predictions, args.molecules)
    references = {m: load_reference(args.data_root, m) for m in sorted({key[0] for key in groups})}
    records = [energy_record(m, 'MP2', ref['mp2'], ref) for m, ref in references.items()]
    for molecule, circuit in sorted(groups):
        if circuit == 'MP2':
            raise ValueError('Prediction exports must contain model predictions; MP2 is calculated from the reference data')
        values = np.concatenate(groups[(molecule, circuit)])
        prediction, complete, missing, nmin, nmax = aggregate(
            values, references[molecule], args.incomplete, f'{molecule}/{circuit}')
        row = energy_record(molecule, circuit, prediction, references[molecule])
        row.update({'Complete exports': complete, 'Incomplete exports': missing,
                    'Min observations': nmin, 'Max observations': nmax, 'Incomplete policy': args.incomplete})
        records.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False, float_format='%.15g')
    print(f'Wrote {len(records)} energy records to {args.output}')


if __name__ == '__main__':
    main()
