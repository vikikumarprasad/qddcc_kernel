# Experiments and numerical results

Run commands from the repository root in the `qddcc_kernel` Conda environment.

| Paper result | Scripts | Saved data under `results/` |
| --- | --- | --- |
| Figure 4b / Table S2: cc-pVDZ learning curves | `run_water_learning_curve.py`, `make_table_s2.py` | `water_learning_curve/` |
| Figure 5 / Table S3: classical water transfer | `reproduce_water_transfer.py`, `../Graphs/Heatmap_Visualization.py` | `water_reproduction/classical_*.csv` |
| Figure 8: blind-molecule energies | `../Graphs/energy_deviance_graphs.py` | `blind/energy_mae.csv` |
| Tables S4–S5 / Figure S14: formaldehyde tuning | `matched_transfer.py`, `cobyla_arm.py`, `gamma_scan.py`, `make_tables_s4_s5.py`, `plot_bandwidth.py` | `final_results_basegrid.csv`, `cobyla_results.csv`, `gamma_scan.csv`, `config.json` |
| Tables S6–S7: computational cost | `bench_costs.py`, `bench_squlearn_scaling.py`, `bench_ccsd.py` | `water_*.json`, `squlearn_*.json`, `ccsd.json` |
| Figure S15: CPMap blocks | `../Graphs/plot_cpmap_blocks.py` | Gates specified in the plotting script |
| Table S8: water bandwidth selection | `reproduce_water_transfer.py`, `reproduce_water_gamma_grid.py`, `water_bandwidth.py` | `water_reproduction/`, `water_quantum_gamma_grid/`, `water_bandwidth/` |

`qk_common.py`, `fast_pqk.py` and `batched_pqk.py` supply the shared data loading,
circuit construction and Pauli-feature evaluation. The plotting scripts also
use the data archives linked in the root README.

## Figures and tables from saved results

```bash
python Graphs/regenerate_available_figures.py
python Experiments/make_table_s2.py
python Experiments/water_bandwidth.py --table-only
mkdir -p paper_results/regenerated_tables
python Experiments/make_tables_s4_s5.py Experiments/results > paper_results/regenerated_tables/tables_s4_s5.tex
```

Figures go to `paper_results/regenerated_figures/`; tables go to
`paper_results/regenerated_tables/`. Figure 8 averages the absolute energy
deviations in `blind/energy_mae.csv` over repetitions and converts hartree to
millihartree. Each molecule has one MP2 record, five Classical, Chebyshev and
Hubregtsen repetitions each, and two MultiControl repetitions.

## Water learning curves

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python Experiments/run_water_learning_curve.py --cv-size 5000 --repetitions 5 --jobs 8
```

The cc-pVDZ experiment samples 1–7% of the 918,384 water training amplitudes,
with five repetitions per model and sample size. Five-fold cross-validation
uses 5,000 amplitudes from each sample; the selected model is refitted on the
full sample and predicts all 231,040 test amplitudes. PQKs use fixed circuit
parameters and cached 1024-shot Pauli samples. Energies use zero singles and
no CCSD iterations. `protocol.json` records the settings, `runs.csv` the 140
individual results, and `summary.csv` their means and standard errors.
Models, predictions and per-run logs are saved in
`paper_results/experiments/water_learning_curve/`; completed runs are reused.

## Formaldehyde tuning

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
work=paper_results/experiments/formaldehyde
python -I Experiments/matched_transfer.py . data "$work" all 8
cp "$work/final_results.csv" "$work/final_results_basegrid.csv"
python -I Experiments/gamma_scan.py "$work" 8
python -I Experiments/cobyla_arm.py . data "$work" 8
```

Selection uses five conformer-grouped folds on 32 source training conformers;
refitting uses all 158 source training conformers. Source evaluation uses the
40 formaldehyde test conformers, and target evaluation uses all amplitudes of
each other molecule, as specified in Table S4. The water scalers are used.
The common grid selects circuit depth, encoding and SVR parameters. The COBYLA
arm optimizes on the same 32 conformers before reselecting SVR parameters.

## Timing measurements

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p paper_results/experiments/timings
python -I Experiments/bench_costs.py . data classical paper_results/experiments/timings/classical.json
python -I Experiments/bench_costs.py . data Chebyshev-direct1024 paper_results/experiments/timings/chebyshev.json
python -I Experiments/bench_costs.py . data MultiControl-direct1024 paper_results/experiments/timings/multicontrol.json
python -I Experiments/bench_squlearn_scaling.py . data Chebyshev 2000 paper_results/experiments/timings/native_chebyshev.json
python -I Experiments/bench_squlearn_scaling.py . data MultiControl 2000 paper_results/experiments/timings/native_multicontrol.json
python -I Experiments/bench_ccsd.py data/geometries paper_results/experiments/timings/ccsd.json
```

Table S6 extrapolates sQUlearn/Aer costs from measurements on up to 2,000
inputs. Direct-feature timings use statevectors followed by binomial sampling.
Timings depend on the machine.

## Water bandwidth selection

Choose new output directories for a training run:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python Experiments/reproduce_water_transfer.py \
  --out paper_results/experiments/water_scores \
  --artifacts paper_results/experiments/water_models --workers 4 --seed 42
python Experiments/reproduce_water_gamma_grid.py \
  --baseline paper_results/experiments/water_scores \
  --out paper_results/experiments/water_gamma_scores \
  --artifacts paper_results/experiments/water_gamma_models
```

These searches use all 10,176 water training amplitudes and five unshuffled
amplitude folds. For ZZFeatureMap-arctan L1, COBYLA runs separately within each
fold and on the full training set. The gamma search repeats optimization for
each bandwidth, using the classical 120-point SVR grid; saved gamma=1 fits
are reused after checking their sampled features and cross-validation scores.
The compact `*_fits.json` file stores the fitted parameters and sampling seeds
needed for this reuse; full optimization histories are saved with the models.

The target `y` is the CCSD doubles amplitude transformed with the water training
target scaler. Gamma is the outer Gaussian-kernel parameter. Pauli features use
1024 independent binomial shots and are checked against Qiskit and sQUlearn.
Transfer means use six target test splits. Table S8 reports one seed for a
quantum configuration chosen from Figure 5. Its classical bandwidth scan
includes choices informed by target scores, as specified in the table.

`water_bandwidth.py` refits the classical winner at each gamma from the saved
cross-validation table. Use `--table-only` to export Table S8 without training.

## Blind-test energy calculation

Figure 8 is plotted from `results/blind/energy_mae.csv`. To calculate energies
from your own per-amplitude export:

```bash
python "Data Generation/Compute_Blind_Energy.py" --predictions predictions.csv
```

The input columns are `Molecule`, `Circuit`, `Expected`, `Predicted` and
`Position` (zero-based amplitude index). Each complete export must contain
exactly one row per stored amplitude, in increasing `Position` order.
`Position` can be omitted for a single export per molecule/circuit. Repeated
exports require explicit positions.
Targets are checked against the stored CCSD amplitudes before evaluation.

The calculation uses the stored Coulomb integrals and amplitudes in their
common orbital basis. MP2 and ML energies use zero singles; the reference uses
stored CCSD singles and doubles. No CCSD iterations are performed.
Complete exports sharing a molecule/circuit label are averaged by position.
Incomplete exports raise an error; `--incomplete exclude` explicitly omits them.
For an export containing partial sequences, `--incomplete position-mean` averages
all available rows at each position and reports the minimum and maximum counts.
These counts can differ across amplitudes.
Partial exports whose targets do not match their stated positions must be
excluded.

Results go to `paper_results/blind/recomputed_energy.csv`. For the Figure 8
layout, the energy table must contain all seven molecules and all five methods
(MP2, Classical, Chebyshev, Hubregtsen and MultiControl), with one configuration
per model family. Plot it separately:

```bash
FIGURE_OUTPUT_DIR=paper_results/blind/recomputed_figures \
python Graphs/energy_deviance_graphs.py --energy paper_results/blind/recomputed_energy.csv
```
