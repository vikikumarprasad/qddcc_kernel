# QDDCC Quantum Kernel Models

Code and data for predicting coupled-cluster doubles amplitudes with classical
and projected quantum kernels. The repository includes training, transfer and
blind-test workflows, saved experiment summaries, and scripts for the manuscript
figures and Supporting Information tables.

[Experiments/README.md](Experiments/README.md) lists the included experiments,
scripts and data, with their settings and commands.

## Environment

Create the Conda environment from the repository root:

```bash
conda env create --solver libmamba -f environment.yml
conda activate qddcc_kernel
```

The environment includes Psi4, Qiskit and sQUlearn. Numerical-library versions
are pinned to the experiment environment; SciPy 1.15.2 supplies the COBYLA
implementation used in the saved optimization runs.

Figure 1(b) labels are typeset with LaTeX. `Graphs/plot_published_shap.py`, run by
`Graphs/regenerate_available_figures.py`, and `Graphs/Shap_Graphs.py` need a system
TeX installation with `dvipng` and the `amsmath`, `braket`, `type1cm` and `type1ec`
packages, for example the Debian/Ubuntu packages `texlive-latex-extra`,
`cm-super-minimal` and `dvipng`.

## Data and results

The large input datasets and result archives are stored on
[Zenodo](https://doi.org/10.5281/zenodo.20792229).

| Archive | Download | Extracted | Contents |
| --- | ---: | ---: | --- |
| `qddcc_data_v1.tar.zst` | 6.6 GB | 21 GB | Features, amplitudes, scalers, geometries and train/test splits under `data/` |
| `qddcc_paper_results_v1.tar.zst` | 4.5 GB | 17 GB | Result CSV files under `paper_results/` and static molecule artwork under `images/` |

Sizes use decimal GB. Download the archives into the repository root, verify
their checksums and extract them:

```bash
sha256sum --check SHA256SUMS
tar --zstd -xf qddcc_data_v1.tar.zst
tar --zstd -xf qddcc_paper_results_v1.tar.zst
```

Small result tables are included in `Experiments/results/`.
Experiment scripts save models, predictions and detailed run logs locally under
`paper_results/`. These generated files are kept outside Git.

### Partial extraction

Individual molecules and basis sets can be extracted after downloading an archive.

| Path | Extracted size | Contents |
| --- | ---: | --- |
| `data/training/STO-3G/` | 0.4 GB | Features, targets, scalers and pickles for seven molecules |
| `data/training/cc-pVDZ/` | 20.4 GB | The corresponding cc-pVDZ data |
| `data/blind/` | 0.3 GB | Blind-molecule descriptors and amplitudes |
| `data/geometries/`, `data/splits/` | < 1 MB total | XYZ files and split manifests |
| `paper_results/energy/`, `paper_results/training/`, `paper_results/water_r2_expr_entg/` | < 1 MB each | Energy, training and circuit-metric summaries |
| `paper_results/transferability/` | 4.5 GB | Scores and per-amplitude transfer predictions |
| `paper_results/water_energy/` | 12 GB | Training-size predictions |
| `paper_results/blind/` | 0.1 GB | Blind-molecule prediction export and energy summary; Figure 8 data are in `Experiments/results/blind/energy_mae.csv` |

For STO-3G water training:

```bash
tar --zstd -xf qddcc_data_v1.tar.zst data/training/STO-3G/Water
```

For the water and formaldehyde transfer studies in SI Sections S5 and S8:

```bash
tar --zstd -xf qddcc_data_v1.tar.zst data/training/STO-3G
tar --zstd -xf qddcc_paper_results_v1.tar.zst paper_results/transferability/STO-3G/Water
```

For Figure 4b training and the reference-energy check:

```bash
tar --zstd -xf qddcc_data_v1.tar.zst data/training/cc-pVDZ/Water data/geometries/training/water
```

The water scalers are required when loading other molecules with
`Experiments/qk_common.py`. Generated figure images are written locally by the
plotting scripts; the archives supply the static molecule artwork they use.

## Repository layout

```text
Data Generation/                 Feature generation and energy calculations
Experiments/                     Experiment scripts and saved numerical results
Graphs/                          Figure scripts

data/
  geometries/training/<MOLECULE>/*.xyz
  geometries/blind/<MOLECULE>.xyz
  splits/
  training/<BASIS_SET>/<MOLECULE>/
    output/
    pickle/
    scaled/
      full_feature_{train,test}.bin
      {5,10,15}_feature_{train,test}.bin
      feature_scaler.bin
      t2_scaler.bin
  blind/<MOLECULE>/
    output/
    pickle/
    scaled/
```

Run commands from the repository root. Trained models are written under `models/`.

## Training and transfer

`QKernel_Methods_Test.py` trains and saves a model. It imports
`FunctionEncoder.py` for feature-encoding functions, `EncodingCircuit.py` for
circuit cloning, and `Training_Functions.py` for the kernel-optimization loss.

For example, train a water model with a Chebyshev projected kernel:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ENCODING_DEGREE=atan \
ENCODING_CIRCUIT=Chebyshev \
NUM_LAYER=1 \
MODEL_TYPE=qsvr \
MOLECULE=Water \
NUM_FEATURES=5 \
QKERNEL=Projected \
DATA_SETUP=Water-Scaled \
RUN_DATE= \
JOB_NUM=0 \
HYPERPARAMETERS=OPTIMIZED \
NUM_QUBITS=5 \
TRAINABLE=False \
BASIS_SET=STO-3G \
MEASURE=XYZ \
PERCENT=1.0 \
RANDOM_SEED=0 \
python QKernel_Methods_Test.py
```

The scripts use environment variables:

| Variable | Values or meaning |
| --- | --- |
| `ENCODING_DEGREE` | Polynomial degree such as `1` or `2`, or `atan` |
| `ENCODING_CIRCUIT` | `YZ_CX`, `Chebyshev`, `ZZ_map`, `ParamZ`, `HighDim`, `Hubregtsen`, `MultiControl`; set `Classical` for `svr` and `krr`, since the value labels prediction exports |
| `MODEL_TYPE` | `qsvr`, `qkrr`, `qgpr`, `svr`, `krr` |
| `HYPERPARAMETERS` | `OPTIMIZED` selects the script's grid for `qsvr`, `qkrr`, `svr` or `krr` |
| `QKERNEL` | `Projected` or `Fidelity` for quantum models |
| `TRAINABLE` | Whether to optimize circuit parameters |
| `MEASURE` | Pauli observables such as `XYZ`, `X`, `Y` or `Z` |
| `PERCENT` | Fraction of the training set to sample |
| `RANDOM_SEED` | Seed for subsampling, initial parameters and simulator sampling |
| `JOB_NUM` | Repetition identifier used in prediction labels and, with `TRAINABLE=False`, in quantum model filenames |

`QKernel_Methods_Test.py` uses `Water-Scaled` processing. The detailed
[experiment instructions](Experiments/README.md) specify the seeds, searches
and evaluation populations used for each reported result.

After training, run `Transferability_Test.py` with the same settings and set
`STRUCTURE` to the target molecule. For example, `MOLECULE=Water` and
`STRUCTURE=Formaldehyde` evaluate a water-trained model on formaldehyde.
Keep `JOB_NUM` and `RANDOM_SEED` consistent between training and evaluation.
The script reloads the saved model, refits its kernel object on source training
data and writes target scores under
`paper_results/transferability/<BASIS_SET>/<MOLECULE>/`. The training and transfer
scripts append to existing CSV files, including files extracted from the results
archive.

## Blind-test results

The Figure 8 energy records are included in
[`Experiments/results/blind/energy_mae.csv`](Experiments/results/blind/energy_mae.csv).
The figure uses the mean absolute energy deviation across the supplied
repetitions. Generate it with:

```bash
python Graphs/energy_deviance_graphs.py
```

`Data Generation/Compute_Blind_Energy.py` evaluates exported predictions using
the integrals and CCSD amplitudes in `data/blind/`. It uses zero predicted singles,
with no CCSD iterations. See [Experiments/README.md](Experiments/README.md#blind-test-energy-calculation)
for the input format and treatment of incomplete exports.
`Test_Blind.py` generates predictions for the molecules listed inside the script,
using the same model settings as training, and writes them under
`paper_results/blind/<MOLECULE>/`.

## Data generation and figures

`Data Generation/Create_Dataset.py` converts pickles to joblib datasets. It
accepts `BASIS`, `STRUCTURE` and `NUM_FILES` (the number of the last pickle file,
for example 200 for STO-3G water), and reuses `data/splits/` by default.
Creating a new split requires an explicit `RANDOM_SEED`. The script writes to
`data/training/<BASIS>/<STRUCTURE>/scaled/`, replacing the extracted files.
`Data Generation/frozen_core_CCSD_Clean.py` reads training geometries and writes
pickles under `data/training/`; use `GEOMETRY_ROOT` or `TRAINING_DATA_ROOT` for
alternative input and output locations.

Generate the figures supported by the available data:

```bash
python Graphs/regenerate_available_figures.py
```

Outputs go to `paper_results/regenerated_figures/`. The command includes
Figures 4b, 5, 8, S14 and S15.

Figure 1(b) can be plotted from `Graphs/figure1b_published_shap.csv`. Those bar
heights were digitized from the manuscript image at its plotted resolution.
A fresh SHAP calculation can be requested with `--include-recomputed-shap`;
it writes a separate output because its values depend on the fitted model and
SHAP settings. Use `--skip-circuit-montage` to omit the Qiskit circuit montage.

The Docker image contains the Conda environment and TeX. From the repository
root, build it and run the figure command with the extracted archives mounted:

```bash
docker build -t qddcc-kernel .
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/images:/app/images:ro" \
  -v "$PWD/paper_results:/app/paper_results" \
  qddcc-kernel
```

Figures are written to `paper_results/regenerated_figures/`. To run another
script, append its command, for example `python Experiments/make_table_s2.py`.

For circuit metrics:

```bash
python "Data Generation/compute_expr_entg_metrics.py" --seed 0
```

`Graphs/plot_cpmap_blocks.py` draws the symbolic C/P blocks of Figure S15.
The supplied metric CSVs contain the CPMap values plotted in Figure 9.
