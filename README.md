# QDDCC Quantum Kernel Models

This project explores quantum kernel methods for DDCC `t2` amplitude prediction.
It contains utilities for modifying Qiskit/sQUlearn feature maps, training
quantum and classical kernel models, and evaluating trained models on transfer
and blind molecule sets.

## Environment

Create the Conda environment from the project root:

```bash
conda env create -f environment.yml
conda activate qddcc_kernel
```

## Data and results from Zenodo

The large files are packaged separately from the source repository.

**Zenodo record:** pending upload. Add the record DOI/link here after the two
archives and `SHA256SUMS` have been uploaded.

| Archive | Download | Extracted | Contents |
| --- | ---: | ---: | --- |
| `qddcc_data_v1.tar.zst` | 6.2 GB | 20 GB | Repaired, ready-to-use `data/` required by the project scripts |
| `qddcc_paper_results_v1.tar.zst` | 4.3 GB | 16 GB | `paper_results/` and generated molecule `images/` |

Download both archives from the Zenodo record into the root of this repository. 
Verify them before extraction:

```bash
sha256sum --check SHA256SUMS
```

Create and activate the Conda environment first if your system does not
already provide `zstd`, then extract the archives from the repository root:

```bash
tar --zstd -xf qddcc_data_v1.tar.zst
tar --zstd -xf qddcc_paper_results_v1.tar.zst
```

After extraction, the code can use `data/`, `paper_results/`, and `images/`
directly; no paths need to be edited. The results archive alone is enough to
inspect the supplied and regenerated figures. Download the data archive as
well to train models or rerun figure scripts that read the dataset.

## Data Layout

Run scripts from the repository root.

```text
data/
  geometries/
    training/<MOLECULE>/*.xyz
    blind/<MOLECULE>.xyz
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

Training models are written under `models/`.

## Script Order

1. `FunctionEncoder.py`

   Helper module. Do not run this directly. `QKernel_Methods_Test.py` imports
   `FunctionEncoder` to wrap feature-containing circuit parameters with custom
   functions such as polynomial or arctangent encodings.

2. `Training_Functions.py`

   Helper module. Do not run this directly. It defines `MSE_Score`, a custom
   sQUlearn kernel loss that can be used by kernel optimization.

3. `QKernel_Methods_Test.py`

   Main training script. Run this first to train and save a model under
   `models/<MOLECULE>/`.

4. `Transferability_Test.py`

   Run after training. It reloads a saved model, refits the kernel object on the
   original training molecule data, evaluates one target molecule from
   `data/training/`, and appends the score under
   `paper_results/transferability/<BASIS_SET>/<MOLECULE>/`.

5. `Test_Blind.py`

   Run after training when blind data is available. It reloads a saved model,
   refits on the training molecule data, evaluates the hard-coded molecules in
   `data/blind/`, and appends MAE values under
   `paper_results/blind/<MOLECULE>/`.

## Reproducing comparable results

We do not need the original saved models or exact paper jobs to run a useful
reproduction. The prepared data use the recovered paper splits, and
`RANDOM_SEED` controls data subsampling, initial circuit parameters, and
simulator sampling. Use the same `RANDOM_SEED` and `JOB_NUM` for training and
evaluation.

A sensible starting point is five repetitions with seeds and job numbers
`0` through `4`, using STO-3G, five features, `PERCENT=1.0`, a projected
kernel, and the circuit/model settings being compared. For each repetition:

1. Run `QKernel_Methods_Test.py` once to train and save the model.
2. Run `Transferability_Test.py` once for each target `STRUCTURE`.
3. Run `Test_Blind.py` once if blind-set results are needed.
4. Regenerate the figures after the result CSVs have been written.

The scripts are configured with environment variables. This example trains a
projected quantum support-vector regressor on water-scaled data:

```bash
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

Useful values:

- `ENCODING_DEGREE`: integer degree such as `1` or `2`, or `atan`.
- `ENCODING_CIRCUIT`: `YZ_CX`, `Chebyshev`, `ZZ_map`, `ParamZ`, `HighDim`,
  `Hubregtsen`, or `MultiControl`.
- `MODEL_TYPE`: `qsvr`, `qkrr`, `qgpr`, `svr`, or `krr`. Use
  `HYPERPARAMETERS=OPTIMIZED` only with model types that define a grid in the
  script: `qsvr`, `qkrr`, `svr`, and `krr`.
- `QKERNEL`: `Projected` or `Fidelity` for quantum models.
- `TRAINABLE`: `True` to optimize kernel parameters, otherwise `False`.
- `MEASURE`: projected-kernel measurement string such as `XYZ`, `X`, `Y`, or
  `Z`.
- `PERCENT`: fraction of each molecule's training set to sample, for example
  `1.0` for all rows or `0.25` for 25 percent.
- `RANDOM_SEED`: controls subsampling, initial circuit parameters, and
  simulator sampling. Pair it with the same `JOB_NUM` for each repetition.

`QKernel_Methods_Test.py` currently assumes `Water-Scaled` processing.

## Transferability Evaluation

Use the same model settings as the training run, then set `STRUCTURE` to the
molecule to test:

```bash
ENCODING_DEGREE=atan \
ENCODING_CIRCUIT=Chebyshev \
NUM_LAYER=1 \
MODEL_TYPE=qsvr \
MOLECULE=Water \
STRUCTURE=Formaldehyde \
NUM_FEATURES=5 \
QKERNEL=Projected \
DATA_SETUP=Water-Scaled \
RUN_DATE= \
JOB_NUM=0 \
HYPERPARAMETERS=OPTIMIZED \
TRAINABLE=False \
BASIS_SET=STO-3G \
MEASURE=XYZ \
RANDOM_SEED=0 \
python Transferability_Test.py
```

`MOLECULE` is the molecule or molecule group used for training. `STRUCTURE` is
the held-out molecule being evaluated.

## Blind Evaluation

Use the same model settings as the training run. `Test_Blind.py` evaluates the
blind molecules listed inside the script:

```bash
ENCODING_DEGREE=atan \
ENCODING_CIRCUIT=Chebyshev \
NUM_LAYER=1 \
MODEL_TYPE=qsvr \
MOLECULE=Water \
STRUCTURE=unused \
NUM_FEATURES=5 \
QKERNEL=Projected \
DATA_SETUP=Water-Scaled \
RUN_DATE= \
JOB_NUM=0 \
HYPERPARAMETERS=OPTIMIZED \
TRAINABLE=False \
BASIS_SET=STO-3G \
MEASURE=XYZ \
RANDOM_SEED=0 \
python Test_Blind.py
```

The blind script writes
`paper_results/blind/<MOLECULE>/Expected_Predicted.csv` and appends
per-molecule MAE results to the configured blind output CSV.

## Data generation and figures

`Data Generation/Create_Dataset.py` is the pickle-to-joblib preprocessing
stage. It accepts `BASIS`, `STRUCTURE`, and `NUM_FILES`, and reuses the
recovered split under `data/splits/` by default. It refuses to create a new
split unless `RANDOM_SEED` is explicitly supplied for a new experiment.

After creating or updating the environment, regenerate every figure currently
supported by the preserved evidence with:

```bash
python Graphs/regenerate_available_figures.py
```

The command writes 21 PNG files under
`paper_results/regenerated_figures/`. It includes heatmaps,
expected/predicted panels, the training and blind-energy plots, reconstructed
STO-3G and cc-pVDZ R²/energy panels, molecule assets, and a reconstructed
circuit montage. The latter marks CPMap as unavailable because that source
implementation is missing.

The published SHAP ranking cannot be reproduced from the recovered model
state. A replacement analysis is therefore excluded by default and can only
be requested explicitly:

```bash
python Graphs/regenerate_available_figures.py --include-recomputed-shap
```

That output is a new deterministic computation, not the published SHAP
figure. `Water_R2_score.png` and `Express_entg.png` also remain unsupported
because their complete source values are absent. 

The earlier XYZ-to-pickle and energy scripts depend on the missing
`helper_CC_ML_spacial.py`.
