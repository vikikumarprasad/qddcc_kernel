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

## Data Layout

Run scripts from the repository root. The data files are not stored in this
repository, but the scripts expect joblib `.bin` files in this shape:

```text
Data/<BASIS_SET>/<MOLECULE>/scaled/
  full_feature_train.bin
  full_feature_test.bin
  <NUM_FEATURES>_feature_train.bin
  <NUM_FEATURES>_feature_test.bin
  feature_scaler.bin
  t2_scaler.bin

Blind_Data/<MOLECULE>/scaled/
  full_feature_train.bin
  feature_scaler.bin
  t2_scaler.bin
```

Training and evaluation outputs are written under `models/`, `transferability/`,
`Blind/`, and CSV result files in the project root.

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
   original training molecule data, evaluates one target molecule from `Data/`,
   and appends the score under `transferability/<BASIS_SET>/<MOLECULE>/`.

5. `Test_Blind.py`

   Run after training when blind data is available. It reloads a saved model,
   refits on the training molecule data, evaluates the hard-coded molecules in
   `Blind_Data/`, and appends MAE values under `Blind/<MOLECULE>/`.

## Training

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
HYPERPARAMETERS=OPTIMIZED \
NUM_QUBITS=5 \
TRAINABLE=False \
MEASURE=XYZ \
PERCENT=1.0 \
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

`QKernel_Methods_Test.py` currently assumes `Water-Scaled` processing and uses
`STO-3G` internally for the training basis.

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
python Test_Blind.py
```

The blind script writes `Blind/<MOLECULE>/Expected_Predicted.csv` and appends
per-molecule MAE results to the configured blind output CSV.
