import os
import csv
import joblib
import json
import time
# !{sys.executable} -m pip install qcircuit
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit import *
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit_aer import Aer
from squlearn.optimizers import Adam, SPSA

from squlearn import Executor

from squlearn.encoding_circuit import QiskitEncodingCircuit, HighDimEncodingCircuit
from squlearn.kernel.qsvr import QSVR
from squlearn.kernel.qgpr import QGPR
from squlearn.encoding_circuit.layered_encoding_circuit import LayeredEncodingCircuit, Layer
from squlearn.kernel import ProjectedQuantumKernel, KernelOptimizer
from Training_Functions import MSE_Score

from Circuit_Classes import *

from joblib import dump, load

degree = int(os.getenv("ENCODING_DEGREE"))
circuit_struct = os.getenv("ENCODING_CIRCUIT")
layers = int(os.getenv("NUM_LAYER"))
model_type = os.getenv("MODEL_TYPE").upper()
molecule_structure = os.getenv("MOLECULE")
num_features = int(os.getenv("NUM_FEATURES"))
kernel_type = os.getenv("QKERNEL").upper()
data_format = os.getenv("DATA_SETUP")
date = os.getenv("RUN_DATE")
hyper_optimized = os.getenv("HYPERPARAMETERS")
trainable = os.getenv("TRAINABLE").upper()
if not date is None:
    input_path = os.path.join(top, rf"{date}/")
else:
    input_path = top
if data_format != "Water-Scaled":
    data_path1 = os.path.join(top, rf"Data/{basis}/{molecule_structure}/scaled/{num_features}_feature_test.bin")
    data_path2 = os.path.join(top, rf"Data/{basis}/{molecule_structure}/scaled/{num_features}_feature_train.bin")
else:
    data_path1 = os.path.join(top, rf"Data/{basis}/{molecule_structure}/scaled/Full_feature_test.bin")
    data_path2 = os.path.join(top, rf"Data/{basis}/{molecule_structure}/scaled/Full_feature_train.bin")

top=os.getcwd()
if model_type != "QSVR" and model_type != "QKRR" and model_type != "QGPR":
    model_path = os.path.join(input_path, rf"models/{trainable}_{kernel_type}_{num_features}_{circuit_struct}_{degree}_{layers}_{model_type}")
    out_path = os.path.join(top, rf"transferability/{trainable}_{kernel_type}_{num_features}_{circuit_struct}-{degree}-{layers}.csv")
else:
    model_path = os.path.join(input_path, rf"models/{circuit_struct}_{degree}_{layers}_{model_type}")
    out_path = os.path.join(top, rf"transferability/{circuit_struct}-{degree}-{layers}.csv")
print(model_path)
with open(model_path, 'rb') as f:
    model_file = joblib.load(f)
    model = model_file["model"]
print(data_path1, data_path2)
with open(data_path1,'rb') as f:
    data_file = joblib.load(f)
    X_test1 = data_file["X"]
    y_test1 = data_file["y"]

with open(data_path2, 'rb') as f:
    data_file = joblib.load(f)
    X_test2 = data_file["X"]
    y_test2 = data_file["y"]

X_test = np.vstack(X_test1, X_test2)
y_test = np.append(y_test1, y_test2)

if data_format == "Water-Scaled":
    with open(os.path.join(top, rf"Data/STO-3G/Water/scaled/feature_scaler.bin"), 'rb') as f:
        water_feature_scaler = joblib.load(f)
    with open(os.path.join(top, rf"Data/STO-3G/Water/scaled/t2_scaler.bin"), 'rb') as f:
        water_t2_scaler = joblib.load(f)
    with open(os.path.join(top, rf"Data/STO-3G/{molecule_structure}/scaled/feature_scaler.bin"), 'rb') as f:
        molecule_feature_scaler = joblib.load(f)
    with open(os.path.join(top, rf"Data/STO-3G/{molecule_structure}/scaled/t2_scaler.bin"), 'rb') as f:
        molecule_t2_scaler = joblib.load(f)
    unscaledX_test = molecule_feature_scaler.inverse_transform(X_test)
    unscaledy_test = molecule_t2_scaler.inverse_transform(y_test.reshape(-1, 1))

    X_test = water_feature_scaler.transform(unscaledX_test)
    y_test = water_t2_scaler.transform(unscaledy_test).flatten()

predictions = model.predict(X_test)
r2 = r2_score(y_test.flatten(), predictions.flatten())
print(f"{molecule_structure} R2: {r2}")
print(out_path)
# Open the file in append mode
with open(out_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Append the label and value as a new row
    writer.writerow([molecule_structure, r2])
