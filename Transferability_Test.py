import os
import csv
import joblib
import json
import time
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

degree = os.getenv("ENCODING_DEGREE")
circuit_struct = os.getenv("ENCODING_CIRCUIT")
layers = int(os.getenv("NUM_LAYER"))
model_type = os.getenv("MODEL_TYPE").upper()
train_molecule = os.getenv("MOLECULE")
molecule_structure = os.getenv("STRUCTURE")
num_features = int(os.getenv("NUM_FEATURES"))
kernel_type = os.getenv("QKERNEL").upper()
data_format = os.getenv("DATA_SETUP")
date = os.getenv("RUN_DATE")
job_num = os.getenv("JOB_NUM")
hyper_optimized = os.getenv("HYPERPARAMETERS")
trainable = os.getenv("TRAINABLE").upper()
basis = os.getenv("BASIS_SET")
measure = os.getenv("MEASURE")
random_seed = int(os.getenv("RANDOM_SEED", "42"))
np.random.seed(random_seed)
top = os.getcwd()

if not date is None:
    input_path = os.path.join(top, rf"{date}/")
else:
    input_path = top

if data_format != "Water-Scaled":
    data_path1 = rf"data/training/{basis}/{molecule_structure}/scaled/{num_features}_feature_test.bin"
    data_path2 = rf"data/training/{basis}/{molecule_structure}/scaled/{num_features}_feature_train.bin"
else:
    data_path1 = rf"data/training/{basis}/{molecule_structure}/scaled/full_feature_test.bin"
    data_path2 = rf"data/training/{basis}/{molecule_structure}/scaled/full_feature_train.bin"

if model_type != "SVR" and model_type != "KRR" and model_type != "GPR":
    model_path = rf"models/{train_molecule}/{trainable}_{kernel_type}_{hyper_optimized}_{num_features}_{circuit_struct}_{degree}_{layers}_{model_type}"
    out_path = rf"paper_results/transferability/{basis}/{train_molecule}/{trainable}_{kernel_type}_{num_features}_{circuit_struct}-{degree}-{layers}-{model_type}"

    if trainable == "FALSE":
        model_path += f"-{job_num}"
        out_path += f"-{job_num}"
    if measure != "":
        model_path += f"_{measure}"
        out_path += f"_{measure}"
    model_path += ".bin"
    out_path += ".csv"
else:
    if hyper_optimized == "OPTIMIZED":
        model_path = rf"models/{train_molecule}/{hyper_optimized}_{model_type}"
        out_path = rf"paper_results/transferability/{basis}/{train_molecule}/{hyper_optimized}_{model_type}"
    else:
        model_path = rf"models/{train_molecule}/{model_type}"
        out_path = rf"paper_results/transferability/{basis}/{train_molecule}/{model_type}"

    model_path += ".bin"
    out_path += ".csv"


#Loads the data, unscales the data and then rescales relative to water
#Training data needs to be reloaded as while kernel kept circuit parameters and hyperparameters
#The models need to be reloaded using the data (this step is very fast unlike initial training)
molecules = train_molecule.split("-")
scale = data_format
total_X_train = np.array([])
total_y_train = np.array([])
for molecule_struct in molecules:
    top=os.getcwd()
    if scale != "Water-Scaled":
        with open(rf'data/training/STO-3G/{molecule_struct}/scaled/{num_features}_feature_train.bin','rb') as f:
            ddcc_train = joblib.load(f)

        with open(rf'data/training/STO-3G/{molecule_struct}/scaled/t2_scaler.bin','rb') as f:
            ddcc_scaler = joblib.load(f)
    else:
        with open(rf'data/training/STO-3G/{molecule_struct}/scaled/full_feature_train.bin','rb') as f:
            ddcc_train = joblib.load(f)

        with open(rf'data/training/STO-3G/{molecule_struct}/scaled/t2_scaler.bin','rb') as f:
            ddcc_scaler = joblib.load(f)

    # print(ddcc_train.keys())
    X_ddcc_train, y_ddcc_train = ddcc_train['X'],ddcc_train['y']

    X_train, y_train = X_ddcc_train, y_ddcc_train

    if scale=="Water-Scaled":
        print("Scaling data")
        with open(rf"data/training/STO-3G/Water/scaled/feature_scaler.bin", 'rb') as f:
            water_feature_scaler = joblib.load(f)
        with open(rf"data/training/STO-3G/Water/scaled/t2_scaler.bin", 'rb') as f:
            water_t2_scaler = joblib.load(f)
        with open(rf"data/training/STO-3G/{molecule_struct}/scaled/feature_scaler.bin", 'rb') as f:
            molecule_feature_scaler = joblib.load(f)
        with open(rf"data/training/STO-3G/{molecule_struct}/scaled/t2_scaler.bin", 'rb') as f:
            molecule_t2_scaler = joblib.load(f)
        print(water_feature_scaler, water_t2_scaler)
        print(molecule_feature_scaler, molecule_t2_scaler)
        unscaledX_train = molecule_feature_scaler.inverse_transform(X_train)
        unscaledy_train = molecule_t2_scaler.inverse_transform(y_train.reshape(-1, 1))

        X_train = np.array(water_feature_scaler.transform(unscaledX_train))
        y_train = np.array(water_t2_scaler.transform(unscaledy_train).flatten())

        top5=np.array([20, 21, 24, 23, 22][::-1])
        top10=np.array([28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])
        top15=np.array([18, 5, 9, 29, 27, 28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])
        print(molecule_struct, X_train.shape)
        X_train = X_train[:, top5]

        if len(total_X_train) == 0:
            total_y_train = y_train
            total_X_train = X_train
        else:
            total_y_train = np.append(total_y_train, y_train)
            total_X_train = np.vstack((total_X_train, X_train))
X_train = total_X_train
y_train = total_y_train

print(model_path, molecule_structure)
with open(model_path, 'rb') as f:
    model_file = joblib.load(f)
    model = model_file["Model"]
print(data_path1, data_path2)
with open(data_path1,'rb') as f:
    data_file = joblib.load(f)
    X_test1 = data_file["X"]
    y_test1 = data_file["y"]

with open(data_path2, 'rb') as f:
    data_file = joblib.load(f)
    X_test2 = data_file["X"]
    y_test2 = data_file["y"]


# Only if not testing your initial training molcule test on all t2 for the transfer target
if train_molecule != molecule_struct:
    X_test = np.vstack((X_test1, X_test2))
    y_test = np.append(y_test1, y_test2)
else:
    X_test = X_test1
    y_test = y_test1

print(X_test.shape, y_test.shape)
#Scale all data relative to water
if data_format == "Water-Scaled":
    with open(os.path.join(top, rf"data/training/STO-3G/Water/scaled/feature_scaler.bin"), 'rb') as f:
        water_feature_scaler = joblib.load(f)
    with open(os.path.join(top, rf"data/training/STO-3G/Water/scaled/t2_scaler.bin"), 'rb') as f:
        water_t2_scaler = joblib.load(f)
    with open(os.path.join(top, rf"data/training/STO-3G/{molecule_structure}/scaled/feature_scaler.bin"), 'rb') as f:
        molecule_feature_scaler = joblib.load(f)
    with open(os.path.join(top, rf"data/training/STO-3G/{molecule_structure}/scaled/t2_scaler.bin"), 'rb') as f:
        molecule_t2_scaler = joblib.load(f)

    unscaledX_test = molecule_feature_scaler.inverse_transform(X_test)
    unscaledy_test = molecule_t2_scaler.inverse_transform(y_test.reshape(-1, 1))

    X_test = water_feature_scaler.transform(unscaledX_test)
    y_test = water_t2_scaler.transform(unscaledy_test).flatten()

    top5=np.array([20, 21, 24, 23, 22][::-1])
    top10=np.array([28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])
    top15=np.array([18, 5, 9, 29, 27, 28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])

    X_test = X_test[:, top5]

#reload model
model.fit(X_train, y_train)
print(model.get_params())
predictions = model.predict(X_test)
r2 = r2_score(y_test.flatten(), predictions.flatten())
print(f"{molecule_structure} R2: {r2}")
print(out_path)

expected_predicted_path = rf"paper_results/transferability/{basis}/{train_molecule}/Expected_Predicted.csv"
#e_p_output = {"Molecule": f"{molecule_structure}" ,"Circuit": f"{circuit_struct}_{degree}_{layers}", "Expected": y_test, "Predicted": predictions}

os.makedirs(f"paper_results/transferability/{basis}/{train_molecule}", exist_ok=True)
# Open the file in append mode
with open(out_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Append the label and value as a new row
    writer.writerow([molecule_structure, r2])

# Combine the lists into a single list of tuples for easier iteration
data_to_write = zip(y_test, predictions)

# Writes the actual t2 and the predicted t2 into a shared csv.
with open(expected_predicted_path, mode='a', newline='') as file:
    fieldnames = ['Molecule', 'Circuit', 'Expected', 'Predicted']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Check if the file is empty to write the header
    if file.tell() == 0:
        writer.writeheader()
    
    # Iterate through the data and write a new row for each pair
    for expected_value, predicted_value in data_to_write:
        row_output = {
            'Molecule': molecule_structure,
            'Circuit': f"{circuit_struct}_{degree}_{layers}",
            'Expected': expected_value,
            'Predicted': predicted_value
        }
        writer.writerow(row_output)
        time.sleep(0.0001)
