import os
import joblib
import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
import qiskit
from qiskit import *
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_aer import AerSimulator
from qiskit_aer import Aer
from squlearn.optimizers import Adam

from squlearn import Executor

from squlearn.encoding_circuit import *
from squlearn.kernel.qsvr import QSVR
from squlearn.kernel.qgpr import QGPR
from squlearn.encoding_circuit.layered_encoding_circuit import LayeredEncodingCircuit, Layer
from squlearn.kernel import ProjectedQuantumKernel, KernelOptimizer, FidelityKernel
from squlearn.kernel.loss import TargetAlignment
from Training_Functions import MSE_Score
from sklearn.kernel_ridge import KernelRidge
from Circuit_Classes import ZZ_featuremap
#from QiskitRegressor import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from joblib import dump, load
from squlearn.kernel import QKRR
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import itertools
from FunctionEncoder import *
import csv

degree = os.getenv("ENCODING_DEGREE")
if not degree == "atan":
    degree = int(degree)
circuit_struct = os.getenv("ENCODING_CIRCUIT")
layers = int(os.getenv("NUM_LAYER"))
model_type = os.getenv("MODEL_TYPE")
molecule_structure = os.getenv("MOLECULE")
num_features = int(os.getenv("NUM_FEATURES"))
kernel_type = os.getenv("QKERNEL")
data_format = os.getenv("DATA_SETUP")
date = os.getenv("RUN_DATE")
hyper_optimized = os.getenv("HYPERPARAMETERS")
num_qubits = int(os.getenv("NUM_QUBITS"))
job_num = os.getenv("JOB_NUM", "0")
trainable = os.getenv("TRAINABLE")
basis = os.getenv("BASIS") or os.getenv("BASIS_SET", "STO-3G")
scale = "Water-Scaled"
measure = os.getenv("MEASURE")
percent_data = float(os.getenv("PERCENT"))
if not 0 < percent_data <= 1:
    raise ValueError("PERCENT must be greater than 0 and at most 1")
random_seed = int(os.getenv("RANDOM_SEED", "42"))
random.seed(random_seed)
np.random.seed(random_seed)
print("Qiksit Version: ",qiskit.__version__)
print("Random seed: ", random_seed)

def learning_rate(iteration, transition_begin = 0, transition_steps = 450, init_value = 0.08, decay_rate = 0.1):
    rate_factor = ((iteration - transition_begin) / transition_steps)
    decayed_value = init_value * (decay_rate ** rate_factor)
    return decayed_value

def degree_relation_function(expression, num_parameters):
    num_params = degree + 1
    if degree <= 0:
        return expression
    output = 0
    # print(expression)
    params = ParameterVector("Phi", num_parameters + num_params)[num_parameters:]
    for i in range(num_params):
        output += params[i] * (expression**i)
    return output

def atan_relation_function(expression, num_parameters):
    num_params = 1
    param = ParameterVector("Phi", num_parameters + num_params)[num_parameters:]
    return param[0] * np.arctan(expression)

def electron_correlation_selection(x_train, y_train, percent):
    chosen_indices = []
    results = []
    num_points_chosen = int(x_train.shape[0]*percent)
    # default_cutoff = 1 * math.pow(10,-7.5)
    # print(default_cutoff)
    for i in range(x_train.shape[0]):
        t2_start = x_train[i][-7]
        doublecheck = x_train[i][-8]
        # if np.abs(t2_start * doublecheck) >  default_cutoff:
        chosen_indices.append(i)
        results.append(np.abs(t2_start * doublecheck))
        # print(f"t2 {t2_start}, Electron Integral {doublecheck}: {doublecheck * t2_start}")
    result_dict = {"index": chosen_indices, "energy": results}
    pairs = list(zip(result_dict["energy"], result_dict["index"]))

    sorted_pairs = sorted(pairs)

    sorted_idx = [idx for energy, idx in sorted_pairs]
    chosen_indices = sorted_idx[:int(num_points_chosen)]
    
    x_train=x_train[chosen_indices]
    y_train=y_train[chosen_indices]
    print(x_train.shape, y_train.shape)
    return x_train, y_train

def probability_selection_scheme(x_train, y_train, percent):
    num_data_points = int(len(x_train) * percent)

    weight_dict = {}
    weight_sum = 0
    for i in range(x_train.shape[0]):
        weight = np.abs((x_train[i][-7] * (x_train[i][8] + x_train[i][12])))
        weight_dict[i] = weight
        weight_sum += weight
        # print(weight_sum)
    # print(weight_sum, weight_sum/(len(x_train)))

    pairs = np.array(list(zip(weight_dict.values(), weight_dict.keys())))
    # print(pairs[:10])
    pairs[:, 0] = pairs[:, 0]/weight_sum
    # print(np.sum(pairs[:, 0]))
    probabilities = pairs[:, 0]
    indices = pairs[:, 1]

    chosen_indices = np.random.choice(indices, size=num_data_points, replace=False, p=probabilities).astype(int)

    x_train=x_train[chosen_indices]
    y_train=y_train[chosen_indices]
    print(x_train.shape, y_train.shape)
    return x_train, y_train

def quantum_kernel_method(circuit_struct, degree, layers, model_type):
    # num_features = os.getenv("NUM_FEATURES")
    if circuit_struct == "YZ_CX":
        encoding_scheme = YZ_CX_EncodingCircuit(num_qubits, num_features, layers)
        encoding_circuit = encoding_scheme.get_circuit(ParameterVector("X",num_features),ParameterVector("P",encoding_scheme.num_parameters))
    elif circuit_struct ==  "Chebyshev":
        encoding_scheme = ChebyshevPQC(num_qubits, num_features,layers, nonlinearity="arctan")
        encoding_circuit = encoding_scheme.get_circuit(ParameterVector("X",num_features),ParameterVector("P",encoding_scheme.num_parameters))
    elif circuit_struct == "ZZ_map":
        encoding_scheme = ZZ_featuremap(num_features, layers, 0, "linear")
        encoding_circuit = encoding_scheme.get_circuit()
    elif circuit_struct == "ParamZ":
        encoding_scheme = ParamZFeatureMap(num_qubits, num_features, layers, entangling=True)
        encoding_circuit = encoding_scheme.get_circuit(ParameterVector("X",num_features),ParameterVector("P",encoding_scheme.num_parameters))
    elif circuit_struct == "HighDim":
        encoding_scheme = HighDimEncodingCircuit(num_features, num_qubits, num_layers=layers)
        encoding_circuit = encoding_scheme.get_circuit(ParameterVector("X",num_features),ParameterVector("P",encoding_scheme.num_parameters))
    elif circuit_struct == "Hubregtsen":
        encoding_scheme = HubregtsenEncodingCircuit(num_features, num_qubits, layers)
        encoding_circuit = encoding_scheme.get_circuit(ParameterVector("X",num_features),ParameterVector("P",encoding_scheme.num_parameters))
    elif circuit_struct == "MultiControl":
        encoding_scheme = MultiControlEncodingCircuit(num_features, num_qubits, layers)
        encoding_circuit = encoding_scheme.get_circuit(ParameterVector("X",num_features),ParameterVector("P",encoding_scheme.num_parameters))
    elif circuit_struct in {"CP_map", "CPMap"}:
        raise NotImplementedError(
            "CPMap is not distributed with this repository. Its implementation "
            "is available only by request from the original CPMap author."
        )
    else:
        raise ValueError(f"Unsupported encoding circuit: {circuit_struct!r}")
    if isinstance(degree, int):
        if degree > 0:
            function_encoded_circuit = FunctionEncoder(encoding_circuit, degree_relation_function, False)
        else:
            function_encoded_circuit = encoding_circuit
    else:
        function_encoded_circuit = FunctionEncoder(encoding_circuit, atan_relation_function, False)
    #if kernel_type == "Fidelity":
    #    function_encoded_circuit = function_encoded_circuit.to_gate()
    encoding_circuit = QiskitEncodingCircuit(function_encoded_circuit, parameter_label=["Phi","P","θ"], feature_label="X")
    num_params = encoding_circuit.num_parameters
    initial_parameters = (np.random.rand(num_params) * 2 * np.pi) - np.pi
    print(initial_parameters)
    print(encoding_circuit.get_circuit(ParameterVector("X", num_features), ParameterVector("P", encoding_circuit.num_parameters) ) )
    if kernel_type == "Projected":
        kernel = ProjectedQuantumKernel(
            encoding_circuit,
            executor=Executor("qasm_simulator", shots=1024, seed=random_seed),
            measurement=measure
        )
    elif kernel_type == "Fidelity": #NOT USED
        kernel = FidelityKernel(
            encoding_circuit,
            executor=Executor(
                Aer.get_backend("aer_simulator"),
                shots=1024,
                seed=random_seed,
            ),
        )

    if date != "" and not date is None:
        start_path = f"models/{date}/{molecule_structure}"
    else:
        start_path = f"models/{molecule_structure}"
    # set up the optimizer
    # adam_opt = Adam(options={"maxiter":1, "lr": learning_rate, "tol":0.001})

    # if trainable kernel set up the kernel optimizer
    if trainable == "True":
        #kta_loss = TargetAlignment()
        kta_loss = MSE_Score(model_type, noisy=True)
        cobyla_opt = COBYLA(maxiter=100)
        kernel = KernelOptimizer(quantum_kernel=kernel, loss = kta_loss, optimizer=cobyla_opt, initial_parameters=initial_parameters)

    if model_type == "qsvr":
        model = QSVR(kernel)
        param_grid = {
            "C":[0.1, 0.25, 0.5, 1, 5, 10],
            "epsilon":[0.1, 0.05, 0.03, 0.08]
        }
    elif model_type == "qgpr":
        model = QGPR(kernel)
    elif model_type == "qkrr":
        model = QKRR(kernel)
        param_grid = {
            'alpha':[0.01, 0.1, 0.2, 0.5, 1, 10]
        }
    print(start_path)
    print("Starting training of model")
    start_time = time.time()
    if hyper_optimized == "OPTIMIZED":
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring="r2",
            verbose=1,
        )
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        print(f"Model Train R2: {grid_search.best_score_}")
        model = grid_search.best_estimator_
        model_name = (
            f"{trainable.upper()}_{kernel_type.upper()}_{hyper_optimized}_"
            f"{num_features}_{circuit_struct}_{degree}_{layers}_{model_type.upper()}"
        )
    else:
        model.fit(X_train, y_train)
        model_name = (
            f"{trainable.upper()}_{kernel_type.upper()}_{hyper_optimized}_"
            f"{num_features}_{circuit_struct}_{degree}_{layers}_{model_type.upper()}"
        )
    end_time = time.time()
    if trainable.upper() == "FALSE":
        model_name += f"-{job_num}"
    if measure:
        model_name += f"_{measure}"
    log_path = f"{start_path}/{model_name}.bin"
    print(log_path)

    print(f"Fitting the model took {end_time-start_time} seconds")
    if trainable == "True":
        print(f"Optimized Parameters: {kernel.get_optimal_parameters()}")
    else:
        print(f"Circuit Parameters: {kernel.parameters}")
    predictions = model.predict(X_test)
    print(f"The r2 score on the test set is {r2_score(y_test.flatten(), predictions.flatten())}")
    print(f"Test MSE: {mean_squared_error(y_test.flatten(), predictions.flatten())}")
    if trainable == "True":
        # loss_history = kta_loss.train_loss_history
        # progress = {"Training Time": end_time-start_time, "Loss History":loss_history, "R2 Score":r2_score(y_test.flatten(), predictions.flatten()), "Model":model}
       progress = {"Training Time": end_time-start_time, "R2 Score":r2_score(y_test.flatten(), predictions.flatten()), "Model":model}
    else:
        progress = progress = {"Training Time": end_time-start_time, "R2 Score":r2_score(y_test.flatten(), predictions.flatten()), "Model":model}
    joblib.dump(progress, log_path)
    path = f"paper_results/training_scores/{percent_data}-{num_features}-{basis}-{molecule_structure}-R2.csv"
    output = [f"{trainable}-{model_type}", f"{circuit_struct}-{str(degree).upper()}-{layers}-{job_num}", r2_score(y_test.flatten(), predictions.flatten())]
    # Open the file in append mode
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Append the label and value as a new row
        writer.writerow(output)
    water_t2_path = f"paper_results/water_energy/{os.path.basename(path)}"

    unscaledy_test = water_t2_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    unscaledy_predictions = water_t2_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    print(unscaledy_predictions.shape)

    # Combine the lists into a single list of tuples for easier iteration
    data_to_write = zip(unscaledy_test, unscaledy_predictions)

    with open(water_t2_path, mode='a', newline='') as file:
        fieldnames = ['Molecule', 'Circuit', 'Expected', 'Predicted', 't2_num']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Check if the file is empty to write the header
        if file.tell() == 0:
            writer.writeheader()
        num_t2s = int((y_test.shape[0]) / 40)
        count = 0
        # Iterate through the data and write a new row for each pair
        for expected_value, predicted_value in data_to_write:
            row_output = {
                'Molecule': f"{molecule_structure}{count // num_t2s}",
                'Circuit': f"{circuit_struct}_{degree}_{layers}_{job_num}",
                'Expected': expected_value,
                'Predicted': predicted_value,
                't2_num': count % num_t2s
            }
            writer.writerow(row_output)
            time.sleep(0.01) #prevent weird writting orders
            count += 1

def classical_kernel_method(model_type):

    if model_type == "svr":
        model = SVR(kernel="rbf")
        param_grid = {
            "kernel":["rbf"],
            "gamma":["auto", 1, 0.1, 0.01, 0.05, 0.001],
            "C":[0.1, 0.5, 1, 5, 10],
            "epsilon":[0.1, 0.05, 0.03, 0.08]
        }
    elif model_type == "krr":
        model = KernelRidge()
        param_grid = {
            'kernel':["rbf","polynomial","linear"],
            'alpha':[0.01, 0.1, 0.2, 0.5, 1, 10],
            'gamma':["scale", "auto", 0.1, 0.01]
        }
    top = os.getcwd()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', verbose=1)
    if date != "" and not date is None:
        start_path = f"models/{date}/{molecule_structure}"
    else:
        start_path = f"models/{molecule_structure}"
    print(top, start_path)
    print("Starting training of model")
    start_time = time.time()
    if hyper_optimized == "OPTIMIZED":
        #model.fit(X_train, y_train)
        #kernel = model.quantum_kernel
        #print(kernel.get_circuit)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        print(f"Model Train R2: {grid_search.best_score_}")
        model = grid_search.best_estimator_
        log_path = start_path + f"/{hyper_optimized}_{model_type.upper()}.bin"
    else:
        model.fit(X_train, y_train)
        log_path = start_path + f"/{model_type.upper()}.bin"
    end_time = time.time()
    
    print(log_path)
    # open(log_path, "w").close()
    print(f"Fitting the model took {end_time-start_time} seconds")

    predictions = model.predict(X_test)
    print(f"The r2 score on the test set is {r2_score(y_test.flatten(), predictions.flatten())}")
    print(f"Test MSE: {mean_squared_error(y_test.flatten(), predictions.flatten())}")

    progress = {"Training Time": end_time-start_time, "R2 Score":r2_score(y_test.flatten(), predictions.flatten()), "Model":model}
    joblib.dump(progress, log_path)
    path = f"paper_results/training_scores/{percent_data}-{num_features}-{basis}-{molecule_structure}-R2.csv"
    output = [f"", f"Classical-{model_type}", r2_score(y_test.flatten(), predictions.flatten())]
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Append the label and value as a new row
        writer.writerow(output)
    water_t2_path = f"paper_results/water_energy/{os.path.basename(path)}"

    unscaledy_test = water_t2_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    unscaledy_predictions = water_t2_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    print(unscaledy_predictions.shape)

    # Combine the lists into a single list of tuples for easier iteration
    data_to_write = zip(unscaledy_test, unscaledy_predictions)

    with open(water_t2_path, mode='a', newline='') as file:
        fieldnames = ['Molecule', 'Circuit', 'Expected', 'Predicted', 't2_num']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Check if the file is empty to write the header
        if file.tell() == 0:
            writer.writeheader()
        num_t2s = int((y_test.shape[0]) / 40)
        count = 0
        # Iterate through the data and write a new row for each pair
        for expected_value, predicted_value in data_to_write:
            row_output = {
                'Molecule': f"{molecule_structure}{count // num_t2s}",
                'Circuit': f"{circuit_struct}_{degree}_{layers}_{job_num}",
                'Expected': expected_value,
                'Predicted': predicted_value,
                't2_num': count % num_t2s}
            writer.writerow(row_output)
            time.sleep(0.01)
            count += 1
time.sleep(int(job_num)*10)
molecules = molecule_structure.split("-")

total_X_test = np.array([])
total_y_test = np.array([])
total_X_train = np.array([])
total_y_train = np.array([])

#This is setup to enable multiple training molecules
for molecule_struct in molecules:
    top=os.getcwd()
    if scale != "Water-Scaled":
        with open(rf'data/training/{basis}/{molecule_struct}/scaled/{num_features}_feature_train.bin','rb') as f:
            ddcc_train = joblib.load(f)

        with open(rf'data/training/{basis}/{molecule_struct}/scaled/{num_features}_feature_test.bin','rb') as f:
            ddcc_test = joblib.load(f)

        with open(rf'data/training/{basis}/{molecule_struct}/scaled/t2_scaler.bin','rb') as f:
            ddcc_scaler = joblib.load(f)
    else:
        with open(rf'data/training/{basis}/{molecule_struct}/scaled/full_feature_train.bin','rb') as f:
            ddcc_train = joblib.load(f)

        with open(rf'data/training/{basis}/{molecule_struct}/scaled/full_feature_test.bin','rb') as f:
            ddcc_test = joblib.load(f)

        with open(rf'data/training/{basis}/{molecule_struct}/scaled/t2_scaler.bin','rb') as f:
            ddcc_scaler = joblib.load(f)

    # print(ddcc_train.keys())
    X_ddcc_train, y_ddcc_train = ddcc_train['X'],ddcc_train['y']
    X_ddcc_test, y_ddcc_test = ddcc_test['X'],ddcc_test['y']

    X_train, y_train = X_ddcc_train, y_ddcc_train
    X_test, y_test = X_ddcc_test, y_ddcc_test

    # Scale all data to the same range (Data was generated by molecule so ranges differ)
    if scale=="Water-Scaled":
        print("Scaling data")
        with open(rf"data/training/{basis}/Water/scaled/feature_scaler.bin", 'rb') as f:
            water_feature_scaler = joblib.load(f)
        with open(rf"data/training/{basis}/Water/scaled/t2_scaler.bin", 'rb') as f:
            water_t2_scaler = joblib.load(f)
        with open(rf"data/training/{basis}/{molecule_struct}/scaled/feature_scaler.bin", 'rb') as f:
            molecule_feature_scaler = joblib.load(f)
        with open(rf"data/training/{basis}/{molecule_struct}/scaled/t2_scaler.bin", 'rb') as f:
            molecule_t2_scaler = joblib.load(f)
        print(water_feature_scaler, water_t2_scaler)
        print(molecule_feature_scaler, molecule_t2_scaler)
        unscaledX_train = molecule_feature_scaler.inverse_transform(X_train)
        unscaledy_train = molecule_t2_scaler.inverse_transform(y_train.reshape(-1, 1))

        if percent_data < 1:
            unscaledX_train, unscaledy_train = probability_selection_scheme(
                unscaledX_train,
                unscaledy_train,
                percent_data,
            )

        X_train = np.array(water_feature_scaler.transform(unscaledX_train))
        y_train = np.array(water_t2_scaler.transform(unscaledy_train).flatten())

        unscaledX_test = molecule_feature_scaler.inverse_transform(X_test)
        unscaledy_test = molecule_t2_scaler.inverse_transform(y_test.reshape(-1, 1))

        X_test = np.array(water_feature_scaler.transform(unscaledX_test))
        y_test = np.array(water_t2_scaler.transform(unscaledy_test).flatten())

        # Select features, we only used 5
        top5=np.array([20, 21, 24, 23, 22][::-1])
        top10=np.array([28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])
        top15=np.array([18, 5, 9, 29, 27, 28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])
        print(molecule_structure, X_train.shape, X_test.shape)
        top = top15[:num_features]
        X_train, X_test = X_train[:, top], X_test[:, top]

        if len(total_X_train) == 0:
            total_y_train = y_train
            total_y_test = y_test
            total_X_train = X_train
            total_X_test = X_test
        else:
            total_y_train = np.append(total_y_train, y_train)
            total_y_test = np.append(total_y_test, y_test)
            total_X_train = np.vstack((total_X_train, X_train))
            total_X_test = np.vstack((total_X_test, X_test))
X_train = total_X_train
y_train = total_y_train
X_test = total_X_test
y_test = total_y_test

# X_train, y_train = X_ddcc_train, y_ddcc_train
# X_test, y_test = X_ddcc_test, y_ddcc_test
#X_train, y_train = X_ddcc_train, y_ddcc_train
#X_test, y_test = np.clip(X_ddcc_test, -1, 1), y_ddcc_test

# X_train = [X_train[i:i+4] for i in range(0,len(X_train),4)]
# X_test = [X_test[i:i+4] for i in range(0,len(X_test),4)]
scaler = ddcc_scaler

print(len(X_train), X_train.shape, X_train.shape)
print(len(y_train), y_train.shape, y_train.shape)

print(X_test.shape)
print(circuit_struct, model_type, degree, layers)

if date != "" and not date is None:
    os.makedirs(f"models/{date}/{molecule_structure}", exist_ok=True)
else:
    os.makedirs(f"models/{molecule_structure}", exist_ok=True)
os.makedirs("paper_results/training_scores", exist_ok=True)
os.makedirs("paper_results/water_energy", exist_ok=True)

if model_type == "QNN":
     pass
#    run_qnn()
elif model_type == "qsvr" or model_type == "qgpr" or model_type == "qkrr":
    quantum_kernel_method(circuit_struct, degree, layers, model_type)
else:
    classical_kernel_method(model_type)
