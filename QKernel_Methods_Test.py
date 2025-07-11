import os
import joblib
import json
import time
# !{sys.executable} -m pip install qcircuit
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
import qiskit
from qiskit import *
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit_aer import Aer
from squlearn.optimizers import Adam, SPSA

from squlearn import Executor

from squlearn.encoding_circuit import *
from squlearn.kernel.qsvr import QSVR
from squlearn.kernel.qgpr import QGPR
from squlearn.encoding_circuit.layered_encoding_circuit import LayeredEncodingCircuit, Layer
from squlearn.kernel import ProjectedQuantumKernel, KernelOptimizer, FidelityKernel
from Training_Functions import MSE_Score
from sklearn.linear_model import Ridge
from Circuit_Classes import ZZ_featuremap
#from QiskitRegressor import *
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from squlearn.kernel import QKRR
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import itertools
from FunctionEncoder import *

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
job_num = os.getenv("JOB_NUM")
trainable = os.getenv("TRAINABLE")
print("Qiksit Version: ",qiskit.__version__)
if model_type == "QNN":
    n_jobs = 16
    os.environ["OMP_NUM_THREADS"] = str(n_jobs)

def learning_rate(iteration, transition_begin = 0, transition_steps = 1, init_value = 0.08, decay_rate = 0.1):
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
#def run_qnn():
#    X_train = X_train.reshape(-1,64,5)
#    X_test = X_test.reshape(-1,64,5)
#    y_train = y_train(-1, 64)
#    y_test = y_test(-1, 64)

#    num_qubits = 5
#    RUD = 1
#    AL = 5

#    optimization_level = 2
#    shots = 3072
    # shots = 1024.0 * 1
#    resilience_level = 1

#    qc = circuit(num_qubits,RUD,AL)

#    observables_labels = ''.join(['I']*(num_qubits-1))+"Z"

#    model = QiskitRegressor(qc,
#                        num_qubits,
#                        AL,
#                        RUD,
#                        'statevector', #'fake',
#                        observables_labels,
#                        channel='ibm_quantum',
#                        instance='ibm-q/open/main',
#                        parameterpath = None,
#                        optimization_level = optimization_level,
#                        resilience_level = resilience_level,
#                        shots = shots,
#                        iterations = 500,
#                        verbose = True,
#                        n_jobs = n_jobs)
#    model.fit(X_train,y_train)

#    predictions = model.predict(X_test)
#    r2 = r2_score(y_test.flatten(), predictions.flatten())
#    mse = mean_squared_error(y_test.flatten(), predictions.flatten())
#    print(f"QNN R2 score is {r2}")
#    print(f"QNN MSE score is {mse}")

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
    else:
        print("Not valid encoder")
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
    initial_parameters = (np.random.rand(num_params) - 0.5) * 2 * np.pi
    print(initial_parameters)
    print(encoding_circuit.get_circuit(ParameterVector("X", num_features), ParameterVector("P", encoding_circuit.num_parameters) ) )
    if kernel_type == "Projected":
        kernel = ProjectedQuantumKernel(
            encoding_circuit,
            executor=Executor(Aer.get_backend("aer_simulator"), shots=1024)
        )
    elif kernel_type == "Fidelity":
        kernel = FidelityKernel(encoding_circuit, executor=Executor(Aer.get_backend("aer_simulator"), shots=1024))
    if date != "" and not date is None:
        start_path = f"models/{date}/{molecule_structure}"
    else:
        start_path = f"models/{molecule_structure}"
    start_path = start_path + f"/{trainable.upper()}_{kernel_type.upper()}"
    # set up the optimizer
    # adam_opt = Adam(options={"maxiter":1, "lr": learning_rate, "tol":0.001})

    # if trainable kernel set up the kernel optimizer
    if trainable == "True":
        cobyla_opt = COBYLA(maxiter=500)
        kta_loss = MSE_Score(model_type, noisy=True)
        kernel = KernelOptimizer(quantum_kernel=kernel, loss = kta_loss, optimizer=cobyla_opt, initial_parameters=initial_parameters)

    if model_type == "qsvr":
        model = QSVR(kernel)
        param_grid = {
            "C":[0.1, 1, 5, 10, 50, 100, 1000],
            "epsilon":[0.1,0.05,0.01,0.005,0.001,0.0001]
        }
    elif model_type == "qgpr":
        model = QGPR(kernel)
    elif model_type == "qkrr":
        model = QKRR(kernel)
        param_grid = {
            'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept':[True, False],
            'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        }
    print(start_path)
    print("Starting training of model")
    start_time = time.time()
    if hyper_optimized == "OPTIMIZED":
        model.fit(X_train, y_train)
            # Extract all the lists of options from the dictionary's values
        all_options_lists = []
        print("Processing dictionary:")
        for key, options in param_grid.items():
            if not isinstance(options, list):
                raise ValueError(f"Value for key '{key}' must be a list of options.")
            print(f"  Key: '{key}', Options: {options}")
            all_options_lists.append(options)

        print("\nGenerating combinations...")
        # Generate all combinations using itertools.product
        # The '*' unpacks the list of lists into separate arguments for product
        best_model = model
        best_r2 = -50
        best_params = {}
        for combination in itertools.product(*all_options_lists):
            combination = dict(zip(param_grid.keys(), combination))
            if model_type == "qsvr":
                kernel = model.quantum_kernel
                temp_model = QSVR(kernel, **combination)
            elif model_type == "qkrr":
                kernel = model._quantum_kernel
                temp_model = QKRR(kernel, **combination)
            temp_model.fit(X_train, y_train)
            predictions = temp_model.predict(X_train)
            r2 = r2_score(y_train.flatten(), predictions.flatten())
            print(f"For {combination}: R2 = {r2}")
            if r2 > best_r2:
                best_r2 = r2
                best_model = temp_model
                best_params = combination
        print(f"Best R2: {best_r2}")
        print(f"Best hyperparameters: {best_params}")
        model = best_model
        #grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", verbose=1)
        #grid_search.fit(X_train, y_train)
        #print(grid_search.best_params_)
        #print(f"Model Train R2: {grid_search.best_score_}")
        #model = grid_search.best_estimator_
        log_path = start_path + f"_{hyper_optimized}_{num_features}_{circuit_struct}_{degree}_{layers}_{model_type.upper()}.bin"
    else:
        model.fit(X_train, y_train)
        log_path = start_path + f"_{num_features}_{circuit_struct}_{degree}_{layers}_{model_type.upper()}.bin"
    end_time = time.time()
    if not job_num is None:
        log_path = log_path + f"-{job_num}"
    print(log_path)
    # open(log_path, "w").close()
    print(f"Fitting the model took {end_time-start_time} seconds")
    if trainable == "True":
        print(f"Optimized Parameters: {kernel.get_optimal_parameters()}")
    else:
        print(f"Circuit Parameters: {kernel.parameters}")
    predictions = model.predict(X_test)
    print(f"The r2 score on the test set is {r2_score(y_test.flatten(), predictions.flatten())}")
    print(f"Test MSE: {mean_squared_error(y_test.flatten(), predictions.flatten())}")
    if trainable == "True":
        loss_history = kta_loss.train_loss_history
        progress = {"Training Time": end_time-start_time, "Loss History":loss_history, "R2 Score":r2_score(y_test.flatten(), predictions.flatten()), "Model":model}
    else:
        progress = progress = {"Training Time": end_time-start_time, "R2 Score":r2_score(y_test.flatten(), predictions.flatten()), "Model":model}
    joblib.dump(progress, log_path)

def classical_kernel_method(model_type):

    if model_type == "svr":
        model = SVR(kernel="rbf")
        param_grid = {
            "kernel":["rbf", "linear", "poly", "sigmoid"],
            "C":[0.1, 1, 5, 10, 50, 100, 1000],
            "gamma":["scale", "auto", 0.1, 0.01, 1, 10, 50, 100],
            "epsilon":[0.1,0.05,0.01,0.005,0.001,0.0001]
        }
    elif model_type == "krr":
        model = Ridge()
        param_grid = {
            'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept':[True, False],
            'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
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

top=os.getcwd()

with open(os.path.join(top,rf'Data/STO-3G/{molecule_structure}/scaled/{num_features}_feature_train.bin'),'rb') as f:
    ddcc_train = joblib.load(f)

with open(os.path.join(top,rf'Data/STO-3G/{molecule_structure}/scaled/{num_features}_feature_test.bin'),'rb') as f:
    ddcc_test = joblib.load(f)

with open(os.path.join(top,rf'Data/STO-3G/{molecule_structure}/scaled/t2_scaler.bin'),'rb') as f:
    ddcc_scaler = joblib.load(f)
# print(ddcc_train.keys())
X_ddcc_train, y_ddcc_train = ddcc_train['X'],ddcc_train['y']
X_ddcc_test, y_ddcc_test = ddcc_test['X'],ddcc_test['y']

# X_train, y_train = X_ddcc_train, y_ddcc_train
# X_test, y_test = X_ddcc_test, y_ddcc_test
X_train, y_train = np.clip(X_ddcc_train, -1, 1), y_ddcc_train
X_test, y_test = np.clip(X_ddcc_test, -1, 1), y_ddcc_test

# X_train = [X_train[i:i+4] for i in range(0,len(X_train),4)]
# X_test = [X_test[i:i+4] for i in range(0,len(X_test),4)]
scaler = ddcc_scaler

print(len(X_train), X_train[0].shape, X_train[-1].shape)
print(len(y_train), y_train[0].shape, y_train[-1].shape)

print(X_test.shape)
print(circuit_struct, model_type, degree, layers)

if date != "" and not date is None:
    os.makedirs(f"models/{date}/{molecule_structure}", exist_ok=True)
else:
    os.makedirs(f"models/{molecule_structure}", exist_ok=True)

if model_type == "QNN":
     pass
#    run_qnn()
elif model_type == "qsvr" or model_type == "qgpr" or model_type == "qkrr":
    quantum_kernel_method(circuit_struct, degree, layers, model_type)
else:
    classical_kernel_method(model_type)
