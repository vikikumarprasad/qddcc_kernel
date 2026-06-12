from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
# from squlearn.encoding_circuit import QiskitEncodingCircuit
import numpy as np

def mitarai(quantumcircuit,num_wires,paramname='x'):
    # encoding as proposed by Mitarai et al.
    num_features = num_wires
    features = ParameterVector(paramname,num_features*2)
    for i in range(num_wires):
        feature_idx = i % num_features  # Calculate the feature index using modulo
        quantumcircuit.ry(np.arcsin(features[feature_idx * 2]), i)
        quantumcircuit.rz(np.arccos(features[feature_idx * 2 + 1] ** 2), i)

def double_angle(quantumcircuit, num_wires,paramname='x'):
    #  creates a circuit that encodes features into wires via angle encoding with an RY then RZ gate
    #  the features are encoded 1-1 onto the qubits
    #  if more wires are passed then features the remaining wires will be filled from the beginning of the feature list
    num_features = num_wires
    features = ParameterVector(paramname,num_features*2)
    for i in range(num_wires):
        feature_index = i % num_features
        quantumcircuit.ry(features[feature_index], i)
        quantumcircuit.rz(features[feature_index], i)

def entangle_cnot(quantumcircuit,num_wires):
    #  entangles all of the wires in a circular fashion using cnot gates
    for i in range(num_wires):
        
        if i == num_wires - 1:
            quantumcircuit.cx(i, 0)
        else:
            quantumcircuit.cx(i, i+1)

def entangle_cz(quantumcircuit,num_wires):
    #  entangles all of the wires in a circular fashion using cz gates
    for i in range(num_wires):
        
        if i == num_wires - 1:
            quantumcircuit.cz(i, 0)
        else:
            quantumcircuit.cz(i, i+1)

def HardwareEfficient(quantumcircuit,num_wires,paramname='theta'):
    parameters = ParameterVector(paramname,num_wires*3)
    for qubit in range(num_wires):
        quantumcircuit.rx(parameters[qubit * 3], qubit)  
        quantumcircuit.rz(parameters[qubit * 3 + 1], qubit)  
        quantumcircuit.rx(parameters[qubit * 3 + 2], qubit)  
    entangle_cnot(quantumcircuit,num_wires)

def circuit(nqubits,RUD=1,AL=1):
    qc = QuantumCircuit(nqubits)
    for i in range(RUD):
        double_angle(qc,nqubits,paramname=f'x{i}')
        qc.barrier()
        for j in range(AL):        
            HardwareEfficient(qc,nqubits,paramname=f'theta{i}_{j}')
            qc.barrier()
    return qc

class DefaultEncoder():
    def __init__(self, num_features, num_qubits, num_layers = 1, degree=0):
        self.num_features = num_features
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.degree = degree
        self.parameterized_encoding = (degree > 0)
    
    def get_circuit(self):
        return self.qc
    
    def get_original(self):
        array = np.array([])
        for i in range(self.num_layers):
            for j in range(self.num_features):
                if self.degree >= 0:
                    array = np.append(array, [0])
                if self.degree >= 1:
                    array = np.append(array, [1])
                array = np.append(array, [0 for _ in range(self.degree - 1)])
        return array
    
    def init_parameters(self, type = 'random'):
        # num_params = self.degree + 1
        # total_params = num_params * (self.num_layers * self.num_features + self.num_features - 1)
        # self.total_params = num_parameters

        if not self.parameterized_encoding:
            if type == "original" or type == "random":
                return np.random.uniform(-np.pi, np.pi, self.total_params)
            else:
                return np.random.uniform(-0.1 * np.pi, 0.1 *np.pi, self.total_params)
        
        match type:
            case "random":
                return np.random.uniform(-np.pi, np.pi, self.total_params)
            
            case "small_random":
                return np.random.uniform(-0.1 * np.pi, 0.1 *np.pi, self.total_params)
            
            case "original":
                return self.get_original()
                
            case "original_small_random":
                return self.get_original() + np.random.uniform(-0.1 * np.pi, 0.1 * np.pi, self.total_params)

class ZZ_featuremap(DefaultEncoder):
    def __init__(self, num_features, num_layers = 1, degree = 0, entanglement = "linear"):
        super().__init__(num_features, num_features, num_layers, degree)
        self.entanglement = "linear"

        if self.parameterized_encoding:
            self.qc = self.zz_qkernel_param()
        else:
            self.qc = self.zz_qkernel()
        
    def zz_qkernel(self):
        num_qubits = self.num_features
        data = ParameterVector("X",self.num_features)

        self.total_params = 0
        qc = QuantumCircuit(num_qubits)
        for i in range(self.num_layers):
            qc.h(range(num_qubits))

            data_indx = 0
            for n in range(num_qubits):
                if data_indx == self.num_features:
                    data_indx = 0
                qc.rz(2.0 * data[data_indx], n)
                data_indx += 1
            for n in range(num_qubits-1):
                qc.cx(n,n+1)
                qc.p(2 * (np.pi - data[n]) * (np.pi - data[n+1]), n+1)
                qc.cx(n,n+1)
            if self.entanglement == "circular":
                qc.cx(num_qubits-1, 0)
                qc.p(2 * (np.pi - data[num_qubits-1]) * (np.pi - data[0]), 0)
                qc.cx(num_qubits-1, 0)
            qc.barrier()
        return qc

    def zz_qkernel_param(self):

        def relation_func(params, feature):
            output = 0
            for i in range(len(params)):
                output += params[i] * (feature**i)
            
            return output
        circular_entangle = (self.entanglement=="circular")
        
        num_params = self.degree + 1
        inputs = ParameterVector("θ", num_params * self.num_layers * ( self.num_features + self.num_features - (not circular_entangle)))

        self.total_params = len(inputs)
        start_index = 0

        num_qubits = self.num_features
        data = ParameterVector("X",self.num_features)
        qc = QuantumCircuit(num_qubits)
        for i in range(self.num_layers):
            qc.h(range(num_qubits))

            data_indx = 0
            for n in range(num_qubits):
                if data_indx == self.num_features:
                    data_indx = 0
                qc.rz(relation_func(inputs[start_index:start_index+num_params], 2.0 * data[data_indx]), n)
                data_indx += 1
                start_index += num_params
            for n in range(num_qubits-1):
                qc.cx(n,n+1)
                qc.p(relation_func(inputs[start_index:start_index+num_params], 2 * (np.pi - data[n]) * (np.pi - data[n+1])), n+1)
                qc.cx(n,n+1)
                start_index += num_params
            if circular_entangle:
                qc.cx(num_qubits-1, 0)
                qc.p(relation_func(inputs[start_index:start_index+num_params], 2 * (np.pi - data[n]) * (np.pi - data[n+1])), n+1)
                start_index += num_params
                qc.cx(num_qubits-1, 0)
            qc.barrier()
        return qc
    
    def get_original(self):
        circular_entangle = (self.entanglement=="circular")
        array = np.array([])
        for i in range(self.num_layers):
            for j in range(self.num_features + self.num_features - (not circular_entangle)):
                if self.degree >= 0:
                    array = np.append(array, [0])
                if self.degree >= 1:
                    array = np.append(array, [1])
                array = np.append(array, [0 for _ in range(self.degree - 1)])
        return array