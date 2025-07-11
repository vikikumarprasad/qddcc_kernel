import numpy as np
from typing import Union
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter, ParameterExpression
from qiskit.circuit import Instruction
from sympy import Integer
import sympy
import symengine as se
from sympy import default_sort_key
from sympy import sympify, ordered, Function, Symbol, atan, Add, Mul, Pow, IndexedBase, S # S is SymPy's singleton for things like 0, 1

def FunctionEncoder(
        encoder_circuit,
        relation_function,
        overwrite_expression = False,
        feature_label = "X",
        parameter_label = "θ",
        combine_type = "Multiply",
        **kwargs
    ):
        num_qubits = encoder_circuit.num_qubits
        modified_circuit = QuantumCircuit(num_qubits)
        num_params = 0
        circuit_length = len(encoder_circuit)
        features = ParameterVector("X", num_qubits)
        #print(encoder_circuit)
        for i in range(circuit_length):
            expression_changed = False
            gate = encoder_circuit[i]
            operation = gate.operation
            qubits = gate.qubits
            clbits = gate.clbits
            # #print(gate)
            new_params = []
            #print(operation.params)
            for expr in operation.params:
                next_round = False
                expression_params = expr.parameters
                base_param_names = [p.name for p in expression_params]
                base_names = [p for p in expression_params if p.name.startswith("X")]
                #print(base_names)
                sympy_expr = expr.sympify()
                additive_terms = sorted(sympy_expr.args, key=lambda i: default_sort_key(i, order='lex'))[::-1]

                #print("Sympy: ", sympy_expr)
                # Determine which Qiskit Parameter objects we need to target

                # additive_terms = list(sympy_expr.args)
                constant_part = S.Zero
                # #print(additive_terms)
                # #print(base_param_names)
                modified_terms = []
                # remaining_parameters = []
                #print(additive_terms)
                if len(base_names) == 1:
                    expression_changed = True
                for term_param in additive_terms:
                    if not len(base_names) > 1:
                        #print("braking early!")          
                        break
                    if term_param.is_number:
                        modified_terms.append(term_param)
                        continue
                    # Keep a list of substitutions to apply to this term
                    substitutions_for_term = {}
                    for symbol in term_param.free_symbols:
                        qiskit_param_for_sym = None
                        parameter_pos = None
                        for i, qiskit_p in enumerate(expr.parameters):
                            #print("Iter: ", symbol)
                            # Compare the SymPy symbol from the term's free_symbols with the SymPy symbol that Qiskit maps 'qiskit_p' to
                            if symbol.name == qiskit_p.name and not qiskit_p.name.startswith(feature_label):
                                qiskit_param_for_sym = qiskit_p
                                parameter_pos = i
                                break
                        check_names = {p.name: p for p in term_param.free_symbols if p.name.startswith(feature_label)}
                        
                        if qiskit_param_for_sym != None and (len(check_names) > 0) or next_round:
                            next_round = False
                            # #print(qiskit_param_for_sym)
                            # #print(term_param.args, type(symbol))
                            # If it's a multiplication and the symbol is directly one of its arguments (factors)
                            # This means it's a multiplicative factor. Substitute with 1.
                            #print("Wooo", symbol)
                            substitutions_for_term[str(symbol)] = S.One
                        elif symbol == term_param:
                            #print(term_param)
                            #print("BAD", symbol)
                            # The term itself is just the symbol (additive role)
                            substitutions_for_term[str(symbol)] = S.Zero
                        else:
                            next_round = isinstance(term_param, se.Function)
                            #print("Funky", symbol)
                    #print(substitutions_for_term.keys())
                    name_map = {p.name: p for p in expr.parameters}
                    # #print("Names: ", name_map)
                    # #print({name_map[name_str]: value
                    #             for name_str, value in substitutions_for_term.items()})
                    mapping = { name_map[name_str]: value
                                for name_str, value in substitutions_for_term.items()
                                if name_str in name_map }
                    #print("Changes: ", mapping)
                    expr = expr.bind(mapping)
                    expression_changed = True
                # #print(type(expr))
                #print("Final expression: ",expr)
                #print(modified_circuit)
                expression_params = expr.parameters
                base_param_names = [p.name for p in expression_params]
                result = 0
                #print("Params: ", num_params)
                #print(expression_changed)
                # #print(base_param_names)
                if overwrite_expression == False and expression_changed:
                    result = relation_function(expr, num_params)
                    temp_param_names = {p.name for p in result.parameters}
                    for name in temp_param_names:
                        if name.startswith("Phi"):
                            num_params += 1
                elif expression_changed:
                    result = 1 if combine_type == "Multiply" else 0
                    #print(result, base_param_names)
                    feature_expr = []
                    for name in base_param_names:
                        if name.startswith(feature_label):
                            #print(name)
                            feature_expr.append(name)
                    feature_expr = [int(feature_expr[i][-2]) for i in range(len(feature_expr))]
                    # features = ParameterVector("X", len(feature_expr))
                    #print(feature_expr)
                    for feature in feature_expr:
                        #print("features", feature)
                        if combine_type == "Multiply":
                            result *= relation_function(features[feature], num_params)
                        else:
                            result += relation_function(features[feature], num_params)
                    #print(result)
                    temp_param_names = {p.name for p in result.parameters}
                    for name in temp_param_names:
                        if name.startswith("Phi"):
                            num_params += 1
                if any(name.startswith(feature_label) for name in base_param_names):
                #     
                    # num_params += increment
                    # result = {result[0]}
                    # #print(result)
                    new_params.append(result)
                else:
                    new_params.append(expr)
                #print(result)
                base_param_names = {p.name for p in expr.parameters}
                for name in base_param_names:
                    if name.startswith("Phi"):
                        num_params += 1
            # #print(new_params)
            if expression_changed:
                new_operation = Instruction(
                    name=operation.name,
                    num_qubits=operation.num_qubits,
                    num_clbits=operation.num_clbits,
                    params=new_params
                )
            else:
                #print("length: ", len(new_params))
                if len(new_params) >= 1:
                    #print(new_params)
                    temp_param_names = {p.name for p in new_params}
                    for name in temp_param_names:
                        if name.startswith("Phi"):
                            num_params += 1
                    #print(temp_param_names, num_params)
                new_operation = operation
            #print(modified_circuit)
            modified_circuit.append(new_operation, qubits, clbits)
        #print(modified_circuit)
        return modified_circuit
