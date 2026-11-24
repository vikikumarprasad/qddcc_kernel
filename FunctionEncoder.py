import numpy as np
from typing import Union
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter, ParameterExpression
from sympy import Integer
import sympy
import symengine as se
from sympy import default_sort_key
from sympy import sympify, ordered, Function, Symbol, atan, Add, Mul, Pow, IndexedBase, S

def FunctionEncoder(
        encoder_circuit,
        relation_function,
        overwrite_expression = False,
        feature_label = "X",
        parameter_label = "Phi",
        combine_type = "Multiply",
        **kwargs
    ):
        num_qubits = encoder_circuit.num_qubits
        modified_circuit = QuantumCircuit(num_qubits)
        num_params = 0
        circuit_length = len(encoder_circuit)
        features = ParameterVector("X", num_qubits)

        for i in range(circuit_length):
            expression_changed = False
            gate = encoder_circuit[i]
            operation = gate.operation
            qubits = gate.qubits
            clbits = gate.clbits

            new_params = []

            for expr in operation.params:
                next_round = False
                expression_params = expr.parameters
                base_param_names = [p.name for p in expression_params]
                base_names = [p for p in expression_params if p.name.startswith("X")]

                sympy_expr = expr.sympify()
                additive_terms = sorted(sympy_expr.args, key=lambda i: default_sort_key(i, order='lex'))[::-1]

                constant_part = S.Zero

                modified_terms = []

                if len(base_names) == 1:
                    expression_changed = True
                for term_param in additive_terms:
                    if not len(base_names) > 1:        
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
                            # Compare the SymPy symbol from the term's free_symbols with the SymPy symbol that Qiskit maps 'qiskit_p' to
                            if symbol.name == qiskit_p.name and not qiskit_p.name.startswith(feature_label):
                                qiskit_param_for_sym = qiskit_p
                                parameter_pos = i
                                break
                        check_names = {p.name: p for p in term_param.free_symbols if p.name.startswith(feature_label)}
                        
                        if qiskit_param_for_sym != None and (len(check_names) > 0) or next_round:
                            next_round = False
                            # If it's a multiplication and the symbol is directly one of its arguments (factors)
                            # This means it's a multiplicative factor. Substitute with 1.
                            substitutions_for_term[str(symbol)] = S.One
                        elif symbol == term_param:
                            # The term itself is just the symbol (additive role)
                            substitutions_for_term[str(symbol)] = S.Zero
                        else:
                            next_round = isinstance(term_param, se.Function)

                    name_map = {p.name: p for p in expr.parameters}

                    mapping = { name_map[name_str]: value
                                for name_str, value in substitutions_for_term.items()
                                if name_str in name_map }

                    expr = expr.bind(mapping)
                    expression_changed = True

                expression_params = expr.parameters
                base_param_names = [p.name for p in expression_params]
                result = 0

                if overwrite_expression == False and expression_changed:
                    result = relation_function(expr, num_params)
                    temp_param_names = {p.name for p in result.parameters}
                    for name in temp_param_names:
                        if name.startswith(parameter_label):
                            num_params += 1
                elif expression_changed:
                    result = 1 if combine_type == "Multiply" else 0

                    feature_expr = []
                    for name in base_param_names:
                        if name.startswith(feature_label):

                            feature_expr.append(name)
                    feature_expr = [int(feature_expr[i][-2]) for i in range(len(feature_expr))]

                    for feature in feature_expr:
                        if combine_type == "Multiply":
                            result *= relation_function(features[feature], num_params)
                        else:
                            result += relation_function(features[feature], num_params)

                    temp_param_names = {p.name for p in result.parameters}
                    for name in temp_param_names:
                        if name.startswith(parameter_label):
                            num_params += 1
                
                if any(name.startswith(feature_label) for name in base_param_names):
                    new_params.append(result)
                else:
                    new_params.append(expr)

                base_param_names = {p.name for p in expr.parameters}
                for name in base_param_names:
                    if name.startswith(parameter_label):
                        num_params += 1

            if expression_changed:
                # Keep the actual gate class (and hence its matrix/definition).
                # A bare Instruction named "rx" is accepted by Aer but cannot be
                # evolved by Qiskit's reference Statevector/Estimator.
                new_operation = operation.base_class(*new_params, label=operation.label)
            else:

                if len(new_params) >= 1:
                    temp_param_names = {p.name for p in new_params}
                    for name in temp_param_names:
                        if name.startswith(parameter_label):
                            num_params += 1
                new_operation = operation
            modified_circuit.append(new_operation, qubits, clbits)

        return modified_circuit
