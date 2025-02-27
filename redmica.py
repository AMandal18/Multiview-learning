Here is the corresponding Python code for the provided C code. The Python code uses libraries such as `numpy` for matrix operations and `scipy` for eigenvalue and eigenvector computations. The code is structured to mimic the functionality of the original C code.

```python
import numpy as np
import os
import sys
import time
from scipy.linalg import eig

class Dataset:
    def __init__(self, number_of_samples, number_of_features, number_of_class_labels, class_labels, data_matrix):
        self.number_of_samples = number_of_samples
        self.number_of_features = number_of_features
        self.number_of_class_labels = number_of_class_labels
        self.class_labels = class_labels
        self.data_matrix = data_matrix

class RhepmMatrix:
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

class Modality:
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

class FeatureSet:
    def __init__(self, relevance, correlation, rhepm_data_matrix, canonical_variables_matrix, basis_vector_matrix):
        self.relevance = relevance
        self.correlation = correlation
        self.rhepm_data_matrix = rhepm_data_matrix
        self.canonical_variables_matrix = canonical_variables_matrix
        self.basis_vector_matrix = basis_vector_matrix

class Feature:
    def __init__(self, relevance, significance, objective_function_value, rhepm_data_matrix):
        self.relevance = relevance
        self.significance = significance
        self.objective_function_value = objective_function_value
        self.rhepm_data_matrix = rhepm_data_matrix

def write_instruction():
    print("s:\tInput Modality-name")
    print("M:\tNumber of Modalities")
    print("f:\tNumber of New Features")
    print("p:\tPath for Input/Output Files")
    print("m:\tMinimum Value of Regularization Parameter")
    print("n:\tMaximum Value of Regularization Parameter")
    print("d:\tIncrement of Regularization Parameter")
    print("o:\tWeight Parameter")
    print("h:\tHelp")
    sys.exit(1)

def read_input_file(filename):
    with open(filename, 'r') as file:
        number_of_samples, number_of_features, number_of_class_labels = map(int, file.readline().split())
        data_matrix = np.zeros((number_of_samples, number_of_features))
        class_labels = np.zeros(number_of_samples, dtype=int)
        for i in range(number_of_samples):
            row = list(map(float, file.readline().split()))
            data_matrix[i] = row[:-1]
            class_labels[i] = int(row[-1])
    return Dataset(number_of_samples, number_of_features, number_of_class_labels, class_labels, data_matrix)

def zero_mean(data_matrix):
    mean = np.mean(data_matrix, axis=1, keepdims=True)
    return data_matrix - mean

def between_set_covariance(data_matrix1, data_matrix2):
    return np.dot(data_matrix1, data_matrix2.T) / data_matrix1.shape[1]

def within_set_covariance(data_matrix):
    return np.dot(data_matrix, data_matrix.T) / data_matrix.shape[1]

def matrix_transpose(data_matrix):
    return data_matrix.T

def matrix_multiplication(data_matrix1, data_matrix2):
    return np.dot(data_matrix1, data_matrix2)

def eigenvalue_eigenvector(data_matrix):
    eigenvalues, eigenvectors = eig(data_matrix)
    return np.real(eigenvalues), np.real(eigenvectors)

def copy_eigenvector(eigenvector_data_matrix, basis_vector_data_matrix):
    basis_vector_data_matrix[:] = eigenvector_data_matrix[:, :basis_vector_data_matrix.shape[1]]

def generate_equivalence_partition(data_matrix, class_labels, epsilon):
    unique_labels = np.unique(class_labels)
    rhepm_data_matrix = np.zeros((len(unique_labels), data_matrix.shape[0]), dtype=int)
    for i, label in enumerate(unique_labels):
        min_val = np.min(data_matrix[class_labels == label])
        max_val = np.max(data_matrix[class_labels == label])
        rhepm_data_matrix[i] = ((data_matrix >= min_val - epsilon) & (data_matrix <= max_val + epsilon)).astype(int)
    return rhepm_data_matrix

def dependency_degree(rhepm_data_matrix):
    confusion_vector = np.sum(rhepm_data_matrix, axis=0) != 1
    gamma = 1 - np.sum(confusion_vector) / rhepm_data_matrix.shape[1]
    return gamma

def form_resultant_equivalence_partition_matrix(equivalence_partition1, equivalence_partition2):
    return equivalence_partition1 & equivalence_partition2

def write_output_file(filename, new_data_matrix, class_labels):
    with open(filename, 'w') as file:
        file.write(f"{new_data_matrix.shape[0]}\t{new_data_matrix.shape[1]}\t{len(np.unique(class_labels))}\n")
        for i in range(new_data_matrix.shape[0]):
            file.write("\t".join(map(str, new_data_matrix[i])) + f"\t{class_labels[i]}\n")

def write_file(filename, new_data_matrix):
    np.savetxt(filename, new_data_matrix, delimiter='\t', fmt='%e')

def write_20_decimal_places_file(filename, new_data_matrix):
    np.savetxt(filename, new_data_matrix, delimiter='\t', fmt='%.20f')

def write_rhepm_file(filename, new_data_matrix):
    np.savetxt(filename, new_data_matrix, delimiter='\t', fmt='%d')

def write_correlation_file(filename, new_data_matrix):
    np.savetxt(filename, new_data_matrix, delimiter='\t', fmt='%.20f')

def write_relevance_file(filename, new_data_matrix):
    np.savetxt(filename, new_data_matrix, delimiter='\t', fmt='%.20f')

def write_lambda_file(filename, new_data_matrix):
    np.savetxt(filename, new_data_matrix, delimiter='\t', fmt='%f')

def write_R_eigenvalue_eigenvector(data_filename, eigenvector_filename, eigenvalue_filename, filename):
    with open(filename, 'w') as file:
        file.write(f"eigenvalue_eigenvector <- function(){{\n")
        file.write(f"\tx <- read.table('{data_filename}')\n")
        file.write(f"\tX <- as.matrix(x)\n")
        file.write(f"\tev <- eigen(X)\n")
        file.write(f"\twrite.table(Re(ev$vec), file='{eigenvector_filename}', sep='\\t', eol='\\n', row.names=FALSE, col.names=FALSE)\n")
        file.write(f"\twrite.table(Re(ev$val), file='{eigenvalue_filename}', sep='\\t', eol='\\n', row.names=FALSE, col.names=FALSE)\n")
        file.write(f"}}\n")
        file.write(f"\neigenvalue_eigenvector();\n")

def main():
    if len(sys.argv) < 2:
        write_instruction()

    modality_filename = None
    path = os.getcwd()
    number_of_modalities = 2
    number_of_new_features = 10
    lambda_minimum = 0.0
    lambda_maximum = 1.0
    delta = 0.1
    epsilon = 0.0
    omega = 0.5
    starting_combination = 0
    extracted_feature_number = 0

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-s':
            modality_filename = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '-M':
            number_of_modalities = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '-f':
            number_of_new_features = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '-p':
            path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '-m':
            lambda_minimum = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '-n':
            lambda_maximum = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '-d':
            delta = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '-o':
            omega = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '-h':
            write_instruction()
        else:
            print("Unrecognised option")
            sys.exit(1)

    if modality_filename is None:
        write_instruction()

    start_time = time.time()
    preprocessing(modality_filename, path, number_of_modalities, number_of_new_features, lambda_minimum, lambda_maximum, delta, epsilon, omega, starting_combination, extracted_feature_number)
    end_time = time.time()
    print(f"\nTOTAL TIME REQUIRED for ReDMiCA={(end_time - start_time) * 1000:.0f} millisec\n")

def preprocessing(modality_filename, path, number_of_modalities, number_of_new_features, lambda_minimum, lambda_maximum, delta, epsilon, omega, starting_combination, extracted_feature_number):
    modality_name = []
    with open(modality_filename, 'r') as file:
        for line in file:
            modality_name.append(line.strip())

    datasets = []
    for name in modality_name:
        filename = os.path.join(path, name)
        datasets.append(read_input_file(filename))

    for i in range(1, number_of_modalities):
        if datasets[i-1].number_of_samples != datasets[i].number_of_samples:
            print(f"Error: Number of Samples in Dataset{i-1} = {datasets[i-1].number_of_samples}\tNumber of Samples in Dataset{i} = {datasets[i].number_of_samples}")
            sys.exit(0)
        for j in range(datasets[i].number_of_samples):
            if datasets[i-1].class_labels[j] != datasets[i].class_labels[j]:
                print(f"Error: Class labels{i-1}[{j+1}] = {datasets[i-1].class_labels[j]}\tClass labels{i}[{j+1}] = {datasets[i].class_labels[j]}")
                sys.exit(0)

    datasets.sort(key=lambda x: x.number_of_features)

    if number_of_new_features > datasets[0].number_of_features or not number_of_new_features:
        number_of_new_features = datasets[0].number_of_features

    transpose_matrices = []
    zero_mean_matrices = []
    for dataset in datasets:
        transpose_matrix = matrix_transpose(dataset.data_matrix)
        transpose_matrices.append(Modality(transpose_matrix))
        zero_mean_matrix = zero_mean(transpose_matrix)
        zero_mean_matrices.append(Modality(zero_mean_matrix))

    output_path = os.path.join(path, "ReDMiCA")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ReDMiCA(output_path, transpose_matrices, zero_mean_matrices, datasets[0].class_labels, [dataset.number_of_features for dataset in datasets], datasets[0].number_of_samples, number_of_new_features, datasets[0].number_of_class_labels, number_of_modalities, lambda_minimum, lambda_maximum, delta, epsilon, omega, starting_combination, extracted_feature_number)

def ReDMiCA(path, transpose_matrices, zero_mean_matrices, class_labels, number_of_features, number_of_samples, number_of_new_features, number_of_class_labels, number_of_modalities, lambda_minimum, lambda_maximum, delta, epsilon, omega, starting_combination, extracted_feature_number):
    count = int((lambda_maximum - lambda_minimum) / delta) + 1
    combination = count ** number_of_modalities

    regularization = np.zeros(number_of_modalities, dtype=int)
    regularization_pointer = np.zeros((combination, number_of_modalities), dtype=int)
    lambda_matrix = np.zeros((combination, number_of_modalities))
    optimal_lambda_objective_function_value = np.zeros((number_of_new_features, number_of_modalities + 1))

    new_featureset = FeatureSet(
        relevance=np.zeros(number_of_new_features),
        correlation=np.zeros(number_of_new_features),
        rhepm_data_matrix=None,
        canonical_variables_matrix=Modality(np.zeros((number_of_samples, number_of_new_features))),
        basis_vector_matrix=[Modality(np.zeros((number_of_features[i], number_of_new_features))) for i in range(number_of_modalities)]
    )

    optimal_featureset = FeatureSet(
        relevance=np.zeros(number_of_new_features),
        correlation=np.zeros(number_of_new_features),
        rhepm_data_matrix=[RhepmMatrix(np.zeros((number_of_class_labels, number_of_samples), dtype=int)) for _ in range(number_of_new_features)],
        canonical_variables_matrix=Modality(np.zeros((number_of_samples, number_of_new_features))),
        basis_vector_matrix=[Modality(np.zeros((number_of_features[i], number_of_new_features))) for i in range(number_of_modalities)]
    )

    for i in range(combination):
        for j in range(number_of_modalities):
            regularization_pointer[i][j] = regularization[j]
            if j == number_of_modalities - 1:
                if regularization[j] == count - 1:
                    regularization[j] = 0
                    for k in range(j-1, -1, -1):
                        if regularization[k] == count - 1:
                            regularization[k] = 0
                        else:
                            regularization[k] += 1
                            break
                else:
                    regularization[j] += 1

    eigenvalue_of_covariance_data_matrix = []
    eigenvector_of_covariance_data_matrix = []
    transpose_eigenvector_of_covariance_data_matrix = []
    inverse_covariance_matrix = []
    covariance_matrix = []
    multiplication_inverse_covariance_matrix1 = []
    multiplication_inverse_covariance_matrix2 = []

    for i in range(number_of_modalities):
        eigenvalue_of_covariance_data_matrix.append(np.zeros(number_of_features[i]))
        eigenvector_of_covariance_data_matrix.append(Modality(np.zeros((number_of_features[i], number_of_features[i]))))
        transpose_eigenvector_of_covariance_data_matrix.append(Modality(np.zeros((number_of_features[i], number_of_features[i]))))
        covariance_matrix.append([Modality(np.zeros((number_of_features[i], number_of_features[j]))) for j in range(number_of_modalities)])

    for i in range(count):
        inverse_covariance_matrix.append(Modality(np.zeros((number_of_features[0], number_of_features[0]))))
        multiplication_inverse_covariance_matrix1.append([Modality(np.zeros((number_of_features[0], number_of_features[0]))) for _ in range(number_of_modalities - 1)])
        multiplication_inverse_covariance_matrix2.append([Modality(np.zeros((number_of_features[j+1], number_of_features[j]))) for j in range(number_of_modalities - 1)])

    if not starting_combination:
        for i in range(number_of_modalities):
            for j in range(number_of_modalities):
                if i == j:
                    covariance_matrix[i][j].data_matrix = within_set_covariance(zero_mean_matrices[i].data_matrix)
                    if lambda_minimum:
                        covariance_matrix[i][j].data_matrix += np.eye(number_of_features[i]) * lambda_minimum
                elif i < j:
                    covariance_matrix[i][j].data_matrix = between_set_covariance(zero_mean_matrices[i].data_matrix, zero_mean_matrices[j].data_matrix)
                else:
                    covariance_matrix[i][j].data_matrix = matrix_transpose(covariance_matrix[j][i].data_matrix)

            eigenvalues, eigenvectors = eigenvalue_eigenvector(covariance_matrix[i][i].data_matrix)
            eigenvector_of_covariance_data_matrix[i].data_matrix = eigenvectors
            transpose_eigenvector_of_covariance_data_matrix[i].data_matrix = matrix_transpose(eigenvectors)

        for t in range(count):
            for i in range(number_of_modalities):
                temp_data_matrix1 = np.zeros((number_of_features[i], number_of_features[i]))
                for j in range(number_of_features[i]):
                    for k in range(number_of_features[i]):
                        if (1 / (eigenvalue_of_covariance_data_matrix[i][k] + t * delta)) >= 0.000001:
                            temp_data_matrix1[j][k] = eigenvector_of_covariance_data_matrix[i].data_matrix[j][k] / (eigenvalue_of_covariance_data_matrix[i][k] + t * delta)
                        else:
                            temp_data_matrix1[j][k] = eigenvector_of_covariance_data_matrix[i].data_matrix[j][k] / 0.000001

                if i:
                    temp_data_matrix2 = np.zeros((number_of_features[i], number_of_features[i]))
                    temp_data_matrix3 = np.zeros((number_of_features[0], number_of_features[i]))
                    temp_data_matrix2 = matrix_multiplication(temp_data_matrix1, transpose_eigenvector_of_covariance_data_matrix[i].data_matrix)
                    temp_data_matrix3 = matrix_multiplication(covariance_matrix[0][i].data_matrix, temp_data_matrix2)
                    multiplication_inverse_covariance_matrix1[t][i-1].data_matrix = matrix_multiplication(temp_data_matrix3, covariance_matrix[i][0].data_matrix)
                    multiplication_inverse_covariance_matrix2[t][i-1].data_matrix = matrix_multiplication(temp_data_matrix2, covariance_matrix[i][i-1].data_matrix)

                    filename = os.path.join(path, f"multiplication_inverse_covariance_matrix1_{t}_{i-1}.txt")
                    write_20_decimal_places_file(filename, multiplication_inverse_covariance_matrix1[t][i-1].data_matrix)

                    filename = os.path.join(path, f"multiplication_inverse_covariance_matrix2_{t}_{i-1}.txt")
                    write_20_decimal_places_file(filename, multiplication_inverse_covariance_matrix2[t][i-1].data_matrix)
                else:
                    inverse_covariance_matrix[t].data_matrix = matrix_multiplication(temp_data_matrix1, transpose_eigenvector_of_covariance_data_matrix[i].data_matrix)
                    filename = os.path.join(path, f"inverse_covariance_matrix_{t}.txt")
                    write_20_decimal_places_file(filename, inverse_covariance_matrix[t].data_matrix)

    else:
        for t in range(count):
            for i in range(number_of_modalities):
                if i:
                    filename = os.path.join(path, f"multiplication_inverse_covariance_matrix1_{t}_{i-1}.txt")
                    multiplication_inverse_covariance_matrix1[t][i-1].data_matrix =
