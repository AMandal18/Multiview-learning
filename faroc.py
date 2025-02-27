import numpy as np
import os
import sys
import time
from scipy.linalg import eigh

class Dataset:
    def __init__(self, data_matrix, class_labels, num_samples, num_features, num_class_labels):
        self.data_matrix = data_matrix
        self.class_labels = class_labels
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_class_labels = num_class_labels

class BasisVector:
    def __init__(self, basis_vector_data_matrix1, basis_vector_data_matrix2):
        self.basis_vector_data_matrix1 = basis_vector_data_matrix1
        self.basis_vector_data_matrix2 = basis_vector_data_matrix2

class FeatureSet:
    def __init__(self, basis_vector_data_matrix1, basis_vector_data_matrix2, canonical_variables_data_matrix, correlation, rhepm_feature):
        self.basis_vector_data_matrix1 = basis_vector_data_matrix1
        self.basis_vector_data_matrix2 = basis_vector_data_matrix2
        self.canonical_variables_data_matrix = canonical_variables_data_matrix
        self.correlation = correlation
        self.rhepm_feature = rhepm_feature

class Feature:
    def __init__(self, rhepm_data_matrix, relevance, significance, objective_function_value):
        self.rhepm_data_matrix = rhepm_data_matrix
        self.relevance = relevance
        self.significance = significance
        self.objective_function_value = objective_function_value

def write_instruction():
    print("1:\tInput File 1")
    print("2:\tInput File 2")
    print("f:\tNumber of New Features")
    print("p:\tPath for Input/Output Files")
    print("m:\tMinimum Value of Regularization Parameter")
    print("n:\tMaximum Value of Regularization Parameter")
    print("d:\tIncrement of Regularization Parameter")
    print("o:\tWeight Parameter")
    print("h:\tHelp")
    sys.exit(1)

def read_input_file(filename):
    with open(filename, 'r') as f:
        num_samples, num_features, num_class_labels = map(int, f.readline().split())
        data_matrix = np.zeros((num_samples, num_features))
        class_labels = np.zeros(num_samples, dtype=int)
        for i in range(num_samples):
            row = list(map(float, f.readline().split()))
            data_matrix[i] = row[:-1]
            class_labels[i] = int(row[-1])
    return Dataset(data_matrix, class_labels, num_samples, num_features, num_class_labels)

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
    eigenvalues, eigenvectors = eigh(data_matrix)
    return eigenvalues, eigenvectors

def FaRoC(path, data_matrix1, data_matrix2, class_labels, lambda_min, lambda_max, delta, num_samples, num_features1, num_features2, num_new_features, num_class_labels, epsilon, omega):
    count = int((lambda_max - lambda_min) / delta) + 1
    if not num_new_features:
        num_new_features = num_features1

    cross_covariance_matrix1 = between_set_covariance(data_matrix1, data_matrix2)
    cross_covariance_matrix2 = matrix_transpose(cross_covariance_matrix1)

    covariance_matrix1 = within_set_covariance(data_matrix1)
    if lambda_min:
        covariance_matrix1 += np.eye(num_features1) * lambda_min

    covariance_matrix2 = within_set_covariance(data_matrix2)
    if lambda_min:
        covariance_matrix2 += np.eye(num_features2) * lambda_min

    eigenvalues1, eigenvectors1 = eigenvalue_eigenvector(covariance_matrix1)
    eigenvalues2, eigenvectors2 = eigenvalue_eigenvector(covariance_matrix2)

    h_data_matrix1 = []
    h_data_matrix2 = []
    for t in range(count):
        temp_matrix1 = np.zeros((num_features1, num_features1))
        for i in range(num_features1):
            for j in range(num_features1):
                if eigenvalues1[j] + t * delta >= 0.000001:
                    temp_matrix1[i][j] = eigenvectors1[i][j] / (eigenvalues1[j] + t * delta)
                else:
                    temp_matrix1[i][j] = eigenvectors1[i][j] / 0.000001
        inverse_covariance_matrix1 = matrix_multiplication(temp_matrix1, matrix_transpose(eigenvectors1))
        h_data_matrix1.append(matrix_multiplication(inverse_covariance_matrix1, cross_covariance_matrix1))

        temp_matrix2 = np.zeros((num_features2, num_features2))
        for i in range(num_features2):
            for j in range(num_features2):
                if eigenvalues2[j] + t * delta >= 0.000001:
                    temp_matrix2[i][j] = eigenvectors2[i][j] / (eigenvalues2[j] + t * delta)
                else:
                    temp_matrix2[i][j] = eigenvectors2[i][j] / 0.000001
        inverse_covariance_matrix2 = matrix_multiplication(temp_matrix2, matrix_transpose(eigenvectors2))
        h_data_matrix2.append(matrix_multiplication(inverse_covariance_matrix2, cross_covariance_matrix2))

    new_h_data_matrix = []
    for i in range(count):
        for j in range(count):
            new_h_data_matrix.append(matrix_multiplication(h_data_matrix1[i], h_data_matrix2[j]))

    basis_vector_set = [BasisVector(np.zeros((num_new_features, num_features1)), np.zeros((num_new_features, num_features2))) for _ in range(count * count)]
    new_feature = [Feature(np.zeros((num_class_labels, num_samples)), 0, 0, 0) for _ in range(count * count)]

    optimal_featureset = FeatureSet(np.zeros((num_features1, num_new_features)), np.zeros((num_features2, num_new_features)), np.zeros((num_samples, num_new_features)), np.zeros(num_new_features), [Feature(np.zeros((num_class_labels, num_samples)), 0, 0, 0) for _ in range(num_new_features)])

    eigenvalue_data_matrix = np.zeros((count * count, num_new_features))
    canonical_variables_data_matrix = np.zeros((count * count, num_samples))
    correlation = np.zeros(count * count)
    objective_function_value = np.zeros(count * count)

    for t in range(num_new_features):
        n = 0
        for i in range(count):
            for j in range(count):
                if not t:
                    eigenvalues, eigenvectors = eigenvalue_eigenvector(new_h_data_matrix[n])
                    basis_vector_set[n].basis_vector_data_matrix1[t] = eigenvectors[:, 0]
                correlation[n] = np.sqrt(eigenvalue_data_matrix[n][t])
                basis_vector_set[n].basis_vector_data_matrix2[t] = matrix_multiplication(cross_covariance_matrix2, basis_vector_set[n].basis_vector_data_matrix1[t])
                cca_variables_data_matrix1 = matrix_multiplication(basis_vector_set[n].basis_vector_data_matrix1[t], data_matrix1)
                cca_variables_data_matrix2 = matrix_multiplication(basis_vector_set[n].basis_vector_data_matrix2[t], data_matrix2)
                canonical_variables_data_matrix[n] = cca_variables_data_matrix1 + cca_variables_data_matrix2
                objective_function_value[n] = rhepm(optimal_featureset, new_feature[n].rhepm_data_matrix, canonical_variables_data_matrix[n], class_labels, num_samples, t, num_class_labels, epsilon, omega)
                n += 1
                print(f"Number of Iteration = {n}")

        max_objective_function_value = np.max(objective_function_value)
        index = np.argmax(objective_function_value)
        if not max_objective_function_value:
            num_new_features = t
            break

        optimal_featureset.basis_vector_data_matrix1[:, t] = basis_vector_set[index].basis_vector_data_matrix1[t]
        optimal_featureset.basis_vector_data_matrix2[:, t] = basis_vector_set[index].basis_vector_data_matrix2[t]
        optimal_featureset.canonical_variables_data_matrix[:, t] = canonical_variables_data_matrix[index]
        optimal_featureset.correlation[t] = correlation[index]
        optimal_featureset.rhepm_feature[t].rhepm_data_matrix = new_feature[index].rhepm_data_matrix

        if not t:
            for i in range(count * count):
                del new_h_data_matrix[i]
            del new_h_data_matrix

        print(f"Number of feature = {t + 1}")

    for i in range(count * count):
        del basis_vector_set[i].basis_vector_data_matrix1
        del basis_vector_set[i].basis_vector_data_matrix2
        del new_feature[i].rhepm_data_matrix

    del cross_covariance_matrix1
    del cross_covariance_matrix2
    del covariance_matrix1
    del covariance_matrix2
    del eigenvalues1
    del eigenvectors1
    del eigenvalues2
    del eigenvectors2
    del h_data_matrix1
    del h_data_matrix2
    del new_h_data_matrix
    del basis_vector_set
    del new_feature
    del eigenvalue_data_matrix
    del canonical_variables_data_matrix
    del correlation
    del objective_function_value

    write_file(os.path.join(path, "basis_vector1.txt"), optimal_featureset.basis_vector_data_matrix1)
    write_file(os.path.join(path, "basis_vector2.txt"), optimal_featureset.basis_vector_data_matrix2)
    write_output_file(os.path.join(path, "canonical_variables.txt"), optimal_featureset.canonical_variables_data_matrix, class_labels, num_samples, num_new_features, num_class_labels)
    write_correlation_file(os.path.join(path, "correlation.txt"), optimal_featureset.correlation)

    del optimal_featureset.basis_vector_data_matrix1
    del optimal_featureset.basis_vector_data_matrix2
    del optimal_featureset.canonical_variables_data_matrix
    del optimal_featureset.correlation
    for i in range(num_new_features):
        del optimal_featureset.rhepm_feature[i].rhepm_data_matrix
    del optimal_featureset.rhepm_feature

def rhepm(optimal_featureset, rhepm_data_matrix, canonical_variables_data_matrix, class_labels, num_samples, num_new_features, num_class_labels, epsilon, omega):
    resultant_equivalence_partition = np.zeros((num_class_labels, num_samples), dtype=int)
    generate_equivalence_partition(rhepm_data_matrix, canonical_variables_data_matrix, class_labels, num_samples, num_class_labels, epsilon)
    relevance = dependency_degree(rhepm_data_matrix, num_samples, num_class_labels)
    if not num_new_features:
        return relevance
    else:
        significance = 0
        j = 0
        for i in range(num_new_features):
            form_resultant_equivalence_partition_matrix(optimal_featureset.rhepm_feature[i].rhepm_data_matrix, rhepm_data_matrix, resultant_equivalence_partition, num_class_labels, num_samples)
            joint_dependency = dependency_degree(resultant_equivalence_partition, num_samples, num_class_labels)
            if joint_dependency - relevance != 0.0:
                j += 1
            significance += joint_dependency - relevance
        if j:
            significance /= j
            return omega * relevance + (1 - omega) * significance
        else:
            return 0

def generate_equivalence_partition(rhepm_data_matrix, data_matrix, class_labels, num_samples, num_class_labels, epsilon):
    label = np.unique(class_labels)
    if len(label) != num_class_labels:
        print("Error: Error in Program.")
        sys.exit(0)
    min_data_matrix = np.zeros(num_class_labels)
    max_data_matrix = np.zeros(num_class_labels)
    for i in range(num_class_labels):
        min_data_matrix[i] = np.min(data_matrix[class_labels == label[i]])
        max_data_matrix[i] = np.max(data_matrix[class_labels == label[i]])
    for i in range(num_class_labels):
        for j in range(num_samples):
            rhepm_data_matrix[i][j] = 0
            if (data_matrix[j] >= min_data_matrix[i] - epsilon) and (data_matrix[j] <= max_data_matrix[i] + epsilon):
                rhepm_data_matrix[i][j] = 1

def dependency_degree(rhepm_data_matrix, num_samples, num_class_labels):
    confusion_vector = np.zeros(num_samples, dtype=int)
    for j in range(num_samples):
        sum_val = np.sum(rhepm_data_matrix[:, j])
        if not sum_val:
            print("Error: Error in RHEPM Computation.")
            sys.exit(0)
        elif sum_val == 1:
            confusion_vector[j] = 0
        else:
            confusion_vector[j] = 1
    gamma = 1 - np.sum(confusion_vector) / num_samples
    return gamma

def form_resultant_equivalence_partition_matrix(equivalence_partition1, equivalence_partition2, resultant_equivalence_partition, num_class_labels, num_samples):
    for i in range(num_class_labels):
        for j in range(num_samples):
            resultant_equivalence_partition[i][j] = 0
            if equivalence_partition1[i][j] and equivalence_partition2[i][j]:
                resultant_equivalence_partition[i][j] = 1

def write_output_file(filename, new_data_matrix, class_labels, num_samples, num_features, num_class_labels):
    with open(filename, 'w') as f:
        f.write(f"{num_samples}\t{num_features}\t{num_class_labels}\n")
        for i in range(num_samples):
            for j in range(num_features):
                f.write(f"{new_data_matrix[i][j]}\t")
            f.write(f"{class_labels[i]}\n")

def write_file(filename, new_data_matrix):
    with open(filename, 'w') as f:
        for row in new_data_matrix:
            for val in row:
                f.write(f"{val}\t")
            f.write("\n")

def write_correlation_file(filename, new_data_matrix):
    with open(filename, 'w') as f:
        for val in new_data_matrix:
            f.write(f"{val}\n")

def main():
    if len(sys.argv) < 3:
        write_instruction()

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    path = os.getcwd()
    num_new_features = 10
    lambda_min = 0.0
    lambda_max = 1.0
    delta = 0.1
    epsilon = 0.0
    omega = 0.5

    for i in range(3, len(sys.argv)):
        if sys.argv[i] == '-f':
            num_new_features = int(sys.argv[i + 1])
        elif sys.argv[i] == '-p':
            path = sys.argv[i + 1]
        elif sys.argv[i] == '-m':
            lambda_min = float(sys.argv[i + 1])
        elif sys.argv[i] == '-n':
            lambda_max = float(sys.argv[i + 1])
        elif sys.argv[i] == '-d':
            delta = float(sys.argv[i + 1])
        elif sys.argv[i] == '-o':
            omega = float(sys.argv[i + 1])
        elif sys.argv[i] == '-h':
            write_instruction()

    if not filename1 or not filename2 or omega < 0 or omega > 1:
        write_instruction()

    start_time = time.time()
    preprocessing(filename1, filename2, path, num_new_features, lambda_min, lambda_max, delta, epsilon, omega)
    end_time = time.time()
    print(f"TOTAL TIME REQUIRED for FaRoC = {(end_time - start_time) * 1000} millisec")

if __name__ == "__main__":
    main()
