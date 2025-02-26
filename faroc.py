import numpy as np
import argparse
import os
import time

class Dataset:
    """ Structure to store dataset information """
    def __init__(self, data_matrix, class_labels):
        self.data_matrix = np.array(data_matrix)
        self.class_labels = np.array(class_labels)
        self.num_samples, self.num_features = self.data_matrix.shape
        self.num_class_labels = len(np.unique(self.class_labels))

def read_input_file(filename):
    """ Reads input dataset from file """
    with open(filename, "r") as f:
        num_samples, num_features, num_class_labels = map(int, f.readline().split())
        data_matrix = []
        class_labels = []
        for _ in range(num_samples):
            line = list(map(float, f.readline().split()))
            data_matrix.append(line[:-1])
            class_labels.append(int(line[-1]))
    return Dataset(data_matrix, class_labels)

def zero_mean(data_matrix):
    """ Centers the dataset around zero by subtracting the mean """
    return data_matrix - np.mean(data_matrix, axis=1, keepdims=True)

def matrix_transpose(matrix):
    """ Returns the transpose of a matrix """
    return np.transpose(matrix)

def matrix_multiplication(matrix1, matrix2):
    """ Performs matrix multiplication """
    return np.dot(matrix1, matrix2)

def between_set_covariance(data_matrix1, data_matrix2):
    """ Computes between-set covariance """
    return matrix_multiplication(data_matrix1, matrix_transpose(data_matrix2)) / data_matrix1.shape[1]

def within_set_covariance(data_matrix):
    """ Computes within-set covariance """
    return matrix_multiplication(data_matrix, matrix_transpose(data_matrix)) / data_matrix.shape[1]

def eigenvalue_eigenvector(matrix):
    """ Computes eigenvalues and eigenvectors of a matrix """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors

def dependency_degree(partition_matrix, num_samples, num_classes):
    """ Computes the dependency degree (gamma) for feature selection """
    confusion_vector = np.sum(partition_matrix, axis=0)
    gamma = 1 - np.count_nonzero(confusion_vector > 1) / num_samples
    return gamma

def form_resultant_equivalence_partition(matrix1, matrix2):
    """ Forms the resultant equivalence partition matrix """
    return np.logical_and(matrix1, matrix2).astype(int)

def generate_equivalence_partition(data_matrix, class_labels, num_samples, num_classes, epsilon):
    """ Generates equivalence partition matrix for feature selection """
    partitions = np.zeros((num_classes, num_samples), dtype=int)
    unique_labels = np.unique(class_labels)
    
    for i, label in enumerate(unique_labels):
        indices = np.where(class_labels == label)[0]
        min_val, max_val = np.min(data_matrix[indices]), np.max(data_matrix[indices])
        partitions[i] = (data_matrix >= (min_val - epsilon)) & (data_matrix <= (max_val + epsilon))

    return partitions

def rhepm(feature_set, rhepm_data_matrix, canonical_variables, class_labels, num_samples, num_new_features, num_classes, epsilon, omega):
    """ Computes the relevance and significance of features """
    relevance = dependency_degree(rhepm_data_matrix, num_samples, num_classes)
    if num_new_features == 0:
        return relevance

    significance = 0
    valid_features = 0
    for i in range(num_new_features):
        resultant_partition = form_resultant_equivalence_partition(feature_set[i], rhepm_data_matrix)
        joint_dependency = dependency_degree(resultant_partition, num_samples, num_classes)
        if joint_dependency - relevance > 0:
            valid_features += 1
        significance += (joint_dependency - relevance)

    significance = significance / valid_features if valid_features > 0 else 0
    return omega * relevance + (1 - omega) * significance

def preprocessing(filename1, filename2, path, num_new_features, lambda_min, lambda_max, delta, epsilon, omega):
    """ Main preprocessing function to perform FaRoC transformation """
    dataset1 = read_input_file(os.path.join(path, filename1))
    dataset2 = read_input_file(os.path.join(path, filename2))

    if dataset1.num_samples != dataset2.num_samples:
        raise ValueError("Number of samples must be the same in both datasets")

    for i in range(dataset1.num_samples):
        if dataset1.class_labels[i] != dataset2.class_labels[i]:
            raise ValueError(f"Class labels do not match at sample {i+1}")

    # Ensure dataset1 has fewer features than dataset2
    if dataset1.num_features > dataset2.num_features:
        dataset1, dataset2 = dataset2, dataset1

    # Check new feature limits
    if num_new_features > min(dataset1.num_features, dataset2.num_features):
        raise ValueError("Number of new features exceeds available features")

    # Compute zero-mean data
    zero_mean_data1 = zero_mean(matrix_transpose(dataset1.data_matrix))
    zero_mean_data2 = zero_mean(matrix_transpose(dataset2.data_matrix))

    # Compute covariance matrices
    cross_covariance = between_set_covariance(zero_mean_data1, zero_mean_data2)
    within_cov1 = within_set_covariance(zero_mean_data1) + lambda_min * np.eye(dataset1.num_features)
    within_cov2 = within_set_covariance(zero_mean_data2) + lambda_min * np.eye(dataset2.num_features)

    # Eigen decomposition
    eigenvals1, eigenvecs1 = eigenvalue_eigenvector(within_cov1)
    eigenvals2, eigenvecs2 = eigenvalue_eigenvector(within_cov2)

    # Compute inverse covariance matrices
    inv_cov1 = np.linalg.inv(eigenvecs1 @ np.diag(eigenvals1) @ matrix_transpose(eigenvecs1))
    inv_cov2 = np.linalg.inv(eigenvecs2 @ np.diag(eigenvals2) @ matrix_transpose(eigenvecs2))

    # Compute transformations
    transformation1 = matrix_multiplication(inv_cov1, cross_covariance)
    transformation2 = matrix_multiplication(inv_cov2, matrix_transpose(cross_covariance))

    # Compute canonical variables
    canonical_variables = matrix_multiplication(transformation1, transformation2)

    # Write output
    np.savetxt(os.path.join(path, "canonical_variables.txt"), canonical_variables)
    print("FaRoC completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Python Implementation of FaRoC Algorithm")
    parser.add_argument("-1", "--file1", required=True, help="Input file 1")
    parser.add_argument("-2", "--file2", required=True, help="Input file 2")
    parser.add_argument("-f", "--features", type=int, default=10, help="Number of new features")
    parser.add_argument("-p", "--path", required=True, help="Path for input/output files")
    parser.add_argument("-m", "--lambda_min", type=float, default=0.0, help="Minimum regularization parameter")
    parser.add_argument("-n", "--lambda_max", type=float, default=1.0, help="Maximum regularization parameter")
    parser.add_argument("-d", "--delta", type=float, default=0.1, help="Increment of regularization parameter")
    parser.add_argument("-o", "--omega", type=float, default=0.5, help="Weight parameter")

    args = parser.parse_args()
    start_time = time.time()
    
    preprocessing(args.file1, args.file2, args.path, args.features, args.lambda_min, args.lambda_max, args.delta, 0.0, args.omega)
    
    end_time = time.time()
    print(f"\nTOTAL TIME REQUIRED for FaRoC = {int((end_time - start_time) * 1000)} milliseconds")

if __name__ == "__main__":
    main()

