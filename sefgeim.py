import numpy as np
import argparse
import os

class Dataset:
    def __init__(self, num_samples, num_features, num_class_labels, class_labels, data_matrix):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_class_labels = num_class_labels
        self.class_labels = class_labels
        self.data_matrix = data_matrix


def read_input_file(filename):
    with open(filename, 'r') as file:
        num_samples, num_features, num_class_labels = map(int, file.readline().split())
        data_matrix = np.zeros((num_samples, num_features))
        class_labels = np.zeros(num_samples, dtype=int)
        
        for i in range(num_samples):
            line = list(map(float, file.readline().split()))
            data_matrix[i, :-1] = line[:-1]
            class_labels[i] = int(line[-1])
        
    return Dataset(num_samples, num_features, num_class_labels, class_labels, data_matrix)


def zero_mean(data_matrix):
    mean = np.mean(data_matrix, axis=1, keepdims=True)
    return data_matrix - mean


def covariance_matrix(data_matrix):
    return np.cov(data_matrix, rowvar=True)


def eigen_decomposition(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]


def matrix_multiplication(matrix1, matrix2):
    return np.dot(matrix1, matrix2)


def compute_within_between_covariance(zero_mean_matrices):
    num_modalities = len(zero_mean_matrices)
    within_covariances = [covariance_matrix(zm) for zm in zero_mean_matrices]
    between_covariances = [[matrix_multiplication(zero_mean_matrices[i], zero_mean_matrices[j].T) / zero_mean_matrices[i].shape[1]
                             for j in range(num_modalities)] for i in range(num_modalities)]
    return within_covariances, between_covariances


def sefgeim_algorithm(datasets, zero_mean_matrices, num_new_features, lambda_min, lambda_max, delta, epsilon, omega):
    num_modalities = len(datasets)
    within_covariances, between_covariances = compute_within_between_covariance(zero_mean_matrices)
    
    best_features = None
    best_score = -np.inf
    
    lambda_values = np.arange(lambda_min, lambda_max + delta, delta)
    for lambda_value in lambda_values:
        eigen_data = [eigen_decomposition(within_cov + lambda_value * between_cov)
                      for within_cov, between_cov in zip(within_covariances, between_covariances)]
        
        selected_features = [eig[1][:, :num_new_features] for eig in eigen_data]
        transformed_data = [matrix_multiplication(feat.T, datasets[i].data_matrix) for i, feat in enumerate(selected_features)]
        final_features = np.mean(transformed_data, axis=0)
        
        score = np.sum(final_features**2)  # Objective function evaluation
        if score > best_score:
            best_score = score
            best_features = final_features
    
    return best_features


def main():
    parser = argparse.ArgumentParser(description="Python Implementation of SeFGeIM Algorithm")
    parser.add_argument('-s', '--input', required=True, help="Input modality filename")
    parser.add_argument('-f', '--features', type=int, default=10, help="Number of new features")
    parser.add_argument('-m', '--lambda_min', type=float, default=0.0, help="Minimum regularization parameter")
    parser.add_argument('-n', '--lambda_max', type=float, default=1.0, help="Maximum regularization parameter")
    parser.add_argument('-d', '--delta', type=float, default=0.1, help="Increment for lambda")
    parser.add_argument('-e', '--epsilon', type=float, default=0.0, help="Epsilon for feature selection")
    parser.add_argument('-o', '--omega', type=float, default=0.5, help="Weight parameter")
    
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        modality_files = [line.strip() for line in f.readlines()]
    
    datasets = [read_input_file(f) for f in modality_files]
    zero_mean_matrices = [zero_mean(ds.data_matrix) for ds in datasets]
    final_features = sefgeim_algorithm(datasets, zero_mean_matrices, args.features, args.lambda_min, args.lambda_max, args.delta, args.epsilon, args.omega)
    
    np.savetxt("sefgeim_features.txt", final_features)
    print("Feature extraction completed and saved to sefgeim_features.txt")

if __name__ == "__main__":
    main()

