import numpy as np
import argparse
import os
import sys  

class Dataset:
    def __init__(self, num_samples, num_features, num_classes, class_labels, data_matrix):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.class_labels = np.array(class_labels)
        self.data_matrix = np.array(data_matrix)


def read_dataset(file_path):
    """ Reads dataset from a file and returns a Dataset object """
    try:
        with open(file_path, "r") as file:
            num_samples, num_features, num_classes = map(int, file.readline().split())
            data = []
            class_labels = []
            
            for line in file:
                values = list(map(float, line.split()))
                data.append(values[:-1])
                class_labels.append(int(values[-1]))

        return Dataset(num_samples, num_features, num_classes, class_labels, np.array(data))

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.", file=sys.stderr)
        sys.exit(1)


def zero_mean(data_matrix):
    """ Center the data matrix by subtracting the mean """
    mean_values = np.mean(data_matrix, axis=1, keepdims=True)
    return data_matrix - mean_values


def covariance_matrix(data_matrix):
    """ Computes within-set covariance matrix """
    return np.dot(data_matrix, data_matrix.T) / data_matrix.shape[1]


def between_set_covariance(data_matrix1, data_matrix2):
    """ Computes between-set covariance matrix """
    return np.dot(data_matrix1, data_matrix2.T) / data_matrix1.shape[1]


def eigenvalue_eigenvector(matrix):
    """ Computes eigenvalues and eigenvectors """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return eigvals[::-1], eigvecs[:, ::-1]  


def preprocess_data(file_name):
    dataset = read_dataset(file_name)
    transposed_matrix = dataset.data_matrix.T
    zero_mean_matrix = zero_mean(transposed_matrix)
    return dataset, zero_mean_matrix


def redmica(file_paths, num_modalities, num_features, lambda_min, lambda_max, delta):
    """ Implements the ReDMiCA Algorithm """

    # Step 1: Read multiple modalities
    datasets = [read_dataset(file) for file in file_paths]
    
    # Ensure all datasets have the same number of samples
    num_samples = datasets[0].num_samples
    for dataset in datasets:
        if dataset.num_samples != num_samples:
            print("Error: Mismatch in the number of samples across modalities.", file=sys.stderr)
            sys.exit(1)
    
    zero_mean_matrices = [zero_mean(dataset.data_matrix.T) for dataset in datasets]

    # Step 2: Compute Covariance Matrices
    within_covariances = [covariance_matrix(zm) for zm in zero_mean_matrices]
    between_covariances = [[between_set_covariance(zero_mean_matrices[i], zero_mean_matrices[j])
                            for j in range(num_modalities)] for i in range(num_modalities)]

    # Step 3: Compute Eigenvalues and Eigenvectors
    eigenvalues = []
    eigenvectors = []
    for i in range(num_modalities):
        vals, vecs = eigenvalue_eigenvector(within_covariances[i])
        eigenvalues.append(vals)
        eigenvectors.append(vecs)

    # Step 4: Regularization
    lambda_values = np.arange(lambda_min, lambda_max, delta)
    best_lambda = lambda_min
    best_objective = -np.inf

    for lmbd in lambda_values:
        # Regularize within covariance matrices
        regularized_covariances = [cov + lmbd * np.identity(cov.shape[0]) for cov in within_covariances]

        # Compute final eigen decomposition
        eigvals, eigvecs = eigenvalue_eigenvector(regularized_covariances[0])  # Assuming first modality
        objective_value = np.sum(eigvals[:num_features])  # Objective function

        if objective_value > best_objective:
            best_lambda = lmbd
            best_objective = objective_value

    print(f"Best lambda: {best_lambda} with objective value: {best_objective}")

    # Step 5: Extract Features
    basis_vectors = [vec[:, :num_features] for vec in eigenvectors]
    canonical_variables = [np.dot(vec.T, zero_mean_matrices[i]) for i, vec in enumerate(basis_vectors)]

    return canonical_variables


def main():
    parser = argparse.ArgumentParser(description="ReDMiCA Algorithm in Python")
    parser.add_argument("-s", "--sources", nargs='+', required=True, help="Input modality files")
    parser.add_argument("-M", "--modalities", type=int, default=2, help="Number of modalities")
    parser.add_argument("-f", "--features", type=int, default=10, help="Number of new features")
    parser.add_argument("-m", "--lambda_min", type=float, default=0.0, help="Minimum lambda value")
    parser.add_argument("-n", "--lambda_max", type=float, default=1.0, help="Maximum lambda value")
    parser.add_argument("-d", "--delta", type=float, default=0.1, help="Lambda step size")
    
    args = parser.parse_args()

    if len(args.sources) != args.modalities:
        print(f"Error: Expected {args.modalities} modalities, but received {len(args.sources)} files.", file=sys.stderr)
        sys.exit(1)

    canonical_vars = redmica(
        args.sources, args.modalities, args.features, args.lambda_min, args.lambda_max, args.delta
    )

    for i, vars in enumerate(canonical_vars):
        print(f"Canonical Variables for Modality {i+1}:")
        print(vars)


if __name__ == "__main__":
    main()

