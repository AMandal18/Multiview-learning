import numpy as np
import os
import sys
import argparse
from scipy.linalg import eig
from pathlib import Path

# Data Structures
class Dataset:
    def __init__(self, samples, features, n_classes, labels, data):
        self.n_samples = samples
        self.n_features = features
        self.n_classes = n_classes
        self.labels = labels
        self.data = data  

class Modality:
    def __init__(self, data):
        self.data = data  

class FeatureSet:
    def __init__(self, n_features, n_modalities, n_samples, n_classes):
        self.basis_vectors = [np.zeros((0,0))]*n_modalities
        self.canonical_vars = np.zeros((n_samples, n_features))
        self.relevance = np.zeros(n_features)
        self.rhepm = [np.zeros((n_classes, n_samples), dtype=int) for _ in range(n_features)]

# Core Functions
def read_dataset(filepath):
    with open(filepath, 'r') as f:
        n_samples, n_features, n_classes = map(int, f.readline().split())
        data = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            row = list(map(float, f.readline().split()))
            data[i] = row[:-1]
            labels[i] = int(row[-1])
    return Dataset(n_samples, n_features, n_classes, labels, data)

def zero_mean(modality):
    mean = np.mean(modality.data, axis=1, keepdims=True)
    return Modality(modality.data - mean)

def compute_covariance(modality1, modality2, n_samples):
    return np.dot(modality1.data, modality2.data.T) / n_samples

def eigenvalue_decomposition(matrix):
    eigenvalues, eigenvectors = eig(matrix)
    idx = eigenvalues.argsort()[::-1]
    return np.real(eigenvalues[idx]), np.real(eigenvectors[:, idx])

def generate_rhepm(feature_values, labels, epsilon=1e-6):
    unique_labels = np.unique(labels)
    rhepm = np.zeros((len(unique_labels), len(labels)), dtype=int)
    for i, label in enumerate(unique_labels):
        class_mask = (labels == label)
        min_val = np.min(feature_values[class_mask]) - epsilon
        max_val = np.max(feature_values[class_mask]) + epsilon
        rhepm[i] = (feature_values >= min_val) & (feature_values <= max_val)
    return rhepm.astype(int)

def dependency_degree(rhepm):
    conflicts = np.sum(rhepm, axis=0) > 1
    return 1.0 - np.sum(conflicts) / rhepm.shape[1]

# Main Algorithm
class SeFGeIM:
    def __init__(self, n_new_features=10, lambda_range=(0.0, 1.0), delta=0.1, omega=0.5):
        self.n_new_features = n_new_features
        self.lambda_min, self.lambda_max = lambda_range
        self.delta = delta
        self.omega = omega
        
    def fit(self, datasets, output_dir):
        self.n_modalities = len(datasets)
        self.n_samples = datasets[0].n_samples
        self.n_classes = datasets[0].n_classes
        self.output_dir = Path(output_dir)
        
        # Preprocess data
        self.zero_mean_modalities = [
            zero_mean(Modality(d.data.T)) for d in datasets  
        ]
        self._check_consistency(datasets)
        self._compute_covariances()
        self._run_feature_extraction()
        
    def _check_consistency(self, datasets):
        # Check sample and label consistency (omitted for brevity)
        pass
    
    def _compute_covariances(self):
        self.covariances = []
        for i in range(self.n_modalities):
            row = []
            for j in range(self.n_modalities):
                if i == j:
                    cov = compute_covariance(self.zero_mean_modalities[i], 
                                           self.zero_mean_modalities[j],
                                           self.n_samples)
                else:
                    cov = compute_covariance(self.zero_mean_modalities[i],
                                           self.zero_mean_modalities[j],
                                           self.n_samples)
                row.append(cov)
            self.covariances.append(row)
    
    def _run_feature_extraction(self):
        lambda_vals = np.arange(self.lambda_min, self.lambda_max + self.delta, self.delta)
        self.feature_set = FeatureSet(self.n_new_features, self.n_modalities,
                                    self.n_samples, self.n_classes)
        
        for feat_idx in range(self.n_new_features):
            best_objective = -np.inf
            best_lambda = None
            
            for lambda_comb in itertools.product(lambda_vals, repeat=self.n_modalities):
                # Compute regularized covariance matrices
                reg_covariances = []
                for i in range(self.n_modalities):
                    reg_cov = self.covariances[i][i] + lambda_comb[i] * np.eye(
                        self.covariances[i][i].shape[0])
                    reg_covariances.append(reg_cov)
                
                # Eigen decomposition and projection (simplified)
                projections = []
                for i in range(self.n_modalities):
                    eigvals, eigvecs = eigenvalue_decomposition(reg_covariances[i])
                    projection = eigvecs[:, :self.n_new_features].T @ self.zero_mean_modalities[i].data
                    projections.append(projection)
                
                # Combine projections and calculate relevance
                combined_proj = np.mean(projections, axis=0)
                rhepm = generate_rhepm(combined_proj[feat_idx], datasets[0].labels)
                gamma = dependency_degree(rhepm)
                
                # Update best solution
                objective = self.omega * gamma  
                if objective > best_objective:
                    best_objective = objective
                    best_lambda = lambda_comb
                    self.feature_set.relevance[feat_idx] = gamma
                    self.feature_set.rhepm[feat_idx] = rhepm
            
            # Save basis vectors and canonical variables
            self._save_results(feat_idx, best_lambda)
    
    def _save_results(self, feat_idx, lambda_comb):
        # Save basis vectors and canonical variables to files
        output_dir = self.output_dir / "SeFGeIM"
        output_dir.mkdir(exist_ok=True)
        
        # Save basis vectors (example for first modality)
        np.savetxt(output_dir / f"basis_vector_{feat_idx}.txt",
                 self.feature_set.basis_vectors[0], fmt="%.8f")
        
        # Save canonical variables
        np.savetxt(output_dir / "canonical_vars.txt",
                 self.feature_set.canonical_vars, fmt="%.8f")

# Command Line Interface
def main():
    parser = argparse.ArgumentParser(description="SeFGeIM Algorithm")
    parser.add_argument('-s', '--modality', required=True, help="Modality file list")
    parser.add_argument('-M', '--modalities', type=int, default=2, help="Number of modalities")
    parser.add_argument('-f', '--features', type=int, default=10, help="Number of new features")
    parser.add_argument('-p', '--path', default=".", help="Input/output path")
    parser.add_argument('-m', '--lambda_min', type=float, default=0.0, help="Minimum lambda")
    parser.add_argument('-n', '--lambda_max', type=float, default=1.0, help="Maximum lambda")
    parser.add_argument('-d', '--delta', type=float, default=0.1, help="Lambda increment")
    parser.add_argument('-o', '--omega', type=float, default=0.5, help="Weight parameter")
    
    args = parser.parse_args()
    
    # Load datasets
    datasets = []
    with open(args.modality) as f:
        for line in f:
            fname = Path(args.path) / line.strip()
            datasets.append(read_dataset(fname))
    
    # Run algorithm
    sefgeim = SeFGeIM(n_new_features=args.features,
                    lambda_range=(args.lambda_min, args.lambda_max),
                    delta=args.delta,
                    omega=args.omega)
    
    start_time = time.time()
    sefgeim.fit(datasets, args.path)
    elapsed = time.time() - start_time
    
    print(f"\nTOTAL TIME REQUIRED: {elapsed:.2f} seconds")
    print("Results saved to:", Path(args.path) / "SeFGeIM")

if __name__ == "__main__":
    main()
