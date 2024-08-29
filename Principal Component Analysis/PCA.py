#########################################################

#PCA: Dimensionality reduction and factor identification.

#########################################################

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components  # Number of principal components to retain
        self.components = None  # Eigenvectors corresponding to the principal components
        self.mean = None  # Mean of the dataset for centering

    def fit(self, X):
        """
        Fit the model to the dataset X by finding the principal components.

        Parameters:
        X (numpy.ndarray): The data to fit the model to, with shape (n_samples, n_features).
        """
        # Mean centering
        self.mean = np.mean(X, axis=0)  # Compute the mean of each feature
        X_centered = X - self.mean  # Center the data by subtracting the mean

        # Compute the covariance matrix of the centered data
        covariance_matrix = np.cov(X_centered, rowvar=False)  # Use rowvar=False to consider columns as features

        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # eigh is optimized for symmetric matrices

        # Sort eigenvectors by descending eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components].T

    def transform(self, X):
        """
        Transform the dataset X to the principal component space.

        Parameters:
        X (numpy.ndarray): The data to transform, with shape (n_samples, n_features).

        Returns:
        numpy.ndarray: The transformed data with shape (n_samples, n_components).
        """
        X_centered = X - self.mean  # Center the data using the mean calculated during fit
        return np.dot(X_centered, self.components.T)  # Project the data onto the principal components

    def reverse_transform(self, X_transformed):
        """
        Reverse the PCA transformation, mapping the data back to the original space.

        Parameters:
        X_transformed (numpy.ndarray): The data in the principal component space, with shape (n_samples, n_components).

        Returns:
        numpy.ndarray: The data mapped back to the original space, with shape (n_samples, n_features).
        """
        return np.dot(X_transformed, self.components) + self.mean  # Reverse the transformation and add the mean

