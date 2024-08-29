###########################################

#Robust PCA: Handling data with outliers.

##########################################

import numpy as np

class RobustPCA:
    def __init__(self, max_iter=1000, tol=1e-7, mu=None, lambda_=None):
        self.max_iter = max_iter  # Maximum number of iterations for the algorithm
        self.tol = tol  # Convergence tolerance
        self.mu = mu  # Step size parameter, if None, it will be set automatically
        self.lambda_ = lambda_  # Weight of the sparse component, if None, it will be set automatically

    def fit(self, X):
        """
        Fit the Robust PCA model to the dataset X by decomposing it into a low-rank matrix and a sparse matrix.

        Parameters:
        X (numpy.ndarray): The data matrix to decompose, with shape (n_samples, n_features).

        Returns:
        L (numpy.ndarray): The low-rank component of the matrix X.
        S (numpy.ndarray): The sparse component of the matrix X.
        """
        m, n = X.shape
        
        # Initialize parameters
        if self.mu is None:
            self.mu = np.prod(X.shape) / (4 * np.sum(np.abs(X)))

        if self.lambda_ is None:
            self.lambda_ = 1 / np.sqrt(max(m, n))

        # Initialize the low-rank (L) and sparse (S) matrices
        L = np.zeros_like(X)
        S = np.zeros_like(X)
        Y = np.zeros_like(X)  # Lagrange multiplier

        for iteration in range(self.max_iter):
            # Singular Value Thresholding (SVT) for low-rank approximation
            U, sigma, Vt = np.linalg.svd(X - S + (1 / self.mu) * Y, full_matrices=False)
            sigma_thresholded = np.maximum(sigma - 1 / self.mu, 0)
            L = np.dot(U, np.dot(np.diag(sigma_thresholded), Vt))

            # Soft-thresholding for the sparse component
            S = np.sign(X - L + (1 / self.mu) * Y) * np.maximum(np.abs(X - L + (1 / self.mu) * Y) - self.lambda_ / self.mu, 0)

            # Update Lagrange multiplier
            Y += self.mu * (X - L - S)

            # Check convergence
            error = np.linalg.norm(X - L - S, 'fro') / np.linalg.norm(X, 'fro')
            if error < self.tol:
                break

        self.L_ = L
        self.S_ = S

    def transform(self, X):
        """
        Transform the dataset X by separating it into low-rank and sparse components using the fitted RPCA model.

        Parameters:
        X (numpy.ndarray): The data matrix to transform, with shape (n_samples, n_features).

        Returns:
        L (numpy.ndarray): The low-rank component of the matrix X.
        S (numpy.ndarray): The sparse component of the matrix X.
        """
        self.fit(X)
        return self.L_, self.S_

    def inverse_transform(self, L, S):
        """
        Reconstruct the original matrix from its low-rank and sparse components.

        Parameters:
        L (numpy.ndarray): The low-rank component of the matrix.
        S (numpy.ndarray): The sparse component of the matrix.

        Returns:
        X (numpy.ndarray): The reconstructed original matrix.
        """
        return L + S





#############################

# Example usage of RobustPCA


#############################
#X = np.random.rand(100, 50)  # Example dataset with 100 samples and 50 features
#X[::10] += np.random.rand(10, 50) * 5  # Add some outliers to the dataset

# Initialize RobustPCA with default parameters
#rpca = RobustPCA(max_iter=1000, tol=1e-7)

# Decompose the data into low-rank and sparse components
#L, S = rpca.transform(X)

# Reconstruct the original data from the low-rank and sparse components
#X_reconstructed = rpca.inverse_transform(L, S)

# L is the low-rank approximation, and S contains the sparse outliers.

