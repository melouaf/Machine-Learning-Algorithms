###############################################################

#Incremental PCA: Real-time data processing and model updating.

###############################################################


import numpy as np

class IncrementalPCA:
    def __init__(self, n_components, batch_size=None):
        self.n_components = n_components  # Number of components to retain
        self.batch_size = batch_size  # Size of each batch for incremental updates
        self.mean_ = None  # Mean of the data
        self.components_ = None  # Principal components
        self.variances_ = None  # Explained variance of each principal component
        self.n_samples_seen_ = 0  # Number of samples processed

    def fit(self, X):
        """
        Fit the Incremental PCA model to the dataset X in batches.

        Parameters:
        X (numpy.ndarray): The data to fit the model to, with shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size = n_samples  # Default to full batch if batch_size is not provided

        # Initialize mean, components, and variances
        self.mean_ = np.zeros(n_features)
        self.components_ = np.zeros((self.n_components, n_features))
        self.variances_ = np.zeros(self.n_components)

        for batch in self._batch_generator(X):
            self._partial_fit(batch)

    def _batch_generator(self, X):
        """
        Generate batches of data from X.

        Parameters:
        X (numpy.ndarray): The data to create batches from, with shape (n_samples, n_features).

        Yields:
        numpy.ndarray: Batches of data with shape (batch_size, n_features).
        """
        n_samples = X.shape[0]
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            yield X[start:end]

    def _partial_fit(self, X_batch):
        """
        Perform a partial fit of the Incremental PCA model using a data batch.

        Parameters:
        X_batch (numpy.ndarray): A batch of data to fit the model to, with shape (batch_size, n_features).
        """
        batch_mean = np.mean(X_batch, axis=0)
        X_batch_centered = X_batch - batch_mean

        # Adjust mean
        total_samples = self.n_samples_seen_ + X_batch.shape[0]
        updated_mean = (self.n_samples_seen_ * self.mean_ + X_batch.shape[0] * batch_mean) / total_samples

        # Update components and variances
        if self.n_samples_seen_ == 0:
            # Initialization
            U, S, Vt = np.linalg.svd(X_batch_centered, full_matrices=False)
            self.components_ = Vt[:self.n_components]
            self.variances_ = S[:self.n_components] ** 2 / (X_batch.shape[0] - 1)
        else:
            # Incremental update
            X_batch_centered -= self.mean_  # Center using previous mean
            covariance_matrix = np.dot(X_batch_centered.T, X_batch_centered) / (X_batch.shape[0] - 1)

            # SVD of the covariance matrix
            U, S, Vt = np.linalg.svd(covariance_matrix, full_matrices=False)
            U = U[:, :self.n_components]
            S = S[:self.n_components]
            Vt = Vt[:self.n_components]

            # Update components
            self.components_ = np.dot(U.T, np.dot(self.components_, U))

            # Update variances
            self.variances_ = (self.n_samples_seen_ * self.variances_ + X_batch.shape[0] * S) / total_samples

        self.mean_ = updated_mean
        self.n_samples_seen_ = total_samples

    def transform(self, X):
        """
        Transform the dataset X using the fitted Incremental PCA model.

        Parameters:
        X (numpy.ndarray): The data to transform, with shape (n_samples, n_features).

        Returns:
        numpy.ndarray: The transformed data with shape (n_samples, n_components).
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_transformed):
        """
        Reverse the Incremental PCA transformation, mapping the data back to the original space.

        Parameters:
        X_transformed (numpy.ndarray): The data in the principal component space, with shape (n_samples, n_components).

        Returns:
        numpy.ndarray: The data mapped back to the original space, with shape (n_samples, n_features).
        """
        return np.dot(X_transformed, self.components_) + self.mean_


