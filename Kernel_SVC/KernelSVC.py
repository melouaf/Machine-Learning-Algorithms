import numpy as np
from scipy import optimize

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  # The variance of the kernel

    def kernel(self, X, Y):
        # Calculate the L2 norms (squared) of each row in X and Y
        X_norm_squared = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm_squared = np.sum(Y ** 2, axis=1).reshape(1, -1)

        # Calculate the squared Euclidean distance matrix
        distances_squared = X_norm_squared + Y_norm_squared - 2 * np.dot(X, Y.T)

        # Apply the Gaussian (RBF) kernel formula
        K = np.exp(-distances_squared / (2 * self.sigma ** 2))

        return K


class Linear:
    def kernel(self, X, Y):
        # Direct matrix multiplication between X and Y^T performs the dot product between all pairs
        return np.dot(X, Y.T)


class KernelSVC:
    def __init__(self, C, kernel, epsilon=1e-3):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.b = None
        self.norm_f = None

    def fit(self, X, y):
        N = y.shape[0]
        self.D = np.diag(y)
        ke = self.kernel(X, X)
        self.X = X

        # Precompute Gram matrix G for efficiency
        G = self.D @ ke @ self.D

        # Lagrange dual problem
        def loss(alpha):
            return -alpha.sum() + 0.5 * alpha @ G @ alpha

        # Partial derivative of Ld on alpha
        def grad_loss(alpha):
            return -np.ones_like(alpha) + G @ alpha

        # Constraints on alpha
        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))

        constraints = (
            {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y), 'jac': lambda alpha: y},
            {'type': 'ineq', 'fun': lambda alpha: b - np.dot(A, alpha), 'jac': lambda alpha: -A}
        )

        optRes = optimize.minimize(fun=loss, x0=np.zeros(N), method='SLSQP', jac=grad_loss, constraints=constraints)
        self.alpha = optRes.x

        # Efficiently identify support vectors and calculate b
        support_mask = (self.alpha > self.epsilon) & (self.alpha < self.C - self.epsilon)
        self.support = X[support_mask]
        support_labels = y[support_mask]
        support_alphas = self.alpha[support_mask]
        self.b = np.mean(support_labels - np.sum(ke[support_mask][:, support_mask] * support_alphas * support_labels, axis=1))

        # RKHS norm of the function f
        self.norm_f = self.alpha @ G @ self.alpha

    def separating_function(self, x):
        return np.dot(self.kernel(x, self.X), self.D @ self.alpha)

    def predict(self, X):
        """Predict y values in {-1, 1}"""
        return np.sign(self.separating_function(X) + self.b)


class OutOfSamplePrediction:
    def __init__(self, X, x, alpha, D, b, sigma=1.):
        self.X = X  # Training data
        self.x = x  # New data points for prediction
        self.sigma = sigma
        self.D = D  # Diagonal matrix of labels for training data
        self.alpha = alpha  # Lagrange multipliers
        self.b = b  # Offset

    def kernel(self):
        # Efficient computation of the RBF kernel using broadcasting
        X2 = np.sum(self.X**2, axis=1)
        x2 = np.sum(self.x**2, axis=1)
        cross_term = np.dot(self.x, self.X.T)
        distances = X2 - 2 * cross_term + x2[:, np.newaxis]
        A = -distances / (2 * self.sigma**2)
        return np.exp(A)

    def separating_function(self):
        res = self.kernel()
        gam = np.dot(self.D, self.alpha)
        final = np.dot(res, gam)
        return final

    def predict(self):
        """Predict y values in {-1, 1}"""
        d = self.separating_function()
        return np.sign(d + self.b)

