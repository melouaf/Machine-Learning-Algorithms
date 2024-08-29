import numpy as np
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_regression

class QuadraticProgramSolver:
    def __init__(self, num_samples, num_features, regularization=10, tolerance=1e-6, alpha=0.5, beta=0.9, max_iterations=500):
        self.num_samples = num_samples
        self.num_features = num_features
        self.regularization = regularization
        self.tolerance = tolerance
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.X, self.y, self.true_weights, self.Q, self.linear_term, self.A, self.b, self.initial_v = self.generate_parameters()

    def generate_parameters(self):
        """Generate parameters for the quadratic program based on synthetic regression data."""
        X, y, true_weights = make_regression(n_samples=self.num_samples, n_features=self.num_features, coef=True, random_state=42)

        Q = np.eye(self.num_samples) / 2
        linear_term = -y

        A = np.concatenate((X.T, -X.T), axis=0)
        b = self.regularization * np.ones(2 * self.num_features)

        initial_v = np.zeros(self.num_samples)

        return X, y, true_weights, Q, linear_term, A, b, initial_v

    def primal_objective(self, weights):
        """Calculate the primal objective value."""
        residuals = self.X @ weights - self.y
        return 0.5 * np.linalg.norm(residuals, 2) ** 2 + self.regularization * np.linalg.norm(weights, 1)

    def dual_objective(self, dual_variable):
        """Calculate the dual objective value."""
        return dual_variable.T @ self.Q @ dual_variable + self.linear_term.T @ dual_variable

    def barrier_objective(self, dual_variable, barrier_param):
        """Calculate the barrier objective function value."""
        constraints = self.b - self.A @ dual_variable
        if np.any(constraints <= 0):
            return float('inf')
        return barrier_param * (dual_variable.T @ self.Q @ dual_variable + self.linear_term.T @ dual_variable) - np.sum(np.log(constraints))

    def compute_gradient(self, dual_variable, barrier_param):
        """Compute the gradient of the barrier objective function."""
        constraints_reciprocal = 1.0 / (self.b - self.A @ dual_variable)
        return barrier_param * (2 * self.Q @ dual_variable + self.linear_term) + self.A.T @ constraints_reciprocal

    def compute_hessian(self, dual_variable, barrier_param):
        """Compute the Hessian matrix of the barrier objective function."""
        constraints_reciprocal_squared = 1.0 / (self.b - self.A @ dual_variable) ** 2
        return 2 * barrier_param * self.Q + self.A.T @ np.diag(constraints_reciprocal_squared) @ self.A

    def line_search(self, objective_func, gradient_func, dual_variable, search_direction):
        """Perform backtracking line search to find the optimal step size."""
        step_size = 1.0
        objective_value = objective_func(dual_variable)
        gradient_dot_direction = gradient_func(dual_variable).T @ search_direction
        
        while step_size > 1e-5:
            new_dual_variable = dual_variable + step_size * search_direction
            if np.all(self.b - self.A @ new_dual_variable > 0) and objective_func(new_dual_variable) <= objective_value + self.alpha * step_size * gradient_dot_direction:
                break
            step_size *= self.beta
        
        return step_size

    def perform_centering_step(self, barrier_param):
        """Perform the centering step for the interior-point method."""
        dual_variable = self.initial_v
        dual_variable_sequence = [dual_variable]

        for _ in range(self.max_iterations):
            gradient = self.compute_gradient(dual_variable, barrier_param)
            hessian = self.compute_hessian(dual_variable, barrier_param)

            search_direction, _ = cg(hessian, gradient, atol=1e-10)

            step_size = self.line_search(lambda x: self.barrier_objective(x, barrier_param), lambda x: self.compute_gradient(x, barrier_param), dual_variable, search_direction)

            new_dual_variable = dual_variable - step_size * search_direction
            dual_variable_sequence.append(new_dual_variable)

            if gradient.T @ search_direction < 2 * self.tolerance:
                break

            dual_variable = new_dual_variable

        return dual_variable_sequence

    def barrier_method(self, barrier_multiplier):
        """Implement the barrier method for solving the quadratic program."""
        dual_variable_sequence = [self.initial_v]
        barrier_param = 1.0

        while len(self.b) / barrier_param > self.tolerance:
            new_dual_variable_sequence = self.perform_centering_step(barrier_param)
            dual_variable_sequence.append(new_dual_variable_sequence[-1])
            barrier_param *= barrier_multiplier

        return dual_variable_sequence

    def run_simulation_and_plot(self, barrier_multipliers):
        """Run simulations for different barrier multipliers and plot the results."""
        simulation_results = []
        optimal_value = float('inf')

        plt.figure(figsize=(10, 8))
        plt.xlabel('Iteration')
        plt.ylabel(Duality Gap')
        plt.title('Convergence of the Barrier Method with Different µ Values')

        for barrier_multiplier in tqdm(barrier_multipliers, desc="Running simulations"):
            dual_variable_sequence = self.barrier_method(barrier_multiplier)
            simulation_results.append(dual_variable_sequence)
            
            objective_values = [self.dual_objective(v) for v in dual_variable_sequence]
            min_objective_value = min(objective_values)
            if min_objective_value < optimal_value:
                optimal_value = min_objective_value

            plt.semilogy([obj - optimal_value for obj in objective_values], label=f'µ={barrier_multiplier}')

        plt.legend()
        plt.grid(True)
        plt.show()



####################
#Usage
####################

# Initialize the solver with problem parameters
#solver = QuadraticProgramSolver(
#    num_samples=1000,        # Number of samples
#    num_features=10,         # Number of features
#    regularization=10,       # Regularization parameter
#    tolerance=1e-6,          # Tolerance for convergence
#    alpha=0.5,               # Line search parameter
#    beta=0.9,                # Line search parameter
#    max_iterations=500       # Maximum number of iterations
#)

# Define barrier multipliers (µ) to test
#barrier_multipliers = [2, 5, 10, 20]  # Different values of µ to observe their effect on convergence

# Run the simulation and plot the results
#solver.run_simulation_and_plot(barrier_multipliers)  # Run the solver and visualize the results







