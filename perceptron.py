import numpy as np
from scipy.integrate import quad

# Define the activation function (e.g., step function for perceptron)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Define the perceptron class
class LebesguePerceptron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    # Forward pass using the step function
    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = step_function(z)
        return y_pred

    # Lebesgue integral-based loss function for the perceptron
    def loss_function(self, X, y_true):
        def integrand(x):
            z = np.dot(x, self.weights) + self.bias
            return 0.5 * np.linalg.norm(step_function(z) - y_true) ** 2  # Loss function for perceptron

        # Integrate the loss function using quad from scipy
        integral_loss, _ = quad(integrand, -np.inf, np.inf)
        return integral_loss

# Example usage
input_size = 2
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
y_true = np.array([0, 1, 1, 0])  # Target labels for OR gate

# Create the perceptron
perceptron = LebesguePerceptron(input_size)

# Calculate the loss using Lebesgue integral
loss = perceptron.loss_function(X, y_true)
print("Loss using Lebesgue integral:", loss)
