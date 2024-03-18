import numpy as np
from scipy.integrate import quad

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the logistic regression class
class LebesgueLogisticRegression:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    # Forward pass using the sigmoid function
    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(z)
        return y_pred

    # Lebesgue integral-based loss function for logistic regression
    def loss_function(self, X, y_true):
        def integrand(x):
            z = np.dot(x, self.weights) + self.bias
            y_pred = sigmoid(z)
            return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)  # Binary cross-entropy loss

        # Integrate the loss function using quad from scipy
        integral_loss, _ = quad(integrand, -np.inf, np.inf)
        return integral_loss

# Example usage
input_size = 2
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
y_true = np.array([0, 1, 1, 0])  # Target labels for XOR gate (non-linear separable)

# Create the logistic regression model
log_reg = LebesgueLogisticRegression(input_size)

# Calculate the loss using Lebesgue integral
loss = log_reg.loss_function(X, y_true)
print("Loss using Lebesgue integral:", loss)
