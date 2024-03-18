import numpy as np

# Define the activation function (e.g., sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the trapezoidal rule for integral approximation
def trapezoidal_integral(f, a, b, n=1000):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2
    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h
    return integral

# Define the neural network class
class IntegralNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    # Forward pass using integral approximation
    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        y_pred = sigmoid(z2)
        return y_pred

    # Integral-based loss function using trapezoidal rule
    def loss_function(self, X, y_true):
        y_pred = self.forward(X)
        loss = lambda y: 0.5 * (y - y_true) ** 2  # Loss function for demonstration
        integral_loss = trapezoidal_integral(loss, 0, 1)  # Assuming y_true is in [0, 1]
        return integral_loss

    # Integral-based backpropagation using trapezoidal rule
    def backward(self, X, y_true, learning_rate=0.01):
        y_pred = self.forward(X)

        # Compute gradients using integral approximation
        grad_loss = lambda y: y - y_true  # Gradient of the loss function
        grad_b2 = trapezoidal_integral(grad_loss, 0, 1)  # Gradient of b2
        grad_W2 = trapezoidal_integral(lambda a: np.outer(a, grad_loss(y_pred)), 0, 1)  # Gradient of W2
        grad_b1 = trapezoidal_integral(lambda z: grad_loss(sigmoid(z)) * sigmoid(z) * (1 - sigmoid(z)), -np.inf, np.inf)  # Gradient of b1
        grad_W1 = trapezoidal_integral(lambda x: np.outer(x, grad_loss(sigmoid(np.dot(x, self.W1) + self.b1)) * sigmoid(np.dot(x, self.W1) + self.b1) * (1 - sigmoid(np.dot(x, self.W1) + self.b1)))), -np.inf, np.inf)  # Gradient of W1

        # Update weights and biases
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1

# Example usage
input_size = 2
hidden_size = 3
output_size = 1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
y_true = np.array([[0], [1], [1], [0]])  # Target labels

# Create the neural network
model = IntegralNeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network using integral-based optimization
epochs = 1000
for epoch in range(epochs):
    model.backward(X, y_true)

# Test the trained model
y_pred = model.forward(X)
print("Predicted output:")
print(y_pred)
