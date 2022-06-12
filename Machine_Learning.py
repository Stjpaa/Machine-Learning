import numpy as np
import mnist as data

# Predict the loss via matrix multiplication
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

# Average the squared loss
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)

# Gradient function to calculate the label
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Classify the result to either 0 or 1 (false or true)
def classify(X, w):
    return np.round(forward(X, w))

# Train the weight // lr is the amount w changes with each iteration
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))


x1, x2, x3, y = np.loadtxt("pizza.txt", skiprows = 1, unpack = True)
X = np.column_stack(((np.ones(x1.size)), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations = 10000, lr = 0.001)




