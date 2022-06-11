import numpy as np

# Predict the loss via matrix multiplication
def predict(X, w):
    return np.matmul(X, w)

# Average the squared loss
def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

# Better function to calculate the average loss
def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

# Train the weight // lr is the amount w changes with each iteration
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

# Read array data into two arrays
x1, x2, x3, y = np.loadtxt("pizza.txt", skiprows = 1, unpack = True)

# Create a (4, n) Matrix
X = np.column_stack(((np.ones(x1.size)), x1, x2, x3))
# Shape Y so it takes (n, 1) Elements
Y = y.reshape(-1, 1)

w = train(X, Y, iterations = 10000, lr = 0.001)
print("\nWeights: %s" % w.T)
print("\nA few predictions:")
for i in range(4):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))





