import numpy as np

# -------------------------
# Helpers
# -------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def mse_loss(y_hat, y):
    e = y_hat - y
    return 0.5 * np.mean(e * e)

# -------------------------
# XOR dataset
# -------------------------
X = np.array([[0.,0.],
              [0.,1.],
              [1.,0.],
              [1.,1.]], dtype=float)   # (N=4,d=2)
y = np.array([0., 1., 1., 0.], dtype=float)  # (N,)

# -------------------------
# Network init (2 -> 2 -> 1)
# -------------------------
np.random.seed(0)
d, h, k = 2, 2, 1
W1 = np.random.randn(h, d) * 0.5   # (h,d)
b1 = np.zeros((h,))                # (h,)
W2 = np.random.randn(k, h) * 0.5   # (k,h)
b2 = np.zeros((k,))                # (k,)

print("Architecture: 2 -> 2 -> 1 (sigmoid hidden, sigmoid output, MSE loss)")
print("Initial params:")
print("W1\n", W1); print("b1\n", b1)
print("W2\n", W2); print("b2\n", b2)
print()

# -------------------------
# Forward and backward (MSE)
# -------------------------
def forward(X, W1, b1, W2, b2):
    z1 = X.dot(W1.T) + b1        # (N,h)
    h1 = sigmoid(z1)             # (N,h)  <-- phi(x;theta)
    z2 = h1.dot(W2.T) + b2       # (N,1)
    y_hat = sigmoid(z2).reshape(-1)  # (N,)
    return y_hat, {'z1': z1, 'h1': h1, 'z2': z2, 'y_hat': y_hat}

def backward_mse(X, y, cache, W2):
    N = X.shape[0]
    z1 = cache['z1']    # (N,h)
    h1 = cache['h1']    # (N,h)
    z2 = cache['z2']    # (N,1)
    y_hat = cache['y_hat']  # (N,)

    # dL/dy_hat = (y_hat - y) / N
    dL_dyhat = (y_hat - y) / N                # (N,)
    dL_dz2 = dL_dyhat.reshape(-1,1) * sigmoid_prime(z2)  # (N,1)

    dW2 = dL_dz2.T.dot(h1)      # (1,h)
    db2 = np.sum(dL_dz2, axis=0) # (1,)

    delta1 = (dL_dz2.dot(W2)).reshape(N, h) * sigmoid_prime(z1)  # (N,h)
    dW1 = delta1.T.dot(X)       # (h,d)
    db1 = np.sum(delta1, axis=0) # (h,)

    return dW1, db1, dW2, db2

# -------------------------
# Training loop (verbose)
# -------------------------
lr = 0.5
epochs = 200
for epoch in range(1, epochs + 1):
    y_hat, cache = forward(X, W1, b1, W2, b2)
    loss = mse_loss(y_hat, y)

    # Print key diagnostics each epoch
    print(f"Epoch {epoch} | Loss {loss:.6f}")
    print(" z1 (pre-activations):\n", cache['z1'])
    print(" phi(x;theta) = h1 (hidden activations):\n", cache['h1'])
    print(" z2 (output pre):\n", cache['z2'].reshape(-1))
    print(" y_hat (output):\n", y_hat)

    dW1, db1, dW2, db2 = backward_mse(X, y, cache, W2)
    print(" Gradients dW2, db2:\n", dW2, db2)
    print(" Gradients dW1, db1:\n", dW1, db1)

    # Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    print(" Updated W1, b1, W2, b2\n")
    # optional separator
    # print("-" * 60)

# -------------------------
# Final evaluation
# -------------------------
print("Training complete.\nFinal forward pass:")
y_hat_final, cache_final = forward(X, W1, b1, W2, b2)
print("Final phi(x;theta):\n", cache_final['h1'])
print("Final outputs y_hat:\n", y_hat_final)
print("Binary preds (threshold 0.5):\n", (y_hat_final >= 0.5).astype(int))
print("Targets:\n", y)
print("Final loss:", mse_loss(y_hat_final, y))
