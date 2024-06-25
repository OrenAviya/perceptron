
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('two_circle.txt')
X = data[:, :-1]  
y = data[:, -1]

#
#  data = np.array([
#     [0.22, -0.98, -1],
#     [1.00, 0.09, -1],
#     [-1.00, 0.10, 1],
#     [0.94, -0.34, -1],
#     [-0.90, 0.43, 1],
#     [0.89, -0.46, -1],
#     [-0.35, -0.94, -1],
#     [-0.56, 0.83, 1],
#     [0.23, 0.97, 1],
#     [0.83, -0.55, -1],
#     [0.58, 0.81, 1],
#     [-0.53, -0.85, -1],
#     [-0.62, 0.79, 1],
#     [-0.85, 0.52, 1],
#     [0.93, 0.36, -1],
#     [-0.97, 0.25, 1],
#     [0.19, -0.98, -1],
#     [-0.91, 0.41, 1],
#     [-0.68, 0.73, 1],
#     [0.98, -0.18, -1],
#     [0.81, 0.59, -1],
#     [-0.99, 0.10, 1],
#     [-0.42, 0.91, 1],
#     [0.54, -0.84, -1],
#     [-0.70, 0.72, 1],
#     [-1.00, 0.01, 1],
#     [0.03, -1.00, -1],
#     [0.31, -0.95, -1],
#     [1.00, 0.04, -1],
#     [-0.99, 0.16, 1],
#     [0.99, -0.17, -1],
#     [-0.64, 0.77, 1],
#     [0.11, 0.99, 1],
#     [0.98, -0.17, -1],
#     [-0.15, -0.99, -1],
#     [-0.99, 0.17, 1],
#     [0.52, -0.85, -1],
#     [-0.40, 0.91, 1],
#     [0.98, -0.17, -1],
#     [0.90, 0.44, -1],
#     [-0.95, 0.31, 1],
#     [-0.48, -0.88, -1],
#     [-0.91, 0.41, 1],
#     [-0.32, 0.95, 1],
#     [-0.99, -0.14, 1],
#     [1.00, -0.05, -1],
#     [-1.00, 0.08, 1],
#     [-0.06, 1.00, 1],
#     [-0.16, 0.99, 1],
#     [0.12, 0.99, 1],
#     [0.97, 0.24, -1],
#     [0.91, 0.41, -1],
#     [-0.57, 0.82, 1],
#     [0.93, 0.36, -1],
#     [0.90, -0.43, -1],
#     [0.75, 0.66, -1],
#     [-0.99, -0.14, 1],
#     [0.37, -0.93, -1]
# ,[-0.87,-0.49,1],[0.77,0.64,-1],[0.3,-0.95,-1],[-0.97,-0.25,1],
#     [-0.88,0.48,1],[0.56,0.83,1],[-0.5,-0.86,-1],[0.31,0.95,1],[-0.86,-0.51,1],
#     [-0.93,0.36,1],[0.01,1.0,1],[1.0,0.05,-1],[-0.02,1.0,1],[-0.92,0.39,1],[-0.8,0.6,1],
#     [-0.44,-0.9,-1],[0.96,0.26,-1],[-0.95,0.32,1],[-0.87,-0.5,1],[-0.98,-0.21,1],
#     [-0.86,0.51,1],[0.81,-0.59,-1],[-0.47,0.88,1],[0.73,0.68,-1],[0.05,1.0,1],[1.0,0.1,-1],[0.26,0.96,1],
#     [-0.42,-0.91,-1],[-0.44,0.9,1],[-0.66,0.75,1],[-0.76,-0.65,1],[0.09,-1.0,-1]])

# X = data[:, :2]
# y = data[:, 2]


# Perceptron algorithm according to the class
def perceptron(X, y, max_iter=1000):
    w = np.zeros(X.shape[1])
    b = 0
    mistakes = 0

    for _ in range(max_iter):
        made_mistake = False
        for i in range(X.shape[0]):
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w += y[i] * X[i]
                b += y[i]
                mistakes += 1
                made_mistake = True
                break  # Exit the current round on the first mistake
        if not made_mistake:
            break
    return w, b, mistakes

# Calculate margin
def calculate_margin(w, b, X, y):
    distances = y * (np.dot(X, w) + b) / np.linalg.norm(w)
    return np.min(distances)

# Brute force margin optimization
def brute_force_margin(X, y, n_samples=1000):
    best_margin = 0
    best_w = None
    best_b = None
    
    for _ in range(n_samples):
        w = np.random.randn(X.shape[1])
        b = np.random.randn()
        margin = calculate_margin(w, b, X, y)
        if margin > best_margin:
            best_margin = margin
            best_w = w
            best_b = b
            
    return best_w, best_b, best_margin

# Run the perceptron algorithm
w_perceptron, b_perceptron, mistakes = perceptron(X, y)
margin_perceptron = calculate_margin(w_perceptron, b_perceptron, X, y)

# Run the brute force optimization
w_optimal, b_optimal, margin_optimal = brute_force_margin(X, y)

print(f"Perceptron Direction Vector: {w_perceptron}")
print(f"Number of Mistakes: {mistakes}")
print(f"Margin achieved by Perceptron: {margin_perceptron}")
print(f"Optimal Margin: {margin_optimal}")

# Plotting the dataset and decision boundaries
plt.figure(figsize=(10, 5))
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

# Plot perceptron decision boundary
x_vals = np.linspace(-2, 2, 100)
y_vals = -(w_perceptron[0] * x_vals + b_perceptron) / w_perceptron[1]
plt.plot(x_vals, y_vals, label='Perceptron Boundary', color='green')

# Plot optimal decision boundary
if w_optimal is not None and b_optimal is not None:
    y_vals_opt = -(w_optimal[0] * x_vals + b_optimal) / w_optimal[1]
    plt.plot(x_vals, y_vals_opt, label='Optimal Boundary', color='purple', linestyle='--')

plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron and Optimal Decision Boundaries')
plt.grid(True)
plt.show()
