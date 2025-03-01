import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function

def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def sigmoid(z):
    return expit(z)

def compute_cost(X, y, theta, reg_lambda):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    reg_term = (reg_lambda / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg_term

def gradient_descent(X, y, theta, alpha, reg_lambda, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y)) + (reg_lambda/m) * np.vstack(([0], theta[1:]))
        theta -= alpha * gradient
    return theta

# Load dataset
df = pd.read_csv('Question3_Final_CP[1].csv')  # Ensure correct filename
X = df.iloc[:, :-1].values  # Input features
y = df.iloc[:, -1].values.reshape(-1, 1)  # Output feature

# Normalize input features
X = normalize(X)
X = np.c_[np.ones(X.shape[0]), X]  # Add bias term

# Initialize theta
theta_init = np.zeros((X.shape[1], 1))

# Parameters to evaluate
params = [
    (100, 0.1, 0.1),
    (1000, 0.2, 1),
    (10000, 0.3, 10)
]

results = {}
for iterations, alpha, reg_lambda in params:
    theta = gradient_descent(X, y, theta_init.copy(), alpha, reg_lambda, iterations)
    cost = compute_cost(X, y, theta, reg_lambda)
    max_theta = np.max(np.abs(theta))  # Ensure we correctly get max theta
    
    results[iterations] = {
        "cost_function": round(float(cost), 2),
        "max_theta": round(float(max_theta), 2)
    }

# Predicting first 10 rows using final model
final_theta = gradient_descent(X, y, theta_init.copy(), 0.3, 10, 10000)
predictions = sigmoid(X @ final_theta)
binary_predictions = (predictions >= 0.5).astype(int)
num_ones_in_first_10 = int(np.sum(binary_predictions[:10]))  # Ensure integer output

# Display results
print("#Iterations | Cost Function | Max Theta")
for key, value in results.items():
    print(f"n={key} | {value['cost_function']} | {value['max_theta']}")

print(f"Number of ones in first 10 rows of prediction: {num_ones_in_first_10}")
