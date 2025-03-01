import pandas as pd
import numpy as np

df = pd.read_csv("Question3_Final_CP 17.csv")

X_features = df[['X1', 'X2', 'X3']].values
y = df[['Y']].values
m = X_features.shape[0]

means = X_features.mean(axis=0)
stds = X_features.std(axis=0)
X_norm = (X_features - means) / stds

X = np.hstack([np.ones((m, 1)), X_norm])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def computeCostReg(X, y, theta, lambd):
    m = X.shape[0]
    h = sigmoid(X @ theta)
    term1 = -y * np.log(h + 1e-15)
    term2 = -(1 - y) * np.log(1 - h + 1e-15)
    unreg_cost = (1 / m) * np.sum(term1 + term2)

    reg_term = (lambd / (2 * m)) * np.sum(theta[1:] ** 2)

    return unreg_cost + reg_term


def gradientReg(X, y, theta, lambd):
    m = X.shape[0]
    h = sigmoid(X @ theta)
    error = h - y

    grad = (1 / m) * (X.T @ error)

    grad[1:] = grad[1:] + (lambd / m) * theta[1:]

    return grad


def gradientDescentReg(X, y, alpha, lambd, iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []

    for _ in range(iterations):
        cost = computeCostReg(X, y, theta, lambd)
        cost_history.append(cost)

        grad = gradientReg(X, y, theta, lambd)
        theta = theta - alpha * grad

    return theta, cost_history


scenarios = [
    (100, 1.0, 1.0),
    (1000, 1.0, 10.0),
    (10000, 2.0, 5.0)
]

for (iters, alpha, lambd) in scenarios:
    theta_final, cost_hist = gradientDescentReg(X, y, alpha, lambd, iters)
    final_cost = cost_hist[-1]
    max_theta = np.max(np.abs(theta_final))

    final_cost_2dec = round(final_cost, 2)
    max_theta_2dec = round(max_theta, 2)

    print(f"After {iters} iterations, alpha={alpha}, lambda={lambd}:")
    print(f"  Cost function (2 decimals)   = {final_cost_2dec}")
    print(f"  Max |theta| (2 decimals)     = {max_theta_2dec}")
    print("------------------------------------------------------")

iters, alpha, lambd = 10000, 2.0, 5.0
theta_final_10000, cost_hist_10000 = gradientDescentReg(X, y, alpha, lambd, iters)

X_first10 = X[:10, :]
h_first10 = sigmoid(X_first10 @ theta_final_10000)

predictions_first10 = (h_first10 >= 0.5).astype(int)
num_zeros_first10 = np.sum(predictions_first10 == 0)

print(f"Number of '0' predictions in the first 10 rows = {num_zeros_first10}")
