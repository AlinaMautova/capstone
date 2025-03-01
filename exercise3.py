import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("Question2_Dataset[1].csv")

# Select input features and output
df_features = df[['X1', 'X2', 'X1^2', 'X1^3', 'X2^2', 'X2^3', 'X1*X2', 'X1^2*X2']]
Y = df['Y'].values.reshape(-1, 1)

# Normalize features (Z-score normalization)
X_norm = (df_features - df_features.mean()) / df_features.std()

# Add bias term (column of ones)
X = np.c_[np.ones(X_norm.shape[0]), X_norm]

# Initialize theta to zeros
theta = np.zeros((X.shape[1], 1))

# Gradient Descent Function
def gradient_descent(X, Y, theta, alpha, iterations):
    m = len(Y)
    J_history = []
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - Y
        gradient = (1/m) * X.T.dot(errors)
        theta -= alpha * gradient
        cost = (1/(2*m)) * np.sum(errors**2)
        J_history.append(cost)
    return theta, J_history

# Run gradient descent for different iteration values
iterations_list = [10, 100, 1000]
alpha = 0.1
results = {}

for num_iter in iterations_list:
    theta_opt, J_hist = gradient_descent(X, Y, np.zeros((X.shape[1], 1)), alpha, num_iter)
    results[num_iter] = {
        "cost_function": int(round(J_hist[-1])),
        "max_theta": int(round(np.max(np.abs(theta_opt))))
    }

# Print results
print("#Iterations\tCost Function\tOptimal Theta")
for num_iter, values in results.items():
    print(f"n={num_iter}\t{values['cost_function']}\t{values['max_theta']}")
