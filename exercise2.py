import pandas as pd
import numpy as np
import math

# 1) Read CSV file
df = pd.read_csv("./datasets/Question2_Final_CP 23.csv")

X_features = df[['X1', 'X2', 'X1^2', 'X1^3', 'X2^2', 'X2^3']].values 
y = df[['Y']].values 

m = X_features.shape[0]

means = X_features.mean(axis=0)
stds  = X_features.std(axis=0)
X_std = (X_features - means) / stds

X = np.hstack([np.ones((m,1)), X_std])

def computeCost(X, y, theta):
    m = X.shape[0]
    errors = X @ theta - y
    return (1/(2*m)) * np.sum(errors**2)

def gradientDescent(X, y, theta, alpha, iterations):
    m = X.shape[0]
    cost_history = []
    for _ in range(iterations):
        cost = computeCost(X, y, theta)
        cost_history.append(cost)
        gradient = (1/m) * (X.T @ (X @ theta - y))
        theta = theta - alpha * gradient
    return theta, cost_history

alpha = 0.01
theta_init = np.zeros((7,1))

theta_10, costHist_10 = gradientDescent(X, y, theta_init, alpha, 10)
theta_100, costHist_100 = gradientDescent(X, y, theta_init, alpha, 100)
theta_1000, costHist_1000 = gradientDescent(X, y, theta_init, alpha, 1000)
theta_10000, costHist_10000 = gradientDescent(X, y, theta_init, alpha, 10000)

finalCost_10 = costHist_10[-1]
finalCost_100 = costHist_100[-1]
finalCost_1000 = costHist_1000[-1]
finalCost_10000 = costHist_10000[-1]

maxTheta_10 = np.max(np.abs(theta_10))
maxTheta_100 = np.max(np.abs(theta_100))
maxTheta_1000 = np.max(np.abs(theta_1000))
maxTheta_10000 = np.max(np.abs(theta_10000))


print("After 10 iterations:")
print("  Cost =", math.ceil(finalCost_10), "  Max|theta| =", math.ceil(maxTheta_10))

print("After 100 iterations:")
print("  Cost =", math.ceil(finalCost_100), "  Max|theta| =", math.ceil(maxTheta_100))

print("After 1000 iterations:")
print("  Cost =", math.ceil(finalCost_1000), "  Max|theta| =", math.ceil(maxTheta_1000))

print("After 10000 iterations:")
print("  Cost =", math.ceil(finalCost_10000), "  Max|theta| =", math.ceil(maxTheta_10000))
