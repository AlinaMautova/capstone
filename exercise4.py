import numpy as np

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0.15, 0.3, 0.45, 0.6], 
              [0.2,  0.4, 0.6,  0.8]]) 

y = np.array([[1], [0]])

input_layer_size   = 4  
hidden_layer1_size = 8  
hidden_layer2_size = 6  
hidden_layer3_size = 4  
hidden_layer4_size = 3  
output_layer_size  = 1  

W1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
               [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
               [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
               [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]])
b1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])

W2 = np.array([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
               [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
               [1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
               [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
               [2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
               [3.2, 3.3, 3.4, 3.5, 3.6, 3.7],
               [3.8, 3.9, 4.0, 4.1, 4.2, 4.3],
               [4.4, 4.5, 4.6, 4.7, 4.8, 4.9]])
b2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

W3 = np.array([[0.2, 0.3, 0.4, 0.5],
               [0.6, 0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2, 1.3],
               [1.4, 1.5, 1.6, 1.7],
               [1.8, 1.9, 2.0, 2.1],
               [2.2, 2.3, 2.4, 2.5]])
b3 = np.array([[0.1, 0.2, 0.3, 0.4]])

W4 = np.array([[0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7],
               [0.8, 0.9, 1.0],
               [1.1, 1.2, 1.3]])
b4 = np.array([[0.1, 0.2, 0.3]])

W5 = np.array([[0.2],
               [0.3],
               [0.4]])
b5 = np.array([[0.1]])

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):

    z1 = X.dot(W1) + b1   
    a1 = leaky_relu(z1)

    z2 = a1.dot(W2) + b2      
    a2 = leaky_relu(z2)

    z3 = a2.dot(W3) + b3 
    a3 = leaky_relu(z3)

    z4 = a3.dot(W4) + b4
    a4 = leaky_relu(z4)

    z5 = a4.dot(W5) + b5
    a5 = sigmoid(z5)

    error = y - a5
    
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch}, Loss: {loss}")

y_pred = a5
print("Final Predictions (a5):\n", y_pred)
