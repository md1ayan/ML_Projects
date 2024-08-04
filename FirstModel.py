import numpy as np
import matplotlib.pyplot as plt

# Generate random data
x = np.sort(np.random.randint(200, 900, 20))
y = np.sort(np.random.randint(1200, 4000, 20))

train_x = x[:16]
train_y = y[:16]

# Normalize the data
train_x_normalized = (train_x - np.mean(train_x)) / np.std(train_x)

# Initial guess for parameters
w = 1
b = 0

# Learning rate
alpha = 0.01

# Number of iterations
iterations = 1000

# Gradient Descent
for _ in range(iterations):
    predictions = w * train_x_normalized + b
    error = predictions - train_y
    w = w - alpha * (2/len(train_x)) * np.dot(error, train_x_normalized)
    b = b - alpha * (2/len(train_x)) * np.sum(error)

# Plotting the results
plt.plot(train_x, w * train_x_normalized + b, c='b', label='Our Prediction')
plt.scatter(train_x, train_y, marker='x', c='r', label='Actual Value')
plt.title("House cost prediction")
plt.xlabel("sqft")
plt.ylabel("cost")
plt.legend()
plt.show()

print("Final w:", w)
print("Final b:", b)