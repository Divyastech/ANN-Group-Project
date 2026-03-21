# =====================================
# COMPETITIVE LEARNING (FROM SCRATCH)
# =====================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Generate Noisy Data
# -------------------------------
np.random.seed(42)

data = np.vstack((
    np.random.randn(50, 2) + [2, 2],
    np.random.randn(50, 2) + [-2, -2]
))


# -------------------------------
# Step 2: Initialize Neurons
# -------------------------------
num_neurons = 2
weights = np.random.rand(num_neurons, 2)


# -------------------------------
# Step 3: Training
# -------------------------------
learning_rate = 0.1
epochs = 20

for epoch in range(epochs):
    for x in data:
        
        # Find winning neuron (minimum distance)
        distances = np.linalg.norm(weights - x, axis=1)
        winner = np.argmin(distances)

        # Update only winner neuron
        weights[winner] += learning_rate * (x - weights[winner])


# -------------------------------
# Step 4: Plot Results
# -------------------------------
plt.scatter(data[:, 0], data[:, 1], label='Data Points')
plt.scatter(weights[:, 0], weights[:, 1], color='red', label='Neurons')
plt.title("Competitive Learning Result")
plt.legend()
plt.show()