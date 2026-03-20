# ================================
# PERCEPTRON MODEL (FROM SCRATCH)
# ================================

import numpy as np

# -------------------------------
# Step 1: Activation Function
# -------------------------------
def step_function(x):
    return 1 if x >= 0 else 0


# -------------------------------
# Step 2: Perceptron Class
# -------------------------------
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        print("Training Started...\n")

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}")
            for i in range(len(X)):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = step_function(linear_output)

                error = y[i] - y_pred

                # Update weights and bias
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

                print(f"Input: {X[i]}, Predicted: {y_pred}, Actual: {y[i]}, Error: {error}")
            print("-" * 40)

        print("Training Completed!\n")


    def predict(self, X):
        predictions = []
        for x in X:
            linear_output = np.dot(x, self.weights) + self.bias
            predictions.append(step_function(linear_output))
        return np.array(predictions)


# -------------------------------
# Step 3: Dataset (AND Gate)
# -------------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])


# -------------------------------
# Step 4: Train Model
# -------------------------------
model = Perceptron(learning_rate=0.1, epochs=10)
model.fit(X, y)


# -------------------------------
# Step 5: Predictions
# -------------------------------
predictions = model.predict(X)

print("Final Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {predictions[i]}, Actual: {y[i]}")


# -------------------------------
# Step 6: Accuracy
# -------------------------------
accuracy = np.mean(predictions == y)
print(f"\nAccuracy: {accuracy * 100:.2f}%")