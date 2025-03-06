import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import joblib


# Define the Perceptron class (Rosenblatt Perceptron)
class Perceptron:
    def __init__(self, input_dim, learning_rate=0.05, epochs=50):
        # Initialize weights and bias with random numbers
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors_ = []  # To track misclassifications per epoch

    def step_function(self, x):
        # Step activation: returns 1 if x >= 0 else 0
        return np.where(x >= 0, 1, 0)

    def forward(self, X):
        # Calculate weighted sum plus bias
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        # Make predictions by applying the step function on the linear output
        linear_output = self.forward(X)
        return self.step_function(linear_output)

    def fit(self, X, y):
        # Train the perceptron using the provided data
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                if error != 0:
                    # Update rule: adjust weights and bias
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += 1
            self.errors_.append(errors)
            print(
                f"Epoch {epoch+1}/{self.epochs}, Misclassifications: {errors}"
            )


def plot_decision_boundary(model, X, y):
    # Define the plot boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # Predict for each point in the mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and data points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap='bwr',
        edgecolor='k',
        alpha=0.7
    )
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def main():
    config_path = os.path.join("config.json")

    # Load configuration parameters from config.json
    with open(config_path, "r") as f:
        config = json.load(f)

    train_ratio = config.get("train_ratio", 0.8)
    random_seed = config.get("random_seed", 42)
    learning_rate = config.get("perceptron", {}).get("learning_rate", 0.05)
    epochs = config.get("perceptron", {}).get("epochs", 50)

    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Load the dataset from CSV
    data_path = os.path.join("data", "synthetic_data.csv")
    df = pd.read_csv(data_path)
    print("Dataset loaded from", data_path)

    # Extract features and target
    X = df[["feature1", "feature2"]].values
    y = df["target"].values

    # Create a train-test split
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_size = int(train_ratio * n_samples)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Initialize and train the perceptron model
    input_dim = X_train.shape[1]
    model = Perceptron(
        input_dim=input_dim,
        learning_rate=learning_rate,
        epochs=epochs
    )
    model.fit(X_train, y_train)

    # Evaluate performance on training data
    train_predictions = model.predict(X_train)
    train_accuracy = np.mean(train_predictions == y_train)
    print(f"Training Accuracy: {train_accuracy:.2f}")

    # Evaluate performance on testing data
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Testing Accuracy: {test_accuracy:.2f}")

    # Display the trained weights and bias
    print("Trained Weights:", model.weights)
    print("Trained Bias:", model.bias)

    # Save metrics to a JSON file
    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        # Convert numpy array to list for JSON compatibility
        "trained_weights": model.weights.tolist(),
        "trained_bias": model.bias,
        "epochs": epochs
    }

    # Define the file path
    file_path = os.path.join("metrics.json")

    # Ensure the directory exists
    os.makedirs(
        os.path.dirname(os.path.abspath(file_path)),
        exist_ok=True
    )

    # Saving the metrics to the file
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to metrics.json")

    # Display the decision boundary using test data
    # plot_decision_boundary(model, X_test, y_test)

    model_dir = os.path.join("model")
    os.makedirs(model_dir, exist_ok=True)
    model_file_path = os.path.join(model_dir, "perceptron_model.pkl")
    joblib.dump(model, model_file_path)
    print("Model saved to", model_file_path)


if __name__ == '__main__':
    main()
