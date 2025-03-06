import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------
# 1. Data Generation
# ------------------------------

# Total number of samples
n_samples = 500
n_samples_per_class = n_samples // 2

# Generate class 0 data: centered at (0, 1) with a slight spread
X0 = np.random.randn(n_samples_per_class, 2) * 0.5 + np.array([0, 1])
y0 = np.zeros(n_samples_per_class)

# Generate class 1 data: centered at (2, 0) with a slight spread
X1 = np.random.randn(n_samples_per_class, 2) * 0.5 + np.array([2, 0])
y1 = np.ones(n_samples_per_class)

# Combine the data from both classes
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# Create a DataFrame with two feature columns and one target column
df = pd.DataFrame(X, columns=["feature1", "feature2"])
df["target"] = y

# Display the first few rows to verify the data
print("First 5 rows of the dataset:")
print(df.head())

# ------------------------------
# 2. Save Data to CSV
# ------------------------------

# Go one folder back and then create the 'data' directory
parent_dir = os.path.join(os.getcwd(), '..', 'data')  # Goes one level up and then to 'data'
os.makedirs(parent_dir, exist_ok=True)

# Define the path to the CSV file
csv_path = os.path.join(parent_dir, "synthetic_data.csv")
df.to_csv(csv_path, index=False)
print(f"Synthetic data saved to {csv_path}")

# ------------------------------
# 3. (Optional) Visualize the Dataset
# ------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.7)
plt.title("Synthetic Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
