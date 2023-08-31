import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv

# Set a random seed for reproducibility
np.random.seed(42)

# Read the dataset
df = pd.read_csv('housing.csv', header=None, delimiter=' ')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Define the test size and random state
test_size = 0.2
random_state = 69

# Calculate the number of test samples
num_test_samples = int(test_size * len(X))

# Shuffle the indices of the data array
indices = np.arange(len(X))
np.random.shuffle(indices)

# Split the data into training and test sets using shuffled indices
X_train = X[indices[num_test_samples:]]
y_train = y[indices[num_test_samples:]]
X_test = X[indices[:num_test_samples]]
y_test = y[indices[:num_test_samples]]

# Define the normal equation function
def normal(X, y):
    X_transpose = np.transpose(X)
    XTX_inv = np.linalg.inv(X_transpose.dot(X))
    theta = XTX_inv.dot(X_transpose).dot(y)
    return theta

# Calculate the theta values for training set using normal equations
theta_train = normal(X_train, y_train)

# Make predictions on training set
y_train_pred = X_train.dot(theta_train)

# Calculate R-squared score for training set
def r2_score(Y, Y_pred):
    numerator = np.sum((Y - Y_pred)**2)
    denominator = np.sum((Y - np.mean(Y))**2)
    score = 1 - numerator / denominator
    return score * 100

r2_train = r2_score(y_train, y_train_pred)
print("R-squared score for training set:", r2_train)

# Make predictions on test set using the theta from training set
y_test_pred = X_test.dot(theta_train)

# Calculate R-squared score for test set
r2_test = r2_score(y_test, y_test_pred)
print("R-squared score for test set:", r2_test)
