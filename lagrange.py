import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

# Parameters
a = 0.0  # Start of the interval
b = 2.0 * np.pi  # End of the interval
n = 100  # Number of data points
std_deviation = 0.0  # Standard deviation of Gaussian noise for epsilon (experiment with different values)
epsilon = np.random.normal(0, std_deviation, n)  # Gaussian noise

# Generate the training set with or without noise
x_train = np.linspace(a, b, n)
y_train = np.sin(x_train + epsilon) if std_deviation > 0 else np.sin(x_train)

# Build the Lagrange interpolation model
lagrange_poly = lagrange(x_train, y_train)

# Generate the test set
x_test = np.linspace(a, b, n)
y_test = np.sin(x_test)

# Calculate train and test errors
train_errors = lagrange_poly(x_train) - y_train
test_errors = lagrange_poly(x_test) - y_test

# Calculate log mean squared errors
train_log_mse = np.log(np.mean(train_errors ** 2))
test_log_mse = np.log(np.mean(test_errors ** 2))

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.scatter(x_train, y_train, label='Training Data', marker='o', color='blue')
plt.plot(x_test, y_test, label='True Function', color='red')
plt.title('Training and Test Data')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_train, lagrange_poly(x_train), label='Lagrange Interpolation', color='green')
plt.title('Lagrange Interpolation Model')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x_train, train_errors, label='Train Errors', color='blue')
plt.title('Train Errors')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x_test, test_errors, label='Test Errors', color='red')
plt.title('Test Errors')
plt.legend()

plt.tight_layout()
plt.show()

# Print train and test log mean squared errors
print(f"Train Log MSE: {train_log_mse}")
print(f"Test Log MSE: {test_log_mse}")
