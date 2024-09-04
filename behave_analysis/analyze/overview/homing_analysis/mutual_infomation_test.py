import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score

# Generate random data
np.random.seed(0)
X_independent = np.random.rand(1000)
Y_independent = np.random.rand(1000)

# Compute mutual information
mi = mutual_info_regression(X_independent.reshape(-1, 1), Y_independent)
print(f"Mutual Information (Independent Variables): {mi[0]:.4f}")

plt.scatter(X_independent, Y_independent, alpha=0.5)
plt.title('Independent Variables')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Generate linear data
X_linear = np.random.rand(1000)
Y_linear = 3 * X_linear + 2

# Compute mutual information
mi = mutual_info_regression(X_linear.reshape(-1, 1), Y_linear)
print(f"Mutual Information (Linear Relationship): {mi[0]:.4f}")

plt.scatter(X_linear, Y_linear, alpha=0.5)
plt.title('Linear Relationship')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Generate non-linear data
X_nonlinear = np.random.rand(1000)
Y_nonlinear = np.sin(6 * np.pi * X_nonlinear)

# Compute mutual information
mi = mutual_info_regression(X_nonlinear.reshape(-1, 1), Y_nonlinear)
print(f"Mutual Information (Non-linear Relationship): {mi[0]:.4f}")

plt.scatter(X_nonlinear, Y_nonlinear, alpha=0.5)
plt.title('Non-linear Relationship')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Generate noisy linear data
X_noisy = np.random.rand(1000)
noise = np.random.normal(0, 0.5, 1000)
Y_noisy = 3 * X_noisy + 2 + noise

# Compute mutual information
mi = mutual_info_regression(X_noisy.reshape(-1, 1), Y_noisy)
print(f"Mutual Information (Noisy Linear Relationship): {mi[0]:.4f}")

plt.scatter(X_noisy, Y_noisy, alpha=0.5)
plt.title('Noisy Linear Relationship')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

functions = [
    ('Linear', lambda x: x),
    ('Quadratic', lambda x: x**2),
    ('Cubic', lambda x: x**3),
    ('Sine', lambda x: np.sin(x)),
    ('Exponential', lambda x: np.exp(x)),
    ('Logarithmic', lambda x: np.log(x + 1))
]

X = np.random.rand(1000)

for name, func in functions:
    Y = func(X)
    mi = mutual_info_regression(X.reshape(-1, 1), Y)
    print(f"Mutual Information ({name}): {mi[0]:.4f}")

