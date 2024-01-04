import numpy as np
import matplotlib.pyplot as plt
def linear_model(x, w, b):
    return w * x + b
np.random.seed(42)
X_train = np.random.rand(50) * 10
y_train = 2 * X_train + 1 + np.random.randn(50) * 2
initial_w = 1.5
initial_b = 0.5

y_pred_initial = linear_model(X_train, initial_w, initial_b)
plt.scatter(X_train, y_train, label='Training Data')
plt.plot(X_train, y_pred_initial, color='red', label='Initial Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Initial Model and Training Data')
plt.show()
modified_w = initial_w * 0.8
modified_b = initial_b * 1.2

y_pred_modified = linear_model(X_train, modified_w, modified_b)
plt.scatter(X_train, y_train, label='Training Data')
plt.plot(X_train, y_pred_initial, color='red', label='Initial Model')
plt.plot(X_train, y_pred_modified, color='green', label='Modified Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Initial and Modified Models with Training Data')
plt.show()
