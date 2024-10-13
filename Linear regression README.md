# Machine-Learning
# Linear Regression 

Using scikit-learn

# Import required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data: X (independent variable) and y (dependent variable)
X = np.array([[1], [2], [3], [4], [5], [6]]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 6])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict values for the test set
y_pred = model.predict(X_test)

# Print the model parameters
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")

# Plot the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()


