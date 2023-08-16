import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data from files
train_data = pd.read_excel("d1l1_Data_for_learning_Wind_speed_at_RochesPoint.xls")
test_data = pd.read_excel("d1l1_Data_for_testing_Wind_speed_at_RochesPoint.xls")

# Separate target variable (y) and features (X) for both training and testing data
y_train = train_data.iloc[:, -1]
X_train = train_data.iloc[:, :-1]

y_test = test_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plotting the results
plt.figure(figsize=(10, 6))

# Training set
plt.scatter(y_train, model.predict(X_train), color='blue', label='Training Set')

# Testing set
plt.scatter(y_test, y_pred, color='red', label='Testing Set')

# Perfect prediction line
x_line = np.linspace(min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max()), 100)
y_line = x_line
plt.plot(x_line, y_line, color='green', linestyle='--', label='Perfect Prediction Line')

plt.title('Actual vs. Predicted Wind Speed')
plt.xlabel('Actual Wind Speed')
plt.ylabel('Predicted Wind Speed')
plt.legend()
plt.grid(True)
plt.show()

print("Mean Squared Error:", mse)
print("R-squared:", r2)
