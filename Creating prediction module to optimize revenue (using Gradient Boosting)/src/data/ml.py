import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
csv_file_path = r"D:\UOM\Creating prediction module to optimize revenue (using Gradient Boosting)\raw data\daily-bike-share.csv"
df = pd.read_csv(csv_file_path)

# Convert categorical columns to categorical data type and create dummy variables

# Select features and target
features = ['temp', 'atemp', 'hum', 'windspeed'] + list(df.columns[df.columns.str.startswith('season')]) + list(df.columns[df.columns.str.startswith('yr')]) + list(df.columns[df.columns.str.startswith('mnth')]) + list(df.columns[df.columns.str.startswith('holiday')]) + list(df.columns[df.columns.str.startswith('weekday')]) + list(df.columns[df.columns.str.startswith('workingday')]) + list(df.columns[df.columns.str.startswith('weathersit')]) 
target = 'rentals'

# Split data into features (X) and target (y)
X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gradient Boosting model with best parameters
best_params = {'learning_rate': 0.1, 'loss': 'squared_error', 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}
gb_model = GradientBoostingRegressor(**best_params)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Create multiple plots to visually inspect the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot actual vs. predicted values
ax1.scatter(y_test, y_pred, alpha=0.5)
ax1.set_xlabel("Actual Rentals")
ax1.set_ylabel("Predicted Rentals")
ax1.set_title("Actual vs. Predicted Rentals")

# Plot residuals
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.5)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel("Predicted Rentals")
ax2.set_ylabel("Residuals")
ax2.set_title("Residuals Plot")

plt.tight_layout()
plt.show()
import joblib

# Save the trained model
model_file_path = "D:/UOM/Creating prediction module to optimize revenue (using Gradient Boosting)/raw data/best_gb_model.pkl"
joblib.dump(gb_model, model_file_path)