import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3  # Text-to-Speech
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = "sales_data.csv"  # Update this path accordingly
sales_data = pd.read_csv(file_path)

# Ensure the dataset has the correct structure
sales_data.columns = ["Year", "Sales"]  # Adjust if necessary
sales_data = sales_data.sort_values(by="Year")

# Splitting data into training and testing sets
X = sales_data[["Year"]]
y = sales_data["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Model Evaluation:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")

# Predict future sales for the next 1-10 years
future_years = np.array(range(sales_data["Year"].max() + 1, sales_data["Year"].max() + 11)).reshape(-1, 1)
future_sales = model.predict(future_years)

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(future_years, future_sales, color='green', label='Predicted Sales')
plt.xlabel("Year")
plt.ylabel("Sales")
plt.legend()
plt.title("Sales Prediction for Next 10 Years")
plt.show()

# Print predicted sales
future_sales_df = pd.DataFrame({"Year": future_years.flatten(), "Predicted Sales": future_sales})
print(future_sales_df)

# Text-to-Speech Function
def speak_predictions():
    engine = pyttsx3.init()
    message = "Predicted sales for the next 10 years are: "
    for year, sales in zip(future_sales_df["Year"], future_sales_df["Predicted Sales"]):
        message += f"Year {year}: {sales:.2f} dollars. "
    engine.say(message)
    engine.runAndWait()

speak_predictions()
