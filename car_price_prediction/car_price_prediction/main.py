import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("car_data.csv")

data["Years_of_Service"] = 2025 - data["Year"]

data.drop(["Year", "Car_Name"], axis=1, inplace=True)

data.replace({
    "Fuel_Type": {"Petrol": 0, "Diesel": 1, "CNG": 2},
    "Seller_Type": {"Dealer": 0, "Individual": 1},
    "Transmission": {"Manual": 0, "Automatic": 1}
}, inplace=True)

X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy (R2 Score):", round(r2_score(y_test, y_pred), 2))


print("\nEnter Car Details for Price Prediction")

fuel = input("Fuel Type (Petrol/Diesel/CNG): ")
years = int(input("Years of Service: "))
showroom_price = float(input("Showroom Price (in Lakhs): "))
owners = int(input("Number of Previous Owners: "))
kms = int(input("Kilometers Driven: "))
seller = input("Seller Type (Dealer/Individual): ")
transmission = input("Transmission (Manual/Automatic): ")


fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}

user_data = np.array([[showroom_price,
                       kms,
                       owners,
                       fuel_map[fuel],
                       seller_map[seller],
                       trans_map[transmission],
                       years]])

predicted_price = model.predict(user_data)

print("\nEstimated Selling Price:",
      round(predicted_price[0], 2), "Lakhs")
