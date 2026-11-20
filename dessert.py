import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# Step 1: Load NASA climate data
climate_df = pd.read_csv("nasa_climate1.csv", skiprows=11)

print("Climate Data:")
print(climate_df.head())

print("\nAvailable Parameters:")
print(climate_df["PARAMETER"].unique())


# Step 2: Select climate variables
rain_df = climate_df[climate_df["PARAMETER"] == "PRECTOTCORR"]
temp_df = climate_df[climate_df["PARAMETER"] == "T2M"]
soil_df = climate_df[climate_df["PARAMETER"] == "GWETTOP"]


# Step 3: Convert climate data into monthly format
months = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

rain_long = rain_df.melt(
    id_vars=["YEAR"],
    value_vars=months,
    var_name="Month",
    value_name="Rain"
)

temp_long = temp_df.melt(
    id_vars=["YEAR"],
    value_vars=months,
    var_name="Month",
    value_name="Temp"
)

soil_long = soil_df.melt(
    id_vars=["YEAR"],
    value_vars=months,
    var_name="Month",
    value_name="Soil"
)


# Step 4: Merge climate variables
climate_monthly = rain_long.merge(
    temp_long, on=["YEAR", "Month"]
)

climate_monthly = climate_monthly.merge(
    soil_long, on=["YEAR", "Month"]
)

print("\nMonthly Climate Data:")
print(climate_monthly.head())


# Step 5: Load NDVI data
ndvi_df = pd.read_csv("India_NDVI.csv")

ndvi_df["date"] = pd.to_datetime(ndvi_df["date"], errors="coerce")

ndvi_df = ndvi_df.dropna(subset=["date", "NDVI"])

ndvi_df["YEAR"] = ndvi_df["date"].dt.year

ndvi_df["Month"] = ndvi_df["date"].dt.strftime("%b").str.upper()

# Convert scaled NDVI values to their original scale
ndvi_df["NDVI"] = ndvi_df["NDVI"] / 10000

ndvi_monthly = (
    ndvi_df.groupby(["YEAR", "Month"])["NDVI"]
    .mean()
    .reset_index()
)

print("\nMonthly NDVI Data:")
print(ndvi_monthly.head())


# Step 6: Merge climate and NDVI data
final_df = climate_monthly.merge(
    ndvi_monthly,
    on=["YEAR", "Month"],
    how="inner"
)

# Remove rows with missing or infinite values
final_df = final_df.replace([np.inf, -np.inf], np.nan)

final_df = final_df.dropna(
    subset=["Rain", "Temp", "Soil", "NDVI"]
)

print("\nFinal Dataset:")
print(final_df.head())

print("\nDataset Shape:", final_df.shape)

if len(final_df) < 5:
    raise ValueError(
        "Not enough matching monthly records. "
        "Check the dates and climate parameters in both datasets."
    )


# Step 7: Plot monthly NDVI
plt.figure(figsize=(10, 5))

plt.plot(final_df["YEAR"].astype(str) + "-" + final_df["Month"],
         final_df["NDVI"])

plt.title("Monthly NDVI")
plt.xlabel("Year and Month")
plt.ylabel("NDVI")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()


# Step 8: Prepare data for machine learning
X = final_df[["Rain", "Temp", "Soil"]]
y = final_df["NDVI"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 9: Train Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Step 10: Make predictions
y_pred = model.predict(X_test)


# Step 11: Evaluate model performance
r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("R2 Score:", round(r2, 4))
print("RMSE:", round(rmse, 4))


# Step 12: Plot actual vs predicted NDVI
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual NDVI")
plt.ylabel("Predicted NDVI")
plt.title("Actual vs Predicted NDVI")
plt.grid(True)
plt.tight_layout()
plt.show()


# Step 13: Display climate variable importance
importance = model.feature_importances_

features = ["Rain", "Temp", "Soil"]

plt.figure(figsize=(7, 5))

plt.bar(features, importance)

plt.title("Climate Variable Importance")
plt.xlabel("Climate Variables")
plt.ylabel("Feature Importance")
plt.tight_layout()
plt.show()



