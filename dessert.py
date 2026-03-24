import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# load climate data
climate_data = pd.read_csv("nasa_climate1.csv", skiprows=11)

rain_data = climate_data[climate_data["PARAMETER"] == "PRECTOTCORR"]
temp_data = climate_data[climate_data["PARAMETER"] == "T2M"]
soil_data = climate_data[climate_data["PARAMETER"] == "GWETTOP"]


# convert to monthly format
months = ["JAN","FEB","MAR","APR","MAY","JUN",
           "JUL","AUG","SEP","OCT","NOV","DEC"]

rain_long = rain_data.melt(id_vars=["YEAR"], value_vars=months,
                           var_name="Month", value_name="Rain")

temp_long = temp_data.melt(id_vars=["YEAR"], value_vars=months,
                           var_name="Month", value_name="Temp")

soil_long = soil_data.melt(id_vars=["YEAR"], value_vars=months,
                           var_name="Month", value_name="Soil")


# merge climate variables
monthly_climate = rain_long.merge(temp_long, on=["YEAR","Month"])
monthly_climate = monthly_climate.merge(soil_long, on=["YEAR","Month"])


# load ndvi data
ndvi_data = pd.read_csv("India_NDVI.csv")

ndvi_data["date"] = pd.to_datetime(ndvi_data["date"])
ndvi_data["YEAR"] = ndvi_data["date"].dt.year
ndvi_data["Month"] = ndvi_data["date"].dt.strftime("%b").str.upper()

ndvi_data["NDVI"] = ndvi_data["NDVI"] / 10000


# group ndvi monthly
ndvi_monthly = ndvi_data.groupby(["YEAR","Month"])["NDVI"].mean().reset_index()


# final dataset
processed_data = monthly_climate.merge(ndvi_monthly, on=["YEAR","Month"])

print(processed_data.head())
print(processed_data.shape)


# simple feature
processed_data["Temp_Rain_Diff"] = processed_data["Temp"] - processed_data["Rain"]


# plot ndvi trend
plt.figure(figsize=(8,4))
plt.plot(processed_data["NDVI"])
plt.title("Monthly NDVI")
plt.xlabel("Time")
plt.ylabel("NDVI")
plt.grid(True)
plt.show()


# basic model
X = processed_data[["Rain","Temp","Soil"]]
y = processed_data["NDVI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("R2 Score:", r2_score(y_test, predictions))

rmse = mean_squared_error(y_test, predictions, squared=False)
print("RMSE:", rmse)


# feature importance
importance = model.feature_importances_
features = ["Rain","Temp","Soil"]

plt.bar(features, importance)
plt.title("Feature Importance")
plt.xlabel("Variables")
plt.ylabel("Importance")
plt.show()