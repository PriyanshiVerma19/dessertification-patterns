# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier   # 🔥 CHANGED (Regressor → Classifier)
from sklearn.metrics import accuracy_score, confusion_matrix

# LOAD CLIMATE DATA
climate_df = pd.read_csv("nasa_climate1.csv", skiprows=11)

# FILTER PARAMETERS
rain_df = climate_df[climate_df["PARAMETER"] == "PRECTOTCORR"]
temp_df = climate_df[climate_df["PARAMETER"] == "T2M"]
soil_df = climate_df[climate_df["PARAMETER"] == "GWETTOP"]

months = [
    "JAN","FEB","MAR","APR","MAY","JUN",
    "JUL","AUG","SEP","OCT","NOV","DEC"
]

# CONVERT TO MONTHLY FORMAT
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

# MERGE CLIMATE DATA
climate_monthly = rain_long.merge(temp_long, on=["YEAR","Month"])
climate_monthly = climate_monthly.merge(soil_long, on=["YEAR","Month"])

# LOAD NDVI DATA
ndvi_df = pd.read_csv("India_NDVI.csv")

ndvi_df["date"] = pd.to_datetime(ndvi_df["date"])
ndvi_df["YEAR"] = ndvi_df["date"].dt.year
ndvi_df["Month"] = ndvi_df["date"].dt.strftime("%b").str.upper()

# Normalize NDVI
ndvi_df["NDVI"] = ndvi_df["NDVI"] / 10000

ndvi_monthly = ndvi_df.groupby(["YEAR","Month"])["NDVI"].mean().reset_index()

# FINAL MERGE
final_df = climate_monthly.merge(ndvi_monthly, on=["YEAR","Month"])

# CHANGE 1: FUTURE NDVI
final_df["NDVI_next"] = final_df["NDVI"].shift(-1)
final_df = final_df.dropna()

# CHANGE 2: CREATE RISK LABEL
def classify(ndvi):
    if ndvi < 0.2:
        return 2   # High Risk
    elif ndvi < 0.4:
        return 1   # Medium Risk
    else:
        return 0   # Low Risk

final_df["Risk"] = final_df["NDVI"].apply(classify)

# VISUALIZATION
# NDVI Trend
plt.figure()
plt.plot(final_df["NDVI"])
plt.title("Monthly NDVI")
plt.xlabel("Time")
plt.ylabel("NDVI")
plt.grid(True)
plt.show()

# 🔥 NEW: Heatmap
plt.figure()
sns.heatmap(final_df[["Rain","Temp","Soil","NDVI"]].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 🔥 CHANGE 3: FEATURES & TARGET
X = final_df[["Rain","Temp","Soil"]]
y = final_df["Risk"]   # 🔥 CHANGED (NDVI → Risk)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 CHANGE 4: MODEL
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 🔥 CHANGE 5: EVALUATION
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# FEATURE IMPORTANCE
importance = model.feature_importances_
features = ["Rain","Temp","Soil"]

plt.figure()
plt.bar(features, importance)
plt.title("Climate Variable Importance")
plt.xlabel("Variables")
plt.ylabel("Importance")
plt.show()

import pickle

pickle.dump(model, open("model.pkl", "wb"))