# Experiment 6: Outlier Handling for Time Series Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Change only these 3 values
file_name = "exp6.csv"
date_col = "Month"
target_col = "Passengers"

# Load Data
df = pd.read_csv(file_name)
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)

dates = df[date_col]
values = df[target_col].astype(float).values

# -------------------------------
# IQR Outlier Detection
# -------------------------------
Q1 = np.percentile(values, 25)
Q3 = np.percentile(values, 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_iqr = (values < lower) | (values > upper)

print("IQR Outliers Found:", outliers_iqr.sum())

# -------------------------------
# Z-Score Outlier Detection
# -------------------------------
mean = np.mean(values)
std = np.std(values)

z_scores = np.abs((values - mean) / std)
outliers_z = z_scores > 2.5

print("Z-Score Outliers Found:", outliers_z.sum())

# -------------------------------
# Plot Outliers
# -------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# IQR Plot
axes[0].plot(dates, values)
axes[0].scatter(dates[outliers_iqr], values[outliers_iqr])
axes[0].set_title("IQR Outlier Detection")

# Z-Score Plot
axes[1].plot(dates, values)
axes[1].scatter(dates[outliers_z], values[outliers_z])
axes[1].set_title("Z-Score Outlier Detection")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# Forecast Function
# -------------------------------
def run_model(data, name):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(12, len(data)):
        X.append(data[i - 12:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X[:split], y[:split])

    pred = model.predict(X[split:])

    pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y[split:].reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    print(f"\n{name}")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))

# -------------------------------
# Original Data
# -------------------------------
run_model(values.copy(), "Before Handling")

# Capping using IQR
cap_values = np.clip(values.copy(), lower, upper)
run_model(cap_values, "After Capping")

# Removal using IQR
remove_values = values[~outliers_iqr]
run_model(remove_values, "After Removal")

# Imputation using IQR
imp_values = values.copy()
median = pd.Series(values).rolling(6, center=True, min_periods=1).median()
imp_values[outliers_iqr] = median[outliers_iqr]

run_model(imp_values, "After Imputation")