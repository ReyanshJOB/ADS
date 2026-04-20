# Experiment 6: Outlier Handling Strategies for Time Series Forecasting
# Install dependencies: pip install pandas scikit-learn numpy matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----- Load Data -----
df = pd.read_csv('exp7_airlinepassenger.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
values = df['Passengers'].values.astype(float)
dates = df['Month']

# ----- Outlier Detection: IQR Method -----
Q1, Q3 = np.percentile(values, 25), np.percentile(values, 75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outlier_mask_iqr = (values < lower) | (values > upper)
print(f"IQR Outliers detected: {outlier_mask_iqr.sum()}")

# ----- Outlier Detection: Z-Score Method -----
z_scores = np.abs((values - values.mean()) / values.std())
outlier_mask_z = z_scores > 2.5
print(f"Z-Score Outliers detected: {outlier_mask_z.sum()}")

# ----- Outlier Detection: Isolation Forest -----
iso = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso.fit_predict(values.reshape(-1, 1))
outlier_mask_iso = iso_labels == -1
print(f"Isolation Forest Outliers detected: {outlier_mask_iso.sum()}")

# ----- Visualise Outliers -----
fig, axes = plt.subplots(3, 1, figsize=(12, 11))
for ax, mask, title, color in zip(
    axes,
    [outlier_mask_iqr, outlier_mask_z, outlier_mask_iso],
    ['IQR Method', 'Z-Score Method (threshold=2.5)', 'Isolation Forest'],
    ['tomato', 'darkorange', 'purple']
):
    ax.plot(dates, values, color='steelblue', linewidth=1.2, label='Passengers')
    ax.scatter(dates[mask], values[mask], color=color, zorder=5, s=60, label='Outlier')
    ax.set_title(f'Outlier Detection – {title}')
    ax.set_ylabel('Passengers')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=150)
plt.show()


# ----- Helper: Train & Evaluate GBR -----
def run_model(data, dates, label):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    SEQ_LEN = 12
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_inv = scaler.inverse_transform(y[split:].reshape(-1, 1)).flatten()
    mae  = mean_absolute_error(y_inv, preds_inv)
    rmse = np.sqrt(mean_squared_error(y_inv, preds_inv))
    mape = np.mean(np.abs((y_inv - preds_inv) / y_inv)) * 100
    print(f"\n{label}")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    return mae, rmse, mape, preds_inv, y_inv, dates[SEQ_LEN + split:SEQ_LEN + split + len(y_inv)]


# ----- Baseline (with outliers) -----
mae_b, rmse_b, mape_b, pred_b, actual_b, tdates_b = run_model(values.copy(), dates, "Baseline (with outliers)")

# ----- Capping / Winsorization -----
values_cap = np.clip(values.copy(), lower, upper)
mae_c, rmse_c, mape_c, pred_c, actual_c, tdates_c = run_model(values_cap, dates, "After Capping (IQR Winsorization)")

# ----- Removal (drop outlier rows) -----
clean_mask = ~outlier_mask_iqr
values_rem = values[clean_mask]
dates_rem = dates[clean_mask].reset_index(drop=True)
mae_r, rmse_r, mape_r, pred_r, actual_r, tdates_r = run_model(values_rem, dates_rem, "After Removal (IQR)")

# ----- Imputation (replace with rolling median) -----
values_imp = values.copy()
s = pd.Series(values_imp)
rolling_med = s.rolling(window=6, min_periods=1, center=True).median()
values_imp[outlier_mask_iqr] = rolling_med[outlier_mask_iqr].values
mae_i, rmse_i, mape_i, pred_i, actual_i, tdates_i = run_model(values_imp, dates, "After Imputation (Rolling Median)")

# ----- Comparison Plot -----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
configs = [
    (axes[0, 0], actual_b, pred_b, tdates_b, f'Baseline  MAE={mae_b:.1f}  RMSE={rmse_b:.1f}'),
    (axes[0, 1], actual_c, pred_c, tdates_c, f'Capping   MAE={mae_c:.1f}  RMSE={rmse_c:.1f}'),
    (axes[1, 0], actual_r, pred_r, tdates_r, f'Removal   MAE={mae_r:.1f}  RMSE={rmse_r:.1f}'),
    (axes[1, 1], actual_i, pred_i, tdates_i, f'Imputation MAE={mae_i:.1f}  RMSE={rmse_i:.1f}'),
]
for ax, actual, pred, td, title in configs:
    ax.plot(td.values, actual, label='Actual', color='steelblue', marker='o', markersize=3)
    ax.plot(td.values, pred,   label='Predicted', color='tomato', linestyle='--', marker='x', markersize=3)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Passengers')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Forecast Comparison: Outlier Handling Strategies', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outlier_comparison.png', dpi=150)
plt.show()

# ----- Summary Table -----
print("\n--- Performance Summary ---")
print(f"{'Strategy':<30} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8}")
print("-" * 56)
for name, mae, rmse, mape in [
    ("Baseline (outliers kept)",  mae_b, rmse_b, mape_b),
    ("Capping (Winsorization)",   mae_c, rmse_c, mape_c),
    ("Removal",                   mae_r, rmse_r, mape_r),
    ("Imputation (Rolling Med)",  mae_i, rmse_i, mape_i),
]:
    print(f"{name:<30} {mae:>8.2f} {rmse:>8.2f} {mape:>8.2f}")
