import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

# Load the econometrics module 2 dataset
df = pd.read_csv('datasets/module_2_data.csv')

# Predict US_STK (US Stock returns) using the other financial variables
feature_cols = ['DXY', 'METALS', 'OIL', 'INTL_STK', 'X13W_TB', 'X10Y_TBY', 'EURUSD']
X = df[feature_cols]
y = df['US_STK']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: Ordinary Least Squares (LS) ---
ls = LinearRegression().fit(X_train, y_train)
y_pred_ls = ls.predict(X_test)
mse_ls = mean_squared_error(y_test, y_pred_ls)
r2_ls = r2_score(y_test, y_pred_ls)

# --- Model 2: LASSO (with cross-validation to pick best alpha) ---
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# --- Test Statistic: Diebold-Mariano test for equal predictive accuracy ---
# H0: Both models have equal predictive accuracy
# H1: One model is significantly better
e_ls = y_test.values - y_pred_ls        # LS residuals
e_lasso = y_test.values - y_pred_lasso  # LASSO residuals
d = e_ls**2 - e_lasso**2                # loss differential
dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d)))
dm_pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))  # two-sided p-value

# --- Results ---
print("=" * 60)
print("LS vs LASSO Comparison — Predicting US Stock Returns")
print("=" * 60)
print(f"\n{'Metric':<30} {'LS':>12} {'LASSO':>12}")
print("-" * 54)
print(f"{'MSE':<30} {mse_ls:>12.8f} {mse_lasso:>12.8f}")
print(f"{'R²':<30} {r2_ls:>12.6f} {r2_lasso:>12.6f}")
print(f"{'# Non-zero coefficients':<30} {len(feature_cols):>12} {np.sum(lasso.coef_ != 0):>12}")

print(f"\nLASSO best alpha: {lasso.alpha_:.6f}")

print(f"\n--- Diebold-Mariano Test ---")
print(f"DM Statistic: {dm_stat:.4f}")
print(f"P-value:      {dm_pvalue:.4f}")

if dm_pvalue < 0.05:
    better = "LASSO" if np.mean(d) > 0 else "LS"
    print(f"Result: Reject H0 — {better} is significantly better (p < 0.05)")
else:
    print("Result: Fail to reject H0 — no significant difference between models")

print(f"\n--- LS Coefficients ---")
for name, coef in zip(feature_cols, ls.coef_):
    print(f"  {name:<15} {coef:>10.6f}")

print(f"\n--- LASSO Coefficients ---")
for name, coef in zip(feature_cols, lasso.coef_):
    marker = "" if coef != 0 else " (shrunk to 0)"
    print(f"  {name:<15} {coef:>10.6f}{marker}")
