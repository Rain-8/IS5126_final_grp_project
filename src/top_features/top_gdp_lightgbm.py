# === LightGBM, using economic factor, target is GDP, find top 10 Economic Features ===

# Install SHAP
# !pip install shap openpyxl --quiet

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
import re
import os

# Load data
file_path = "../../dataset/cleaned_all_data.xlsx"
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns

# Load classification file
classified="../../dataset/classified_columns.xlsx"
classified=pd.read_excel(classified)
economic_columns = classified[classified["Category"] == "economic factor"]["Column"].str.strip().tolist()

# Define target column
target = 'Regional Gross Domestic Product (RMB 10000)'
print([col for col in df.columns if target in col])

# Filter valid economic columns
economic_columns = [col for col in economic_columns if col in df.columns]
for col_to_remove in ['year', 'regionalism code', 'area', target]:
    if col_to_remove in economic_columns:
        economic_columns.remove(col_to_remove)

print(f"Features going into model: {economic_columns}")

# Filter rows where target is not null
df_econ = df[economic_columns + [target]]
df_econ = df_econ[df_econ[target].notna()]

# Define features and target
X = df_econ[economic_columns]
y = df_econ[target]

# Sanitize column names before DataFrame creation
import re
from collections import Counter

def sanitize_column(name):
    return re.sub(r"[^\w]", "_", name)

# Sanitize
X_numeric = X.select_dtypes(include=[np.number])
sanitized_columns = [sanitize_column(col) for col in X_numeric.columns]

# Make them unique if duplicates exist
counts = Counter()
unique_sanitized_columns = []
for col in sanitized_columns:
    counts[col] += 1
    if counts[col] > 1:
        unique_sanitized_columns.append(f"{col}_{counts[col]}")
    else:
        unique_sanitized_columns.append(col)

# Scale and assign unique column names
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
X_scaled_df = pd.DataFrame(X_scaled, columns=unique_sanitized_columns)


# Final safety check
if X_scaled.shape[1] != len(sanitized_columns):
    print("\nERROR: Mismatch in column dimensions!")
else:
    print("\nColumns match. Proceeding...")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Train LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=10, random_state=42)
lgb_model.fit(X_train, y_train)

# SHAP Analysis
explainer = shap.Explainer(lgb_model)
shap_values = explainer(X_train)

# SHAP Summary Plot
print("\nGenerating SHAP summary plot...")
shap.summary_plot(shap_values, X_train, plot_type="bar")

# Feature importance from LightGBM
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Evaluation
y_pred = lgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nTop 10 Features (by LightGBM):")
print(feature_importances.head(10))
print(f"\nR²: {r2:.4f}, RMSE: {rmse:.2f}")

# Plot Top 10 Feature Importance
plt.figure(figsize=(10, 6))
top10 = feature_importances.head(10)
plt.barh(top10['Feature'][::-1], top10['Importance'][::-1])
plt.title("Top 10 Important Features (LightGBM)")
plt.xlabel("Importance")
plt.tight_layout()


save_dir = "../../results/top_features"
os.makedirs(save_dir, exist_ok=True)

# Save the plot
save_path = os.path.join(save_dir, "top10_lgbm_importance.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

print("Min target value:", y.min())
print("Max target value:", y.max())
print("Mean target value:", y.mean())

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"Relative MAE (%): {mae / y_test.mean() * 100:.2f}%")