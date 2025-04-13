# Install required libraries
# !pip install shap xgboost openpyxl --quiet

# --- Imports ---
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt
import os

# --- Load dataset ---
# Load data
file_path = "../../dataset/cleaned_all_data.xlsx"
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns
print("Loading data...")
df = df.drop(columns=[col for col in df.columns if col.endswith('.1')], errors='ignore')

# --- Define features and target ---
target = "Industrial sulfur dioxide production (ton)"

combined_features = [
    # Economic
    "Regional Gross Domestic Product (RMB 10000)",
    "Value added of the secondary industry (10000 yuan)",
    "Value added of the tertiary industry (10000 yuan)",
    "Value added of the primary industry (10000 yuan)",



    # Pollution Treatment
    "Industrial wastewater discharge reaches the standard (10000 tons)",
    "Industrial smoke and dust removal capacity (ton)",
    "Domestic sewage treatment rate (%)",
    "Harmless treatment rate of household waste (%)",
    "Comprehensive utilization rate of industrial solid waste (%)",
    "Comprehensive utilization rate of general industrial solid waste (%)",
    "Centralized treatment rate of sewage treatment plant (%)"
]

# Filter valid columns
available_features = [f for f in combined_features if f in df.columns]
df_model = df[available_features + [target]].dropna()

print("Features going into model:", available_features)
print("Target:", target)

# --- Prepare data ---
X = df_model[available_features]
y = df_model[target]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Train XGBoost Regressor ---
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# --- SHAP Explanation ---
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# === SUMMARY PLOT ===
# Create and save summary plot
shap.summary_plot(
    shap_values,
    features=X_test,
    feature_names=available_features,
    show=False, 
    plot_size=(14, 10)
)

# Save summary plot before showing
save_dir = "../../results/top_combined_features"
os.makedirs(save_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "combined_top_pollution_summary.png"), bbox_inches='tight', dpi=300)
plt.show()  

# === BAR PLOT ===
# Create and save SHAP bar plot
shap.plots.bar(shap_values, max_display=10, show=False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "combined_top_pollution_bar.png"), bbox_inches='tight', dpi=300)
plt.show()  # show second plot

