# === RandomForestRegressor using economic factor, target is GDP, find top 10 Economic Features ===

# Install SHAP
# !pip install shap openpyxl --quiet

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
import os


# Load the dataset
file_path = "../../dataset/cleaned_all_data.xlsx"
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]

classified="../../dataset/classified_columns.xlsx"
classified=pd.read_excel(classified)

# Get economic factor columns
economic_columns = classified[classified["Category"] == "economic factor"]["Column"].str.strip().tolist()
target = 'Regional Gross Domestic Product (RMB 10000)'

print([col for col in df.columns if "Regional Gross Domestic Product (RMB 10000)" in col])


# Filter only if all columns exist
economic_columns = [col for col in economic_columns if col in df.columns]
# Remove target from feature list (in case it's wrongly included)
if 'year' in economic_columns:
    economic_columns.remove('year')
if 'regionalism code' in economic_columns:
    economic_columns.remove('regionalism code')

if target in economic_columns:
    economic_columns.remove(target)
print(f"Features going into model: {economic_columns}")

df_econ = df[economic_columns + [target]]


df_econ = df_econ[df_econ[target].notna()]

# Prepare features and target
X = df_econ[economic_columns]
y = df_econ['Regional Gross Domestic Product (RMB 10000)'].values[:, 0] if df_econ['Regional Gross Domestic Product (RMB 10000)'].ndim > 1 else df_econ['Regional Gross Domestic Product (RMB 10000)']

print(type(y))  
print(y.shape)  # should be (n_samples,)


# Scale features (optional but recommended)
X_numeric = X.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)



# Train Random Forest
model = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42)

# Define K-Fold (you can change n_splits)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on full dataset
cv_scores = cross_val_score(model, X_scaled_df, y, cv=kf, scoring='r2')

print("K-Fold R² Scores:", cv_scores)
print("Mean R² Score:", np.mean(cv_scores))

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)


# SHAP explanation (TreeExplainer is better for tree models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_array = shap_values[0].values if hasattr(shap_values[0], "values") else shap_values[0]
else:
    shap_array = shap_values.values if hasattr(shap_values, "values") else shap_values

# Ensure it's 2D
shap_array = np.array(shap_array)

# Check shape for debugging
print(f"SHAP array shape: {shap_array.shape}")
print(f"X_test shape: {X_test.shape}")

# Compute mean absolute SHAP values
mean_importance = np.abs(shap_array).mean(axis=0)

# Construct the importance DataFrame
shap_df = pd.DataFrame({
    'Feature': X_test.columns,   # use X_test because it was used in explainer
    'SHAP Importance': mean_importance
}).sort_values(by='SHAP Importance', ascending=False)

# Show top 10
print("Top 10 Economic Features:")
print(shap_df.head(10))

print("Train Score:", model.score(X_train, y_train))
print("Test Score:", model.score(X_test, y_test))


# Visualize

shap.summary_plot(
    shap_array,
    X_test,
    plot_type="bar",
    max_display=10,
    show=False,
    plot_size=(12, 6)
)

plt.subplots_adjust(left=0.35, top=0.95, bottom=0.1, right=0.95)

# Save the plot
save_dir = "../../results/top_features"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "top_gdp_rf_shap.png"), bbox_inches='tight', dpi=300)
plt.show()


