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


# Load data
file_path = "../../dataset/cleaned_all_data.xlsx"
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns

# Load classification file
classified="../../dataset/classified_columns.xlsx"
classified=pd.read_excel(classified)

# Get economic factor columns
pollution_columns = classified[classified["Category"] == "pollution factor"]["Column"].str.strip().tolist()
target = 'Industrial sulfur dioxide emissions (tons)'

print([col for col in df.columns if "Industrial sulfur dioxide emissions (tons)" in col])


# Filter only if all columns exist
pollution_columns = [col for col in pollution_columns if col in df.columns]

# Remove target from feature list (in case it's wrongly included)
if 'year' in pollution_columns:
    pollution_columns.remove('year')
if 'regionalism code' in pollution_columns:
    pollution_columns.remove('regionalism code')

if target in pollution_columns:
    pollution_columns.remove(target)
print(f"Features going into model: {pollution_columns}")

pollution_columns = list(dict.fromkeys(pollution_columns))

from collections import Counter
dup_check = Counter(pollution_columns)
duplicates = [k for k, v in dup_check.items() if v > 1]
print("Remaining duplicates (if any):", duplicates)

print(f"Final features ({len(pollution_columns)}): {pollution_columns}")

df_econ = df[pollution_columns + [target]]


df_econ = df_econ[df_econ[target].notna()]

# Prepare features and target
X = df_econ[pollution_columns]
y = df_econ['Industrial sulfur dioxide emissions (tons)'].values[:, 0] if df_econ['Industrial sulfur dioxide emissions (tons)'].ndim > 1 else df_econ['Industrial sulfur dioxide emissions (tons)']

print(type(y))  # should be <class 'pandas.core.series.Series'>
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
print("Top 10 Pollution Features:")
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

# Adjust spacing manually
plt.subplots_adjust(left=0.35, top=0.95, bottom=0.1, right=0.95)

save_dir = "../../results/top_features"
os.makedirs(save_dir, exist_ok=True)

# Save the plot
save_path = os.path.join(save_dir, "top_pollution.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)


plt.show()


