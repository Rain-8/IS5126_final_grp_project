# === Secondary Industry Value Added: DiD Analysis with Clustering ===

# --- Install required packages ---
# !pip install pandas scikit-learn openpyxl matplotlib statsmodels --quiet

# --- Imports ---
import os
os.environ["OMP_NUM_THREADS"]="2"

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import statsmodels.formula.api as smf
import seaborn as sns
# --- Load Dataset ---
file_path = "../../dataset/cleaned_all_data.xlsx"
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns
print("Loading data...")
df = df.drop(columns=[col for col in df.columns if col.endswith(".1")], errors='ignore')

# --- Clustering on pre-2015 data ---
target_feature = "Value added of the secondary industry (10000 yuan)"
region_col = "area"
year_col = "year"

df = df[[region_col, year_col, target_feature]].dropna()
df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
df = df.dropna(subset=[year_col])
df[year_col] = df[year_col].astype(int)

pre_policy_df = df[df[year_col] < 2015]
region_sec_add = pre_policy_df.groupby(region_col)[target_feature].mean().reset_index()
region_sec_add.rename(columns={target_feature: "add_secondary_pre2015"}, inplace=True)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
region_sec_add['group'] = kmeans.fit_predict(region_sec_add[['add_secondary_pre2015']])
cluster_centers = kmeans.cluster_centers_.flatten()
treated_label = np.argmax(cluster_centers)
region_sec_add['group'] = region_sec_add['group'].apply(lambda x: 'treated' if x == treated_label else 'control')

# Save region groupings
region_sec_add.to_csv("../../results/cluster_analysis/sec_add_groups.csv", index=False)

# --- Merge Clusters into Full Dataset ---
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]
df = df.drop(columns=[col for col in df.columns if col.endswith(".1")], errors='ignore')
df = df[df["area"].notna() & df["year"].notna()]
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
df = df[df['year'] <= 2019]

groups = pd.read_csv("../../results/cluster_analysis/sec_add_groups.csv")
df = df.merge(groups, on="area", how="inner")
df['treated'] = df['group'].map({'treated': 1, 'control': 0})
df['post2015'] = df['year'].apply(lambda x: 1 if x >= 2016 else 0)
df['treated_post'] = df['treated'] * df['post2015']

# --- Pre/Post Trend Plot ---
print("Plotting Secondary Industry Value Added trends...")
agg = df.groupby(['year', 'group'])[target_feature].mean().reset_index()

plt.figure(figsize=(10,6))
for grp in agg['group'].unique():
    subset = agg[agg['group'] == grp]
    plt.plot(subset['year'], subset[target_feature], marker='o', label=grp.title())

plt.axvline(x=2015, color='gray', linestyle='--', label='Policy Year (2015)')
plt.title("Secondary Industry Value Added: Treated vs Control Regions")
plt.xlabel("Year")
plt.ylabel("Value added of the secondary industry (10000 yuan)")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_dir = "../../results/cluster_analysis"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "secondary_industry_treated_vs_control.png"), dpi=300, bbox_inches='tight')
plt.show()

# --- DiD Regression ---
print("\nRunning DiD Regression...")
df_treat = df[df[target_feature].notna()]
did_model = smf.ols(
    formula=f"Q('{target_feature}') ~ treated + post2015 + treated_post + C(year)",
    data=df_treat
).fit()
print(did_model.summary())

att = did_model.params.get("treated_post", None)
print(f"\nEstimated Policy Effect (ATT): {att:.2f}" if att is not None else "No ATT coefficient found.")

# --- Multivariate Panel Regression with GDP ---
# if "Regional Gross Domestic Product (RMB 10000)" in df.columns:
#     print("\nRunning panel regression with GDP control...")
#     df_panel = df.dropna(subset=[target_feature, "Regional Gross Domestic Product (RMB 10000)"])
#     panel_model = smf.ols(
#         formula=f"Q('{target_feature}') ~ treated + post2015 + treated_post + Q('Regional Gross Domestic Product (RMB 10000)') + C(year)",
#         data=df_panel
#     ).fit()
#     print(panel_model.summary())
# else:
#     print("GDP feature not found for panel regression.")

# Predict using the DiD model
df_treat['DiD_Predicted'] = did_model.predict(df_treat)

# Plot actual vs predicted for treated group
treated_plot = df_treat[df_treat['treated'] == 1].groupby('year')[[target_feature, 'DiD_Predicted']].mean().reset_index()

plt.figure(figsize=(10,6))
plt.plot(treated_plot['year'], treated_plot[target_feature], label='Actual (Treated)', marker='o')
plt.plot(treated_plot['year'], treated_plot['DiD_Predicted'], label='DiD Predicted', linestyle='--', marker='x')
plt.axvline(2015, color='gray', linestyle='--', label='Policy Year (2015)')
plt.title("Actual vs DiD Predicted (Treated Group)")
plt.xlabel("Year")
plt.ylabel(target_feature)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "did_prediction_treated_group.png"), dpi=300, bbox_inches='tight')
plt.show()



