# Install required packages
# !pip install pandas scikit-learn openpyxl matplotlib --quiet
import os

os.environ["OMP_NUM_THREADS"] = "2"
# Import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


#Load dataset
file_path = "../../dataset/cleaned_all_data.xlsx"
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns
print("Loading data...")
df = df.drop(columns=[col for col in df.columns if col.endswith(".1")], errors='ignore')

# Check relevant columns
target_feature = "Industrial sulfur dioxide production (ton)"
region_col = "area"  # change if your region/city column is named differently
year_col = "year"

# Keep only rows with valid data
df = df[[region_col, year_col, target_feature]].dropna()

# Filter pre-2015 data for baseline emissions
pre_policy_df = df[df[year_col] < 2015]

# Aggregate emissions per region (mean pre-2015 emissions)
region_emissions = pre_policy_df.groupby(region_col)[target_feature].mean().reset_index()
region_emissions.rename(columns={target_feature: "avg_pre2015_emissions"}, inplace=True)

print("\nSample of average pre-2015 emissions per region:")
print(region_emissions.head())

# Clustering into treated (high emitters) vs control (low emitters)
kmeans = KMeans(n_clusters=2, random_state=42)
region_emissions['group'] = kmeans.fit_predict(region_emissions[['avg_pre2015_emissions']])

# Optional: Label clusters consistently as treated (high) and control (low)
cluster_centers = kmeans.cluster_centers_.flatten()
treated_label = np.argmax(cluster_centers)
region_emissions['group'] = region_emissions['group'].apply(lambda x: 'treated' if x == treated_label else 'control')

# Visualize cluster distribution
plt.figure(figsize=(10, 5))
plt.hist(
    region_emissions[region_emissions['group'] == 'treated']['avg_pre2015_emissions'],
    bins=30, alpha=0.7, label='Treated', color='red'
)
plt.hist(
    region_emissions[region_emissions['group'] == 'control']['avg_pre2015_emissions'],
    bins=30, alpha=0.7, label='Control', color='blue'
)
plt.title("Pre-2015 Emissions: Treated vs Control Regions")
plt.xlabel("Average SO₂ Emissions (ton)")
plt.ylabel("Number of Regions")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_dir = "../../results/cluster_analysis"
os.makedirs(save_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "pre_2015_SO2_treated_vs_control.png"), bbox_inches='tight', dpi=300)
plt.show()



# --- Pre-2015 Avg SO₂ Emissions Line Plot for Treated vs Control ---

# Filter for pre-2015
pre_policy_df = df[df['year'] < 2015]

# Merge treatment group info
pre_policy_df = pre_policy_df.merge(region_emissions[['area', 'group']], on='area', how='left')

# Group by year and group to get average emissions
pre_yearly_avg = pre_policy_df.groupby(['year', 'group'])['Industrial sulfur dioxide production (ton)'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
for grp in pre_yearly_avg['group'].unique():
    subset = pre_yearly_avg[pre_yearly_avg['group'] == grp]
    plt.plot(subset['year'], subset['Industrial sulfur dioxide production (ton)'],
             marker='o', label=grp.title())

plt.title("Pre-2015 Average SO₂ Emissions Over Time by Group")
plt.xlabel("Year")
plt.ylabel("Avg SO₂ Emissions (ton)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
save_path = "../../results/cluster_analysis/pre_2015_SO2_emissions_by_year.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()








# Save output mapping
region_emissions.to_csv("../../results/cluster_analysis/region_treatment_groups.csv", index=False)
print("\nRegion groupings saved to region_treatment_groups.csv")





import statsmodels.formula.api as smf




# Load treatment assignment
print("Loading treatment group assignments...")
assignments = pd.read_csv("../../results/cluster_analysis/region_treatment_groups.csv")  # From k-means output

#  Merge on region (area)
df = df.merge(assignments, how="left", left_on="area", right_on="area")

# Keep necessary columns
target = "Industrial sulfur dioxide production (ton)"
df = df[['area', 'year', target, 'group']].dropna()

#  Construct DiD variables
df['treated'] = (df['group'] == 'treated').astype(int)
df['post2015'] = (df['year'] >= 2015).astype(int)
df['treated_post'] = df['treated'] * df['post2015']

print("\nFinal data shape:", df.shape)
print(df[['treated', 'post2015', 'treated_post']].tail())

# DiD Regression using OLS
print("\nRunning Difference-in-Differences Regression...\n")
model = smf.ols(
    formula="Q('Industrial sulfur dioxide production (ton)') ~ treated + post2015 + treated_post + C(year)",
    data=df
).fit()


# Print summary
print(model.summary())

# Get estimated ATT (policy effect)
att = model.params.get("treated_post", None)
print(f"\nEstimated Policy Effect (ATT): {att:.2f}" if att is not None else "No ATT coefficient found.")


# Merge treatment info
# df = df.merge(assignments, on="area", how="inner")
df['treated'] = df['group'].map({'treated': 1, 'control': 0})
df['post2015'] = df['year'].apply(lambda x: 1 if x >= 2015 else 0)
df['treated_post'] = df['treated'] * df['post2015']

# Target: Industrial sulfur dioxide production (ton)
target_emission = "Industrial sulfur dioxide production (ton)"

# Plot Pre/Post Emission Trends
print(" Plotting SO₂ emission trends...")
agg = df.groupby(['year', 'group'])[target_emission].mean().reset_index()

plt.figure(figsize=(10,6))
for grp in agg['group'].unique():
    subset = agg[agg['group'] == grp]
    plt.plot(subset['year'], subset[target_emission], marker='o', label=grp.title())

plt.axvline(x=2015, color='gray', linestyle='--', label='Policy Year (2015)')
plt.title("SO₂ Emissions: Treated vs Control Regions")
plt.xlabel("Year")
plt.ylabel("SO₂ Emissions (ton)")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs(save_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "post_2015_SO2_treated_vs_control.png"), bbox_inches='tight', dpi=300)
plt.show()
plt.show()


