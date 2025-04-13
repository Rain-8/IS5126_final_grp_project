# Imports
import os

os.environ["OMP_NUM_THREADS"] = "2"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score



# === Merge cluster info to full year-wise dataset ===
file_path = "../../dataset/cleaned_all_data.xlsx"
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns
print("Loading data...")

df = df[df['year'].between(2009, 2019)]  # Restrict years

# Top GDP influencing features (from SHAP)
# Silhouette Score for k=3: 0.730
features = [
    "Value added of the tertiary industry (10000 yuan)",
    "Value added of the secondary industry (10000 yuan)",
    "Total retail sales of consumer goods in society (10000 yuan)",
    "Education expenditure (10000 yuan)",
    "Year end balance of various loans from financial institutions (RMB 10000)",
    "Expenditure within the general budget of local finance (10000 yuan)",
    "Year end balance of savings for urban and rural residents (10000 yuan)"
]



# Drop NA values
df_cluster = df[['area', 'year'] + features].dropna()

# Average over years (per city)
df_grouped = df_cluster.groupby('area')[features].mean().reset_index()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_grouped[features])

# Run KMeans clustering
k = 3  # You can test different values
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_grouped['Cluster'] = kmeans.fit_predict(X_scaled)


df_full = pd.read_excel(file_path)
df_full = df_full.loc[:, ~df_full.columns.duplicated()]
df_full = df_full[df_full['year'].between(2009, 2019)]

# Merge clusters
cluster_map = df_grouped[['area', 'Cluster']]
df_full = df_full.merge(cluster_map, on='area', how='left')


# === Create post-policy variable ===
df_full['post'] = df_full['year'].apply(lambda x: 1 if x >= 2015 else 0)

# === Plot pollution (SO2) per cluster over time ===
pollution_col = "Industrial sulfur dioxide emissions (tons)"

agg_cluster = df_full.groupby(['year', 'Cluster'])[pollution_col].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=agg_cluster, x='year', y=pollution_col, hue='Cluster', palette="Set2", marker='o')
plt.axvline(2015, color='red', linestyle='--', label='Policy Year (2015)')
plt.title(f"{pollution_col} Trend by Economic Cluster")
plt.ylabel("Avg SO₂ Emissions (tons)")
plt.grid(True)
plt.legend()
plt.tight_layout()
save_dir = "../../results/cluster_analysis"
os.makedirs(save_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "pollution_trend_wrt_gdp_cluster.png"), bbox_inches='tight', dpi=300)
plt.show()