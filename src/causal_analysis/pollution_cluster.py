# === Clustering by 3 of top treatment factors ===

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

file_path = "../../dataset/cleaned_all_data.xlsx"
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns
print("Loading data...")

# Filter for years between 2015 and 2019
df_filtered = df[(df['year'] >= 2009) & (df['year'] <= 2019)].copy()

features = [
    # "Industrial sulfur dioxide production (ton)",
    # "Industrial nitrogen oxide emissions (tons)",
    # "Annual average concentration of inhalable fine particulate matter (micrograms/cubic meter)",
    "Industrial wastewater discharge volume (10000 tons)",
    "Industrial smoke and dust emissions (ton)",
    "Industrial electricity consumption (10000 kWh)"
    # "Comprehensive utilization rate of general industrial solid waste (%)"
]

# Drop NA values
df_cluster = df[['area', 'year'] + features].dropna()

# Average over years (per city)
df_grouped = df_cluster.groupby('area')[features].mean().reset_index()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_grouped[features])

# Run KMeans clustering
k = 2  # You can test different values
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_grouped['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate silhouette score
score = silhouette_score(X_scaled, df_grouped['Cluster'])
print(f"\nSilhouette Score for k={k}: {score:.3f}")
if score > 0.5:
    print("Good clustering. Well-separated clusters.")
elif score > 0.3:
    print("Moderate clustering. Some overlap may exist.")
else:
    print("Poor clustering. Consider increasing k or changing features.\n")

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Get coordinates
x = df_grouped[features[1]]
y = df_grouped[features[0]]
z = df_grouped[features[2]]
labels = df_grouped['Cluster']

ax.view_init(elev=20, azim=105)  # elev是仰角，azim是方位角

# Plot each cluster with a different color
scatter = ax.scatter(x, y, z, c=labels, cmap='viridis', s=50)

# Label axes
ax.set_xlabel(features[1])
ax.set_ylabel(features[0])
ax.set_zlabel(features[2])
ax.set_title('3D KMeans Clustering (k=2)')

# Add legend
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)
save_dir = "../../results/cluster_analysis"
os.makedirs(save_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "pollution_clusters.png"), bbox_inches='tight', dpi=300)

plt.show()

# Show city-cluster mapping
df_grouped[['area', 'Cluster']]

