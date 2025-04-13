# Install packages
# !pip install openpyxl scikit-learn matplotlib seaborn --quiet

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Load cleaned data
# Load data
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

# Silhouette Score for k=2: 0.559
# features = [
#     "Value added of the secondary industry (10000 yuan)",
#     "Value added of the tertiary industry (10000 yuan)",
#     "Value added of the primary industry (10000 yuan)",
#     "The proportion of added value of the primary industry to GDP (%)",
#     "Number of museums",
#     "Number of full-time primary school teachers (person)",
#     "Total retail sales of consumer goods in society (10000 yuan)",
#     "Average number of employees on duty (10000 people)",
#     "Per capita regional GDP (yuan)",
#     "The proportion of added value of the tertiary industry to GDP (%)"
# ]

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

# Calculate silhouette score
score = silhouette_score(X_scaled, df_grouped['Cluster'])
print(f"\nSilhouette Score for k={k}: {score:.3f}")
if score > 0.5:
    print("Good clustering. Well-separated clusters.")
elif score > 0.3:
    print("Moderate clustering. Some overlap may exist.")
else:
    print("Poor clustering. Consider increasing k or changing features.\n")

# PCA for 2D plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_grouped['Cluster'], palette="Set2")
plt.title("City Clusters based on Top GDP Influencing Features (High, Mediuim, Low)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
save_dir = "../../results/cluster_analysis"
os.makedirs(save_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "gdp_clusters.png"), bbox_inches='tight', dpi=300)
plt.show()

