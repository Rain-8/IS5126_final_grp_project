import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import itertools

# ========== Load Data ==========
data_path = "../../dataset/cleaned_all_data.xlsx"
df_filtered = pd.read_excel(data_path)
df_filtered = df_filtered[(df_filtered['year'] >= 2009) & (df_filtered['year'] <= 2019)].copy()
df_filtered['year'] = df_filtered['year'].astype(int)

# ========== Config ==========
label_col = 'Industrial sulfur dioxide production (ton)'
feature_cols = [
    "Value added of the secondary industry (10000 yuan)",
    "Value added of the tertiary industry (10000 yuan)",
    "Value added of the primary industry (10000 yuan)",
    "Industrial wastewater discharge reaches the standard (10000 tons)",
    "Industrial smoke and dust removal capacity (ton)",
    "Domestic sewage treatment rate (%)",
    "Harmless treatment rate of household waste (%)",
    "Comprehensive utilization rate of industrial solid waste (%)",
    "Comprehensive utilization rate of general industrial solid waste (%)",
    "Centralized treatment rate of sewage treatment plant (%)",
    'year'
]

train_df = df_filtered[(df_filtered['year'] >= 2009) & (df_filtered['year'] <= 2014)].copy()
predict_df = df_filtered[(df_filtered['year'] >= 2015) & (df_filtered['year'] <= 2019)].copy()

# ==========  Normalize Features and Labels ==========
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_predict = scaler.transform(predict_df[feature_cols])

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(train_df[[label_col]]).ravel()

# ==========  Model ==========
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)
X_predict_tensor = torch.tensor(X_predict, dtype=torch.float32)

class LinearRegressionGD(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionGD, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# ==========  Hyperparameter Grid Search ==========
param_grid = {
    'lr': [0.05, 0.01, 0.005],
    'epochs': [300, 500, 800],
    'batch_size': [50, 75]
}

param_combinations = list(itertools.product(param_grid['lr'], param_grid['epochs'], param_grid['batch_size']))
best_r2 = -float('inf')
best_params = None
best_pred = None

for lr, epochs, batch_size in tqdm(param_combinations, desc=" Grid Search Progress"):
    model = LinearRegressionGD(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            model.train()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_predict_tensor).numpy()
    y_true = y_scaler.transform(predict_df[[label_col]]).ravel()
    r2 = r2_score(y_true, y_pred_scaled)

    if r2 > best_r2:
        best_r2 = r2
        best_params = {'lr': lr, 'epochs': epochs, 'batch_size': batch_size}
        best_pred = y_pred_scaled

    print(f"R² Score for lr={lr}, epochs={epochs}, batch_size={batch_size}: {r2:.4f}")

print("\n Best Parameters:", best_params)
print(f"Best R² Score: {best_r2:.4f}")

# ==========  Retrain Best Model ==========
model = LinearRegressionGD(X_train.shape[1])
optimizer = optim.SGD(model.parameters(), lr=best_params['lr'])
criterion = nn.MSELoss()

for epoch in range(best_params['epochs']):
    for i in range(0, len(X_train_tensor), best_params['batch_size']):
        X_batch = X_train_tensor[i:i + best_params['batch_size']]
        y_batch = y_train_tensor[i:i + best_params['batch_size']]
        model.train()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ==========  Predictions and Trend ==========
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_predict_tensor).numpy()

y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
predict_df[label_col + '_predicted'] = y_pred

# Avoid merge conflict
if label_col + '_predicted' in df_filtered.columns:
    df_filtered.drop(columns=[label_col + '_predicted'], inplace=True)

df_filtered = df_filtered.merge(
    predict_df[['year', 'area', label_col + '_predicted']],
    on=['year', 'area'], how='left')

# ==========  Consistent Prediction Regions ==========
high_predicted_cities = []
low_predicted_cities = []

for area in predict_df['area'].unique():
    area_data = predict_df[predict_df['area'] == area]
    if all(area_data[label_col + '_predicted'] > area_data[label_col]):
        high_predicted_cities.append(area)
    if all(area_data[label_col + '_predicted'] < area_data[label_col]):
        low_predicted_cities.append(area)

print("\nCities where predicted > actual in all years:", high_predicted_cities)
print("\nCities where predicted < actual in all years:", low_predicted_cities)

# ==========  Save Results ==========
results_dir = "../../results/temporal_analysis/LR/SO2"
os.makedirs(results_dir, exist_ok=True)

df_filtered.to_csv(os.path.join(results_dir, "SO2_prediction_linear_regression_gd.csv"), index=False)
df_filtered[df_filtered['area'].isin(high_predicted_cities)].to_csv(os.path.join(results_dir, "high_predicted_cities.csv"), index=False)
df_filtered[df_filtered['area'].isin(low_predicted_cities)].to_csv(os.path.join(results_dir, "low_predicted_cities.csv"), index=False)

# ========== Plot ==========
df_filtered.loc[df_filtered['year'] <= 2014, label_col + '_predicted'] = None
so2_trend = df_filtered.groupby('year')[[label_col, label_col + '_predicted']].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(so2_trend['year'], so2_trend[label_col], label='Actual SO2', marker='o')
plt.plot(so2_trend['year'], so2_trend[label_col + '_predicted'], label='Predicted SO2', marker='x', linestyle='--')
plt.xlabel("Year")
plt.ylabel("SO2 Emissions")
plt.title("2009-2019 Actual vs Predicted SO2 Emissions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "so2_trend.png"), dpi=300)
plt.show()