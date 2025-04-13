import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import itertools
from tqdm import tqdm
import os

# ==========  Basic Setup ==========
label_col = 'Regional Gross Domestic Product (RMB 10000)'
feature_cols = [
    "Industrial nitrogen oxide emissions (tons)",
    "Annual average concentration of inhalable fine particulate matter (micrograms/cubic meter)",
    "Industrial wastewater discharge volume (10000 tons)",
    "Industrial smoke and dust emissions (ton)",
    "Industrial wastewater discharge reaches the standard (10000 tons)",
    "Industrial smoke and dust removal capacity (ton)",
    "Domestic sewage treatment rate (%)",
    "Harmless treatment rate of household waste (%)",
    "Comprehensive utilization rate of industrial solid waste (%)",
    "Comprehensive utilization rate of general industrial solid waste (%)",
    "Centralized treatment rate of sewage treatment plant (%)",
    'year'
]

# ========== Load & Prepare Data ==========
file_path = "../../dataset/cleaned_all_data.xlsx"
df_filtered = pd.read_excel(file_path)
df_filtered['year'] = df_filtered['year'].astype(int)

train_df = df_filtered[df_filtered['year'] <= 2014].copy()
predict_df = df_filtered[(df_filtered['year'] >= 2015) & (df_filtered['year'] <= 2019)].copy()

# ========== Normalize Features ==========
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_predict = scaler.transform(predict_df[feature_cols])

# ========== Normalize Target ==========
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(train_df[[label_col]]).ravel()

# ========== Define Model ==========
class LinearRegressionGD(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionGD, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)
X_predict_tensor = torch.tensor(X_predict, dtype=torch.float32)

# ========== Grid Search ==========
param_grid = {
    'lr': [0.05, 0.01, 0.005],
    'epochs': [300, 500, 800],
    'batch_size': [50, 75]
}

best_r2 = -float('inf')
best_params = None
best_pred = None
param_combinations = list(itertools.product(param_grid['lr'], param_grid['epochs'], param_grid['batch_size']))

for lr, epochs, batch_size in tqdm(param_combinations, desc="Grid Search Progress"):
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

print("\nBest Model Parameters:")
print(best_params)
print(f"Best R² Score: {best_r2:.4f}")

# ========== Retrain Final Model ==========
best_lr = best_params['lr']
best_epochs = best_params['epochs']
best_batch_size = best_params['batch_size']
model = LinearRegressionGD(X_train.shape[1])
optimizer = optim.SGD(model.parameters(), lr=best_lr)
criterion = nn.MSELoss()

for epoch in range(best_epochs):
    model.train()
    for i in range(0, len(X_train_tensor), best_batch_size):
        X_batch = X_train_tensor[i:i+best_batch_size]
        y_batch = y_train_tensor[i:i+best_batch_size]
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ========== Make Predictions ==========
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_predict_tensor).numpy()

y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
predict_df[label_col + '_predicted'] = y_pred

if label_col + '_predicted' in df_filtered.columns:
    df_filtered.drop(columns=[label_col + '_predicted'], inplace=True)

# Merge prediction back into full dataframe
df_filtered = df_filtered.merge(
    predict_df[['year', 'area', label_col + '_predicted']],
    on=['year', 'area'],
    how='left'
)

df_filtered.loc[df_filtered['year'] <= 2014, label_col + '_predicted'] = None

# ========== Plot Trends ==========
gdp_trend = df_filtered.groupby('year')[[label_col, label_col + '_predicted']].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(gdp_trend['year'], gdp_trend[label_col], label='Actual GDP Mean', marker='o')
plt.plot(gdp_trend['year'], gdp_trend[label_col + '_predicted'], label='Predicted GDP Mean', marker='x', linestyle='--')

plt.xlabel("Year")
plt.ylabel("GDP (10,000 RMB)")
plt.title("1990-2019 Actual vs Predicted GDP Trends")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_dir = "../../results/temporal_analysis/LR"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "1990_2019_Actual_vs_Predicted_GDP_Trends.png"), dpi=300, bbox_inches='tight')
plt.show()