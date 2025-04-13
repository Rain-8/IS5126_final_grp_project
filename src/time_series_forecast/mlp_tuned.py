# ============================
#  Install Libraries (if needed)
# ============================
# !pip install pandas matplotlib scikit-learn tensorflow openpyxl scikit-learn-intelex --quiet

# ============================
#  Import Libraries
# ============================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearnex import patch_sklearn
patch_sklearn()

# ============================
# Global Configs
# ============================
forecast_steps = 3  # Number of steps to forecast

# ============================
# Load & Clean Data
# ============================
file_path = "../../dataset/cleaned_all_data.xlsx"
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]
df = df.drop(columns=[col for col in df.columns if col.endswith(".1")], errors='ignore')

# Encode non-numeric columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
label_encoders = {}
for col in non_numeric_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# ============================
# Feature Selection
# ============================
combined_features = [
    "Value added of the secondary industry (10000 yuan)",
    "Value added of the tertiary industry (10000 yuan)",
    "Value added of the primary industry (10000 yuan)",
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
    "Centralized treatment rate of sewage treatment plant (%)"
]

target_cols = [
    "Regional Gross Domestic Product (RMB 10000)",
    "Industrial sulfur dioxide production (ton)"
]

imputer = SimpleImputer(strategy='mean')
df[combined_features + target_cols] = imputer.fit_transform(df[combined_features + target_cols])

# Feature selection
k_best_gdp = SelectKBest(score_func=f_regression, k=8)
k_best_so2 = SelectKBest(score_func=f_regression, k=10)

X_gdp = df[combined_features]
y_gdp = df[target_cols[0]]
X_new_gdp = k_best_gdp.fit_transform(X_gdp, y_gdp)
gdp_selected_features = [combined_features[i] for i in k_best_gdp.get_support(indices=True)]

X_so2 = df[combined_features]
y_so2 = df[target_cols[1]]
X_new_so2 = k_best_so2.fit_transform(X_so2, y_so2)
so2_selected_features = [combined_features[i] for i in k_best_so2.get_support(indices=True)]

# Scaling
feature_scaler_gdp = MinMaxScaler()
df_gdp = df[gdp_selected_features + [target_cols[0]]]
# df_gdp[gdp_selected_features] = feature_scaler_gdp.fit_transform(df_gdp[gdp_selected_features])
df_gdp.loc[:, gdp_selected_features] = feature_scaler_gdp.fit_transform(df_gdp[gdp_selected_features])

target_scaler_gdp = MinMaxScaler()
df_gdp[[target_cols[0]]] = target_scaler_gdp.fit_transform(df_gdp[[target_cols[0]]])

feature_scaler_so2 = MinMaxScaler()
df_so2 = df[so2_selected_features + [target_cols[1]]]
df_so2[so2_selected_features] = feature_scaler_so2.fit_transform(df_so2[so2_selected_features])
target_scaler_so2 = MinMaxScaler()
df_so2[[target_cols[1]]] = target_scaler_so2.fit_transform(df_so2[[target_cols[1]]])

# Sequence creation
def create_multistep_sequences(data, features, target_col, time_steps=5, forecast_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_steps + 1):
        X.append(data.iloc[i:i+time_steps][features].values)
        y.append(data.iloc[i+time_steps:i+time_steps+forecast_steps][target_col].values)
    return np.array(X), np.array(y)

X_seq_gdp, y_seq_gdp = create_multistep_sequences(df_gdp, gdp_selected_features, [target_cols[0]], forecast_steps=forecast_steps)
X_seq_so2, y_seq_so2 = create_multistep_sequences(df_so2, so2_selected_features, [target_cols[1]], forecast_steps=forecast_steps)

X_train_gdp, X_test_gdp, y_train_gdp, y_test_gdp = train_test_split(X_seq_gdp, y_seq_gdp, test_size=0.2, random_state=42)
X_train_so2, X_test_so2, y_train_so2, y_test_so2 = train_test_split(X_seq_so2, y_seq_so2, test_size=0.2, random_state=42)

# === Modified MLP Forecasting Script with Hyperparameter Tuning ===

# === Modified MLP Forecasting Script with Hyperparameter Tuning ===

# Use combinations of (hidden layer size, learning rate)
hyperparam_grid = [
    {'hidden_units': 64, 'lr': 0.001},
    {'hidden_units': 64, 'lr': 0.01},
    {'hidden_units': 128, 'lr': 0.001},
    {'hidden_units': 128, 'lr': 0.01}
]

best_model_gdp = None
best_model_so2 = None
best_loss_gdp = float('inf')
best_loss_so2 = float('inf')
best_config_gdp = None
best_config_so2 = None

# Update model building to accept hyperparameters
def build_mlp_model(input_shape, forecast_steps, hidden_units, lr):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(hidden_units, activation='relu'),
        Dense(hidden_units // 2, activation='relu'),
        Dense(forecast_steps)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Helper function to predict and inverse transform
def predict_and_inverse_transform(model, X_test, target_scaler):
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    return target_scaler.inverse_transform(y_pred)

# Try all combinations
def run_grid_search(X_train, y_train, X_test, y_test, target_name):
    global best_model_gdp, best_model_so2, best_loss_gdp, best_loss_so2, best_config_gdp, best_config_so2
    for config in hyperparam_grid:
        print(f"Training {target_name} with config: {config}")
        input_shape = X_train.shape[1] * X_train.shape[2]
        forecast_steps = y_train.shape[1]

        model = build_mlp_model(input_shape, forecast_steps, config['hidden_units'], config['lr'])
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit(
            X_train.reshape(X_train.shape[0], -1),
            y_train.reshape(y_train.shape[0], -1),
            validation_split=0.2,
            epochs=100,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )
        loss, mae = model.evaluate(X_test.reshape(X_test.shape[0], -1), y_test.reshape(y_test.shape[0], -1), verbose=0)
        print(f"{target_name} | Loss: {loss:.4f} | MAE: {mae:.4f}")

        if target_name == "GDP" and loss < best_loss_gdp:
            best_loss_gdp = loss
            best_model_gdp = model
            best_config_gdp = config
        elif target_name == "SO₂" and loss < best_loss_so2:
            best_loss_so2 = loss
            best_model_so2 = model
            best_config_so2 = config

# Run grid search for GDP and SO2
run_grid_search(X_train_gdp, y_train_gdp, X_test_gdp, y_test_gdp, "GDP")
run_grid_search(X_train_so2, y_train_so2, X_test_so2, y_test_so2, "SO₂")

# === Predictions ===
y_pred_gdp = predict_and_inverse_transform(best_model_gdp, X_test_gdp, target_scaler_gdp)
y_pred_so2 = predict_and_inverse_transform(best_model_so2, X_test_so2, target_scaler_so2)
y_test_gdp_original = target_scaler_gdp.inverse_transform(y_test_gdp.reshape(y_test_gdp.shape[0], -1))
y_test_so2_original = target_scaler_so2.inverse_transform(y_test_so2.reshape(y_test_so2.shape[0], -1))

# === Final Metrics ===
print(f"\nBest GDP Config: {best_config_gdp} | MAPE: {mean_absolute_percentage_error(y_test_gdp_original, y_pred_gdp):.4f}")
print(f"Best SO₂ Config: {best_config_so2} | MAPE: {mean_absolute_percentage_error(y_test_so2_original, y_pred_so2):.4f}")

# === Save Final Models ===
best_model_gdp.save_weights("../../model/MLP/mlp_gdp_best.weights.h5")
best_model_so2.save_weights("../../model/MLP/mlp_so2_best.weights.h5")
pd.DataFrame(y_pred_gdp).to_csv("../../results/temporal_analysis/MLP/y_pred_gdp_best.csv", index=False)
pd.DataFrame(y_pred_so2).to_csv("../../results/temporal_analysis/MLP/y_pred_so2_best.csv", index=False)

# === Plot as before ===
os.makedirs("../../results/temporal_analysis/MLP/best", exist_ok=True)

def plot_predictions(y_test, y_pred, sample_idx, target_name="GDP"):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, forecast_steps + 1), y_test[sample_idx], label="Actual")
    plt.plot(range(1, forecast_steps + 1), y_pred[sample_idx], label="Predicted", linestyle="--")
    plt.title(f"Sample {sample_idx}: Forecast - {target_name}")
    plt.xlabel("Years Ahead (t+1 to t+3)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"../../results/temporal_analysis/MLP/best/sample_{target_name}_{sample_idx}.png", dpi=300)
    plt.close()

def plot_all_together(y_test, y_pred, target_name="GDP"):
    plt.figure(figsize=(10, 5))
    for i in range(1, 6):
        plt.plot(range(1, forecast_steps + 1), y_test[i], label=f"Actual {target_name} - Sample {i}")
        plt.plot(range(1, forecast_steps + 1), y_pred[i], linestyle="--", label=f"Predicted {target_name} - Sample {i}")
    plt.title(f"Forecast for Samples 1–5: {target_name}")
    plt.xlabel("Years Ahead")
    plt.ylabel(target_name)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../results/temporal_analysis/MLP/best/all_{target_name}.png", dpi=300)
    plt.close()

def plot_continuous_predictions(y_test, y_pred, target_name="GDP"):
    actual_series = np.concatenate([y_test[i] for i in range(1, 6)])
    predicted_series = np.concatenate([y_pred[i] for i in range(1, 6)])
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(actual_series) + 1), actual_series, label=f"Actual {target_name}", marker='o')
    plt.plot(range(1, len(actual_series) + 1), predicted_series, label=f"Predicted {target_name}", marker='x', linestyle='--')
    plt.title(f"Forecast for Samples 1–5 (Continuous): {target_name}")
    plt.xlabel("Forecasted Year Index")
    plt.ylabel(target_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../../results/temporal_analysis/MLP/best/continuous_{target_name}.png", dpi=300)
    plt.close()

for i in range(1, 6):
    plot_predictions(y_test_gdp_original, y_pred_gdp, i, target_name="GDP")
    plot_predictions(y_test_so2_original, y_pred_so2, i, target_name="SO₂")

plot_all_together(y_test_gdp_original, y_pred_gdp, target_name="GDP")
plot_all_together(y_test_so2_original, y_pred_so2, target_name="SO₂")
plot_continuous_predictions(y_test_gdp_original, y_pred_gdp, target_name="GDP")
plot_continuous_predictions(y_test_so2_original, y_pred_so2, target_name="SO₂")
