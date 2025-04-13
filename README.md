# IS5126_final_project


 
IS5126_final_project/
├── dataset/                         # All dataset files
│   ├── China database.xlsx          # Raw data
│   ├── classified_columns.xlsx      # Metadata for grouped features
│   ├── cleaned_all_data.xlsx        # Final cleaned dataset
│   └── data.py                      # Helper script for data loading/cleaning

├── model/                           # Saved models (MLP, CNN, LSTM, etc.)

├── results/                         # Plots, tables, and experiment results

├── src/                             # Source code organized by module
│   ├── causal_analysis/             # Clustering + Difference-in-Differences
│   ├── time_series_forecast/        # CNN/LSTM forecasting models
│   └── top_features/                # Feature selection using RF + SHAP

├── .gitattributes                   # Git settings
├── .gitignore                       # Ignore Python/VSCode artifacts
├── README.md                        # Project overview and documentation
└── requirements.txt                 # Python dependencies

