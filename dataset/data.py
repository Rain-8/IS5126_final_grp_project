# === Data Preprocessing ===

import pandas as pd
import numpy as np
import os

data_path = "China database.xlsx"
output_file_path = "cleaned_all_data.xlsx"


if os.path.exists(data_path):
    # Load entire dataset
    df = pd.read_excel(data_path)

    # Replace zeros with NaN in percentage/ratio columns only (optional)
    cols_to_clean = [
        'Domestic sewage treatment rate (%)',
        'Harmless treatment rate of household waste (%)',
        'Comprehensive utilization rate of industrial solid waste (%)',
        'Comprehensive utilization rate of general industrial solid waste (%)'
    ]
    df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)

    # Interpolate grouped by area for time-series filling
    df = df.groupby('area').apply(lambda group: group.interpolate(method='linear')).reset_index(drop=True)

    # Forward-fill and backward-fill remaining missing values
    # df.fillna(method='ffill', inplace=True)
    # df.fillna(method='bfill', inplace=True)
    df.ffill(inplace=True)  # Forward fill
    df.bfill(inplace=True)  # Backward fill


    # Save cleaned dataset
    df.to_excel(output_file_path, index=False)

    print("Cleaned full dataset shape:", df.shape)
    df.head()

else:
    print("Original Excel file not found. Please upload it.")