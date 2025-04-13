import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno
import warnings
import os
warnings.filterwarnings('ignore')

# Set English display
plt.rcParams['font.sans-serif'] = ['Arial']  # Use Arial font
plt.rcParams['axes.unicode_minus'] = True

# Create directory for saving plots
def create_plots_directory(directory='eda_plots'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# 1. Check basic information
def check_basic_info(df, plots_dir):
    print("Basic Data Information:")
    print(f"Data Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nData Preview:")
    print(df.head())
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Check missing values
    print("\nMissing Values Statistics:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage(%)': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # Visualize missing values
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title('Missing Values Visualization')
    plt.savefig(f"{plots_dir}/missing_values_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

# 2. Handle missing values
def handle_missing_values(df):
    # For numeric variables, fill missing values with median
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # If there are categorical variables, they could be filled with mode
    
    return df

# 3. Analyze distributions
def analyze_distributions(df, features, plots_dir):
    # Histograms and density plots
    for feature in features:
        if feature in df.columns:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            sns.histplot(df[feature], kde=True)
            plt.title(f'{feature} Distribution Histogram')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[feature])
            plt.title(f'{feature} Boxplot')
            plt.xlabel(feature)
            
            plt.tight_layout()
            # Create a valid filename by replacing invalid characters
            safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace(' ', '_')
            plt.savefig(f"{plots_dir}/distribution_{safe_feature_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Check for outliers
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
            
            if not outliers.empty:
                print(f"{feature} has {len(outliers)} outliers detected")
                print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
                print(f"Outlier range: {df[feature].min()} - {df[feature].max()}\n")

# 4. Correlation analysis
def correlation_analysis(df, features, plots_dir):
    # Select specified features for correlation analysis
    features_df = df[features]
    
    # Calculate correlation matrix
    corr_matrix = features_df.corr()
    
    # Visualize correlation matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Output highly correlated feature pairs
    print("\nHighly Correlated Feature Pairs (|correlation| > 0.7):")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr, columns=['Feature 1', 'Feature 2', 'Correlation'])
        high_corr_df = high_corr_df.sort_values('Correlation', ascending=False)
        print(high_corr_df)
        # Save high correlation data to CSV
        high_corr_df.to_csv(f"{plots_dir}/high_correlations.csv", index=False)
    else:
        print("No highly correlated feature pairs found")

# 5. Grouped analysis and trend analysis (assuming there's a time or region column)
def grouped_analysis(df, features, group_by_col='Year', plots_dir=None):
    if group_by_col in df.columns:
        # View trends over time
        for feature in features:
            if feature in df.columns:
                plt.figure(figsize=(12, 6))
                df.groupby(group_by_col)[feature].mean().plot(marker='o')
                plt.title(f'{feature} Trend by {group_by_col}')
                plt.xlabel(group_by_col)
                plt.ylabel(f'Average {feature}')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                if plots_dir:
                    safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace(' ', '_')
                    plt.savefig(f"{plots_dir}/trend_{safe_feature_name}_by_{group_by_col}.png", dpi=300, bbox_inches='tight')
                plt.close()

# 6. Data standardization/normalization
def normalize_data(df, features, plots_dir):
    # Create copies for standardized and normalized data
    std_scaled_df = df.copy()
    minmax_scaled_df = df.copy()
    
    # Standardization (Z-score)
    std_scaler = StandardScaler()
    std_scaled_df[features] = std_scaler.fit_transform(df[features])
    
    # Normalization (Min-Max)
    minmax_scaler = MinMaxScaler()
    minmax_scaled_df[features] = minmax_scaler.fit_transform(df[features])
    
    # Compare original data with standardized data
    for feature in features[:3]:  # Show comparison for first 3 features only
        if feature in df.columns:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            sns.histplot(df[feature], kde=True)
            plt.title(f'Original Data: {feature}')
            
            plt.subplot(1, 3, 2)
            sns.histplot(std_scaled_df[feature], kde=True)
            plt.title(f'Standardized: {feature}')
            
            plt.subplot(1, 3, 3)
            sns.histplot(minmax_scaled_df[feature], kde=True)
            plt.title(f'Normalized: {feature}')
            
            plt.tight_layout()
            
            safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace(' ', '_')
            plt.savefig(f"{plots_dir}/normalization_comparison_{safe_feature_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return std_scaled_df, minmax_scaled_df

# 7. Group features and analyze relationships within groups
def analyze_feature_groups(df, features, plots_dir):
    # Divide features into economic and pollution groups
    economic_features = [f for f in features if any(keyword in f.lower() for keyword in ['gdp', 'value added', 'industry', 'economic'])]
    pollution_features = [f for f in features if any(keyword in f.lower() for keyword in ['emissions', 'particulate', 'wastewater', 'smoke', 'dust'])]
    treatment_features = [f for f in features if any(keyword in f.lower() for keyword in ['treatment', 'utilization', 'removal'])]
    
    # Analyze economic indicators vs pollution indicators with scatter plot matrix
    if economic_features and pollution_features:
        # Select first 2 features from each group for display
        eco_selection = economic_features[:2]
        pol_selection = pollution_features[:2]
        
        # Economic indicators vs Pollution indicators
        plt.figure(figsize=(15, 10))
        for i, eco_feat in enumerate(eco_selection):
            for j, pol_feat in enumerate(pol_selection):
                if eco_feat in df.columns and pol_feat in df.columns:
                    plt.subplot(len(eco_selection), len(pol_selection), i*len(pol_selection)+j+1)
                    sns.scatterplot(x=df[eco_feat], y=df[pol_feat])
                    plt.xlabel(eco_feat.split('(')[0])
                    plt.ylabel(pol_feat.split('(')[0])
                    plt.title(f'Economic vs Pollution')
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/economic_vs_pollution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze relationship between pollution and treatment
    if pollution_features and treatment_features:
        # Select first 2 features from each group for display
        pol_selection = pollution_features[:2]
        treat_selection = treatment_features[:2]
        
        # Pollution indicators vs Treatment indicators
        plt.figure(figsize=(15, 10))
        for i, pol_feat in enumerate(pol_selection):
            for j, treat_feat in enumerate(treat_selection):
                if pol_feat in df.columns and treat_feat in df.columns:
                    plt.subplot(len(pol_selection), len(treat_selection), i*len(treat_selection)+j+1)
                    sns.scatterplot(x=df[pol_feat], y=df[treat_feat])
                    plt.xlabel(pol_feat.split('(')[0])
                    plt.ylabel(treat_feat.split('(')[0])
                    plt.title(f'Pollution vs Treatment')
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/pollution_vs_treatment.png", dpi=300, bbox_inches='tight')
        plt.close()

# 8. Principal Component Analysis (PCA)
def perform_pca(df, features, plots_dir):
    from sklearn.decomposition import PCA
    
    # Prepare data
    X = df[features].copy()
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    
    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{plots_dir}/pca_explained_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Output important features for each principal component
    print("\nPCA Principal Component Feature Contributions:")
    components_df = pd.DataFrame(pca.components_, columns=features)
    for i, component in enumerate(components_df.index):
        importance = pd.Series(abs(components_df.iloc[i]), index=components_df.columns)
        top_features = importance.nlargest(3).index.tolist()
        print(f"Principal Component {i+1} (Explained Variance: {explained_variance[i]:.2%}):")
        print(f"  - Main Features: {', '.join(top_features)}")
    
    # Save PCA loadings to CSV
    loadings_df = pd.DataFrame(pca.components_.T, index=features, 
                              columns=[f'PC{i+1}' for i in range(len(features))])
    loadings_df.to_csv(f"{plots_dir}/pca_loadings.csv")
    
    # Plot scatter plot of first two principal components
    if pca_result.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%})')
        plt.title('PCA: First Two Principal Components')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{plots_dir}/pca_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return pca, pca_result

# Main analysis workflow
def perform_eda(df, features, plots_dir='eda_plots'):
    # Create directory for saving plots
    plots_dir = create_plots_directory(plots_dir)
    print(f"All plots will be saved to: {os.path.abspath(plots_dir)}")
    
    print("=" * 50)
    print("Starting Exploratory Data Analysis")
    print("=" * 50)
    
    # 1. Check basic information
    print("\n1. Basic Data Information Check")
    check_basic_info(df, plots_dir)
    
    # 2. Handle missing values
    print("\n2. Handle Missing Values")
    df = handle_missing_values(df)
    
    # 3. Analyze distributions
    print("\n3. Distribution Analysis")
    analyze_distributions(df, features, plots_dir)
    
    # 4. Correlation analysis
    print("\n4. Correlation Analysis")
    correlation_analysis(df, features, plots_dir)
    
    # 5. Grouped analysis (if time or region column exists)
    print("\n5. Grouped Analysis")
    if 'Year' in df.columns:
        grouped_analysis(df, features, 'Year', plots_dir)
    elif 'Region' in df.columns:
        grouped_analysis(df, features, 'Region', plots_dir)
    else:
        print("No time or region column found in data, skipping grouped analysis")
    
    # 6. Data standardization/normalization
    print("\n6. Data Standardization/Normalization")
    std_scaled_df, minmax_scaled_df = normalize_data(df, features, plots_dir)
    
    # 7. Feature group analysis
    print("\n7. Feature Group Analysis")
    analyze_feature_groups(df, features, plots_dir)
    
    # 8. Principal component analysis
    print("\n8. Principal Component Analysis (PCA)")
    pca, pca_result = perform_pca(df, features, plots_dir)
    
    # Save processed dataframes
    df.to_csv(f"{plots_dir}/processed_data.csv", index=False)
    std_scaled_df.to_csv(f"{plots_dir}/standardized_data.csv", index=False)
    minmax_scaled_df.to_csv(f"{plots_dir}/normalized_data.csv", index=False)
    
    print("\nEDA Analysis Completed!")
    print(f"All results have been saved to: {os.path.abspath(plots_dir)}")
    
    return df, std_scaled_df, minmax_scaled_df, pca, pca_result

# Define feature list
combined_features = [
    # Economic
    "Regional Gross Domestic Product (RMB 10000)",
    "Value added of the secondary industry (10000 yuan)",
    "Value added of the tertiary industry (10000 yuan)",
    "Value added of the primary industry (10000 yuan)",
    
    # Pollution
    "Industrial nitrogen oxide emissions (tons)",
    "Annual average concentration of inhalable fine particulate matter (micrograms/cubic meter)",
    "Industrial wastewater discharge volume (10000 tons)",
    "Industrial smoke and dust emissions (ton)",
    
    # Pollution Treatment
    "Industrial wastewater discharge reaches the standard (10000 tons)",
    "Industrial smoke and dust removal capacity (ton)",
    "Domestic sewage treatment rate (%)",
    "Harmless treatment rate of household waste (%)",
    "Comprehensive utilization rate of industrial solid waste (%)",
    "Comprehensive utilization rate of general industrial solid waste (%)",
    "Centralized treatment rate of sewage treatment plant (%)"
]

# Example call (prepare your data first)
df = pd.read_excel('China database.xlsx')

df, std_scaled_df, minmax_scaled_df, pca, pca_result = perform_eda(df, combined_features)
