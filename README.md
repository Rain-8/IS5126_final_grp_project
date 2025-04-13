# 📊 IS5126 – Exploratory Data Analysis

## 📁 Data Source

This project investigates a large-scale panel dataset capturing **China’s economic and environmental indicators** spanning from **1990 to 2022**. The dataset includes **9,901 observations and 190 variables**, with information covering:

- Regional GDP and industry-specific economic metrics
- Pollution indicators (e.g., emissions, particulate matter)
- Environmental treatment efforts (e.g., wastewater treatment, solid waste utilization)

## 🧪 Exploratory Data Analysis (EDA)

### 1. **Scope and Focus**
- Focused on core economic and environmental variables with consistent data availability.
- Excluded features with over 90% missing values (mostly in social services or auxiliary metrics).

### 2. **Data Quality**
- Key variables like GDP and emissions have manageable missing data (~6–10%).
- Severe sparsity in some columns; analysis prioritizes well-documented variables.

### 3. **Outliers and Distributions**
- Most features (e.g., GDP, emissions) are heavily right-skewed.
- Outliers are present, reflecting real disparities across provinces.

### 4. **Correlation Patterns**
- Strong intra-category correlations (e.g., secondary and tertiary industries).
- Pollution and treatment efforts positively correlated in high-discharge regions.
- Economic vs. pollution relationships are less consistent, hinting at policy heterogeneity.

### 5. **Integrated Feature Comparison**
- Grouped variables into 3 main clusters:
  - Economic
  - Pollution
  - Treatment
- Scatter plots show that higher GDP generally aligns with higher pollution.
- High-emission regions often exhibit stronger investment in treatment infrastructure.

### 6. **Dimensionality Reduction via PCA**
- Principal Component 1 (24% variance): Dominated by GDP and industry value added.
- Principal Component 2 (14%): Driven by emissions and treatment capacity.
- PC1 vs. PC2 plot reveals patterns that may correspond to time or regional clusters.

### 7. **Normalization Techniques**
- Compared raw, standardized (Z-score), and min-max normalized data.
- Normalization helps facilitate downstream modeling and clustering.

### 8. **Key Takeaways**
- Economic growth is often accompanied by increased pollution.
- However, many high-output regions also implement pollution control.
- Missing data and extreme outliers require cautious interpretation.
- Suggested follow-ups:
  - Region-wise policy impact analysis
  - Advanced imputation and transformation methods
  - Robust statistical techniques for skewed distributions
"""
