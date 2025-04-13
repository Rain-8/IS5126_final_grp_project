# IS5126_final_project

Exploratory Data Analysis Report on China Environmental and Economic Data
This report presents a refined exploration of a dataset containing economic and environmental indicators for China spanning from 1990 to 2022. The dataset encompasses 9,901 observations and 190 variables, which together illuminate patterns in areas such as regional GDP, industrial emissions, air quality, and the efficacy of pollution control measures. The goal of this analysis is to identify core trends, highlight potential data quality issues, and provide insights that could guide deeper research or policy evaluations.
1. Introduction and Data Scope
The dataset brings together a broad range of metrics related to China’s economic development (including GDP and industry-specific value added) and environmental factors (such as emission volumes, particulate matter concentrations, and waste treatment rates). With nearly a hundred different regional or temporal categories, the data shed light on significant disparities across provinces and over time. Preliminary checks reveal substantial missing data in several columns, particularly those concerning social services, along with numerous outliers in both economic and pollution indicators. These issues require careful handling to ensure robust conclusions.
2. Data Quality and Missing Values
A key characteristic of this dataset is the presence of notable missingness. While some economically and environmentally critical indicators—such as GDP, nitrogen oxide emissions, and wastewater discharge—have only moderate levels of missing data (generally around 6–10%), certain other variables exceed a 90% missing rate, limiting their practical utility. A missing-values matrix confirms that entire columns are sparsely populated and may not be suitable for rigorous analysis.
  Figure 1: missing_values_matrix.png
(This figure should show a visual matrix of missing values across rows and columns.)
Given that the most important economic and pollution variables are more consistently recorded, the analysis primarily focuses on those better-documented features.
3. Distribution and Outlier Assessment
Most economic and pollution measures exhibit strong right-skewness, reflecting a majority of observations clustered at lower or moderate values but also a subset of extremely high values. For example, Regional Gross Domestic Product (RMB 10,000) ranges from as low as 26,600 to as high as 446,530,000 in these units, underscoring the stark economic discrepancies among regions. The histogram and boxplot for GDP offer a clear illustration of this skew and outlier presence:
 Figure 2: distribution_Regional_Gross_Domestic_Product_(RMB_10000).png
(A combined histogram and boxplot highlighting the range and skewness for Regional GDP.)
Pollution metrics display similarly wide variations. Nitrogen oxide emissions, for instance, range from only a few dozen tons in some areas to nearly 200,000 tons in others:
  Figure 3: distribution_Industrial_nitrogen_oxide_emissions_(tons).png
(A combined histogram and boxplot illustrating the right-skewed distribution and extreme outliers for NOx emissions.)
Certain treatment measures also reveal unusual or anomalous values. The domestic sewage treatment rate, for example, occasionally appears to exceed 100% or drop below 0%, possibly due to reporting inconsistencies: 
 Insert Figure 4: distribution_Domestic_sewage_treatment_rate_(%).png
(A combined histogram and boxplot showing the distribution for sewage treatment rates, including any anomalies.)
4. Correlation Analysis
Examining the relationships among variables clarifies how economic performance correlates with pollution outputs and treatment efforts. A comprehensive correlation heatmap helps visualize these interdependencies:
  Figure 5: correlation_heatmap.png
(A heatmap capturing pairwise correlations among selected features.)
As expected, GDP-related metrics (e.g., value added for secondary and tertiary industries) show very strong positive correlations with one another. Meanwhile, pollution discharge volume correlates positively with the volume of treated pollutants, implying that areas with greater total discharges typically invest more resources in treatment. Despite these trends, correlations between economic indicators and environmental variables are less consistent, hinting at complex, context-dependent relationships.
5. Integrated Feature Analysis
To gain a more coherent picture, variables are grouped into three broad categories: economic (e.g., GDP, industry value added), pollution (e.g., emissions, particulate matter), and pollution treatment (e.g., sewage treatment rates, pollutant removal capacities). Scatter plots pairing economic measures with pollution indicators reveal a generally positive association—regions with higher economic output also tend to have higher emission levels:
  Figure 6: economic_vs_pollution.png
(A grid or set of scatter plots comparing selected economic vs. pollution variables.) 
A similar approach for pollution indicators vs. treatment measures highlights the extent to which high-emission areas also show increased pollution-control capacity:
  Figure 7: pollution_vs_treatment.png
(A set of scatter plots comparing pollution indicators with treatment or utilization rates.)
Some regions with substantial emissions indeed invest heavily in pollutant abatement, though the overall effectiveness of these efforts varies significantly.
6. Principal Component Analysis (PCA)
Conducting PCA on standardized data highlights the underlying dimensions shaping the dataset. The first principal component, capturing roughly 24% of the variance, is dominated by large-scale economic measures such as total GDP and industry-specific value added. The second component, explaining around 14%, revolves around industrial emissions and wastewater treatment. Subsequent components feature varying blends of pollution and treatment metrics, as well as primary-industry contributions.
 Insert Figure 8: pca_explained_variance.png
(A bar chart showing explained variance by each principal component, with a cumulative variance line.)
When plotting the first two principal components, a distinct pattern often emerges, potentially indicating clusters by region or time:
Insert Figure 9: pca_scatter.png
(A scatter plot of PC1 vs. PC2, visualizing how observations cluster in a reduced-dimensional space.)
7. Normalization for Comparative Analysis
Given the wide array of measurement units—ranging from monetary sums in tens of thousands of RMB to tons of emissions and percentages for treatment rates—normalizing or standardizing the data helps put these variables on a more comparable scale. Standardization (Z-scores) reveals which areas deviate most from global means, while min-max scaling confines all values to [0,1] for simpler comparisons. The following figure contrasts the original, standardized, and min-max transformed versions of an economic indicator:
  Figure 10: normalization_comparison_Regional_Gross_Domestic_Product_(RMB_10000).png
(A triptych or multi-panel figure comparing original, standardized, and min-max normalized distributions for Regional GDP.)
Though these transformations do not eliminate outliers, they facilitate clustering or other advanced modeling by mitigating unit-based discrepancies.
8. Conclusions and Recommendations
Overall, the dataset captures China’s economic and environmental evolution over more than three decades, showcasing pronounced disparities and a nuanced interplay between growth and pollution. High economic output correlates positively with pollution levels, yet many high-emission regions also exhibit advanced treatment measures, demonstrating at least partial mitigation efforts. Despite these insights, data-quality concerns—such as missing values and outliers—necessitate caution when interpreting results. A few strategies are suggested for deeper study:
1.	Year-by-year or region-by-region analysis to track policy impacts and local differences.
2.	Enhanced data cleaning, including domain-informed handling of impossible or extreme outliers, and improved imputation for missing values.
3.	Further dimensionality reduction or feature engineering to manage the high correlation among certain economic indicators.
4.	Consideration of logarithmic transforms or robust methods for extremely skewed variables.
Altogether, this exploratory analysis provides preliminary evidence of how economic activity can drive pollution while also spurring investment in environmental management. More specialized investigations—combining detailed regional information, policy timelines, and domain-specific knowledge—could yield a fuller understanding of the dynamic links between China’s economic development and its environmental challenges.


 
