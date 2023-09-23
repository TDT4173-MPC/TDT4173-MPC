import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_targets = pd.read_parquet('/Users/petterdalhaug/Documents/GitHub/data_analysis/analysis/data/A/train_targets.parquet')
X_train_observed = pd.read_parquet('/Users/petterdalhaug/Documents/GitHub/data_analysis/analysis/data/A/X_train_observed.parquet')
X_train_estimated = pd.read_parquet('/Users/petterdalhaug/Documents/GitHub/data_analysis/analysis/data/A/X_train_estimated.parquet')

print(train_targets)
real_data = pd.merge(train_targets, X_train_observed, on='date_forecast')
# estimated_data = pd.merge(train_targets, X_train_estimated, on='date_forecast')


# real_corr = real_data.corr()
# estimated_corr = estimated_data.corr()

# # Extracting correlation with the target column
# real_corr_with_target = real_corr[['your_target_column_here']]
# estimated_corr_with_target = estimated_corr[['your_target_column_here']]

# plt.figure(figsize=(12, 8))
# sns.heatmap(real_corr, annot=True, cmap='coolwarm')
# plt.title("Correlation Matrix for Real Data")
# plt.show()

# plt.figure(figsize=(12, 8))
# sns.heatmap(estimated_corr, annot=True, cmap='coolwarm')
# plt.title("Correlation Matrix for Estimated Data")
# plt.show()