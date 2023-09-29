import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def structure_data(x_train_obs, x_train_est, x_target):
    split_value = x_train_est['date_forecast'].iloc[0]
    split_index = x_target[x_target['time'] == split_value].index[0]
    x_target_obs = x_target.iloc[:split_index]
    x_target_est = x_target.iloc[split_index:]
    x_train_obs_resampled = x_train_obs.set_index('date_forecast').resample('1H').mean()
    x_train_est_resampled = x_train_est.set_index('date_calc').resample('1H').mean()
    x_train_est_resampled = x_train_est_resampled.drop(columns=['date_forecast'])
    x_target_obs_resampled = x_target_obs.set_index('time').resample('1H').mean()
    x_target_est_resampled = x_target_est.set_index('time').resample('1H').mean()
    return x_train_obs_resampled, x_train_est_resampled, x_target_obs_resampled, x_target_est_resampled

def calculate_correlations(data, target):
    correlations = data.apply(lambda x: x.corr(target))
    sorted_correlations = correlations.abs().sort_values(ascending=True)
    sorted_correlations = sorted_correlations.dropna()
    return sorted_correlations

def plot_correlation_matrix(data, title, save=False, show=True):
    plt.figure(figsize=(16, 10))
    plt.title(title)
    sns.heatmap(data.to_frame(), annot=True, cmap='coolwarm', cbar=True, linewidths=0.5)
    plt.tight_layout(pad=3)
    if save:
        plt.savefig(f"Analysis/Mathias/plots/{title}.png")
    if show:
        plt.show()

def create_correlation_matrix(data, target):
    corr_matrix = data.corrwith(target)
    corr_matrix_sorted = corr_matrix.sort_values(ascending=False)
    corr_matrix_sorted = corr_matrix_sorted.dropna()
    corr_matrix_sorted.index = corr_matrix_sorted.index.str.replace(r':(\w+)', r' [\1]', regex=True)
    return corr_matrix_sorted

def create_correlation_files():

    # Factory A
    x_target = pd.read_parquet("Analysis/data/A/train_targets.parquet")
    x_train_obs = pd.read_parquet("Analysis/data/A/X_train_observed.parquet")
    x_train_est = pd.read_parquet("Analysis/data/A/X_train_estimated.parquet")
    x_train_obs_resampled, x_train_est_resampled, x_target_obs_resampled, x_target_est_resampled = structure_data(x_train_obs, x_train_est, x_target)
    corr_matrix = create_correlation_matrix(x_train_est_resampled, x_target_est_resampled['pv_measurement'])
    plot_correlation_matrix(corr_matrix, "Location A estimated correlation", save=True, show=False)

    # Factory B
    x_target = pd.read_parquet("Analysis/data/B/train_targets.parquet")
    x_train_obs = pd.read_parquet("Analysis/data/B/X_train_observed.parquet")
    x_train_est = pd.read_parquet("Analysis/data/B/X_train_estimated.parquet")
    x_train_obs_resampled, x_train_est_resampled, x_target_obs_resampled, x_target_est_resampled = structure_data(x_train_obs, x_train_est, x_target)
    corr_matrix = create_correlation_matrix(x_train_est_resampled, x_target_est_resampled['pv_measurement'])
    plot_correlation_matrix(corr_matrix, "Location B estimated correlation", save=True, show=False)

    # Factory C
    x_target = pd.read_parquet("Analysis/data/C/train_targets.parquet")
    x_train_obs = pd.read_parquet("Analysis/data/C/X_train_observed.parquet")
    x_train_est = pd.read_parquet("Analysis/data/C/X_train_estimated.parquet")
    x_train_obs_resampled, x_train_est_resampled, x_target_obs_resampled, x_target_est_resampled = structure_data(x_train_obs, x_train_est, x_target)
    corr_matrix = create_correlation_matrix(x_train_est_resampled, x_target_est_resampled['pv_measurement'])
    plot_correlation_matrix(corr_matrix, "Location C estimated correlation", save=True, show=False)


if __name__ == "__main__":
    x_target = pd.read_parquet("Analysis/data/A/train_targets.parquet")
    x_train_obs = pd.read_parquet("Analysis/data/A/X_train_observed.parquet")
    x_train_est = pd.read_parquet("Analysis/data/A/X_train_estimated.parquet")
    x_test = pd.read_parquet("Analysis/data/A/X_test_estimated.parquet")
    print(x_target.head())
    print(x_train_obs.head())
    print(x_train_est.head())
    print(x_test.head(100))

