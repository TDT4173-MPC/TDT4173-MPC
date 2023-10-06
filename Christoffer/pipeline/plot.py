import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_features(csv_path, feature_x, feature_y):
    """
    Plot feature_y against feature_x from a CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Check if the features exist in the dataframe
    if feature_x not in df.columns or feature_y not in df.columns:
        print(f"Error: One or both of the features '{feature_x}' and '{feature_y}' do not exist in the CSV.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature_x], df[feature_y], alpha=0.5)
    plt.title(f"{feature_y} vs {feature_x}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_features.py <path_to_csv> <feature_x> <feature_y>")
        sys.exit(1)

    csv_path = sys.argv[1]
    feature_x = sys.argv[2]
    feature_y = sys.argv[3]

    plot_features(csv_path, feature_x, feature_y)
