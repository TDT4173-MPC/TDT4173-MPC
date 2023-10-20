import numpy as np
import pandas as pd
import h2o

def create_submission(pred_A, pred_B, pred_C, output_file="submission.csv"):
    """
    Create a Kaggle submission file.

    Parameters:
    - pred_A, pred_B, pred_C: Arrays containing predictions.
    - output_file: Name of the output CSV file.

    Returns:
    - None. Writes the submission to a CSV file.
    """
    
    # Concatenate predictions
    predictions = np.concatenate([pred_A, pred_B, pred_C])

    # Create an id array
    ids = np.arange(0, len(predictions))

    # Create a DataFrame
    df = pd.DataFrame({
        'id': ids,
        'prediction': predictions
    })

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


# Start the H2O cluster (locally)
h2o.init(max_mem_size="10G")

# Load test data
test_A = pd.read_parquet("preprocessing/data/test_A.parquet").dropna().drop(columns="date_forecast")
test_B = pd.read_parquet("preprocessing/data/test_B.parquet").dropna().drop(columns="date_forecast")
test_C = pd.read_parquet("preprocessing/data/test_C.parquet").dropna().drop(columns="date_forecast")

# Load models
load_directory = "Mathias/models/h2o_models"
model_A = h2o.load_model(f"{load_directory}/A/GBM_grid_1_AutoML_35_20231019_222437_model_15")
model_B = h2o.load_model(f"{load_directory}/B/GBM_grid_1_AutoML_36_20231020_12442_model_15")
model_C = h2o.load_model(f"{load_directory}/C/StackedEnsemble_AllModels_1_AutoML_37_20231020_42445")

# Make predictions
pred_A = model_A.predict(h2o.H2OFrame(test_A)).as_data_frame().values.flatten()
pred_B = model_B.predict(h2o.H2OFrame(test_B)).as_data_frame().values.flatten()
pred_C = model_C.predict(h2o.H2OFrame(test_C)).as_data_frame().values.flatten()

# Make negative predictions 0
pred_A[pred_A < 0] = 0
pred_B[pred_B < 0] = 0
pred_C[pred_C < 0] = 0

# Create submission
create_submission(pred_A, pred_B, pred_C, output_file="Mathias/submission.csv")

# Stop the H2O cluster
# h2o.cluster().shutdown()

