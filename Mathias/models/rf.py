# Data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Models
from sklearn.ensemble import RandomForestRegressor

# Utils
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


# Read in the data
data_path = './preprocessing/data'
obs_A = pd.read_parquet(f'{data_path}/obs_A.parquet').drop(columns='date_forecast')
est_A = pd.read_parquet(f'{data_path}/est_A.parquet').drop(columns='date_forecast')
obs_B = pd.read_parquet(f'{data_path}/obs_B.parquet').drop(columns='date_forecast')
est_B = pd.read_parquet(f'{data_path}/est_B.parquet').drop(columns='date_forecast')
obs_C = pd.read_parquet(f'{data_path}/obs_C.parquet').drop(columns='date_forecast')
est_C = pd.read_parquet(f'{data_path}/est_C.parquet').drop(columns='date_forecast')

test_A = pd.read_parquet(f'{data_path}/test_A.parquet').dropna().drop(columns='date_forecast')
test_B = pd.read_parquet(f'{data_path}/test_B.parquet').dropna().drop(columns='date_forecast')
test_C = pd.read_parquet(f'{data_path}/test_C.parquet').dropna().drop(columns='date_forecast')

# Concatenate
# A = pd.concat([obs_A, est_A])
# B = pd.concat([obs_B, est_B])
# C = pd.concat([obs_C, est_C])
A = obs_A
B = obs_B
C = obs_C

print(A.columns)

# Split to features and labels
X_A = A.drop(columns=['pv_measurement'])
y_A = A['pv_measurement']
X_B = B.drop(columns=['pv_measurement'])
y_B = B['pv_measurement']
X_C = C.drop(columns=['pv_measurement'])
y_C = C['pv_measurement']

# Split into train and test
# X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, shuffle=False)
# X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, shuffle=False)
# X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, shuffle=False)

# Train models
# Initialize StackingRegressor with the base models and a meta-model
rf_A = RandomForestRegressor(n_estimators=100, max_depth=15)
rf_B = RandomForestRegressor(n_estimators=100, max_depth=15)
rf_C = RandomForestRegressor(n_estimators=100, max_depth=15)

# Train the models
print('Training models...')
rf_A.fit(X_A, y_A)
print('A done')
rf_B.fit(X_B, y_B)
print('B done')
rf_C.fit(X_C, y_C)
print('C done')

# Predict
pred_A = rf_A.predict(test_A)
pred_B = rf_B.predict(test_B)
pred_C = rf_C.predict(test_C)

# Clip negative values to 0
pred_A = np.clip(pred_A, 0, None)
pred_B = np.clip(pred_B, 0, None)
pred_C = np.clip(pred_C, 0, None)

print(f'Score A: {rf_A.score(X_A, y_A)}')
print(f'Score B: {rf_B.score(X_B, y_B)}')
print(f'Score C: {rf_C.score(X_C, y_C)}')

# Interpret the model
# Get feature importances
importances_A = rf_A.feature_importances_
importances_B = rf_B.feature_importances_
importances_C = rf_C.feature_importances_

# Get feature names
feature_names_A = X_A.columns
feature_names_B = X_B.columns
feature_names_C = X_C.columns

# Sort feature importances in descending order
indices_A = np.argsort(importances_A)[::-1]
indices_B = np.argsort(importances_B)[::-1]
indices_C = np.argsort(importances_C)[::-1]

# Rearrange feature names so they match the sorted feature importances
names_A = [feature_names_A[i] for i in indices_A]
names_B = [feature_names_B[i] for i in indices_B]
names_C = [feature_names_C[i] for i in indices_C]

# Print the feature ranking
print('Feature ranking for A:')
for i in range(X_A.shape[1]):
    print(f'{i + 1}. {names_A[i]} ({importances_A[indices_A[i]]})')

print('Feature ranking for B:')
for i in range(X_B.shape[1]):
    print(f'{i + 1}. {names_B[i]} ({importances_B[indices_B[i]]})')

print('Feature ranking for C:')
for i in range(X_C.shape[1]):
    print(f'{i + 1}. {names_C[i]} ({importances_C[indices_C[i]]})')



# Create submission
create_submission(pred_A, pred_B, pred_C, output_file="./Mathias/submission.csv")



"""
Feature rankings:
Feature ranking for A:
1. direct_rad:W (0.7216056393428908)
2. diffuse_rad:W (0.10699770989069707)
3. sun_azimuth:d (0.0197125093950962)
4. clear_sky_rad:W (0.014539907263424455)
5. wind_speed_u_10m:ms (0.010009812077161407)
6. wind_speed_v_10m:ms (0.00907633452444233)
7. visibility:m (0.008116899774543948)
8. total_cloud_cover:p (0.007952767358554895)
9. wind_speed_10m:ms (0.007711950706161025)
10. clear_sky_energy_1h:J (0.0076138900057908816)
11. direct_rad_1h:J (0.0075925167414849235)
12. effective_cloud_cover:p (0.007565693456896919)
13. t_1000hPa:K (0.007111095992725387)
14. relative_humidity_1000hPa:p (0.007084214237301578)
15. air_density_2m:kgm3 (0.006642478384373463)
16. diffuse_rad_1h:J (0.006503069848117982)
17. sun_elevation:d (0.006097141527080492)
18. month (0.006033208042392453)
19. dew_point_2m:K (0.005300457140359334)
20. absolute_humidity_2m:gm3 (0.004712625068950012)
21. pressure_100m:hPa (0.0037032558608695153)
22. sfc_pressure:hPa (0.003533633427518731)
23. msl_pressure:hPa (0.0035287009264269903)
24. year (0.003207549531858333)
25. pressure_50m:hPa (0.003080156419589852)
26. time_of_day (0.0026885769782002394)
27. snow_water:kgm2 (0.002268001638310856)
28. is_in_shadow:idx (9.440430014987456e-06)
29. is_day:idx (7.475771815869555e-07)
30. elevation:m (1.6431583418104023e-08)
31. calc_time_before (0.0)

Feature ranking for B:
1. sun_elevation:d (0.6057953317124253)
2. direct_rad:W (0.1433656249457944)
3. year (0.02747268571793926)
4. month (0.02033737537867466)
5. clear_sky_rad:W (0.019586531777633504)
6. sun_azimuth:d (0.016406975818030335)
7. diffuse_rad:W (0.012733069804946378)
8. direct_rad_1h:J (0.012369956618852691)
9. t_1000hPa:K (0.012144103118033059)
10. relative_humidity_1000hPa:p (0.010757596527161)
11. dew_point_2m:K (0.010524855475092063)
12. visibility:m (0.010445201822459919)
13. wind_speed_u_10m:ms (0.010416113694561626)
14. clear_sky_energy_1h:J (0.009687113236648453)
15. wind_speed_v_10m:ms (0.008505195994030133)
16. absolute_humidity_2m:gm3 (0.00816133712091355)
17. wind_speed_10m:ms (0.0077442471804643675)
18. effective_cloud_cover:p (0.007736529688438999)
19. air_density_2m:kgm3 (0.007511697852540563)
20. total_cloud_cover:p (0.006649782802991218)
21. diffuse_rad_1h:J (0.0064026075784686136)
22. pressure_100m:hPa (0.005849956945248969)
23. pressure_50m:hPa (0.005320326899471471)
24. sfc_pressure:hPa (0.005268260084208127)
25. msl_pressure:hPa (0.005168289848464599)
26. snow_water:kgm2 (0.0019339871630893544)
27. time_of_day (0.0016577553405327289)
28. is_in_shadow:idx (2.7743925239330906e-05)
29. is_day:idx (1.970532026139457e-05)
30. elevation:m (4.060738392004891e-08)
31. calc_time_before (0.0)

Feature ranking for C:
1. sun_elevation:d (0.5872690370542192)
2. direct_rad:W (0.0971182810556481)
3. direct_rad_1h:J (0.07985664653890256)
4. clear_sky_rad:W (0.06127637173548156)
5. clear_sky_energy_1h:J (0.055719905450674456)
6. t_1000hPa:K (0.01807640442508774)
7. visibility:m (0.008390991115626745)
8. dew_point_2m:K (0.008346609604545266)
9. diffuse_rad:W (0.007463209469964769)
10. relative_humidity_1000hPa:p (0.006914493726328476)
11. effective_cloud_cover:p (0.006191348191776316)
12. wind_speed_u_10m:ms (0.006176914951936598)
13. total_cloud_cover:p (0.005803247960789516)
14. wind_speed_10m:ms (0.005654092867664247)
15. diffuse_rad_1h:J (0.005562887003718124)
16. wind_speed_v_10m:ms (0.005491149684525656)
17. air_density_2m:kgm3 (0.004761412420632298)
18. sun_azimuth:d (0.004662261194527309)
19. absolute_humidity_2m:gm3 (0.004462509999229851)
20. year (0.003839695625940314)
21. sfc_pressure:hPa (0.0030600819562290014)
22. pressure_100m:hPa (0.002871864634062971)
23. msl_pressure:hPa (0.0028143031503359145)
24. pressure_50m:hPa (0.0026472315416908453)
25. month (0.0024824774941553693)
26. snow_water:kgm2 (0.002165886981023832)
27. time_of_day (0.0009147890619922817)
28. is_in_shadow:idx (5.336558152579116e-06)
29. is_day:idx (5.455187822934272e-07)
30. elevation:m (1.3026355701269745e-08)
31. calc_time_before (0.0)
"""