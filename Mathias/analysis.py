import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("Analysis/data/A/X_train_observed.parquet")

print(f'Shape: {df.shape}', end='\n\n\r')
print(f'Head: {df.head()}', end='\n\n\r')
print(f'Statistics: {df.describe()}', end='\n\n\r')