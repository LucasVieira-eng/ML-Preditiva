import tensorflow as tf
import pandas as pd
import numpy as np

test_df = pd.read_csv('test_FD001.txt', sep=' ', header=None)
train_df = pd.read_csv('train_FD001.txt', sep=' ', header=None)

# Remove empty columns (from extra spaces)
train_df = train_df.dropna(axis=1, how='all')

# Assign column names
num_features = train_df.shape[1]
columns = ['engine_no', 'cycle'] + [f'feature_{i}' for i in range(1, num_features-1)]
train_df.columns = columns

# Calculate Y: cycles remaining for each row
train_df['max_cycle'] = train_df.groupby('engine_no')['cycle'].transform('max')
train_df['cycles_remaining'] = train_df['max_cycle'] - train_df['cycle']

# Prepare X and Y
X_train = train_df.iloc[:, 2:num_features-1].values  # sensor measurements
Y_train = train_df['cycles_remaining'].values

