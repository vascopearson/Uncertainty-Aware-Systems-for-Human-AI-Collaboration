import pandas as pd
import numpy as np
import os
import re

# Read the main alerts data and expert predictions
alerts = pd.read_parquet('../../alert_data/alerts.parquet')
train_data = alerts.loc[alerts["month"].isin([3, 4, 5, 6])]
test = alerts.loc[alerts["month"] == 7]
expert_predictions = pd.read_parquet('./expert_predictions.parquet')

# Get test predictions from the main expert_predictions file
test_predictions = expert_predictions.loc[test.index]

print("=== ORIGINAL DATA ===")
print(f"Test data shape: {test.shape}")
print(f"Test predictions shape: {test_predictions.shape}")
print(f"Test index range: {test.index.min()} to {test.index.max()}")
print()

# Define all noise settings and seeds
noise_settings = [
    ('1.0', '0.3'),
    ('1.5', '0.4'), 
    ('2.0', '0.5')
]
seeds = range(5)  # 0 to 4

# Process each combination
for noise_1, noise_2 in noise_settings:
    for seed in seeds:
        setting = f"{noise_1}_{noise_2}_seed_{seed}"
        drift_file = f'../../drift_alert_data/test_alert_data_noisy_{noise_1}_{noise_2}_seed_{seed}.parquet'
        
        print(f"=== PROCESSING SETTING: {setting} ===")
        print(f"Drift file: {drift_file}")
        
        try:
            # Read drift data
            drift_data = pd.read_parquet(drift_file)
            print(f"Drift data shape: {drift_data.shape}")
            print(f"Drift data index range: {drift_data.index.min()} to {drift_data.index.max()}")
            
            # Determine the offset based on the noise setting
            if noise_1 == '1.0':
                offset = 1000000
            elif noise_1 == '1.5':
                offset = 2000000
            elif noise_1 == '2.0':
                offset = 3000000
                        
            # Adjust drift data indices to match the original test data range
            adjusted_drift_data = drift_data.copy()
            adjusted_drift_data.index = adjusted_drift_data.index - offset
                        
            # Get predictions for the adjusted drift indices from the main expert_predictions file
            # We need to find the corresponding predictions by matching the adjusted indices
            drift_predictions = expert_predictions.loc[adjusted_drift_data.index]
            
            # Adjust the drift predictions indices to match the original test data range
            drift_predictions.index = drift_predictions.index + offset
            
            # Combine test predictions with drift predictions
            combined_predictions = pd.concat([test_predictions, drift_predictions])
            
            print(f"Combined predictions shape: {combined_predictions.shape}")
            
            # Save the combined predictions
            output_path = f'./test_expert_predictions_{setting}.parquet'
            combined_predictions.to_parquet(output_path)
            
            print(f"Saved: {output_path}")
            print()
            
        except Exception as e:
            print(f"Error processing {setting}: {e}")
            print()

# Get train predictions and probability of error from the main files
train_predictions = expert_predictions.loc[train_data.index]

# Save the train files
train_predictions_output_path = f'./train_expert_predictions.parquet'

train_predictions.to_parquet(train_predictions_output_path)
print(f"Saved: {train_predictions_output_path}")

print("=== COMPLETED ===")