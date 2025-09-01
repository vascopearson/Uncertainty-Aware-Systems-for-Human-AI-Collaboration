import pandas as pd
import numpy as np
import os
import re
import yaml

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

with open('../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

os.makedirs('../capacity_constraints', exist_ok=True)

# Load data (contains data from month 4-8)
alert_data = pd.read_parquet("../alert_data/alerts.parquet")
alert_data = cat_checker(alert_data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

batch_size = 100  # Define the batch size
deferral_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

# Create batch vectors for the different test sets
for file_name in os.listdir('../drift_alert_data'):
    if file_name.endswith('.parquet'):
        match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
        end_name = f"{match.group(1)}"
        if match.group(2) == '0':

            # Load test data
            test = alert_data[alert_data['month'] == 7]
            test_noisy = pd.read_parquet(f'../drift_alert_data/{file_name}')
            test = pd.concat([test, test_noisy])

            # Create batch vector
            indices = test.index.to_numpy()
            np.random.shuffle(indices)

            num_batches = len(indices) // batch_size + (1 if len(indices) % batch_size != 0 else 0)
            batch_vector = np.zeros(len(indices), dtype=int)

            for i in range(num_batches):
                if i == num_batches - 1:
                    batch_vector[i * batch_size:] = i
                else:
                    batch_vector[i * batch_size:(i + 1) * batch_size] = i

            # Create a DataFrame with only the batch column and the index
            batch_df = pd.DataFrame({
                'index': test.index,
                'batch': pd.Categorical(batch_vector)
            }).set_index('index')

            batch_df = batch_df.sort_index()
            batch_df.to_csv(f"../capacity_constraints/batch_{end_name}.csv")


# Create human capacity matrices H for different number of experts and deferral rates
for num_experts in range(1, 6):
    for deferral_rate in deferral_rates:
        # Initialize the capacity matrix H
        H = np.zeros((num_batches, num_experts))
        
        # Calculate the number of instances each expert can handle per batch
        capacity_per_expert_per_batch = int((batch_size * deferral_rate) / num_experts)
        
        H.fill(capacity_per_expert_per_batch)
        H_df = pd.DataFrame(H, columns=[f"Expert_{i+1}" for i in range(num_experts)])
        H_df.to_csv(f"../capacity_constraints/H_{num_experts}_experts_{int(deferral_rate * 100)}_percent.csv", index=False)
