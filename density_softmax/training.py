import pandas as pd
import yaml
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder

import torch
import torch.optim as optim

from density_softmax_h import RealNVP
from hpo_realnvp import RealNVPHPO

def log_scale(values):
    return [np.sign(x) * np.log(np.abs(x)) if x != 0 else 0 for x in values]

def min_max_normalize(array):

    min_val = array.min()
    max_val = array.max()

    # Avoid division by zero
    if max_val != min_val:
        normalized_array = (array - min_val) / (max_val - min_val)
    else:
        normalized_array = np.zeros_like(array)  # All elements become 0 if max_val equals min_val

    return normalized_array

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    return new_data

with open('../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)
lambda_weight=data_cfg['lambda']

# Data loading
data = pd.read_parquet('../alert_data/alerts.parquet')
data = cat_checker(data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']
NUMERIC_COLS = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count', 'customer_age', 
                'days_since_request', 'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h', 
                'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'bank_months_count', 'proposed_credit_limit',
                'session_length_in_minutes', 'device_distinct_emails_8w', 'device_fraud_count']

# One-hot encoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(data[CATEGORICAL_COLS])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names(CATEGORICAL_COLS), index=data.index)

# Combine the all columns except CATEGORICAL_COLS with the one-hot encoded features
data = pd.concat([data.drop(CATEGORICAL_COLS, axis=1), encoded_cat_df], axis=1)

# Split data
train = data.loc[(data["month"].isin([3,4,5]))]
val = data.loc[data["month"] == 6]

X_train = train.drop(columns = ['fraud_bool','model_score','month'])
y_train = train['fraud_bool']

X_val = val.drop(columns = ['fraud_bool','model_score','month']) 
y_val = val['fraud_bool']

# Standardize only the numeric features based on the training data
scaler = StandardScaler()
X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
X_val[NUMERIC_COLS] = scaler.transform(X_val[NUMERIC_COLS])

# Load test data
test = data.loc[data["month"] == 7]

# Load noisy test data and encode categorical data
test_noisy = pd.read_parquet('../drift_alert_data/test_alert_data_noisy_2.0_0.5_seed_0.parquet')
encoded_cats = encoder.transform(test_noisy[CATEGORICAL_COLS])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names(CATEGORICAL_COLS), index=test_noisy.index)
test_noisy = pd.concat([test_noisy.drop(CATEGORICAL_COLS, axis=1), encoded_cat_df], axis=1)

# Define test features
X_test = test.drop(columns = ["month",'model_score', "fraud_bool"])
y_test = test["fraud_bool"]
X_test_noisy = test_noisy.drop(columns = ["month",'model_score', "fraud_bool"])
y_test_noisy = test_noisy["fraud_bool"]

# Standardize the test data
X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])
X_test_noisy[NUMERIC_COLS] = scaler.transform(X_test_noisy[NUMERIC_COLS])



#### Training density softmax ####

# MLP features
latent_features_train = pd.read_csv('../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_train_ds_features.csv', index_col=0)
latent_features_train = torch.tensor(latent_features_train.values, dtype=torch.float32)
latent_features_val = pd.read_csv(f'../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_val_ds_features.csv', index_col=0)
latent_features_val = torch.tensor(latent_features_val.values, dtype=torch.float32)

#"""
# Train RealNVP model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
realnvp_model = RealNVP(num_coupling_layers=6, input_dim=latent_features_train.shape[1], hidden_dim=128).to(device)
realnvp_optimizer = optim.Adam(realnvp_model.parameters(), lr=1e-4)
best_val_loss = float('inf')

# Training loop for RealNVP
if not (os.path.exists('../density_softmax/best_realnvp_model.pth')):
    for epoch in range(300):
        realnvp_optimizer.zero_grad()
        loss = realnvp_model.log_loss(latent_features_train)
        loss.backward()
        realnvp_optimizer.step()
        print(f'Epoch [{epoch+1}/300], Loss: {loss.item():.4f}')

        # Validation Step
        val_loss = realnvp_model.log_loss(latent_features_val).item()

        print(f'Epoch [{epoch+1}/300], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = realnvp_model
            os.makedirs('../density_softmax/', exist_ok=True)
            torch.save(best_model, os.path.join('../density_softmax/', 'best_realnvp_model.pth'))
            print("Saved best RealNVP model.")

# Load the best model for future use
realnvp_model = torch.load('../density_softmax/best_realnvp_model.pth')
realnvp_model.to(device)
realnvp_model.eval()
#"""

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not (os.path.exists('../density_softmax/best_realnvp_model.pth')):
    # Perform HPO for the RealNVP model.
    parameters = {
        'total': 100,
        'startups': 10,
        'params': {
            'num_coupling_layers': {'range': [2, 10]},
            'hidden_dim': {'range': [64, 256]},
            'learning_rate': {'range': [1e-5, 1e-3], 'log': True},
            'batch_size': {'range': [10, 128], 'log': True},
            'num_epochs': {'range': [50, 300]}
        }
    }

    hpo = RealNVPHPO(latent_features_train, latent_features_val, parameters=parameters, path='../density_softmax', patience=10)
    hpo.initialize_optimizer()

    # After HPO, load the best model
    realnvp_model = torch.load('../density_softmax/best_realnvp_model.pth')
    realnvp_model.to(device)
    realnvp_model.eval()
"""


# Train density models (RealNVP) for each expert

expert_pred = pd.read_parquet(f'../L2D/synthetic_experts/train_expert_predictions.parquet')

experts = list(expert_pred.columns)

X_train_exp = X_train.copy()
X_val_exp = X_val.copy()


for n in [1,5,20,40]:
    for seed in range(5):
        np.random.seed(seed)
        assignment_indices = np.random.choice(X_train_exp.index, size=len(X_train_exp) // n, replace=False)
        X_train_exp[f'assignment'] = 0
        X_train_exp.loc[assignment_indices, f'assignment'] = 1

        assignment_indices_val = np.random.choice(X_val_exp.index, size=len(X_val_exp) // n, replace=False)
        X_val_exp[f'assignment'] = 0
        X_val_exp.loc[assignment_indices_val, f'assignment'] = 1

        train_set_exp = X_train_exp.loc[X_train_exp[f'assignment'] == 1]
        train_set_exp = train_set_exp.drop(columns = f'assignment')
        X_train_exp = X_train_exp.drop(columns = f'assignment')

        val_set_exp = X_val_exp.loc[X_val_exp[f'assignment'] == 1]
        val_set_exp = val_set_exp.drop(columns = f'assignment')
        X_val_exp = X_val_exp.drop(columns = f'assignment')

        X_train_tensor = torch.tensor(train_set_exp.values, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(val_set_exp.values, dtype=torch.float32).to(device)
        latent_features_train = pd.read_csv('../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_train_ds_features.csv', index_col=0)
        latent_features_train = torch.tensor(latent_features_train.values, dtype=torch.float32)
        latent_features_val = pd.read_csv(f'../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_val_ds_features.csv', index_col=0)
        latent_features_val = torch.tensor(latent_features_val.values, dtype=torch.float32)

        if not (os.path.exists(f'../density_softmax/expert_density_models/n_{n}/seed_{seed}/')):
            os.makedirs(f'../density_softmax/expert_density_models/n_{n}/seed_{seed}/')

        if not (os.path.exists(f'../density_softmax/expert_density_models/n_{n}/seed_{seed}/best_realnvp_model.pth')):
            print(f"Fitting RealNVP for, TSize 1/{n}, TSeed {seed}")

            # Train RealNVP model
            best_val_loss = float('inf')
            realnvp_model = RealNVP(num_coupling_layers=6, input_dim=latent_features_train.shape[1], hidden_dim=128).to(device)
            realnvp_optimizer = optim.Adam(realnvp_model.parameters(), lr=1e-4)

            for epoch in range(300):
                realnvp_optimizer.zero_grad()
                loss = realnvp_model.log_loss(latent_features_train)
                loss.backward()
                realnvp_optimizer.step()
                print(f'Epoch [{epoch+1}/300], Loss: {loss.item():.4f}')

                # Validation Step
                val_loss = realnvp_model.log_loss(latent_features_val).item()

                print(f'Epoch [{epoch+1}/300], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

                # Save the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = realnvp_model
                    os.makedirs('../density_softmax/', exist_ok=True)
                    torch.save(best_model, os.path.join(f'../density_softmax/expert_density_models/n_{n}/seed_{seed}', 'best_realnvp_model.pth'))
                    print("Saved best RealNVP model.")

            # Load the best model for evaluation
            realnvp_model = torch.load(f'../density_softmax/expert_density_models/n_{n}/seed_{seed}/best_realnvp_model.pth')
            realnvp_model.to(device)
            realnvp_model.eval()