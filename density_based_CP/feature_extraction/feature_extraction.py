import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import os
import re

def cat_checker(data, features, cat_dict):
    """
    Ensures categorical columns have the correct categories.
    """
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    return new_data

# Function to extract second-to-last layer features
def extract_features_from_layer(model, X, device):
    """
    Extracts features from the second-to-last layer of the model.

    Parameters:
    model (torch.nn.Module): The trained model.
    X (pd.DataFrame): Input data.
    device (torch.device): Device to run the model on.

    Returns:
    np.array: Extracted features.
    """
    model.eval()  # Set the model to evaluation mode
    X = torch.tensor(X.values, dtype=torch.float32).to(device)  # Convert input data to a tensor and move to the appropriate device

    activations = X
    # Perform forward pass through the network up to the second-to-last layer
    for layer in model.network[:-2]:  # Exclude the last layer and the sigmoid activation
        activations = layer(activations)
    
    return activations.detach().cpu().numpy()  # Move the activations back to CPU and convert to numpy array

# Load the trained MLP model
model = torch.load('../../density_based_CP/feature_extraction/best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load configuration file
with open('../../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

# Load data
data = pd.read_parquet('../../alert_data/alerts.parquet')
data = cat_checker(data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

# Define column names from configuration
LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']
NUMERIC_COLS = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count', 'customer_age', 
                'days_since_request', 'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h', 
                'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'bank_months_count', 'proposed_credit_limit',
                'session_length_in_minutes', 'device_distinct_emails_8w', 'device_fraud_count']

# One-hot encoding for categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(data[CATEGORICAL_COLS])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names(CATEGORICAL_COLS), index=data.index)

# Combine all columns except CATEGORICAL_COLS with the one-hot encoded features
data = pd.concat([data.drop(CATEGORICAL_COLS, axis=1), encoded_cat_df], axis=1)

# Split data into training, validation, and test sets
train = data.loc[(data["month"].isin([3,4,5,6]))]
test = data.loc[data["month"] == 7]
train_density_softmax = data.loc[(data["month"].isin([3,4,5]))]
val_density_softmax = data.loc[(data["month"].isin([6]))]

# Define train features and labels
X_train = train.drop(columns = ['fraud_bool','model_score','month'])
y_train = train['fraud_bool']
X_train_ds = train_density_softmax.drop(columns = ['fraud_bool','model_score','month'])
y_train_ds = train_density_softmax['fraud_bool']
X_val_ds = val_density_softmax.drop(columns = ['fraud_bool','model_score','month'])
y_val_ds = val_density_softmax['fraud_bool']

# Standardize only the numeric features based on the training data
scaler = StandardScaler()
X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
scaler_ds_train = StandardScaler()
X_train_ds[NUMERIC_COLS] = scaler.fit_transform(X_train_ds[NUMERIC_COLS])
scaler_ds_val = StandardScaler()
X_val_ds[NUMERIC_COLS] = scaler.fit_transform(X_val_ds[NUMERIC_COLS])

# Extract features from the second-to-last layer
features_train = extract_features_from_layer(model, X_train, 'cpu')
features_train_ds = extract_features_from_layer(model, X_train_ds, 'cpu')
features_val_ds = extract_features_from_layer(model, X_val_ds, 'cpu')

# Convert to DataFrame
features_train_df = pd.DataFrame(features_train, index=train.index, 
                                      columns=[f'feature_{i+1}' for i in range(features_train.shape[1])])
features_train_ds_df = pd.DataFrame(features_train_ds, index=train_density_softmax.index, 
                                      columns=[f'feature_{i+1}' for i in range(features_train_ds.shape[1])])
features_val_ds_df = pd.DataFrame(features_val_ds, index=val_density_softmax.index, 
                                      columns=[f'feature_{i+1}' for i in range(features_val_ds.shape[1])])

# Save the extracted features
os.makedirs('../../density_based_CP/feature_extraction/processed_data', exist_ok=True)
features_train_df.to_csv('../../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_train_features.csv', index=True)
features_train_ds_df.to_csv('../../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_train_ds_features.csv', index=True)
features_val_ds_df.to_csv('../../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_val_ds_features.csv', index=True)

# Loop through files in the drift_alert_data directory
for file_name in os.listdir('../../drift_alert_data'):
    if file_name.endswith('.parquet'):
        match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
        end_name = f"{match.group(1)}_seed_{match.group(2)}"

        # Load noisy test data
        test_noisy = pd.read_parquet(f'../../drift_alert_data/{file_name}')
        encoded_cats = encoder.transform(test_noisy[CATEGORICAL_COLS])
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names(CATEGORICAL_COLS), index=test_noisy.index)
        test_noisy = pd.concat([test_noisy.drop(CATEGORICAL_COLS, axis=1), encoded_cat_df], axis=1)

        # Define test features and labels
        X_test = test.drop(columns = ['fraud_bool','model_score','month']) 
        y_test = test['fraud_bool']
        X_test_noisy = test_noisy.drop(columns = ["month",'model_score', "fraud_bool"])
        y_test_noisy = test_noisy["fraud_bool"]

        # Standardize only the numeric features based on the training data
        X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])
        X_test_noisy[NUMERIC_COLS] = scaler.transform(X_test_noisy[NUMERIC_COLS])
        X_test = pd.concat([X_test, X_test_noisy])

        # Extract features from the second-to-last layer
        features_test = extract_features_from_layer(model, X_test, 'cpu')

        # Convert to DataFrame
        features_test_df = pd.DataFrame(features_test, index=X_test.index, 
                                              columns=[f'feature_{i+1}' for i in range(features_test.shape[1])])

        # Save the extracted features
        features_test_df.to_csv(f'../../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_test_features_{end_name}.csv', index=True)

print("Features extracted and saved successfully.")