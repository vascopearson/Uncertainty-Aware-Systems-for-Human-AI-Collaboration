import pandas as pd
import yaml
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from sklearn.metrics import roc_auc_score
#from hpo_mlp import HPO
from single_mlp import train_single_mlp


def cat_checker(data, features, cat_dict):
    """
    Ensures categorical columns have the correct categories.
    """
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    return new_data

# Load configuration file
with open('../../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

# Data loading
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

# Split data into training and validation sets
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

# Define hyperparameters for optimization
parameters = {
    'total': 100,
    'startups': 10,
    'params': {
        'hidden_layer_sizes': {'range': [[200, 50], [128, 50], [128, 20], [128, 128, 50], [128, 128, 20]]},
        'learning_rate': {'range': [1e-5, 1e-3], 'log': True},
        'weight_decay': {'range': [1e-5, 1e-3], 'log': True},
        'dropout_rate': {'range': [0.0, 0.5]},
        'num_epochs': {'range': [20, 200]},
        'batch_size': {'range': [10, 128], 'log': True},
    }
}

# Hyperparameter Optimization if the best model does not exist
#if not (os.path.exists('../../density_based_CP/feature_extraction/best_model.pth')):
#    hpo = HPO(X_train, X_val, y_train, y_val, train_w=y_train.replace([0, 1], [data_cfg['lambda'], 1]), val_w=y_val.replace([0, 1], [data_cfg['lambda'], 1]), parameters=parameters, path='../../density_based_CP/feature_extraction', patience=10)
#    hpo.initialize_optimizer()

# Train MLP by training a single MLP
if not (os.path.exists('../feature_extraction/best_model.pth')):
    print('Training MLP')
    train_single_mlp(X_train, X_val, y_train, y_val, [128,50], 0.00003096, 0.000384, 73, 0.056, '../feature_extraction', lambda_weight=data_cfg['lambda'])


# Load the trained model
model = torch.load('./best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # Set the model to evaluation mode

def output(data, model):
    """
    Generates output probabilities from the model.
    """
    model.eval()
    data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(data_tensor).cpu().numpy()
        probabilities = 1 / (1 + np.exp(-logits))  # Apply sigmoid to get probabilities
    return probabilities

# Load test data
test = data.loc[data["month"] == 7]

# Load noisy test data and encode categorical data
test_noisy = pd.read_parquet('../../drift_alert_data/test_alert_data_noisy_1.0_0.3_seed_0.parquet')
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

# Evaluate on the training set
train_preds = output(X_train, model)
train_preds = (train_preds >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_train, train_preds).ravel()
avg_cost_model = (data_cfg['lambda']*fp + fn)/(tn+fp+fn+tp)

selected_model = {}
selected_model['fpr_train'] = float(fp/(fp+tn))
selected_model['fnr_train'] = float(fn/(fn+tp))
selected_model['prev_train'] = float(y_train.mean())
selected_model['cost_train'] = float(avg_cost_model)

print(f"Training Set -- Model: {avg_cost_model:.3f}")

# Evaluate on the validation set
val_preds = output(X_val, model)
val_preds = (val_preds >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, val_preds).ravel()
avg_cost_model = (data_cfg['lambda']*fp + fn)/(tn+fp+fn+tp)

selected_model['fpr_val'] = float(fp/(fp+tn))
selected_model['fnr_val'] = float(fn/(fn+tp))
selected_model['prev_val'] = float(y_val.mean())
selected_model['cost_val'] = float(avg_cost_model)

print(f"Val Set -- Model: {avg_cost_model:.5f}")

# Evaluate on the test set
test_preds = output(X_test, model)
test_preds = (test_preds >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
avg_cost_model = (data_cfg['lambda']*fp + fn)/(tn+fp+fn+tp)

selected_model['fpr_test'] = float(fp/(fp+tn))
selected_model['fnr_test'] = float(fn/(fn+tp))
selected_model['prev_test'] = float(y_test.mean())
selected_model['cost_test'] = float(avg_cost_model)

print(f"Test Set -- Model: {avg_cost_model:.5f}")

# Evaluate on the noisy test set
noisy_test_preds = output(X_test_noisy, model)
noisy_test_preds = (noisy_test_preds >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test_noisy, noisy_test_preds).ravel()
avg_cost_model = (data_cfg['lambda']*fp + fn)/(tn+fp+fn+tp)

selected_model['fpr_test'] = float(fp/(fp+tn))
selected_model['fnr_test'] = float(fn/(fn+tp))
selected_model['prev_test'] = float(y_test.mean())
selected_model['cost_test'] = float(avg_cost_model)

print(f"Noisy Test Set -- Model: {avg_cost_model:.5f}")

# Calibration and ROC results 

# Original test set
probs_test = pd.Series(output(X_test, model).flatten(), index=X_test.index)

# Rebalance the validation set
e_c = y_val.replace([0,1], [data_cfg['lambda'],1]).mean()
reb_1 = (y_val.mean()/e_c)
reb_0 = ((1-y_val).mean()*data_cfg['lambda']/e_c)
n_min = len(y_test.loc[y_test==0])
n_max = int(n_min*reb_1/reb_0)
oversampled = pd.concat([y_test.loc[y_test == 0], y_test.loc[y_test == 1].sample(replace=True, n = n_max, random_state=42)]).index
outcomes = y_test.loc[oversampled]

# Calculate calibration curve
prob_true, prob_pred = calibration_curve(y_true = outcomes, y_prob = probs_test.loc[oversampled], strategy='quantile', n_bins = 10)
print("Calibration on the test set: ", np.mean(np.abs(prob_true - prob_pred))*100)

# Calculate the ROC-AUC for the original test set
test_roc_auc = roc_auc_score(y_true = outcomes, y_score = probs_test.loc[oversampled])
print(f"Test ROC-AUC: {test_roc_auc:.5f}")

