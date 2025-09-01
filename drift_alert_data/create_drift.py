import pandas as pd
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load YAML configuration file
with open('../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

# Load data
alert_data = pd.read_parquet('../alert_data/alerts.parquet')
print(alert_data['fraud_bool'].value_counts())

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    return new_data

alert_data = cat_checker(alert_data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

# Define numeric columns
NUMERIC_COLS = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count', 'customer_age', 
                'days_since_request', 'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h', 
                'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'bank_months_count', 'proposed_credit_limit',
                'session_length_in_minutes', 'device_distinct_emails_8w', 'device_fraud_count', 'month']

test = alert_data.loc[alert_data["month"] == 7]


# NUMERIC DISCRETE are all bounded below
NUMERIC_DISCRETE = ['prev_address_months_count', 'current_address_months_count', 'zip_count_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'bank_months_count', 'device_distinct_emails_8w', 'device_fraud_count']
#days_since_request, velocity_6h, velocity_24h, velocity_4w, proposed_credit_limit, session_length_in_minutes (all are bounded below except intended_balcon_amount)
NUMERIC_CONTINUOUS = ['days_since_request', 'intended_balcon_amount', 'velocity_6h', 'velocity_24h', 'velocity_4w', 'proposed_credit_limit', 'session_length_in_minutes']
FLAG_COLS = ['email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'has_other_cards', 'foreign_request', 'keep_alive_session']
NUMERIC_CONTINUOUS_LIMITED = ['name_email_similarity', 'model_score']
NUMERIC_DISCRETE_LIMITED = ['income', 'customer_age', 'month']


# Noise parameters:
noise_params = [[1.0, 0.3], [1.5, 0.4], [2.0, 0.5]]

# Create noisy test sets
for seed in range(5):
    # Set seed
    np.random.seed(seed)
    
    # Load test data
    test = alert_data.loc[alert_data["month"] == 7]

    for noise_std_dev, flip_prob in noise_params:

        # Add Gaussian noise to NUMERIC_CONTINUOUS columns
        scaler = StandardScaler()
        test[NUMERIC_CONTINUOUS] = scaler.fit_transform(test[NUMERIC_CONTINUOUS])
        for col in NUMERIC_CONTINUOUS:
            noise = np.random.normal(0, noise_std_dev, size=test[col].shape)
            test.loc[:,col] += noise
        test[NUMERIC_CONTINUOUS] = scaler.inverse_transform(test[NUMERIC_CONTINUOUS])
        NUMERIC_CONTINUOUS_EXCEPT_BALCON = [col for col in NUMERIC_CONTINUOUS if col != 'intended_balcon_amount']
        test[NUMERIC_CONTINUOUS_EXCEPT_BALCON] = test[NUMERIC_CONTINUOUS_EXCEPT_BALCON].clip(lower=0)

        # Add Gaussian noise to NUMERIC_CONTINUOUS_LIMITED columns
        for col in NUMERIC_CONTINUOUS_LIMITED:
            noise = np.random.normal(0, noise_std_dev, size=test[col].shape)
            test.loc[:,col] += noise
            test.loc[:,col] = test[col].clip(0, 1)

        # Add Gaussian noise to NUMERIC_DISCRETE columns
        scaler_discrete = StandardScaler()
        test[NUMERIC_DISCRETE] = scaler_discrete.fit_transform(test[NUMERIC_DISCRETE])
        for col in NUMERIC_DISCRETE:
            noise = np.random.normal(0, noise_std_dev, size=test[col].shape)
            test.loc[:,col] += noise
        test[NUMERIC_DISCRETE] = scaler_discrete.inverse_transform(test[NUMERIC_DISCRETE])
        for col in NUMERIC_DISCRETE:
            test.loc[:,col] = np.round(test[col])
            if col in ['prev_address_months_count', 'device_distinct_emails_8w']:
                test.loc[:,col] = test[col].clip(lower=-1)
            else:
                test.loc[:,col] = test[col].clip(lower=0)

        # Add Gaussian noise to NUMERIC_DISCRETE_LIMITED columns
        min_vals = test[NUMERIC_DISCRETE_LIMITED].min()
        max_vals = test[NUMERIC_DISCRETE_LIMITED].max()
        scaler_discrete_limited = StandardScaler()
        test[NUMERIC_DISCRETE_LIMITED] = scaler_discrete_limited.fit_transform(test[NUMERIC_DISCRETE_LIMITED])
        for col in NUMERIC_DISCRETE_LIMITED:
            noise = np.random.normal(0, noise_std_dev, size=test[col].shape)
            test.loc[:,col] += noise
        test[NUMERIC_DISCRETE_LIMITED] = scaler_discrete_limited.inverse_transform(test[NUMERIC_DISCRETE_LIMITED])
        for col in NUMERIC_DISCRETE_LIMITED:
            if col == 'customer_age':
                test.loc[:,col] = np.round(test[col] / 10) * 10  # Round to the nearest 10
            else:
                test.loc[:,col] = np.round(test[col])
            if col == 'month':
                test.loc[:,col] = test[col].clip(lower=min_vals[col])
            else:
                test.loc[:,col] = test[col].clip(lower=min_vals[col], upper=max_vals[col])


        # Randomly flip FLAG_COLS
        for col in FLAG_COLS:
            mask = np.random.rand(len(test)) < flip_prob
            test.loc[mask, col] = 1 - test.loc[mask, col]

        # Randomly flip categorical labels
        CATEGORICAL_COLS = data_cfg['data_cols']['categorical']
        for col in CATEGORICAL_COLS:
            unique_values = test[col].unique()
            mask = np.random.rand(len(test)) < flip_prob
            test.loc[mask, col] = np.random.choice(unique_values, size=mask.sum())

        # Change index so that it is different from original test data
        test.index = test.index+1000000

        # Randomly sample 25% of the instances (so that the test set will have 20% noisy instances)
        sampled_test = test.sample(frac=0.25, random_state=42)

        # Save the noisy data to a parquet file
        sampled_test.to_parquet(f'../drift_alert_data/test_alert_data_noisy_{noise_std_dev}_{flip_prob}_seed_{seed}.parquet', index=True)

print("Noisy test data saved.")
