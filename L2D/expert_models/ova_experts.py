import numpy as np
import pandas as pd
import os
import yaml
import pickle
import re
import hpo

os.makedirs('../../results/figs/expert_models', exist_ok=True)

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

def sig(x):
    return 1/(1+np.exp(-x))

def output(data, model, init_score):
    return sig(model.predict(data,raw_score=True) + init_score)


with open('../../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

# Load data (contains data from month 4-8)
alert_data = pd.read_parquet("../../alert_data/alerts.parquet")
alert_data = cat_checker(alert_data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

# Split into training and validation sets
train_exp = alert_data[alert_data['month'].isin([3, 4, 5])]
val_exp = alert_data[alert_data['month'] == 6]

# Load expert predictions
expert_pred = pd.read_parquet(f'../../L2D/synthetic_experts/train_expert_predictions.parquet')
expert_pred.index.name = 'case_id'

# Combine data with expert predictions
train_exp = train_exp.merge(expert_pred, on='case_id', how='left')
val_exp = val_exp.merge(expert_pred, on='case_id', how='left')


experts = list(expert_pred.columns)

for expert in experts:
    for n in [40,20,5,1]:
        for seed in range(5):
            np.random.seed(seed)
            
            # Randomly pick ~1/n of train_exp rows
            assignment_indices = np.random.choice(train_exp.index, size=len(train_exp) // n, replace=False)
            train_exp[f'assignment_{expert}'] = 0
            train_exp.loc[assignment_indices, f'assignment_{expert}'] = 1

            # Randomly pick ~1/n of val_exp rows
            assignment_indices_val = np.random.choice(val_exp.index, size=len(val_exp) // n, replace=False)
            val_exp[f'assignment_{expert}'] = 0
            val_exp.loc[assignment_indices_val, f'assignment_{expert}'] = 1

            # Subset the training data only where assignment_{expert} == 1
            train_set_exp = train_exp.loc[train_exp[f'assignment_{expert}'] == 1]
            train_set_exp = cat_checker(train_set_exp, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])
            train_exp = train_exp.drop(columns = f'assignment_{expert}')
            
            # Subset the validation data
            val_set_exp = val_exp.loc[val_exp[f'assignment_{expert}'] == 1]
            val_set_exp = cat_checker(val_set_exp, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])
            val_exp = val_exp.drop(columns = f'assignment_{expert}')

            # Create sample weights based on fraud_bool
            train_w = train_set_exp['fraud_bool'].replace([0,1],[data_cfg['lambda'],1])
            val_w = val_set_exp['fraud_bool'].replace([0,1],[data_cfg['lambda'],1])

            # Drop columns we don’t want to feed into the model
            train_x = train_set_exp.drop(columns = ['fraud_bool', 'month', f'assignment_{expert}']+list(expert_pred.columns)) # The expert models are trained taking into account the model_score!
            val_x = val_set_exp.drop(columns = ['fraud_bool', 'month', f'assignment_{expert}']+list(expert_pred.columns))

            # The y-label is 1 if expert’s prediction matches fraud_bool, else 0
            train_y = (train_set_exp[expert] == train_set_exp['fraud_bool']).astype(int)
            val_y = (val_set_exp[expert] == val_set_exp['fraud_bool']).astype(int)

            if not (os.path.exists(f'../../L2D/expert_models/ova/{expert}/n_{n}/seed_{seed}/')):
                os.makedirs(f'../../L2D/expert_models/ova/{expert}/n_{n}/seed_{seed}/')

            if not (os.path.exists(f'../../L2D/expert_models/ova/{expert}/n_{n}/seed_{seed}/best_model.pickle')):
                opt = hpo.HPO(train_x,val_x,train_y,val_y,train_w, val_w, method = 'TPE', path = f'../../L2D/expert_models/ova/{expert}/n_{n}/seed_{seed}/')
                opt.initialize_optimizer(data_cfg['data_cols']['categorical'], 10)


for n in [40,20,5,1]:
    for seed in range(5):

        # Get predictions on the different test sets
        for file_name in os.listdir('../../drift_alert_data'):
            if file_name.endswith('.parquet'):
                match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
                end_name = f"{match.group(1)}_seed_{match.group(2)}"

                # Load test data
                test = alert_data[alert_data['month'] == 7]
                test_noisy = pd.read_parquet(f'../../drift_alert_data/{file_name}')
                test = pd.concat([test, test_noisy])

                X_test = test.drop(columns = ['fraud_bool','model_score','month'])

                # Load classifier_h
                with open(f"../../L2D/classifier_h/selected_model/best_model.pickle", 'rb') as fp:
                    classifier_h = pickle.load(fp)
                with open(f"../../L2D/classifier_h/selected_model/model_properties.yaml", 'r') as fp:
                    classifier_h_properties = yaml.safe_load(fp)

                # Make predictions with classifier_h
                h_preds = output(X_test, classifier_h, classifier_h_properties['init_score'])

                test = cat_checker(test, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])
                test = alert_data[alert_data['month'] == 7]
                test_noisy = pd.read_parquet(f'../../drift_alert_data/{file_name}')
                test = pd.concat([test, test_noisy])
                X_test = test.drop(columns = ['month','fraud_bool'])

                # Create a DataFrame to store expert predictions
                preds = pd.DataFrame(index = test.index, columns = os.listdir(f'../../L2D/expert_models/ova/'))
                for expert in os.listdir(f'../../L2D/expert_models/ova'):

                    name = str(n) + str(seed) + expert + end_name

                    # Load the previously-trained model for this expert
                    with open(f"../../L2D/expert_models/ova/{expert}/n_{n}/seed_{seed}/best_model.pickle", "rb") as input_file:
                        model = pickle.load(input_file)

                    preds.loc[:, expert] = model.predict_proba(X_test)[:,1]

                preds.loc[:,'classifier_h'] = np.maximum(h_preds,  1-h_preds)
                preds.loc[:,'classifier_h_confidence'] = h_preds


                os.makedirs(f'../../L2D/deferral/l2d_predictions/n_{n}/seed_{seed}', exist_ok=True)

                with open(f"../../L2D/deferral/l2d_predictions/n_{n}/seed_{seed}/ova_{end_name}.pkl", "wb") as out_file:
                    pickle.dump(preds, out_file)



for n in [40,20,5,1]:
    for seed in range(5):
        # Get experts average predicted probability of correctness on the training data
        train_expert_predictions = pd.read_parquet(f'../../L2D/synthetic_experts/train_expert_predictions.parquet')
        train = alert_data[alert_data['month'].isin([3, 4, 5, 6])]
        X_train = train.drop(columns = ['month','fraud_bool'])
    
        # Dictionary to store the average probabilities
        expert_avg_probabilities = {}
    
        for expert in os.listdir(f'../../L2D/expert_models/ova'):
            with open(f"../../L2D/expert_models/ova/{expert}/n_{n}/seed_{seed}/best_model.pickle", "rb") as input_file:
                model = pickle.load(input_file)
            preds = model.predict_proba(X_train)[:,1]
            expert_avg_probabilities[expert] = preds.mean()
            print("TrDataSize: {n}, Seed: {seed}\n", expert_avg_probabilities)
    
        # Save probabilities to a CSV file
        os.makedirs(f'../../capacity_constraints/n_{n}/seed_{seed}', exist_ok=True)
        expert_avg_probabilities_df = pd.DataFrame(expert_avg_probabilities, index=[0])
        expert_avg_probabilities_df.to_csv(f'../../capacity_constraints/n_{n}/seed_{seed}/expert_avg_probabilities.csv', index=False)