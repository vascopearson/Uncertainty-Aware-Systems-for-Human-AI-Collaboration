import pandas as pd
import yaml
import numpy as np
import pickle
import os
import re
import torch


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

def min_max_normalize(array):

    min_val = array.min()
    max_val = array.max()

    # Handle the case where max_val equals min_val to avoid division by zero
    if max_val != min_val:
        normalized_array = (array - min_val) / (max_val - min_val)
    else:
        normalized_array = np.zeros_like(array)  # All elements become 0 if max_val equals min_val

    return normalized_array

# Define the logit function (inverse sigmoid)
def logit(p):
    return np.log(p / (1 - p))

# Function to adjust logits based on density score and average correctness probabilities
def adjust_logits(logit_classifier, avg_prob, density_score):
    logit_avg_prob = logit(avg_prob)  # Transform avg probability to logit
    adjusted_logit = (density_score * logit_classifier) + ((1 - density_score) * logit_avg_prob)
    return adjusted_logit

with open('../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)
lambda_weight=data_cfg['lambda']


for n in [40,20,5,1]:
    for seed in range(5):
        print(f"n: {n}, seed: {seed}")
        
        avg_prob_df = pd.read_csv(f'../capacity_constraints/n_{n}/seed_{seed}/expert_avg_probabilities.csv')
        expert_avg_probabilities = avg_prob_df.iloc[0].to_dict()

        # Get predictions on the different test sets
        for file_name in os.listdir('../drift_alert_data'):
            if file_name.endswith('.parquet'):
                match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
                #if match.group(2) != '0':
                #    continue
                end_name = f"{match.group(1)}_seed_{match.group(2)}"
                print(end_name)


                # Load the best LightGBM model
                with open(f'../L2D/classifier_h/selected_model/best_model.pickle', 'rb') as infile:
                    lgb_model = pickle.load(infile)

                # Load the initial score used in LightGBM
                with open(f'../L2D/classifier_h/selected_model/model_properties.yaml', 'r') as infile:
                    lgb_model_properties = yaml.safe_load(infile)

                init_score = lgb_model_properties['init_score']

                # Load RealNVP model
                realnvp_model = torch.load('../density_softmax/best_realnvp_model.pth')
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                realnvp_model.to(device)
                realnvp_model.eval()

                # Load test data
                data = pd.read_parquet('../alert_data/alerts.parquet')
                data = cat_checker(data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

                test = data.loc[(data["month"] == 7)]

                X_test = test.drop(columns = ["month", 'model_score', "fraud_bool"])
                X_test_exp = test.drop(columns = ["month", "fraud_bool"])
                y_test = test["fraud_bool"]

                test_noisy = pd.read_parquet(f'../drift_alert_data/{file_name}')

                X_test_noisy = test_noisy.drop(columns = ["month", 'model_score', "fraud_bool"])
                X_test_noisy_exp = test_noisy.drop(columns = ["month", "fraud_bool"])
                y_test_noisy = test_noisy["fraud_bool"]

                X_combined_lgbm = pd.concat([X_test, X_test_noisy])
                X_combined_lgbm_exp = pd.concat([X_test_exp, X_test_noisy_exp])
                y_combined_lgbm = pd.concat([y_test, y_test_noisy])
                X_origin_lgbm = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_test_noisy))])

                # Extract latent features
                latent_features_combined = pd.read_csv(f'../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_test_features_{end_name}.csv', index_col=0)
                latent_features_combined = torch.tensor(latent_features_combined.values, dtype=torch.float32)
                latent_features_train = pd.read_csv('../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_train_ds_features.csv', index_col=0)
                latent_features_train = torch.tensor(latent_features_train.values, dtype=torch.float32)

                # Get the density scores from the RealNVP model                
                density_scores = realnvp_model.score_samples(latent_features_combined)
                density_scores = density_scores.detach().cpu().numpy()  # Move to CPU and convert to numpy array
                density_scores = density_scores.astype(np.float64)
                max_density = density_scores.max()
                density_scores_train = realnvp_model.score_samples(latent_features_train).detach().cpu().numpy().astype(np.float64)
                max_train_score = density_scores_train.max()
                density_scores = density_scores/max_train_score
                density_scores = np.exp(density_scores)
                density_scores = min_max_normalize(density_scores)

                # Get the logits or raw scores from the LightGBM model
                lgb_logits = lgb_model.predict(X_combined_lgbm, raw_score=True) + init_score

                # Adjust logits using the density scores from the RealNVP model
                adjusted_logits = lgb_logits*density_scores

                # Convert adjusted logits to probabilities using the sigmoid function
                adjusted_probabilities = 1 / (1 + np.exp(-(adjusted_logits)))
                lgb_probs = 1 / (1 + np.exp(-(lgb_logits)))

                # Predictions based on these adjusted probabilities
                adjusted_predictions = (adjusted_probabilities >= 0.5).astype(int)

                # Define predictions dataframe
                preds = pd.DataFrame(index = X_combined_lgbm.index, columns = os.listdir(f'../L2D/expert_models/ova/'))

                preds.loc[:,'classifier_h'] = np.maximum(adjusted_probabilities,  1-adjusted_probabilities)
                preds.loc[:,'classifier_h_confidence'] = adjusted_probabilities

                for expert in os.listdir(f'../L2D/expert_models/ova'):

                    with open(f"../L2D/expert_models/ova/{expert}/n_{n}/seed_{seed}/best_model.pickle", "rb") as input_file:
                        model = pickle.load(input_file)

                    # Load RealNVP model
                    realnvp_model = torch.load(f'../density_softmax/expert_density_models/n_{n}/seed_{seed}/best_realnvp_model.pth')
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    realnvp_model.to(device)
                    realnvp_model.eval()

                    # Get the density scores from the RealNVP model                
                    density_scores = realnvp_model.score_samples(latent_features_combined)
                    density_scores = density_scores.detach().cpu().numpy()
                    density_scores = density_scores.astype(np.float64)
                    max_density = density_scores.max()
                    density_scores_train = realnvp_model.score_samples(latent_features_train).detach().cpu().numpy().astype(np.float64)
                    max_train_score = density_scores_train.max()
                    density_scores = density_scores/max_train_score
                    density_scores = np.exp(density_scores)
                    density_scores = min_max_normalize(density_scores)

                    # Get the logits or raw scores from the LightGBM model
                    lgb_expert_logits = model.predict(X_combined_lgbm_exp, raw_score=True)

                    # Transform each expert's average probability of correctness to logits
                    avg_prob = expert_avg_probabilities[expert]
                
                    # Adjust the logits using the density scores and the logit of average probability of correctness
                    adjusted_expert_logits = [
                        adjust_logits(logit_classifier, avg_prob, density_score)
                        for logit_classifier, density_score in zip(lgb_expert_logits, density_scores)
                    ]
                    adjusted_expert_logits = np.array(adjusted_expert_logits)
                
                    # Convert adjusted logits to probabilities using the sigmoid function
                    adjusted_expert_probabilities = 1 / (1 + np.exp(-(adjusted_expert_logits)))
                
                    # Predictions based on these adjusted probabilities
                    adjusted_predictions_exp = (adjusted_expert_probabilities >= 0.5).astype(int)

                    preds.loc[:, expert] = adjusted_expert_probabilities

                os.makedirs(f'../L2D/deferral/l2d_ds_predictions/n_{n}/seed_{seed}', exist_ok=True)

                with open(f"../L2D/deferral/l2d_ds_predictions/n_{n}/seed_{seed}/ova_{end_name}.pkl", "wb") as out_file:
                    pickle.dump(preds, out_file)