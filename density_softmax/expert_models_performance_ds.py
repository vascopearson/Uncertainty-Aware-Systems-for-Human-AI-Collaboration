import os
import re
import numpy as np
import pandas as pd
import pickle
import torch
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# Helper Functions 

def load_data_config():
    """Load a YAML or other config file as needed."""
    import yaml
    with open('../data/data_config.yaml', 'r') as infile:
        return yaml.safe_load(infile)

def load_test_data(alert_data, file_name=None):
    """
    If file_name is provided, load the noisy data from that file and combine it
    with the original month=7 test data. If file_name is None, return only the
    original month=7 data.
    """
    test = alert_data[alert_data['month'] == 7]
    if file_name is not None:
        test_noisy = pd.read_parquet(f'../drift_alert_data/{file_name}')
        test = pd.concat([test, test_noisy])
    return test

def get_X_test(test_df, drop_cols=('fraud_bool', 'month')):
    """Drop columns that are not features."""
    return test_df.drop(columns=list(drop_cols), errors='ignore')

def reweight_outcomes(test_df, data_cfg):
    """
    Implements the reweighting logic using the validation set (month=6),
    oversampling positives based on data_cfg['lambda'].
    Returns (index_of_oversampled, outcomes_series) based on FRAUD_BOOL.
    """
    # Read the full data again to get the validation set (month=6)
    data = pd.read_parquet('../alert_data/alerts.parquet')
    val = data[data["month"] == 6]
    y_val = val['fraud_bool']

    # Weighted average cost factor
    e_c = y_val.replace([0, 1], [data_cfg['lambda'], 1]).mean()
    reb_1 = (y_val.mean() / e_c)                         
    reb_0 = ((1 - y_val).mean() * data_cfg['lambda'] / e_c) 

    y_test = test_df['fraud_bool']
    n_min = len(y_test[y_test == 0])  # number of negative samples
    n_max = int(n_min * reb_1 / reb_0)

    # Oversample the positives
    oversampled_index = pd.concat([
        y_test[y_test == 0],
        y_test[y_test == 1].sample(replace=True, n=n_max, random_state=42)
    ]).index

    return oversampled_index, y_test.loc[oversampled_index]

def load_expert_model(expert, n, seed):
    """
    Load the previously-trained one-vs-all model (LightGBM or otherwise)
    for a given expert, n, and seed.
    """
    model_path = f"../L2D/expert_models/ova/{expert}/n_{n}/seed_{seed}/best_model.pickle"
    with open(model_path, "rb") as input_file:
        return pickle.load(input_file)

def load_realnvp_model(n, seed):
    """
    Load the pre-trained RealNVP model for a given n and seed.
    Adjust path as needed.
    """
    model_path = f"../density_softmax/expert_density_models/n_{n}/seed_{seed}/best_realnvp_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    realnvp_model = torch.load(model_path, map_location=device)
    realnvp_model.eval()
    return realnvp_model

def min_max_normalize(array):
    """Simple min-max normalization, for example."""
    mn, mx = np.min(array), np.max(array)
    if mx - mn < 1e-12:
        return np.ones_like(array) * 0.5
    return (array - mn) / (mx - mn)

# Define the logit function (inverse sigmoid)
def logit(p):
    return np.log(p / (1 - p))

def adjust_logits(logit_classifier, avg_prob_correct, density_score):
    """
    Given the raw LGBM logit (logit_classifier), the expert’s average
    probability of correctness (avg_prob_correct), and the RealNVP density
    score (density_score), produce the adjusted logit.
    """
    logit_avg_prob = logit(avg_prob_correct)  # Transform avg probability to logit
    adjusted_logit = (density_score * logit_classifier) + ((1 - density_score) * logit_avg_prob)
    return adjusted_logit

def calculate_ece(prob_true, prob_pred):
    """Calculate Expected Calibration Error (ECE) in percentage."""
    ece = np.mean(np.abs(prob_true - prob_pred)) * 100
    return ece


# Main Evaluation Routines

def evaluate_expert_models_realnvp(
    alert_data,
    data_cfg,
    n_values=[40, 20, 5, 1],
    seeds=range(5)
):
    """
    This routine:
      1. Iterates over n_values, seeds.
      2. For each noise file (or 'no_noise'), loads test data + expert predictions.
      3. Loads each expert's LGBM model + RealNVP model, adjusts logits, obtains predictions.
      4. Uses “expert correctness” as the label:
         label = (expert_pred[expert] == test['fraud_bool']).astype(int)
      5. Reweights the test distribution.
      6. Computes ROC AUC, ECE for each scenario.
    """

    # Predefine noise sets and result containers
    noise_mapping = {
        "1.0_0.3": "low",
        "1.5_0.4": "medium",
        "2.0_0.5": "high",
    }
    noise_categories = ["no_noise", "low", "medium", "high"]

    roc_results = { cat: {n: [] for n in n_values} for cat in noise_categories }
    ece_results = { cat: {n: [] for n in n_values} for cat in noise_categories }

    for n in n_values:
        for seed in seeds:

            # Load RealNVP for this (n, seed)
            realnvp_model = load_realnvp_model(n, seed)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            realnvp_model.to(device)

            # Load average correctness probabilities for each expert
            avg_probabilities_path = f'../capacity_constraints/n_{n}/seed_{seed}/expert_avg_probabilities.csv'
            avg_prob_df = pd.read_csv(avg_probabilities_path)
            expert_avg_probabilities = avg_prob_df.iloc[0].to_dict()  # dict: expert -> average prob

            # Evaluate each noise file
            drift_path = '../drift_alert_data'
            for file_name in os.listdir(drift_path):
                if not file_name.endswith('.parquet'):
                    continue
                match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
                if not match:
                    continue
                noise_str = match.group(1)  # e.g. "1.0_0.3"
                noise_type = noise_mapping[noise_str]

                # ---- (A) Load test data + expert predictions ----
                test = load_test_data(alert_data, file_name)

                # Load the test_expert_predictions_{noise_str}_seed_{...}.parquet
                expert_pred_path = f'../L2D/synthetic_experts/test_expert_predictions_{noise_str}_seed_{match.group(2)}.parquet'
                test_expert_pred = pd.read_parquet(expert_pred_path)
                test_expert_pred.index.name = 'case_id'

                # Reweight based on FRAUD_BOOL
                oversampled_idx, _ = reweight_outcomes(test, data_cfg)

                # Prepare features for LightGBM
                X_test = get_X_test(test)

                # Load features used for density estimation with RealNVP
                realnvp_features_path = f'../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_test_features_{noise_str}_seed_{match.group(2)}.csv'
                latent_df = pd.read_csv(realnvp_features_path, index_col=0)
                latent_tensor = torch.tensor(latent_df.values, dtype=torch.float32, device=device)

                # Score with RealNVP
                with torch.no_grad():
                    density_scores = realnvp_model.score_samples(latent_tensor)
                density_scores = density_scores.detach().cpu().numpy()  # to CPU
                density_scores = density_scores.astype(np.float64)

                # Load train features for density estimation and perform normalization step
                latent_features_train = pd.read_csv('../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_train_ds_features.csv', index_col=0)
                latent_features_train = torch.tensor(latent_features_train.values, dtype=torch.float32)
                density_scores_train = realnvp_model.score_samples(latent_features_train).detach().cpu().numpy().astype(np.float64)
                max_train_score = density_scores_train.max()
                density_scores = density_scores/max_train_score
                density_scores = np.exp(density_scores)
                density_scores = min_max_normalize(density_scores)

                # Evaluate each expert model
                experts_path = f'../L2D/expert_models/ova'
                experts_list = os.listdir(experts_path)
                for expert in experts_list:
                    # (B) Load the LGBM model
                    model = load_expert_model(expert, n, seed)

                    # (C) Predict raw logits with LGBM
                    lgb_logits = model.predict(X_test, raw_score=True)
                    lgb_logits = np.array(lgb_logits)

                    # (D) For each row, incorporate RealNVP + avg prob
                    avg_prob_correct = expert_avg_probabilities[expert]
                    adjusted_logits = []
                    for raw_lgb_logit, dens_score in zip(lgb_logits, density_scores):
                        new_val = adjust_logits(raw_lgb_logit, avg_prob_correct, dens_score)
                        adjusted_logits.append(new_val)
                    adjusted_logits = np.array(adjusted_logits)

                    # (E) Convert to probabilities
                    adjusted_probs = 1.0 / (1.0 + np.exp(-adjusted_logits))
                    y_pred_prob = pd.Series(adjusted_probs, index=test.index)

                    # (F) Expert correctness label
                    y_expert_correct = (test_expert_pred[expert] == test['fraud_bool']).astype(int)
                    y_true_oversampled      = y_expert_correct.loc[oversampled_idx]
                    y_pred_prob_oversampled = y_pred_prob.loc[oversampled_idx]

                    # (G) Compute ROC, ECE
                    roc_auc_reweighted = roc_auc_score(y_true_oversampled, y_pred_prob_oversampled)
                    roc_results[noise_type][n].append(roc_auc_reweighted)

                    prob_true, prob_pred = calibration_curve(
                        y_true=y_true_oversampled,
                        y_prob=y_pred_prob_oversampled,
                        strategy='quantile',
                        n_bins=10
                    )
                    ece_score = calculate_ece(prob_true, prob_pred)
                    ece_results[noise_type][n].append(ece_score)

            # ---- (H) Handle the no_noise case (just month=7 data) ----
            test_original = load_test_data(alert_data, file_name=None)
            oversampled_idx, _ = reweight_outcomes(test_original, data_cfg)

            # Load the “no_noise” expert predictions (filter by index of test_original) 
            expert_pred_path = f'../L2D/synthetic_experts/test_expert_predictions_2.0_0.5_seed_0.parquet'
            test_expert_pred = pd.read_parquet(expert_pred_path)
            test_expert_pred = test_expert_pred.loc[test_original.index]

            X_test_no_noise = get_X_test(test_original)

            # RealNVP features for no-noise case (filter by index of test_original) 
            no_noise_features_path = f'../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_test_features_2.0_0.5_seed_0.csv'
            latent_df = pd.read_csv(no_noise_features_path, index_col=0)
            latent_df = latent_df.loc[test_original.index]
            latent_tensor = torch.tensor(latent_df.values, dtype=torch.float32, device=device)
            
            # Score with RealNVP
            with torch.no_grad():
                density_scores = realnvp_model.score_samples(latent_tensor)
            density_scores = density_scores.detach().cpu().numpy()
            density_scores = density_scores.astype(np.float64)
            
            # Load train features for density estimation and perform normalization step
            latent_features_train = pd.read_csv('../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_train_ds_features.csv', index_col=0)
            latent_features_train = torch.tensor(latent_features_train.values, dtype=torch.float32)
            density_scores_train = realnvp_model.score_samples(latent_features_train).detach().cpu().numpy().astype(np.float64)
            max_train_score = density_scores_train.max()
            density_scores = density_scores/max_train_score
            density_scores = np.exp(density_scores)
            density_scores = min_max_normalize(density_scores)

            # Evaluate each expert
            for expert in os.listdir(f'../L2D/expert_models/ova'):
                model = load_expert_model(expert, n, seed)

                # Raw LGBM logits
                lgb_logits = model.predict(X_test_no_noise, raw_score=True)
                lgb_logits = np.array(lgb_logits)

                avg_prob_correct = expert_avg_probabilities[expert]
                adjusted_logits = []
                for raw_lgb_logit, dens_score in zip(lgb_logits, density_scores):
                    new_val = adjust_logits(raw_lgb_logit, avg_prob_correct, dens_score)
                    adjusted_logits.append(new_val)
                adjusted_logits = np.array(adjusted_logits)

                adjusted_probs = 1.0 / (1.0 + np.exp(-adjusted_logits))
                y_pred_prob = pd.Series(adjusted_probs, index=test_original.index)

                y_expert_correct = (test_expert_pred[expert] == test_original['fraud_bool']).astype(int)
                y_true_oversampled      = y_expert_correct.loc[oversampled_idx]
                y_pred_prob_oversampled = y_pred_prob.loc[oversampled_idx]

                roc_auc_reweighted = roc_auc_score(y_true_oversampled, y_pred_prob_oversampled)
                roc_results["no_noise"][n].append(roc_auc_reweighted)

                prob_true, prob_pred = calibration_curve(
                    y_true=y_true_oversampled,
                    y_prob=y_pred_prob_oversampled,
                    strategy='quantile',
                    n_bins=10
                )
                ece_score = calculate_ece(prob_true, prob_pred)
                ece_results["no_noise"][n].append(ece_score)

    return roc_results, ece_results


def compute_summary_and_plot_realnvp(
    roc_results,
    ece_results,
    n_values=[40, 20, 5, 1]
):
    """
    Identical to your existing compute_summary_and_plot, but referencing
    the realnvp-based results.
    """
    noise_categories = roc_results.keys()  # same set as ece_results
    summary_roc = {}
    summary_ece = {}

    for noise_setting in noise_categories:
        summary_roc[noise_setting] = {}
        summary_ece[noise_setting] = {}

        for n in n_values:
            values_roc = np.array(roc_results[noise_setting][n])
            mean_roc   = np.mean(values_roc)
            ci_roc     = stats.t.interval(0.95, len(values_roc)-1, loc=mean_roc, scale=stats.sem(values_roc))

            summary_roc[noise_setting][n] = {
                'mean': mean_roc,
                'ci_lower': ci_roc[0],
                'ci_upper': ci_roc[1]
            }

            values_ece = np.array(ece_results[noise_setting][n])
            mean_ece   = np.mean(values_ece)
            ci_ece     = stats.t.interval(0.95, len(values_ece)-1, loc=mean_ece, scale=stats.sem(values_ece))

            summary_ece[noise_setting][n] = {
                'mean': mean_ece,
                'ci_lower': ci_ece[0],
                'ci_upper': ci_ece[1]
            }

    def plot_metric_vs_n(summary_dict, metric_name, ylim, file_suffix=''):
        noise_order = ['no_noise', 'low', 'medium', 'high']
        ns = [1/40, 1/20, 1/5, 1]  
        plt.figure(figsize=(5,4))
        
        for noise_setting in noise_order:
            means = [summary_dict[noise_setting][n]['mean'] for n in [40, 20, 5, 1]]
            lows  = [summary_dict[noise_setting][n]['ci_lower'] for n in [40, 20, 5, 1]]
            ups   = [summary_dict[noise_setting][n]['ci_upper'] for n in [40, 20, 5, 1]]

            errs_lower = np.array(means) - np.array(lows)
            errs_upper = np.array(ups) - np.array(means)

            plt.errorbar(
                ns, means, 
                yerr=[errs_lower, errs_upper], 
                fmt='-o', capsize=5, label=noise_setting
            )

        plt.xlabel('Data Availability (log scale)', fontsize=14)
        plt.ylabel(metric_name, fontsize=14) 
        plt.title(f'{metric_name} vs. Data Availability \n (Distance-Aware)', fontsize=16)
        plt.xscale('log')
        plt.xticks(ns, labels=['1/40', '1/20', '1/5', '1/1'], fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(*ylim)
        plt.grid(True)
        plt.legend(loc="best", fontsize=12)
        plt.tight_layout()
        plt.savefig(f'../results/figs/expert_models/{metric_name}_vs_n_realnvp{file_suffix}.pdf', format='pdf', dpi=300)
        plt.show()

    # Plot ROC AUC
    plot_metric_vs_n(summary_roc, 'ROC AUC', ylim=(0.5, 0.65), file_suffix='')
    # Plot ECE
    plot_metric_vs_n(summary_ece, 'ECE (%)', ylim=(3, 13), file_suffix='')

    return summary_roc, summary_ece


# Main Script Start 

if __name__ == "__main__":
    import torch 

    # 1) Load config, base data
    data_cfg = load_data_config()
    alert_data = pd.read_parquet("../alert_data/alerts.parquet")

    # 2) Evaluate expert models WITH RealNVP-based logit adjustments (ALL REWEIGHTED)
    roc_results_rnvp, ece_results_rnvp = evaluate_expert_models_realnvp(
        alert_data=alert_data,
        data_cfg=data_cfg,
        n_values=[40, 20, 5, 1],
        seeds=range(5)
    )

    # 3) Summarize & Plot
    summary_roc_rnvp, summary_ece_rnvp = compute_summary_and_plot_realnvp(
        roc_results_rnvp, 
        ece_results_rnvp,
        n_values=[40, 20, 5, 1]
    )
    
    # Save summary results to pickle file
    with open('../results/expert_model_ds_summary_metrics.pkl', 'wb') as f:
        pickle.dump({
            'summary_roc': summary_roc_rnvp,
            'summary_ece': summary_ece_rnvp
        }, f)    
