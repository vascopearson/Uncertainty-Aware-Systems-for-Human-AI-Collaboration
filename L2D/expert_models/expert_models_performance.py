import os
import re
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# Helper Functions 

def load_data_config():
    """Load a YAML or other config file as needed."""
    import yaml
    with open('../../data/data_config.yaml', 'r') as infile:
        return yaml.safe_load(infile)

def load_test_data(alert_data, file_name=None):
    """
    If file_name is provided, load the noisy data from that file and combine it
    with the original month=7 test data. If file_name is None, return only the
    original month=7 data.
    """
    test = alert_data[alert_data['month'] == 7]
    if file_name is not None:
        test_noisy = pd.read_parquet(f'../../drift_alert_data/{file_name}')
        test = pd.concat([test, test_noisy])
    return test

def get_X_test(test_df, drop_cols=('fraud_bool', 'month')):
    """Drop columns that are not features."""
    return test_df.drop(columns=list(drop_cols), errors='ignore')

def reweight_outcomes(test_df, data_cfg):
    """
    Implements the reweighting logic using the validation set (month=6).
    Oversamples positives based on the ratio derived from data_cfg['lambda'].
    Returns (index_of_oversampled, outcomes_series).
    """
    # Get the validation set (month=6)
    data = pd.read_parquet('../../alert_data/alerts.parquet')
    val = data[data["month"] == 6]
    y_val = val['fraud_bool']

    # Weighted average cost factor
    e_c = y_val.replace([0, 1], [data_cfg['lambda'], 1]).mean()
    reb_1 = (y_val.mean() / e_c)                       # reweight for positives
    reb_0 = ((1 - y_val).mean() * data_cfg['lambda'] / e_c)  # reweight for negatives

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
    Load the previously-trained one-vs-all model for a given expert, n, and seed.
    """
    model_path = f"../../L2D/expert_models/ova/{expert}/n_{n}/seed_{seed}/best_model.pickle"
    with open(model_path, "rb") as input_file:
        return pickle.load(input_file)

def calculate_ece(prob_true, prob_pred):
    """Calculate Expected Calibration Error (ECE) in percentage."""
    ece = np.mean(np.abs(prob_true - prob_pred)) * 100
    return ece



# Main Evaluation Routines

def evaluate_expert_models(
    alert_data,
    data_cfg,
    n_values=[40, 20, 5, 1],
    seeds=range(5)
):
    """
    Main routine that:
      1. Iterates over n_values, seeds.
      2. For each noise file (or 'no_noise'), loads test data.
      3. Loads each expert's model and computes predictions (ROC AUC, ECE)
         on the RE-WEIGHTED distribution.
      4. Stores results in dictionaries.
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

    # For each data-splitting scenario
    for n in n_values:
        for seed in seeds:

            # Evaluate each noise file
            for file_name in os.listdir('../../drift_alert_data'):
                if not file_name.endswith('.parquet'):
                    continue
                match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
                if not match:
                    continue
                noise_type = noise_mapping[match.group(1)]

                # ---- (A) Load test data (with noise) and expert predictions----
                test = load_test_data(alert_data, file_name)

                X_test = get_X_test(test)
                oversampled_idx, outcomes = reweight_outcomes(test, data_cfg)
                
                # Load expert predictions
                expert_pred = pd.read_parquet(f'../../L2D/synthetic_experts/test_expert_predictions_{match.group(1)}_seed_{match.group(2)}.parquet')
                expert_pred.index.name = 'case_id'

                # ---- (B) Evaluate each expert model on this test set ----
                experts_path = f'../../L2D/expert_models/ova'
                experts_list = os.listdir(experts_path)
                for expert in experts_list:
                    model = load_expert_model(expert, n, seed)

                    # Model predictions
                    y_pred_prob_array = model.predict_proba(X_test)[:, 1]
                    y_pred_prob = pd.Series(y_pred_prob_array, index=test.index)
                    
                    # Grab reweighted portion for the metric
                    expert_correctness = (expert_pred[expert] == test['fraud_bool']).astype(int)
                    y_true_oversampled = expert_correctness.loc[oversampled_idx]
                    y_pred_prob_oversampled = y_pred_prob.loc[oversampled_idx]  

                    # ROC AUC on oversampled distribution
                    roc_auc_reweighted = roc_auc_score(y_true_oversampled, y_pred_prob_oversampled)
                    roc_results[noise_type][n].append(roc_auc_reweighted)

                    # ECE on oversampled distribution
                    prob_true, prob_pred = calibration_curve(
                        y_true=y_true_oversampled,
                        y_prob=y_pred_prob_oversampled,
                        strategy='quantile',
                        n_bins=10
                    )
                    ece_score = calculate_ece(prob_true, prob_pred)
                    ece_results[noise_type][n].append(ece_score)

            # ---- (C) Handle the no_noise case (just the real test data from month=7) ----
            test_original = load_test_data(alert_data, file_name=None)  # Just month=7

            X_test_no_noise = get_X_test(test_original)
            oversampled_idx, outcomes = reweight_outcomes(test_original, data_cfg)
            
            # Load expert predictions
            expert_pred = pd.read_parquet(f'../../L2D/synthetic_experts/test_expert_predictions_2.0_0.5_seed_0.parquet')
            expert_pred = expert_pred.loc[test_original.index]
            expert_pred.index.name = 'case_id'

            for expert in os.listdir(f'../../L2D/expert_models/ova'):
                model = load_expert_model(expert, n, seed)

                y_pred_prob_array = model.predict_proba(X_test_no_noise)[:, 1]
                y_pred_prob = pd.Series(y_pred_prob_array, index=test_original.index)

                expert_correctness = (expert_pred[expert] == test_original['fraud_bool']).astype(int)
                y_true_oversampled = expert_correctness.loc[oversampled_idx]
                y_pred_prob_oversampled = y_pred_prob.loc[oversampled_idx]  

                # ROC AUC on oversampled distribution
                roc_auc_reweighted = roc_auc_score(y_true_oversampled, y_pred_prob_oversampled)
                roc_results["no_noise"][n].append(roc_auc_reweighted)

                # ECE on oversampled distribution
                prob_true, prob_pred = calibration_curve(
                    y_true=y_true_oversampled,
                    y_prob=y_pred_prob_oversampled,
                    strategy='quantile',
                    n_bins=10
                )
                ece_score = calculate_ece(prob_true, prob_pred)
                ece_results["no_noise"][n].append(ece_score)

    # Return the two result dictionaries
    return roc_results, ece_results


def compute_summary_and_plot(
    roc_results,
    ece_results,
    n_values=[40, 20, 5, 1]
):
    """
    1. Compute mean & confidence intervals for ROC/ECE across seeds.
    2. Plot the results as in your original code.
    """

    noise_categories = roc_results.keys()  # same set as ece_results
    summary_roc = {}
    summary_ece = {}

    for noise_setting in noise_categories:
        summary_roc[noise_setting] = {}
        summary_ece[noise_setting] = {}

        for n in n_values:
            # Reweighted ROC
            values_roc = np.array(roc_results[noise_setting][n])
            mean_roc   = np.mean(values_roc)
            ci_roc     = stats.t.interval(0.95, len(values_roc)-1, loc=mean_roc, scale=stats.sem(values_roc))

            summary_roc[noise_setting][n] = {
                'mean': mean_roc,
                'ci_lower': ci_roc[0],
                'ci_upper': ci_roc[1]
            }

            # Reweighted ECE
            values_ece = np.array(ece_results[noise_setting][n])
            mean_ece   = np.mean(values_ece)
            ci_ece     = stats.t.interval(0.95, len(values_ece)-1, loc=mean_ece, scale=stats.sem(values_ece))

            summary_ece[noise_setting][n] = {
                'mean': mean_ece,
                'ci_lower': ci_ece[0],
                'ci_upper': ci_ece[1]
            }

    # --- Plotting ---
    def plot_metric_vs_n(summary_dict, metric_name, ylim, file_suffix=''):
        """
        General function to plot a metric (ROC or ECE) vs. data availability (1/n).
        summary_dict is like summary_roc or summary_ece.
        """
        noise_order = ['no_noise', 'low', 'medium', 'high']
        ns = [1/40, 1/20, 1/5, 1]  # 1/n in the same order
        plt.figure(figsize=(5,4))
        
        for noise_setting in noise_order:
            means = [summary_dict[noise_setting][n]['mean'] for n in [40, 20, 5, 1]]
            lows  = [summary_dict[noise_setting][n]['ci_lower'] for n in [40, 20, 5, 1]]
            ups   = [summary_dict[noise_setting][n]['ci_upper'] for n in [40, 20, 5, 1]]

            errs_lower = np.array(means) - np.array(lows)
            errs_upper = np.array(ups) - np.array(means)

            plt.errorbar(ns, means, yerr=[errs_lower, errs_upper], 
                         fmt='-o', capsize=5, label=noise_setting)

        plt.xlabel('Data Availability (log scale)', fontsize=14)
        plt.ylabel(metric_name, fontsize=14) 
        plt.title(f'{metric_name} vs. Data Availability', fontsize=16)
        plt.xscale('log')
        plt.xticks(ns, labels=['1/40', '1/20', '1/5', '1/1'], fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(*ylim)
        plt.grid(True)
        plt.legend(loc="best", fontsize=12)
        plt.tight_layout()
        plt.savefig(f'../../results/figs/expert_models/{metric_name}_vs_n_combined{file_suffix}.pdf', format='pdf', dpi=300)
        plt.show()

    # Plot ROC AUC (reweighted)
    plot_metric_vs_n(summary_roc, 'ROC AUC', ylim=(0.5, 0.65), file_suffix='')
    # Plot ECE (reweighted)
    plot_metric_vs_n(summary_ece, 'ECE (%)', ylim=(3, 13), file_suffix='')

    return summary_roc, summary_ece


# Main Script

if __name__ == "__main__":
    # 1) Load config, base data
    data_cfg = load_data_config()
    alert_data = pd.read_parquet("../../alert_data/alerts.parquet")

    # 2) Evaluate expert models (ALL REWEIGHTED)
    roc_results, ece_results = evaluate_expert_models(
        alert_data=alert_data,
        data_cfg=data_cfg,
        n_values=[40, 20, 5, 1],
        seeds=range(5)
    )

    # 3) Summarize & Plot
    summary_roc, summary_ece = compute_summary_and_plot(
        roc_results, 
        ece_results,
        n_values=[40, 20, 5, 1]
    )
    
    # Save summary results to pickle file
    with open('../../results/expert_model_summary_metrics.pkl', 'wb') as f:
        pickle.dump({
            'summary_roc': summary_roc,
            'summary_ece': summary_ece
        }, f)    
