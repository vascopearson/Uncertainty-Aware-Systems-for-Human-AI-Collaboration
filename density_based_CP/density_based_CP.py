import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import re

from density_based_CP_functions import calculate_coverage_metrics, subset_data, learn_density_estimator

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
with open('../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

# Load data
data = pd.read_parquet('../alert_data/alerts.parquet')
data = cat_checker(data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

# Split data into training and calibration sets
train = data.loc[(data["month"].isin([3,4,5,6]))]  # Month 6 is calibration

# Drop unnecessary columns
X_train = train.drop(columns=['fraud_bool','model_score','month'])
y_train = train['fraud_bool']

# Initialize the results dataframes
os.makedirs('../results', exist_ok=True)
perc_of_noisy_in_null = pd.DataFrame(columns=[f'alpha_{alpha}' for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]])
perc_of_noisy_identified = pd.DataFrame(columns=[f'alpha_{alpha}' for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]])

# Define alphas
n_alpha = 50
alphas = [i / n_alpha for i in range(n_alpha + 1)]

# Lists to store results
all_null_proportions_ood = []
all_ood_proportions = []
end_names = []

accuracies_tr_list = {}
entry_proportions_tr_list = {}
null_proportions_tr_list = {}
accuracies_test_list = {}
entry_proportions_test_list = {}
null_proportions_test_list = {}

train_done = 0

# Low, medium and high noise
accepted_noises = {"1.0_0.3": "low noise", "1.5_0.4": "medium noise", "2.0_0.5": "high noise"}

# Loop through files in the drift_alert_data directory
for file_name in os.listdir('../drift_alert_data'):
    if file_name.endswith('.parquet'):
        match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
        if not match:
            continue

        noise_str = match.group(1)
        noise_seed = match.group(2)
        # Skip if not in our accepted set OR not seed 0
        if noise_str not in accepted_noises: # or noise_seed != '0':  #if we want plots use only one seed, otherwise remove this condition
            continue
        
        end_name = f"{noise_str}_seed_{noise_seed}"
        print(f"Noise: {end_name} ({accepted_noises[noise_str]})")

        # Load test data
        test = data.loc[data["month"] == 7]
        test_noisy = pd.read_parquet(f'../drift_alert_data/test_alert_data_noisy_{end_name}.parquet')
        test = pd.concat([test, test_noisy])

        X_test = test.drop(columns=['fraud_bool','model_score','month']) 
        y_test = test['fraud_bool']

        # Load features
        features_train = pd.read_csv('../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_train_features.csv', index_col=0)
        features_test  = pd.read_csv(f'../density_based_CP/feature_extraction/processed_data/BAF_second_to_last_layer_test_features_{end_name}.csv', index_col=0)

        # Prepare data for conformal prediction
        Z = list(zip(features_train.values.tolist(), y_train.tolist()))
        Y = np.unique(y_train)

        # Calibration set size
        p = {
           0: len(data.loc[(data["month"]==6) & (data["fraud_bool"]==0)]),
           1: len(data.loc[(data["month"]==6) & (data["fraud_bool"]==1)])
        }
        print("Calibration set size by class:", p)

        # Bandwidth for KDE
        bdwth = 0.2

        null_proportions_ood = []
        ood_proportions = []

        # Train the density estimator and get density scores
        p_hat_list = []
        log_density = []
        for y in Y:
            print(f"Performing density estimation for class {y}")
            X_tr, X_cal = subset_data(Z, y, p[y])
            p_hat_y = learn_density_estimator(X_tr, bdwth)
            p_hat_list.append(p_hat_y)
            log_density.append(p_hat_y.score_samples(X_cal))
        print("Density estimation finished!")

        # Compute density for train & test
        train_density = []
        test_density  = []
        for i, y in enumerate(Y):
            print(f"Getting density scores for class {y}")
            train_density.append(np.exp(p_hat_list[i].score_samples(features_train.values)))
            test_density.append(np.exp(p_hat_list[i].score_samples(features_test.values)))
        print("Density scores calculated!")

        # Loop over alpha
        for alpha in alphas:
            C_train = np.zeros((features_train.shape[0], len(Y)), dtype=bool)
            C_test = np.zeros((features_test.shape[0], len(Y)), dtype=bool)

            # t_hat
            t_hat_list = []
            for i2, y in enumerate(Y):
                t_hat_y = np.percentile(np.exp(log_density[i2]), 100*alpha)
                t_hat_list.append(t_hat_y)
            print("Quantiles have been calculated for alpha=", alpha)

            # Build the prediction sets
            if train_done == 0:
                for i2,y in enumerate(Y):
                    C_train[:, i2] = (train_density[i2] >= t_hat_list[i2])
            for i2,y in enumerate(Y):
                C_test[:, i2] = (test_density[i2] >= t_hat_list[i2])

            if train_done == 0:
                train[f'prediction_set_{alpha}'] = list(C_train)
            test[f'prediction_set_{alpha}'] = list(C_test)

            # Proportion of null predictions from the noisy portion
            null_preds = test[f'prediction_set_{alpha}'].apply(lambda x: np.all(np.array(x) == False))
            null_ood   = null_preds[test.index.isin(test_noisy.index)].sum()
            total_null = null_preds.sum()
            null_proportion_ood = null_ood / total_null if total_null>0 else 0
            null_proportions_ood.append(null_proportion_ood)

            # Proportion of OOD data identified
            total_ood = len(test_noisy)
            ood_proportion = null_ood / total_ood if total_ood>0 else 0
            ood_proportions.append(ood_proportion)

        # Save results
        end_names.append(end_name)
        all_null_proportions_ood.append(null_proportions_ood)
        all_ood_proportions.append(ood_proportions)

        if train_done == 0:
            train.to_parquet('../density_based_CP/feature_extraction/processed_data/BAF_train_with_prediction_sets.parquet')
        test.to_parquet(f'../density_based_CP/feature_extraction/processed_data/BAF_test_with_prediction_sets_{end_name}.parquet')
        train_done = 1
        print(f"Prediction sets saved to parquet for {end_name}.")

        # Coverage on training set
        metrics_train = calculate_coverage_metrics(Z, alphas, Z, Y, p, bandwidth=bdwth)
        (accuracies_tr, entry_proportions_tr, null_proportions_tr,
         class_accuracies_tr, class_entry_proportions_tr, class_null_proportions_tr) = metrics_train

        accuracies_tr_list[end_name]         = accuracies_tr
        entry_proportions_tr_list[end_name]  = entry_proportions_tr
        null_proportions_tr_list[end_name]   = null_proportions_tr

        # Coverage on test set
        Z_test = list(zip(features_test.values.tolist(), y_test.tolist()))
        metrics_test = calculate_coverage_metrics(Z_test, alphas, Z, Y, p, bandwidth=bdwth)
        (accuracies_test, entry_proportions_test, null_proportions_test,
         class_accuracies_test, class_entry_proportions_test, class_null_proportions_test) = metrics_test

        accuracies_test_list[end_name]         = accuracies_test
        entry_proportions_test_list[end_name]  = entry_proportions_test
        null_proportions_test_list[end_name]   = null_proportions_test

        # Plot coverage on train & test
        os.makedirs('../results/figs/conformal_coverage_plots', exist_ok=True)
        plt.figure(figsize=(6, 6))
        # Training
        plt.plot(alphas, accuracies_tr, label='Classification Accuracy (Train)', linestyle='-', color='royalblue')
        plt.plot(alphas, entry_proportions_tr, label='Proportion of {0,1} entries (Train)', linestyle='--', color='royalblue', alpha=0.5)
        plt.plot(alphas, null_proportions_tr, label='Proportion of null set (Train)', linestyle=':', color='royalblue')
        # Test
        plt.plot(alphas, accuracies_test, label='Classification Accuracy (Test)', linestyle='-', color='orangered')
        plt.plot(alphas, entry_proportions_test, label='Proportion of {0,1} entries (Test)', linestyle='--', color='orangered', alpha=0.5)
        plt.plot(alphas, null_proportions_test, label='Proportion of null set (Test)', linestyle=':', color='orangered')
        plt.xlabel('Alpha', fontsize=16)
        plt.ylabel('Proportion', fontsize=16)
        plt.title(f'Conformal Coverage: {accepted_noises[noise_str]}', fontsize=18)
        plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15), fontsize=12)
        plt.tight_layout()
        plt.savefig(f'../results/figs/conformal_coverage_plots/conformal_coverage_{end_name}.pdf',
                    format='pdf', dpi=300, bbox_inches='tight')

# Create label_mapping and label_order for plotting
label_mapping = {
    "1.0_0.3_seed_0": "Low noise",
    "1.5_0.4_seed_0": "Medium noise",
    "2.0_0.5_seed_0": "High noise"
}
label_order = ["1.0_0.3_seed_0", "1.5_0.4_seed_0", "2.0_0.5_seed_0"]

os.makedirs('../results/figs/proportion_noisy_in_nulls', exist_ok=True)
# Plot the proportion of null set predictions from noisy data (all lines in one plot)
plt.figure(figsize=(5,4))
for end_name in label_order:
    if end_name in end_names:
        i = end_names.index(end_name)
        label = label_mapping.get(end_name, end_name)
        plt.plot(alphas[1:], all_null_proportions_ood[i][1:], linestyle='-', label=label)
plt.xlabel('Alpha', fontsize=14)
plt.ylabel('Proportion of Null Predictions\n from Noisy Data', fontsize=14)
plt.title('Null Predictions from Noisy Data', fontsize=16)
plt.ylim(0.2,1.0)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.legend(fontsize=12)
plt.savefig('../results/figs/proportion_noisy_in_nulls/proportion_null_predictions_noisy_data_combined.pdf',
            format='pdf', dpi=300)

os.makedirs('../results/figs/proportion_noisy_id', exist_ok=True)
# Plot the proportion of noisy data identified (all lines in one plot)
plt.figure(figsize=(5,4))
for end_name in label_order:
    if end_name in end_names:
        i = end_names.index(end_name)
        label = label_mapping.get(end_name, end_name)
        plt.plot(alphas, all_ood_proportions[i], linestyle='-', label=label)
plt.xlabel('Alpha', fontsize=14)
plt.ylabel('Proportion of Noisy Data Identified', fontsize=14)
plt.title('Noisy Data Identified', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.legend(fontsize=12)
plt.savefig('../results/figs/proportion_noisy_id/proportion_noisy_data_identified_combined.pdf',
            format='pdf', dpi=300)

# Create a figure and axes for a 2x2 grid of subplots (since we only have 3 lines to plot, we can skip extra subplot)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[1,1].axis('off')  # We'll only use 3 subplots

# label_mapping for titles
label_mapping_small = {
    "1.0_0.3_seed_0": "Low noise",
    "1.5_0.4_seed_0": "Medium noise",
    "2.0_0.5_seed_0": "High noise"
}
label_order_small = ["1.0_0.3_seed_0", "1.5_0.4_seed_0", "2.0_0.5_seed_0"]
plot_positions = [(0,0), (0,1), (1,0)]

# Plot coverage metrics for each noise level
for idx, end_name in enumerate(label_order_small):
    if end_name in end_names:
        row, col = plot_positions[idx]
        ax = axes[row, col]

        ax.plot(alphas, accuracies_tr_list[end_name], label='Accuracy (Train)', linestyle='-', color='royalblue')
        ax.plot(alphas, entry_proportions_tr_list[end_name], label='Prop of {0,1} (Train)', linestyle='--', color='royalblue', alpha=0.5)
        ax.plot(alphas, null_proportions_tr_list[end_name], label='Null set (Train)', linestyle=':', color='royalblue')

        ax.plot(alphas, accuracies_test_list[end_name], label='Accuracy (Test)', linestyle='-', color='orangered')
        ax.plot(alphas, entry_proportions_test_list[end_name], label='Prop of {0,1} (Test)', linestyle='--', color='orangered', alpha=0.5)
        ax.plot(alphas, null_proportions_test_list[end_name], label='Null set (Test)', linestyle=':', color='orangered')

        ax.set_xlabel('Alpha', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title(label_mapping_small[end_name], fontsize=14)
        ax.grid(False)

# Add legend to the last subplot
handles, labels = ax.get_legend_handles_labels()
axes[1,1].legend(handles, labels, loc='center', fontsize=12)
plt.tight_layout()
plt.savefig('../results/figs/conformal_coverage_plots/conformal_coverage_combined_low_medium_high.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
