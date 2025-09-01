import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml


with open('../../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)
lamda = data_cfg['lambda']

# Load alert data
alert_data = pd.read_parquet('../../alert_data/alerts.parquet')
test = alert_data.loc[alert_data["month"] == 7]
y_test = test['fraud_bool']

# Define noise levels and seeds
noise_levels = ['1.0_0.3', '1.5_0.4', '2.0_0.5']
seeds = range(5)

# Calculate the average misclassification cost for the expert team for each noise level
for noise_level in noise_levels:
    total_misclassification_cost = 0
    for seed in seeds:
        # Load noisy test data
        noisy_test_data = pd.read_parquet(
            f'../../drift_alert_data/test_alert_data_noisy_{noise_level}_seed_{seed}.parquet'
        )
        noisy_y_test = noisy_test_data['fraud_bool']

        # Concatenate the (clean) test labels and noisy test labels
        y_test = pd.concat([test['fraud_bool'], noisy_y_test], ignore_index=True)

        # Load expert predictions; each column corresponds to an expert
        expert_predictions = pd.read_parquet(
            f'../../L2D/synthetic_experts/test_expert_predictions_{noise_level}_seed_{seed}.parquet'
        )

        # Add true labels as a column.
        expert_predictions['true_labels'] = y_test.values

        # 1) Get the list of columns that are the experts. 
        prediction_cols = [col for col in expert_predictions.columns if col != 'true_labels']

        # 2) Separate expert predictions from true labels
        preds_df = expert_predictions[prediction_cols]  # DataFrame of shape (N, #experts)
        true_labels = expert_predictions['true_labels']  # Series of shape (N,)

        # 3) Align them
        preds_df, true_labels = preds_df.align(true_labels, axis=0)

        # 4) Compute misclassification cost across all experts
        false_positives_df = (preds_df != true_labels[:, None]) & (preds_df == 1)
        false_negatives_df = (preds_df != true_labels[:, None]) & (preds_df == 0)

        # 5) Compute the rates
        fp_rate = false_positives_df.mean().mean()  # average false positive rate (all experts)
        fn_rate = false_negatives_df.mean().mean()  # average false negative rate (all experts)

        # Weighted by the cost factor lambda for false positives
        misclassification_cost = fp_rate * lamda + fn_rate

        # Sum the misclassification cost for this seed
        total_misclassification_cost += misclassification_cost

    # Calculate average misclassification cost for this noise level
    avg_misclassification_cost = total_misclassification_cost / len(seeds)
    print(f'Average misclassification cost for noise level {noise_level}: {avg_misclassification_cost}')

    

# Load the expert parameters data
expert_params = pd.read_parquet('../../L2D/synthetic_experts/expert_parameters.parquet')

# Drop the columns you don't want in the heatmap
expert_params_filtered = expert_params.drop(columns=['fn_beta', 'fp_beta', 'alpha'])

# Modify row labels to use 'Expert#0', 'Expert#1', etc.
expert_params_filtered.index = [f'Expert#{i}' for i in range(expert_params_filtered.shape[0])]

# Plot heatmap
os.makedirs('../../results/figs/expert_properties', exist_ok=True)
plt.figure(figsize=(9, 6))
ax = sns.heatmap(
    -expert_params_filtered, 
    cmap='coolwarm', 
    center=0, 
    annot=False,
    vmin=-1, 
    vmax=1
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
#sns.heatmap(-expert_params_filtered, cmap='coolwarm', center=0, annot=False, vmin = -1, vmax = 1)
plt.title("Weight Vector Heatmap", fontsize=18)
plt.xticks(rotation=90)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(f'../../results/figs/expert_properties/weight_vector_heatmap.pdf', format='pdf', dpi=300)
plt.clf()





# Define the paths for the expert predictions, classifier predictions, and the data
expert_predictions_path = '../../L2D/synthetic_experts/'
data_path = '../../drift_alert_data/'
classifier_predictions_path = '../../L2D/classifier_h/selected_model/'

# Load the relevant expert prediction files for seed_0
expert_files = sorted([f for f in os.listdir(expert_predictions_path) if 'seed_0' in f and 'test_expert_predictions' in f])

# Load the relevant data files for seed_0
data_files = sorted([f for f in os.listdir(data_path) if 'seed_0' in f and 'test_alert_data_noisy' in f])

# Load the relevant classifier prediction files for seed_0
classifier_files = sorted([f for f in os.listdir(classifier_predictions_path) if f.endswith('seed_0.parquet')])


# Function to calculate the correctness matrix
def calculate_correctness_matrix(df, true_label_col):
    expert_columns = df.columns[df.columns != true_label_col]
    n_experts = len(expert_columns)
    correctness_matrix = np.zeros((n_experts, n_experts))

    for i, row_expert in enumerate(expert_columns):
        for j, col_expert in enumerate(expert_columns):
            if i != j:
                row_correct = (df[row_expert] == df[true_label_col])
                col_incorrect = (df[col_expert] != df[true_label_col])
                correctness_matrix[i, j] = np.mean(row_correct & col_incorrect)
    
    return correctness_matrix, expert_columns

# Loop through the 5 pairs of expert and data files, and also the classifier files
for i, (expert_file, data_file, classifier_file) in enumerate(zip(expert_files, data_files, classifier_files)):
    print(f'Processing pair {i+1}: {expert_file}, {data_file}, {classifier_file}')

    # Load original test data
    data = pd.read_parquet('../../alert_data/alerts.parquet')
    test_data = data.loc[data["month"] == 7]
    data_df = pd.read_parquet(os.path.join(data_path, data_file))
    data_df = pd.concat([test_data, data_df])
    
    # Load the expert predictions
    expert_df = pd.read_parquet(os.path.join(expert_predictions_path, expert_file))
    expert_df.columns = [f'Expert#{j}' for j in range(expert_df.shape[1])]

    # Load the classifier predictions from the .pkl file
    classifier_predictions = pd.read_parquet(os.path.join(classifier_predictions_path, classifier_file))

    # Add the classifier predictions to the expert DataFrame as another expert
    expert_df['Classifier'] = (classifier_predictions > 0.5).astype(int)
    
    # Get true labels from data
    true_labels = data_df['fraud_bool']

    # Add true labels to the expert predictions
    expert_df['true_labels'] = true_labels

    # Generate the correctness matrix
    correctness_matrix, expert_columns = calculate_correctness_matrix(expert_df, true_label_col='true_labels')

    # Plot the heatmap for this pair
    plt.figure(figsize=(5, 5))
    sns.heatmap(correctness_matrix, annot=False, fmt=".2f", cmap="magma", xticklabels=expert_columns, yticklabels=expert_columns, vmin=0, vmax=0.35)
    plt.title(f"Fraction of Instances where\nRow Expert is Correct and\nColumn Expert is Incorrect")
    plt.xlabel("Column Expert")
    plt.ylabel("Row Expert")
    plt.tight_layout()
    plt.savefig(f'../../results/figs/expert_properties/correctness_comparison_{data_file}.pdf', format='pdf', dpi=300)
    plt.clf()
    
