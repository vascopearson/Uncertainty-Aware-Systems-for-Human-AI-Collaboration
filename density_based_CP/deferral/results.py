import numpy as np
import pandas as pd
import os
import re
import yaml
from sklearn.metrics import confusion_matrix
from scipy import stats
import pickle


# Load the data configuration for the lambda value
with open('../../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)
lambda_value = data_cfg['lambda']

# Calculate misclassification costs and plot
def calculate_cost(y_true, y_pred, lambda_value):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (lambda_value * fp + fn) / (tn + fp + fn + tp)


########## Calculate misclassification cost of each method on each setting ###########

# Dictionary to store the assignment matrices, the model/expert predictions and the test labels
assignments = {}
expert_predictions = {}
model_preds = {}
preds = {}
test_labels = {}

# Base directories for the assignment matrices
base_dirs = {
    'cp': '../../results/deferral_results/assignment_matrices/',
    'ds': '../../results/deferral_results/ds_assignment_matrices/',
    'l2d': '../../results/deferral_results/l2d_assignment_matrices/',
    'rl': '../../results/deferral_results/rl_assignment_matrices/',
    'random': '../../results/deferral_results/random_assignment_matrices/'
}

# Load the test data labels (without noise)
alert_data = pd.read_parquet(f'../../alert_data/alerts.parquet')
test_labels_no_noise = alert_data.loc[(alert_data["month"] == 7)]['fraud_bool']

# Iterate through the files and save the assignment matrices
for file_name in os.listdir('../../drift_alert_data'):
    if file_name.endswith('.parquet'):
        match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
        end_name = f"{match.group(1)}_seed_{match.group(2)}"
        noise = match.group(1)
        noise_seed = match.group(2)
        test_labels[end_name] = pd.concat([test_labels_no_noise, pd.read_parquet(f'../../drift_alert_data/{file_name}')['fraud_bool']])
        for nbr_experts in range(1, 6):
            for deferral_rate in range(10, 60, 10):
                for method, base_dir in base_dirs.items():
                    for n in [40, 20, 5, 1]:
                        for seed in range(5):
                            file_path = f'{base_dir}assignments_noise_{noise}_seed_{noise_seed}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{seed}.parquet'
                            var_name = f'{method}_assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{seed}'
                            if os.path.exists(file_path):
                                assignments[var_name] = pd.read_parquet(file_path)

        expert_predictions[f'assignments_noise_{end_name}'] = pd.read_parquet(f'../../L2D/synthetic_experts/test_expert_predictions_{end_name}.parquet')
        model_preds[f'assignments_noise_{end_name}'] = pd.read_parquet(f'../../L2D/classifier_h/selected_model/model_test_preds_{end_name}.parquet')
        preds[f'preds_{end_name}'] = pd.concat([model_preds[f'assignments_noise_{end_name}'], expert_predictions[f'assignments_noise_{end_name}']], axis=1)

# Initialize a dictionary to store misclassification costs for each setting
results = {}

# Calculate misclassification costs for each setting

for key in assignments.keys():
    match = re.search(r"(\w+)_assignments_noise_(\d+\.\d+_\d+\.\d+)_seed_(\d+)", key)
    if match:
        method = match.group(1)
        noise_end_name = match.group(2)
        end_name = f"{match.group(2)}_seed_{match.group(3)}"
        parts = key.split('_')
        noise_seed = int(parts[6])
        nbr_experts = int(parts[7])
        deferral_rate = int(parts[9])
        n = int(re.search(r"1over(\d+)", key).group(1))
        seed = int(re.search(r"seed(\d+)", key).group(1))

        if f'preds_{end_name}' in preds:
            assignment_matrix = assignments[key]
            prediction_matrix = preds[f'preds_{end_name}']
            # Ensure the matrices are aligned correctly by index
            prediction_matrix = prediction_matrix.loc[assignment_matrix.index]
            # Select the same number of columns from prediction_matrix as assignment_matrix
            num_columns = assignment_matrix.shape[1]
            filtered_prediction_matrix = prediction_matrix.iloc[:, :num_columns]
            # Element-wise multiplication and sum along the rows to get a single prediction per row
            final_predictions = pd.DataFrame((assignment_matrix.values * filtered_prediction_matrix.values).sum(axis=1), index=assignment_matrix.index, columns=[key])

            y_true = test_labels[end_name].loc[final_predictions.dropna().index.tolist()]
            y_pred = final_predictions.loc[y_true.index]
            cost = calculate_cost(y_true, y_pred, lambda_value)

            # Create a unique identifier for the setting
            setting_identifier = f'{noise_end_name}_{nbr_experts}_{deferral_rate}_1over{n}'

            # Store the cost in the results dictionary
            if setting_identifier not in results:
                results[setting_identifier] = {'cp': [], 'ds': [], 'l2d': [], 'rl': [], 'random': []}
            results[setting_identifier][method].append(cost)
            
with open("../../results/results.pkl", "wb") as f:
    pickle.dump(results, f)

# Initialize DataFrame to store misclassification costs
results_df = pd.DataFrame(columns=['Setting', 'cp', 'ds,' 'l2d', 'rl', 'random'])

# Calculate the mean and 95% confidence intervals and apply bold to the lowest cost
for setting_identifier, methods in results.items():
    row = {'Setting': setting_identifier}
    mean_costs = {}

    training_size = setting_identifier.split('_')[-1]
    
    for method, costs in methods.items():
        mean_cost = np.mean(costs)
        sem = stats.sem(costs)
        conf_interval = sem * stats.t.ppf((1 + 0.95) / 2., len(costs) - 1)
        mean_costs[method] = (mean_cost*100, conf_interval*100)
        row[method] = f"{mean_cost*100:.2f} ± {conf_interval*100:.2f}"

    # Find the two lowest mean costs
    sorted_methods = sorted(mean_costs, key=lambda k: mean_costs[k][0])
    min_method = sorted_methods[0]  # Lowest cost (bold)
    second_min_method = sorted_methods[1]  # Second lowest cost (underline)

    # Apply bold formatting to the minimum mean cost
    row[min_method] = f"\\textbf{{{mean_costs[min_method][0]:.2f}}} ± {mean_costs[min_method][1]:.2f}"
    
    # Apply underline formatting to the second-lowest mean cost
    row[second_min_method] = f"\\underline{{{mean_costs[second_min_method][0]:.2f}}} ± {mean_costs[second_min_method][1]:.2f}"

    # Append the row to the DataFrame
    results_df = results_df.append(row, ignore_index=True)

# Replace underscores with spaces in the 'Setting' column
results_df['Setting'] = results_df['Setting'].str.replace('_', ' ')

# Split the 'Setting' column into 'Noise', 'N_experts', and 'Deferral_rate'
results_df[['Noise', 'N_experts', 'Deferral_rate', 'Training_size']] = results_df['Setting'].str.extract(r'(\d+\.\d+\s\d+\.\d+)\s(\d+)\s(\d+)\s1over(\d+)')

# Drop the original 'Setting' column
results_df = results_df.drop(columns=['Setting'])

# Reorder the columns
results_df = results_df[['Noise', 'N_experts', 'Deferral_rate', 'Training_size', 'cp', 'ds', 'l2d', 'rl', 'random']]

# Split the DataFrame by 'Training_size' and 'Noise'
split_dfs = [df for _, df in results_df.groupby(['Training_size', 'Noise'])]

# Define the LaTeX multi-column headers
multi_column_header = '''
\\begin{tabular}{ccccccccc}
\\toprule
\\multicolumn{4}{c}{\\textbf{Setting}} & \\multicolumn{5}{c}{\\textbf{Deferral Strategy}} \\\\
\\cmidrule(lr){1-4} \\cmidrule(lr){5-9}
Noise & NE & DR & TS & CP & DS & L2D & RL & Random \\\\
\\midrule
'''

# Convert each split DataFrame to LaTeX format and save to separate files
os.makedirs('../../results/latex_tables', exist_ok=True)
for idx, split_df in enumerate(split_dfs):
    latex_table = split_df.to_latex(index=False, escape=False, header=False)
    latex_table = multi_column_header + latex_table
    with open(f'../../results/latex_tables/latex_MC_table_part_{idx+1}.tex', 'w') as f:
        f.write(latex_table)
    print(latex_table)


########## Proportion of instances assigned to each strategy by the optimization problem ##########

def calculate_proportions_and_statistics(base_dir, method, noise_end_name, nbr_experts, deferral_rate, n, seeds):
    proportions = []
    std_devs = []
    
    for training_seed in seeds:
        for noise_seed in seeds:
            file_path = f'{base_dir}/count_{method}/count_{method}_noise_{noise_end_name}_seed_{noise_seed}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{training_seed}.csv'
            if os.path.exists(file_path):
                # Load the CSV file
                data = pd.read_csv(file_path, header='infer')

                # Calculate the proportion of instances assigned to the method across all batches
                proportion = np.array(data / 100)  # Each batch has 100 instances

                proportions.append(proportion.mean(axis = 1)[0])

                # Calculate the standard deviation across batches
                std_dev = proportion.std(axis = 1)
                std_devs.append(std_dev[0])
    
    # Calculate mean and confidence intervals across seeds
    mean_proportion = np.mean(proportions)
    mean_std_dev = np.mean(std_devs)
    sem_proportion = stats.sem(proportions)
    sem_std_dev = stats.sem(std_devs)
    ci_proportion = sem_proportion * stats.t.ppf((1 + 0.95) / 2., len(proportions) - 1)
    ci_std_dev = sem_std_dev * stats.t.ppf((1 + 0.95) / 2., len(std_devs) - 1)

    return mean_proportion, ci_proportion, mean_std_dev, ci_std_dev

def calculate_alpha_statistics(base_dir, noise_end_name, nbr_experts, deferral_rate, n, seeds):
    mean_alphas_fixed = []
    std_devs = []
    
    for training_seed in seeds:
        for noise_seed in seeds:
            file_path = f'{base_dir}/alphas/alphas_noise_{noise_end_name}_seed_{noise_seed}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{training_seed}.csv'
            if os.path.exists(file_path):
                # Load the CSV file
                data = pd.read_csv(file_path, header='infer')

                # Calculate the alphas_array across all batches
                alphas_array = np.array(data)  # Each batch has 100 instances

                mean_alphas_fixed.append(alphas_array.mean(axis = 1)[0])

                # Calculate the standard deviation across batches
                std_dev = alphas_array.std(axis = 1)
                std_devs.append(std_dev[0])
    
    # Calculate mean and confidence intervals across seeds
    mean_alphas = np.mean(mean_alphas_fixed)
    mean_std_dev = np.mean(std_devs)
    sem_alphas = stats.sem(mean_alphas_fixed)
    sem_std_dev = stats.sem(std_devs)
    ci_alphas = sem_alphas * stats.t.ppf((1 + 0.95) / 2., len(mean_alphas_fixed) - 1)
    ci_std_dev = sem_std_dev * stats.t.ppf((1 + 0.95) / 2., len(std_devs) - 1)

    return mean_alphas, ci_alphas, mean_std_dev, ci_std_dev

# Initialize the DataFrame to store results
results_df = pd.DataFrame(columns=['Setting', 'Mean_Proportion_L2D', 'Std_Dev_L2D', 'Mean_Proportion_RL', 'Std_Dev_RL', 'Mean_Alphas', 'Std_Dev_Alphas'])

# Iterate through the files and save the assignment matrices
for noise_end_name in ['1.0_0.3', '1.5_0.4', '2.0_0.5']:
    for nbr_experts in range(1, 6):
        for deferral_rate in range(10, 60, 10):
            for n in [40,20,5,1]:
                # Initialize dictionary to store L2D and RL results
                row = {'Setting': f'{noise_end_name}_{nbr_experts}_{deferral_rate}_1over{n}'}

                for method in ['rl', 'l2d']:
                    base_dir = '../../results/deferral_results'
                    mean_proportion, ci_proportion, mean_std_dev, ci_std_dev = calculate_proportions_and_statistics(base_dir, method, noise_end_name, nbr_experts, deferral_rate, n, range(5))
                    mean_alphas, ci_alphas, mean_alphas_std_dev, ci_alphas_std_dev = calculate_alpha_statistics(base_dir, noise_end_name, nbr_experts, deferral_rate, n, range(5))
                        
                    # Create a setting identifier
                    setting_identifier = f'{noise_end_name}_{nbr_experts}_{deferral_rate}_1over{n}'
                    
                    # Fill the appropriate columns based on the method
                    if method == 'l2d':
                        row['Mean_Proportion_L2D'] = f"{100*mean_proportion:.2f}\% ± {100*ci_proportion:.2f}"
                        row['Std_Dev_L2D'] = f"{mean_std_dev:.2f} ± {ci_std_dev:.2f}"
                    else:
                        row['Mean_Proportion_RL'] = f"{100*mean_proportion:.2f}\% ± {100*ci_proportion:.2f}"
                        row['Std_Dev_RL'] = f"{mean_std_dev:.2f} ± {ci_std_dev:.2f}"
                        row['Mean_Alphas'] = f"{mean_alphas:.2f} ± {ci_alphas:.4f}"
                
                # Append the row to the DataFrame
                results_df = results_df.append(row, ignore_index=True)

# Save the results to a CSV file
results_df.to_csv('../../results/proportion_and_std_dev_statistics.csv', index=False)

# Replace underscores with spaces in the 'Setting' column for better readability
results_df['Setting'] = results_df['Setting'].str.replace('_', ' ')

# Split the 'Setting' column into 'Noise', 'N_experts', 'Deferral_rate', and 'Training_size'
results_df[['Noise', 'N_experts', 'Deferral_rate', 'Training_size']] = results_df['Setting'].str.extract(r'(\d+\.\d+\s\d+\.\d+)\s(\d+)\s(\d+)\s1over(\d+)')

# Drop the original 'Setting' column and reorder the columns
results_df = results_df[['Noise', 'N_experts', 'Deferral_rate', 'Training_size', 'Mean_Proportion_RL', 'Mean_Alphas']]

# Split the DataFrame by 'Training_size' and 'Noise'
split_dfs = [df for _, df in results_df.groupby(['Training_size', 'Noise'])]

# Define the LaTeX multi-column headers
multi_column_header = '''
\\begin{tabular}{cccccc}
\\toprule
\\multicolumn{4}{c}{\\textbf{Setting}} & \\textbf{Proportion assigned by RL} & \\textbf{Selected Alpha} \\\\
\\cmidrule(lr){1-4} \\cmidrule(lr){5-5} \\cmidrule(lr){6-6}
Noise & NE & DR & TS &  \\\\
\\midrule
'''

# Convert each split DataFrame to LaTeX format and save to separate files
for idx, split_df in enumerate(split_dfs):
    latex_table = split_df.to_latex(index=False, escape=False, header=False)
    latex_table = multi_column_header + latex_table + '\\bottomrule\n\\end{tabular}'
    with open(f'../../results/latex_tables/latex_alpha_table_part_{idx+1}.tex', 'w') as f:
        f.write(latex_table)
    print(latex_table)

