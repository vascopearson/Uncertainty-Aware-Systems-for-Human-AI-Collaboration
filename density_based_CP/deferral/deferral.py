import pandas as pd
import yaml
import pickle
import numpy as np
import os
from ortools.sat.python import cp_model
import json

# Set random seed
np.random.seed(42)

# Load the YAML configuration file
with open('../../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)


def calculate_weights(df, alphas):
    n_alphas = len(alphas)

    # Initialize a series to hold the alpha_null values for each row
    alpha_null_series = pd.Series(index=df.index, dtype=float)
    
    # Initialize a list to hold the proportions of null set predictions for each alpha value
    null_set_counts = [0] * n_alphas
    total_rows = len(df)
    
    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Iterate over the alpha values from 0.0 to 1.0 in steps of 0.1
        for i, alpha in enumerate(alphas):
            prediction_set_col = f'prediction_set_{alpha}'
            if np.all(row[prediction_set_col] == False):
                alpha_null_series.at[index] = alpha
                null_set_counts[i] += 1
                break
    
    # Convert counts to proportions
    cumulative_counts = np.cumsum(null_set_counts)
    null_set_proportions = [count / total_rows for count in cumulative_counts]

    # Calculate the predicted null proportion list and the weight of each instance
    pred_null_set_proportions = [0] * n_alphas
    weights = [0] * n_alphas
    for i in range(1, n_alphas):
        prev_alpha = alphas[i - 1]
        prev_proportion = null_set_proportions[i - 1]
        
        # Fit a line between (prev_alpha, prev_proportion) and (1, 1)
        slope = (1 - prev_proportion) / (1 - prev_alpha)
        intercept = 1 - slope
        
        current_alpha = alphas[i]
        pred_null_set_proportions[i] = slope * current_alpha + intercept
        weights[i] = pred_null_set_proportions[i] / null_set_proportions[i]

        # Clip the weight to a maximum value of 1
        if weights[i] > 1:
            weights[i] = 1
    
    # Create a DataFrame for the weights with the same index as alpha_null_series
    weights_df = pd.Series([weights[alphas.index(alpha)] for alpha in alpha_null_series], index=df.index, name='weight')

    # Convert the alpha_null_series to a DataFrame and add the weights column
    alpha_null_weights_df = alpha_null_series.to_frame(name='alpha_null')
    alpha_null_weights_df['weight'] = weights_df
    
    return alpha_null_weights_df


def convert_assignments_to_dataframe(assignment, index_mapping, experts):
    # Define the column names
    columns = ['classifier_h'] + [f'standard#{i}' for i in range(int(experts))]

    # Initialize an empty DataFrame to hold the assignments
    assignment_df = pd.DataFrame(0, index=index_mapping.keys(), columns=columns)

    # Populate the DataFrame with assignment values
    for (i, j), value in assignment.items():
        row_index = index_mapping[i]
        assignment_df.iloc[row_index, j] = value

    return assignment_df


def optimize_assignment_and_coverage(batches, preds_matrix, alphas, n_experts, deferral_rate):
    final_assignment = pd.DataFrame(columns=['classifier_h'] + [f'standard#{i}' for i in range(int(n_experts))])
    print("final_assignment: ", final_assignment)
    best_alphas ={}
    max_objective = {}
    final_counts_rl_instances = {}
    final_counts_l2d_instances = {}

    # Load the capacity matrix
    capacities = pd.read_csv(f'../../capacity_constraints/H_{int(n_experts)}_experts_{deferral_rate}_percent.csv')

    # Extract expert columns (limit to num_experts_to_use)
    expert_columns = [f'standard#{i}' for i in range(int(n_experts))]
    expert_avg_columns = [f'standard#{i}_avg' for i in range(int(n_experts))]

    # Loop through each batch and filter the weight and capacity matrix
    for batch_id in batches['batch'].unique():
        max_objective[batch_id] = 0
        max_objective_batch = -np.inf
        batch_objective = {}
        best_alpha = None
        final_count_rl_instances = 0
        final_count_l2d_instances = 0
        batch_indices = batches[batches['batch'] == batch_id].index

        alpha_null_weights_df = calculate_weights(preds_matrix.loc[batch_indices], alphas)

        alphas_to_optimize = alphas[:10]

        if len(batch_indices) == len(batches[batches['batch'] == 0]):
            for alpha in alphas_to_optimize:

                # Initialize batch objective
                batch_objective[alpha] = 0

                # Initialize the weight matrix
                weight_matrix_columns = ['classifier_h'] + expert_columns
                batch_weight_matrix = pd.DataFrame(index=preds_matrix.loc[batch_indices].index, columns=weight_matrix_columns)

                # Fill the weight matrix
                count_rl_instances = 0
                count_l2d_instances = 0
                for index, row in preds_matrix.loc[batch_indices].iterrows(): 
                    if all(row[f'prediction_set_{alpha}'] == [False, False]):
                        count_rl_instances += 1
                        batch_weight_matrix.at[index, 'classifier_h'] = 0
                        for expert_col, avg_col in zip(expert_columns, expert_avg_columns):
                            batch_weight_matrix.at[index, expert_col] = row[avg_col]
                    else:
                        count_l2d_instances += 1
                        batch_weight_matrix.at[index, 'classifier_h'] = row['classifier_h'] * alpha_null_weights_df.loc[index,'weight']
                        for expert_col in expert_columns:
                            batch_weight_matrix.at[index, expert_col] = row[expert_col] * alpha_null_weights_df.loc[index,'weight']

                H = capacities.iloc[batch_id,:int(n_experts)]
                H.index = batch_weight_matrix.columns[1:]  # Rename to match batch_weight_matrix columns except classifier_h
                H['classifier_h'] = len(batch_indices) - capacities.iloc[batch_id,:int(n_experts)].sum()
                H = H.reindex(batch_weight_matrix.columns)
                H = H.astype(int)

                # Ensure H['classifier_h'] is non-negative
                if H['classifier_h'] < 0:
                    H['classifier_h'] = 0

                # Optimization
                model = cp_model.CpModel()
                nb_instances = len(batch_indices)
                nb_decision_makers = len(weight_matrix_columns)

                # Variables
                A = {}
                for i in range(nb_instances):
                    for j in range(nb_decision_makers):
                        A[i, j] = model.NewBoolVar(f'A[{i},{j}]')

                # Objective
                objective = []
                for i in range(nb_instances):
                    for j in range(nb_decision_makers):
                        objective.append(batch_weight_matrix.iloc[i,j]*A[i, j])
                model.Maximize(sum(objective))

                # Constraints
                for j in range(nb_decision_makers):
                    model.Add(
                        sum(A[i, j] for i in range(nb_instances)) == H.iloc[j]
                    )

                for i in range(nb_instances):
                    model.Add(
                        sum(A[i, j] for j in range(nb_decision_makers)) == 1
                    )

                # Solve the model
                solver = cp_model.CpSolver()
                status = solver.Solve(model)

                if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                    assignment = {(batch_indices[i], j): solver.Value(A[i, j])
                                  for i in range(nb_instances)
                                  for j in range(nb_decision_makers)}
                    
                    index_mapping = {idx: i for i, idx in enumerate(batch_weight_matrix.index)}
                    assignment = convert_assignments_to_dataframe(assignment, index_mapping, n_experts)
                    for i in range(nb_instances):
                        for j in range(nb_decision_makers):
                            batch_objective[alpha] += batch_weight_matrix.iloc[i,j]*assignment.iloc[i,j]
                    print(f'Batch {batch_id} optimized for alpha {alpha}. Objective: {batch_objective}')

                else:
                    print(f'No solution found for batch {batch_id} with alpha {alpha}.')

                # Compare batch_objective for all alphas and choose the best alpha
                if batch_objective[alpha] > max_objective_batch:
                    max_objective_batch = batch_objective[alpha]
                    best_alpha = alpha
                    final_count_rl_instances = count_rl_instances
                    final_count_l2d_instances = count_l2d_instances
                    assignment_batch = assignment.copy()

            print(f"Optimization objective for batch {batch_id}: {max_objective_batch}. Obtained with alpha = {best_alpha}")
            max_objective[batch_id] = max_objective_batch
            final_assignment = final_assignment.append(assignment_batch)
            best_alphas[batch_id] = [best_alpha]
            final_counts_rl_instances[batch_id] = [final_count_rl_instances]
            final_counts_l2d_instances[batch_id] = [final_count_l2d_instances]
            print("max_objective: ", max_objective)
            print("best_alphas: ", best_alphas)
            print("final_assignment: ", final_assignment)

    return best_alphas, max_objective, final_assignment, final_counts_rl_instances, final_counts_l2d_instances



def density_softmax(batches, preds_matrix, n_experts, deferral_rate):

    # Filter the columns to get probabilities of correctness
    filtered_columns = preds_matrix.filter(regex='^(standard|classifier_h)(?!.*(confidence|avg)$)').columns
    expert_columns = [f'standard#{i}' for i in range(int(n_experts))]
    filtered_columns = ['classifier_h'] + expert_columns
    preds_matrix = preds_matrix[filtered_columns]

    # Load the capacity matrix
    capacities = pd.read_csv(f'../../capacity_constraints/H_{n_experts}_experts_{deferral_rate}_percent.csv')
    # Initialize final assignment matrix
    final_assignment = pd.DataFrame(index=preds_matrix.index, columns=filtered_columns, data=0)

    # Loop through each batch and filter the preds and capacity matrix
    for batch_id in batches['batch'].unique():
        batch_indices = batches[batches['batch'] == batch_id].index
        if len(batch_indices) == len(batches[batches['batch'] == 0]):
            # Initialize batch objective
            batch_objective = 0
            preds_batch = preds_matrix.loc[batch_indices] # Weight Matrix
            H = capacities.iloc[batch_id,:int(n_experts)]
            H.index = preds_batch.columns[1:]
            H['classifier_h'] = len(batch_indices) - capacities.iloc[batch_id,:int(n_experts)].sum()
            H = H.reindex(preds_batch.columns)
            H = H.astype(int) # Capacity Matrix

            # Optimization
            model = cp_model.CpModel()
            nb_instances = len(batch_indices)
            nb_decision_makers = len(preds_batch.columns)
            
            # Variables
            A = {}
            for i in range(nb_instances):
                for j in range(nb_decision_makers):
                    A[i, j] = model.NewBoolVar(f'A[{i},{j}]')
            
            # Objective
            objective = []
            for i in range(nb_instances):
                for j in range(nb_decision_makers):
                    objective.append(preds_batch.iloc[i,j]*A[i, j])
            model.Maximize(sum(objective))
            
            # Constraints
            for j in range(nb_decision_makers):
                model.Add(
                    sum(A[i, j] for i in range(nb_instances)) == H.iloc[j]
                )
            for i in range(nb_instances):
                model.Add(
                    sum(A[i, j] for j in range(nb_decision_makers)) == 1
                )
            
            # Solve the model
            solver = cp_model.CpSolver()
            status = solver.Solve(model)

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                assignment = {(batch_indices[i], j): solver.Value(A[i, j])
                              for i in range(nb_instances)
                              for j in range(nb_decision_makers)}
                
                index_mapping = {idx: i for i, idx in enumerate(preds_batch.index)}
                assignment = convert_assignments_to_dataframe(assignment, index_mapping, n_experts)
                for i in range(nb_instances):
                    for j in range(nb_decision_makers):
                        batch_objective += preds_batch.iloc[i,j]*assignment.iloc[i,j]
                print(f'Batch {batch_id} optimized. Objective: {batch_objective}')

            else:
                print(f'No solution found for batch {batch_id}.')

            final_assignment = final_assignment.append(assignment)
            
    # Remove all rows that where not assigned (batch didn't reach the necessary size)
    final_assignment = final_assignment.loc[~(final_assignment == 0).all(axis=1)]
    print(final_assignment)
    return final_assignment


def l2d_baseline(batches, preds_matrix, n_experts, deferral_rate):

    # Filter the columns to get probabilities of correctness
    filtered_columns = preds_matrix.filter(regex='^(standard|classifier_h)(?!.*(confidence|avg)$)').columns
    expert_columns = [f'standard#{i}' for i in range(int(n_experts))]
    filtered_columns = ['classifier_h'] + expert_columns
    preds_matrix = preds_matrix[filtered_columns]

    # Load the capacity matrix
    capacities = pd.read_csv(f'../../capacity_constraints/H_{int(n_experts)}_experts_{deferral_rate}_percent.csv')

    # Initialize final assignment matrix
    final_assignment = pd.DataFrame(index=preds_matrix.index, columns=filtered_columns, data=0)

    # Loop through each batch and filter the preds and capacity matrix
    for batch_id in batches['batch'].unique():
        batch_indices = batches[batches['batch'] == batch_id].index
        if len(batch_indices) == len(batches[batches['batch'] == 0]):
            preds_batch = preds_matrix.loc[batch_indices]
            H = capacities.iloc[batch_id,:int(n_experts)]
            H.index = preds_batch.columns[1:]
            H['classifier_h'] = len(batch_indices) - capacities.iloc[batch_id,:int(n_experts)].sum()
            H = H.astype(int)

            # Initialize capacity tracker
            current_capacity = H.copy()

            # Iterate over each instance in the batch
            for idx in preds_batch.index:
                # Get the sorted probabilities for the instance
                sorted_probs = preds_batch.loc[idx].sort_values(ascending=False)

                # Assign the instance to the decision-maker with the highest probability that has not exceeded capacity
                for decision_maker in sorted_probs.index:
                    if current_capacity[decision_maker] > 0:
                        final_assignment.at[idx, decision_maker] = 1
                        current_capacity[decision_maker] -= 1
                        break

    # Remove all rows that where not assigned (batch didn't reach the necessary size)
    final_assignment = final_assignment.loc[~(final_assignment == 0).all(axis=1)]

    return final_assignment


def random_baseline(batches, preds_matrix, n_experts, deferral_rate):
    # Filter the columns to get probabilities of correctness and limit to n_experts
    filtered_columns = preds_matrix.filter(regex='^(standard|classifier_h)(?!.*(confidence|avg)$)').columns
    expert_columns = [f'standard#{i}' for i in range(int(n_experts))]
    filtered_columns = ['classifier_h'] + expert_columns
    preds_matrix = preds_matrix[filtered_columns]

    # Load the capacity matrix
    capacities = pd.read_csv(f'../../capacity_constraints/H_{int(n_experts)}_experts_{deferral_rate}_percent.csv')

    # Initialize final assignment matrix
    final_assignment = pd.DataFrame(index=preds_matrix.index, columns=filtered_columns, data=0)

    # Loop through each batch and filter the preds and capacity matrix
    for batch_id in batches['batch'].unique():
        batch_indices = batches[batches['batch'] == batch_id].index
        if len(batch_indices) == len(batches[batches['batch'] == 0]):
            preds_batch = preds_matrix.loc[batch_indices]
            H = capacities.iloc[batch_id, :int(n_experts)]
            H.index = preds_batch.columns[1:]
            H['classifier_h'] = len(batch_indices) - capacities.iloc[batch_id, :int(n_experts)].sum()
            H = H.astype(int)

            # Initialize capacity tracker
            current_capacity = H.copy()

            # Iterate over each instance in the batch
            for idx in preds_batch.index:
                available_decision_makers = current_capacity[current_capacity > 0].index.tolist()

                if available_decision_makers:
                    # Randomly choose a decision-maker from the available ones
                    chosen_decision_maker = np.random.choice(available_decision_makers)
                    final_assignment.at[idx, chosen_decision_maker] = 1
                    current_capacity[chosen_decision_maker] -= 1

    # Remove all rows that where not assigned (batch didn't reach the necessary size)
    final_assignment = final_assignment.loc[~(final_assignment == 0).all(axis=1)]

    return final_assignment


def rl_baseline(batches, preds_matrix, n_experts, deferral_rate):

    # Filter the columns to get probabilities of correctness and limit to n_experts
    filtered_columns = preds_matrix.filter(regex='^(standard|classifier_h|prediction_set)').columns
    expert_columns = [f'standard#{i}' for i in range(int(n_experts))]
    filtered_columns = ['classifier_h'] + expert_columns
    cp_columns = preds_matrix.filter(regex='^prediction_set').columns.tolist()
    cp_preds_matrix = preds_matrix[cp_columns]
    preds_matrix = preds_matrix[filtered_columns]

    # Load the capacity matrix
    capacities = pd.read_csv(f'../../capacity_constraints/H_{int(n_experts)}_experts_{deferral_rate}_percent.csv')

    # Initialize final assignment matrix
    final_assignment = pd.DataFrame(index=preds_matrix.index, columns=filtered_columns, data=0)

    # Loop through each batch and filter the preds and capacity matrix
    for batch_id in batches['batch'].unique():
        batch_indices = batches[batches['batch'] == batch_id].index
        if len(batch_indices) == len(batches[batches['batch'] == 0]):
            preds_batch = preds_matrix.loc[batch_indices]
            cp_preds_batch = cp_preds_matrix.loc[batch_indices]
            H = capacities.iloc[batch_id, :int(n_experts)]
            H.index = preds_batch.columns[1:]  # Excluding 'classifier_h'
            total_expert_capacity = H.sum()
            H['classifier_h'] = len(batch_indices) - total_expert_capacity
            H = H.astype(int)

            # Initialize capacity tracker
            current_capacity = H.copy()

            chosen_prediction_set = f"prediction_set_0.{int(deferral_rate) // 10}"

            if chosen_prediction_set:
                print(f"Using {chosen_prediction_set} for assignment.")

                # Get indexes where prediction set is [False, False]
                null_set_indexes = cp_preds_batch[cp_preds_batch[chosen_prediction_set].apply(lambda x: np.all(np.array(x) == False))].index
                assigned_null_set_indexes = []

                # Randomly assign these indexes to experts until their capacities are reached
                for idx in null_set_indexes:
                    available_decision_makers = current_capacity[expert_columns][current_capacity[expert_columns] > 0].index.tolist()
                    if available_decision_makers:
                        # Randomly choose a decision-maker from the available ones
                        chosen_decision_maker = np.random.choice(available_decision_makers)
                        final_assignment.at[idx, chosen_decision_maker] = 1
                        current_capacity[chosen_decision_maker] -= 1
                        assigned_null_set_indexes.append(idx)

                # Assign remaining instances to the classifier_h
                remaining_indexes = preds_batch.index.difference(assigned_null_set_indexes)
                final_assignment.loc[remaining_indexes, 'classifier_h'] = 1

            else:
                # If no suitable prediction set is found, assign all instances to classifier_h
                final_assignment.loc[batch_indices, 'classifier_h'] = 1

    # Remove all rows that where not assigned (batch didn't reach the necessary size)
    final_assignment = final_assignment.loc[~(final_assignment == 0).all(axis=1)]

    return final_assignment



# Get tables, varying training data seed and noise seed
# Define alphas to get weights for weights for the probabilities
n_alpha = 50
alpha_list = [i / n_alpha for i in range(n_alpha + 1)]

# Load the cp settings from the JSON file
settings_file_path = "../../results/deferral_results/settings_cp.json"
with open(settings_file_path, "r") as f:
    settings_cp = json.load(f)
    
# Loop through each setting and save the value of each variable to a corresponding variable
for setting in settings_cp:
    noise, noise_seed, nbr_experts, deferral_rate, n, tr_seed = setting
    end_name = f"{noise}_seed_{noise_seed}"
    print(setting)
    print(end_name)
    if not (os.path.exists(f'../../results/deferral_results/assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')):

        # Load alert data with CP sets
        train_CP = pd.read_parquet('../../density_based_CP/feature_extraction/processed_data/BAF_train_with_prediction_sets.parquet')
        test_CP = pd.read_parquet(f'../../density_based_CP/feature_extraction/processed_data/BAF_test_with_prediction_sets_{end_name}.parquet')

        # Load batch vector
        batches_df = pd.read_csv(f'../../capacity_constraints/batch_{noise}.csv', index_col='index')

        # Open and load the probability of correctness of each expert
        with open(f"../../L2D/deferral/l2d_predictions/n_{n}/seed_{tr_seed}/ova_{end_name}.pkl", "rb") as in_file:
            prob_correctness = pickle.load(in_file)

        # Load the average probability of correctness of each expert on the training set
        expert_avg_probs = pd.read_csv(f'../../capacity_constraints/n_{n}/seed_{tr_seed}/expert_avg_probabilities.csv')
        expert_avg_probs = expert_avg_probs.add_suffix('_avg')
        repeated_avg_probs = pd.concat([expert_avg_probs] * len(test_CP), ignore_index=True)
        repeated_avg_probs.index = test_CP.index

        # Join the test data with the prob of correctness data
        test_with_prob_correctness = test_CP.join(prob_correctness)
        test_with_prob_correctness = test_with_prob_correctness.join(repeated_avg_probs)

        # Filter the columns
        filtered_columns = test_with_prob_correctness.filter(regex='^(standard|prediction_set|classifier_h)').columns
        filtered_test_with_prob_correctness = test_with_prob_correctness[filtered_columns]

        # Calculate assignment matrices
        os.makedirs('../../results/deferral_results/assignment_matrices', exist_ok=True)
        os.makedirs('../../results/deferral_results/alphas', exist_ok=True)
        os.makedirs('../../results/deferral_results/count_rl', exist_ok=True)
        os.makedirs('../../results/deferral_results/count_l2d', exist_ok=True)

        alphas, objective, assignment, count_rl, count_l2d = optimize_assignment_and_coverage(batches_df, filtered_test_with_prob_correctness, alpha_list, nbr_experts, deferral_rate)
        assignment.to_parquet(f'../../results/deferral_results/assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')
        pd.DataFrame(alphas).to_csv(f'../../results/deferral_results/alphas/alphas_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.csv', index=False)
        pd.DataFrame(count_rl).to_csv(f'../../results/deferral_results/count_rl/count_rl_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.csv', index=False)
        pd.DataFrame(count_l2d).to_csv(f'../../results/deferral_results/count_l2d/count_l2d_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.csv', index=False)



# Load the ds settings from the JSON file
settings_file_path = "../../results/deferral_results/settings_ds.json"
with open(settings_file_path, "r") as f:
    settings_ds = json.load(f)
    
# Loop through each setting and save the value of each variable to a corresponding variable
for setting in settings_ds:
    noise, noise_seed, nbr_experts, deferral_rate, n, tr_seed = setting
    end_name = f"{noise}_seed_{noise_seed}"
    if not (os.path.exists(f'../../results/deferral_results/ds_assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')):
        print(setting)
        # Load batch vector
        batches_df = pd.read_csv(f'../../capacity_constraints/batch_{noise}.csv', index_col='index')

        # Open and load the probability of correctness of each expert with density-softmax models
        with open(f"../../L2D/deferral/l2d_ds_predictions/n_{n}/seed_{tr_seed}/ova_{end_name}.pkl", "rb") as in_file:
            prob_correctness_ds = pickle.load(in_file)

        # Calculate assignment matrices
        os.makedirs('../../results/deferral_results/ds_assignment_matrices', exist_ok=True)

        assignment = density_softmax(batches_df, prob_correctness_ds, nbr_experts, deferral_rate)
        assignment.to_parquet(f'../../results/deferral_results/ds_assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')


 
# Load the l2d settings from the JSON file
settings_file_path = "../../results/deferral_results/settings_l2d.json"
with open(settings_file_path, "r") as f:
    settings_l2d = json.load(f)
    
# Loop through each setting and save the value of each variable to a corresponding variable
for setting in settings_l2d:
    noise, noise_seed, nbr_experts, deferral_rate, n, tr_seed = setting
    end_name = f"{noise}_seed_{noise_seed}"
    if not (os.path.exists(f'../../results/deferral_results/l2d_assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')):
                        
        # Load alert data with CP sets
        train_CP = pd.read_parquet('../../density_based_CP/feature_extraction/processed_data/BAF_train_with_prediction_sets.parquet')
        test_CP = pd.read_parquet(f'../../density_based_CP/feature_extraction/processed_data/BAF_test_with_prediction_sets_{end_name}.parquet')

        # Load batch vector
        batches_df = pd.read_csv(f'../../capacity_constraints/batch_{noise}.csv', index_col='index')

        # Open and load the probability of correctness of each expert
        with open(f"../../L2D/deferral/l2d_predictions/n_{n}/seed_{tr_seed}/ova_{end_name}.pkl", "rb") as in_file:
            prob_correctness = pickle.load(in_file)

        # Load the average probability of correctness of each expert on the training set
        expert_avg_probs = pd.read_csv(f'../../capacity_constraints/n_{n}/seed_{tr_seed}/expert_avg_probabilities.csv')
        expert_avg_probs = expert_avg_probs.add_suffix('_avg')
        repeated_avg_probs = pd.concat([expert_avg_probs] * len(test_CP), ignore_index=True)
        repeated_avg_probs.index = test_CP.index

        # Join the test data with the prob of correctness data
        test_with_prob_correctness = test_CP.join(prob_correctness)
        test_with_prob_correctness = test_with_prob_correctness.join(repeated_avg_probs)

        # Filter the columns
        filtered_columns = test_with_prob_correctness.filter(regex='^(standard|prediction_set|classifier_h)').columns
        filtered_test_with_prob_correctness = test_with_prob_correctness[filtered_columns]

        # Calculate assignment matrices
        os.makedirs('../../results/deferral_results/l2d_assignment_matrices', exist_ok=True)

        assignment = l2d_baseline(batches_df, filtered_test_with_prob_correctness, nbr_experts, deferral_rate)
        assignment.to_parquet(f'../../results/deferral_results/l2d_assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')



# Load the rl settings from the JSON file
settings_file_path = "../../results/deferral_results/settings_rl.json"
with open(settings_file_path, "r") as f:
    settings_rl = json.load(f)
    
# Loop through each setting and save the value of each variable to a corresponding variable
for setting in settings_rl:
    noise, noise_seed, nbr_experts, deferral_rate, n, tr_seed = setting
    end_name = f"{noise}_seed_{noise_seed}"
    if not (os.path.exists(f'../../results/deferral_results/rl_assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')):
                        
        # Load alert data with CP sets
        train_CP = pd.read_parquet('../../density_based_CP/feature_extraction/processed_data/BAF_train_with_prediction_sets.parquet')
        test_CP = pd.read_parquet(f'../../density_based_CP/feature_extraction/processed_data/BAF_test_with_prediction_sets_{end_name}.parquet')

        # Load batch vector
        batches_df = pd.read_csv(f'../../capacity_constraints/batch_{noise}.csv', index_col='index')

        # Open and load the probability of correctness of each expert
        with open(f"../../L2D/deferral/l2d_predictions/n_{n}/seed_{tr_seed}/ova_{end_name}.pkl", "rb") as in_file:
            prob_correctness = pickle.load(in_file)

        # Load the average probability of correctness of each expert on the training set
        expert_avg_probs = pd.read_csv(f'../../capacity_constraints/n_{n}/seed_{tr_seed}/expert_avg_probabilities.csv')
        expert_avg_probs = expert_avg_probs.add_suffix('_avg')
        repeated_avg_probs = pd.concat([expert_avg_probs] * len(test_CP), ignore_index=True)
        repeated_avg_probs.index = test_CP.index

        # Join the test data with the prob of correctness data
        test_with_prob_correctness = test_CP.join(prob_correctness)
        test_with_prob_correctness = test_with_prob_correctness.join(repeated_avg_probs)

        # Filter the columns
        filtered_columns = test_with_prob_correctness.filter(regex='^(standard|prediction_set|classifier_h)').columns
        filtered_test_with_prob_correctness = test_with_prob_correctness[filtered_columns]

        # Calculate assignment matrices
        os.makedirs('../../results/deferral_results/rl_assignment_matrices', exist_ok=True)

        assignment = rl_baseline(batches_df, filtered_test_with_prob_correctness, nbr_experts, deferral_rate)
        assignment.to_parquet(f'../../results/deferral_results/rl_assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')



# Load the random settings from the JSON file
settings_file_path = "../../results/deferral_results/settings_random.json"
with open(settings_file_path, "r") as f:
    settings_random = json.load(f)
    
# Loop through each setting and save the value of each variable to a corresponding variable
for setting in settings_random:
    noise, noise_seed, nbr_experts, deferral_rate, n, tr_seed = setting
    end_name = f"{noise}_seed_{noise_seed}"
    if not (os.path.exists(f'../../results/deferral_results/random_assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')):

        # Load alert data with CP sets
        train_CP = pd.read_parquet('../../density_based_CP/feature_extraction/processed_data/BAF_train_with_prediction_sets.parquet')
        test_CP = pd.read_parquet(f'../../density_based_CP/feature_extraction/processed_data/BAF_test_with_prediction_sets_{end_name}.parquet')

        # Load batch vector
        batches_df = pd.read_csv(f'../../capacity_constraints/batch_{noise}.csv', index_col='index')

        # Open and load the probability of correctness of each expert
        with open(f"../../L2D/deferral/l2d_predictions/n_{n}/seed_{tr_seed}/ova_{end_name}.pkl", "rb") as in_file:
            prob_correctness = pickle.load(in_file)

        # Load the average probability of correctness of each expert on the training set
        expert_avg_probs = pd.read_csv(f'../../capacity_constraints/n_{n}/seed_{tr_seed}/expert_avg_probabilities.csv')
        expert_avg_probs = expert_avg_probs.add_suffix('_avg')
        repeated_avg_probs = pd.concat([expert_avg_probs] * len(test_CP), ignore_index=True)
        repeated_avg_probs.index = test_CP.index

        # Join the test data with the prob of correctness data
        test_with_prob_correctness = test_CP.join(prob_correctness)
        test_with_prob_correctness = test_with_prob_correctness.join(repeated_avg_probs)

        # Filter the columns
        filtered_columns = test_with_prob_correctness.filter(regex='^(standard|prediction_set|classifier_h)').columns
        filtered_test_with_prob_correctness = test_with_prob_correctness[filtered_columns]

        # Calculate assignment matrices
        os.makedirs('../../results/deferral_results/random_assignment_matrices', exist_ok=True)

        assignment = random_baseline(batches_df, filtered_test_with_prob_correctness, nbr_experts, deferral_rate)
        assignment.to_parquet(f'../../results/deferral_results/random_assignment_matrices/assignments_noise_{end_name}_{nbr_experts}_experts_{deferral_rate}_deferral_rate_1over{n}_traininig_size_seed{tr_seed}.parquet')