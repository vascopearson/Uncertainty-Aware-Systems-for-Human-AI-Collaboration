import numpy as np
from sklearn.neighbors import KernelDensity

def subset_data(Z, Y, p):
    """
    Splits the data into training and calibration subsets based on the given class and number of calibration instances.

    Parameters:
    Z (list): List of tuples containing data points and their corresponding labels.
    Y (int): The class label to filter the data.
    p (int): The number of instances to use for calibration.

    Returns:
    X_tr (np.array): Training subset.
    X_cal (np.array): Calibration subset.
    """
    X_y = [x for x, y in Z if y == Y]
    X_tr = np.array(X_y[:-p])  # All except the last p instances
    X_cal = np.array(X_y[-p:])  # The last p instances
    return X_tr, X_cal

def learn_density_estimator(X_tr, bandwidth=0.1):
    """
    Trains a Kernel Density Estimator on the given training data.

    Parameters:
    X_tr (np.array): Training data.
    bandwidth (float): Bandwidth parameter for the Kernel Density Estimator.

    Returns:
    kde (KernelDensity): Trained Kernel Density Estimator.
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(X_tr)
    return kde

def quantile(density_estimator, X_cal, alpha):
    """
    Calculates the quantile threshold for the given calibration data and confidence level.

    Parameters:
    density_estimator (KernelDensity): Trained Kernel Density Estimator.
    X_cal (np.array): Calibration data.
    alpha (float): Confidence level.

    Returns:
    float: Quantile threshold.
    """
    log_density = density_estimator.score_samples(X_cal)
    return np.percentile(np.exp(log_density), 100 * alpha)

def cp_training_algorithm(Z, Y, alpha, p, bandwidth=0.1):
    """
    Trains the density-based conformal prediction model.

    Parameters:
    Z (list): List of tuples containing data points and their corresponding labels.
    Y (list): List of unique class labels.
    alpha (float): Confidence level.
    p (dict): Dictionary containing the number of calibration instances for each class.
    bandwidth (float): Bandwidth parameter for the Kernel Density Estimator.

    Returns:
    p_hat_list (list): List of trained Kernel Density Estimators for each class.
    t_hat_list (list): List of quantile thresholds for each class.
    """
    p_hat_list = []
    t_hat_list = []

    for y in Y:
        print(f"Performing density estimation for class {y}")
        X_tr, X_cal = subset_data(Z, y, p[y])
        p_hat_y = learn_density_estimator(X_tr, bandwidth)
        print(f"Calculating quantile for class {y}")
        t_hat_y = quantile(p_hat_y, X_cal, alpha)
        p_hat_list.append(p_hat_y)
        t_hat_list.append(t_hat_y)

    return p_hat_list, t_hat_list

def cp_prediction_algorithm(X, p_hat_list, t_hat_list, Y):
    """
    Makes predictions using the density-based conformal prediction model.

    Parameters:
    X (np.array): Data points to make predictions on.
    p_hat_list (list): List of trained Kernel Density Estimators for each class.
    t_hat_list (list): List of quantile thresholds for each class.
    Y (list): List of unique class labels.

    Returns:
    C (np.array): Boolean array indicating the predicted classes for each data point.
    """
    C = np.zeros((X.shape[0], len(Y)), dtype=bool)

    for i, (y, p_hat_y, t_hat_y) in enumerate(zip(Y, p_hat_list, t_hat_list)):
        log_density = p_hat_y.score_samples(X)
        density = np.exp(log_density)
        C[:, i] = density >= t_hat_y

    return C

def calculate_coverage_metrics(data, alphas, Z, Y, p, bandwidth):
    """
    Calculates coverage metrics for the density-based conformal prediction model.

    Parameters:
    data (list): List of tuples containing data points and their corresponding labels.
    alphas (list): List of confidence levels.
    Z (list): List of tuples containing data points and their corresponding labels.
    Y (list): List of unique class labels.
    p (dict): Dictionary containing the number of calibration instances for each class.
    bandwidth (float): Bandwidth parameter for the Kernel Density Estimator.

    Returns:
    accuracies (list): List of accuracies for each confidence level.
    entry_proportions (list): List of entry proportions for each confidence level.
    null_proportions (list): List of null proportions for each confidence level.
    class_accuracies (dict): Dictionary of accuracies for each class and confidence level.
    class_entry_proportions (dict): Dictionary of entry proportions for each class and confidence level.
    class_null_proportions (dict): Dictionary of null proportions for each class and confidence level.
    """
    accuracies = []
    entry_proportions = []
    null_proportions = []
    class_accuracies = {y: [] for y in Y}
    class_entry_proportions = {y: [] for y in Y}
    class_null_proportions = {y: [] for y in Y}

    X = np.array([x for x, y in data])
    y_true_all = np.array([y for x, y in data])

    # Train the density estimator and get density scores
    p_hat_list = []
    log_density = []
    for y in Y:
        print(f"Performing density estimation for class {y}")
        X_tr, X_cal = subset_data(Z, y, p[y])
        p_hat_y = learn_density_estimator(X_tr, bandwidth)
        p_hat_list.append(p_hat_y)
        log_density.append(p_hat_y.score_samples(X_cal))
    print("Density estimation finished!")

    # Get density score for features
    density = []
    for i, (y, p_hat_y) in enumerate(zip(Y, p_hat_list)):
        print(f"Getting density scores for class {y}")
        density.append(np.exp(p_hat_y.score_samples(X)))
    print(f"Density scores calculated!")

    for alpha in alphas:
        C = np.zeros((X.shape[0], len(Y)), dtype=bool)

        print(f"Fitting Density-Based Conformal Prediction with alpha = {alpha}")
        # Calculating the quantile
        t_hat_list = []
        for y_index, y in enumerate(Y):
            print(f"Calculating quantile for class {y}")
            t_hat_y = np.percentile(np.exp(log_density[y_index]), 100 * alpha)
            t_hat_list.append(t_hat_y)
        print("Quantiles have been calculated!")

        correct_predictions = 0
        full_set_entries = 0
        total_nulls = 0
        correct_predictions_per_class = {y: 0 for y in Y}
        full_set_entries_per_class = {y: 0 for y in Y}
        total_nulls_per_class = {y: 0 for y in Y}
        total_per_class = {y: 0 for y in Y}

        # Get predictions for all data points
        for i, (y, t_hat_y) in enumerate(zip(Y, t_hat_list)):
            C[:, i] = density[i] >= t_hat_y

        for i, y_true in enumerate(y_true_all):
            total_per_class[y_true] += 1
            if C[i].sum() > 1:
                full_set_entries += 1
                full_set_entries_per_class[y_true] += 1
            elif C[i].sum() == 0:
                total_nulls += 1
                total_nulls_per_class[y_true] += 1
            if y_true in Y[C[i]]:
                correct_predictions += 1
                correct_predictions_per_class[y_true] += 1

        total = len(data)
        accuracies.append(correct_predictions / total)
        entry_proportions.append(full_set_entries / total)
        null_proportions.append(total_nulls / total)

        for y in Y:
            class_total = total_per_class[y]
            if class_total > 0:
                class_accuracies[y].append(correct_predictions_per_class[y] / class_total)
                class_entry_proportions[y].append(full_set_entries_per_class[y] / class_total)
                class_null_proportions[y].append(total_nulls_per_class[y] / class_total)
            else:
                class_accuracies[y].append(0)
                class_entry_proportions[y].append(0)
                class_null_proportions[y].append(0)

        print(accuracies, entry_proportions, null_proportions)

    return accuracies, entry_proportions, null_proportions, class_accuracies, class_entry_proportions, class_null_proportions