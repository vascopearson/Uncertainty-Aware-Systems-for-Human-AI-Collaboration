# %%
import pandas as pd
import yaml
import numpy as np
import hpo
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import re


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

n_jobs = 10

# Data loading
data = pd.read_parquet(f'../../alert_data/alerts.parquet')
data = cat_checker(data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

train = data.loc[(data["month"].isin([3, 4, 5]))]
val = data.loc[data["month"] == 6]

X_train = train.drop(columns = ['fraud_bool','model_score','month'])
y_train = train['fraud_bool']

X_val = val.drop(columns = ['fraud_bool','model_score','month']) 
y_val = val['fraud_bool']

w_train = y_train.replace([0,1],[data_cfg['lambda'],1])
w_val = y_val.replace([0,1],[data_cfg['lambda'],1])

p_train = (y_train*w_train).sum()/(w_train.sum())
p_val = (y_val*w_val).sum()/(w_val.sum())

init_train = np.log((p_train)/(1-p_train))
init_val = np.log((p_val)/(1-p_val))

n = 0
for param_space_dic in os.listdir('../../L2D/classifier_h/param_spaces/'):
    with open('../../L2D/classifier_h/param_spaces/' + param_space_dic, 'r') as infile:
        param_space = yaml.safe_load(infile)

    for initial in np.arange(init_train, init_train + 2, 0.2):
        param_space['init_score'] = initial
        os.makedirs(f'../../L2D/classifier_h/models/', exist_ok=True)
        
        if not (os.path.exists(f'../../L2D/classifier_h/models/model_{n}')):
            opt = hpo.HPO(X_train,X_val,y_train,y_val,w_train,w_val, parameters = param_space, method = 'TPE', path = f"../../L2D/classifier_h/models/model_{n}")
            opt.initialize_optimizer(data_cfg['categorical_dict'], n_jobs)
            n +=1
        else:
            print('model is trained')
            n +=1

Trials = []

for model in os.listdir('../../L2D/classifier_h/models/'):
    study = int(model.split('_')[-1])
    with open('../../L2D/classifier_h/models/' + model + '/history.yaml', 'r') as infile:
        param_hist = yaml.safe_load(infile)

    with open('../../L2D/classifier_h/models/' + model + '/config.yaml', 'r') as infile:
        conf = yaml.safe_load(infile)
    
    temp = pd.DataFrame(param_hist)
    temp['study'] = study
    temp['max_depth_max'] = conf['params']['max_depth']['range'][1]
    Trials.append(temp)

Trials = pd.concat(Trials)
Trials = Trials.reset_index(drop = True)
Trials['study'] = Trials['study'].astype(int)
a = Trials

selec_ix = a.loc[a['ll'] == a['ll'].min(),'study'].to_numpy()[0]

selected_model_path = f'../../L2D/classifier_h/models/model_{selec_ix}'

with open(f'{selected_model_path}/best_model.pickle', 'rb') as infile:
    model = pickle.load(infile)

with open(f'{selected_model_path}/config.yaml', 'r') as infile:
    model_cfg = yaml.safe_load(infile)


# Load test data
test = data.loc[(data["month"] == 7)]

X_test = test.drop(columns = ["month",'model_score', "fraud_bool"])
y_test = test["fraud_bool"]
w_test = y_test.replace([0,1],[data_cfg['lambda'],1])

p_test = (y_test*w_test).sum()/(w_test.sum())
init_test = np.log((p_test)/(1-p_test))

selected_model = dict()
init_score = model_cfg['init_score']
selected_model['init_score'] = float(init_score)
selected_model['threshold'] = 0.5


################# RESULTS #################

# Training set results
model_preds = pd.Series(output(X_train, model, init_score) >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_train, model_preds).ravel()
avg_cost_model = (data_cfg['lambda']*fp + fn)/(tn+fp+fn+tp)

selected_model['fpr_train'] = float(fp/(fp+tn))
selected_model['fnr_train'] = float(fn/(fn+tp))
selected_model['prev_train'] = float(y_train.mean())
selected_model['cost_train'] = float(avg_cost_model)

tn, fp, fn, tp = confusion_matrix(y_train, np.ones(len(y_train))).ravel()

print(f"Training Set: {avg_cost_model:.3f}.")


# Validation set results
model_preds = pd.Series(output(X_val, model, init_score) >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_val, model_preds).ravel()
avg_cost_model = (data_cfg['lambda']*fp + fn)/(tn+fp+fn+tp)

selected_model['fpr_val'] = float(fp/(fp+tn))
selected_model['fnr_val'] = float(fn/(fn+tp))
selected_model['prev_val'] = float(y_val.mean())
selected_model['cost_val'] = float(avg_cost_model)

tn, fp, fn, tp = confusion_matrix(y_val, np.ones(len(y_val))).ravel()

print(f"Validation Set: {avg_cost_model:.3f}.")


# Test set results
model_preds = pd.Series(output(X_test, model, init_score) > 0.5, index=X_test.index).astype(int)

# Save model_preds with indices from X_test
model_preds_df = model_preds.to_frame(name='model_preds')

tn, fp, fn, tp = confusion_matrix(y_test, model_preds).ravel()
av_cost_model = (data_cfg['lambda']*fp + fn)/(tn+fp+fn+tp)

selected_model['fpr_test'] = float(fp/(fp+tn))
selected_model['fnr_test'] = float(fn/(fn+tp))
selected_model['prev_test'] = float(y_test.mean())
selected_model['cost_test'] = float(av_cost_model)

tn, fp, fn, tp = confusion_matrix(y_test, np.ones(len(y_test))).ravel()
print(f"Test Set: {av_cost_model:.5f}.")


# Noisy subset Test set results
noise_levels = {}
for file_name in os.listdir('../../drift_alert_data'):
    if file_name.endswith('.parquet'):
        match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
        noise_level = match.group(1)
        end_name = f"{match.group(1)}_seed_{match.group(2)}"

        # Load noisy test data
        test_noisy = pd.read_parquet(f'../../drift_alert_data/{file_name}')

        X_test_noisy = test_noisy.drop(columns = ["month",'model_score', "fraud_bool"])
        y_test_noisy = test_noisy["fraud_bool"]
        w_test_noisy = y_test_noisy.replace([0,1],[data_cfg['lambda'],1])

        p_test_noisy = (y_test_noisy*w_test_noisy).sum()/(w_test_noisy.sum())
        init_test_noisy = np.log((p_test_noisy)/(1-p_test_noisy))

        selected_model = dict()
        init_score = model_cfg['init_score']
        selected_model['init_score'] = float(init_score)
        selected_model['threshold'] = 0.5

        # Noisy Test set results
        model_preds_noisy = pd.Series(output(X_test_noisy, model, init_score) >= 0.5, index=X_test_noisy.index).astype(int)

        # Save model_preds with indices from X_test
        model_preds_df = model_preds.to_frame(name='model_preds')
        model_preds_noisy_df = model_preds_noisy.to_frame(name='model_preds')
        model_preds_df = pd.concat([model_preds_df, model_preds_noisy_df])
        model_preds_df.to_parquet(f'../../L2D/classifier_h/selected_model/model_test_preds_{end_name}.parquet')

        tn, fp, fn, tp = confusion_matrix(y_test_noisy, model_preds_noisy).ravel()
        avg_cost_model = (data_cfg['lambda']*fp + fn)/(tn+fp+fn+tp)

        selected_model['fpr_test'] = float(fp/(fp+tn))
        selected_model['fnr_test'] = float(fn/(fn+tp))
        selected_model['prev_test'] = float(y_test.mean())
        selected_model['cost_test'] = float(avg_cost_model)

        tn, fp, fn, tp = confusion_matrix(y_test, np.ones(len(y_test))).ravel()


# Print avg noisy test set results
noise_levels = {}
for file_name in os.listdir('../../drift_alert_data'):
    if file_name.endswith('.parquet'):
        match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
        noise_level = match.group(1)
        end_name = f"{match.group(1)}_seed_{match.group(2)}"

        # Load noisy test data
        test_noisy = pd.read_parquet(f'../../drift_alert_data/{file_name}')

        X_test_noisy = test_noisy.drop(columns = ["month",'model_score', "fraud_bool"])
        X_test_noisy = pd.concat([X_test, test_noisy.drop(columns=["month", 'model_score', "fraud_bool"])])
        y_test_noisy = test_noisy["fraud_bool"]
        y_test_noisy = pd.concat([y_test, test_noisy["fraud_bool"]])
        w_test_noisy = pd.concat([w_test, test_noisy["fraud_bool"].replace([0,1],[data_cfg['lambda'],1])])
        w_test_noisy = y_test_noisy.replace([0,1],[data_cfg['lambda'],1])

        p_test_noisy = (y_test_noisy*w_test_noisy).sum()/(w_test_noisy.sum())
        init_test_noisy = np.log((p_test_noisy)/(1-p_test_noisy))

        selected_model = dict()
        init_score = model_cfg['init_score']
        selected_model['init_score'] = float(init_score)
        selected_model['threshold'] = 0.5

        # Noisy Test set results
        model_preds_noisy = pd.Series(output(X_test_noisy, model, init_score) >= 0.5, index=X_test_noisy.index).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test_noisy, model_preds_noisy).ravel()
        avg_cost_model = (data_cfg['lambda']*fp + fn)/(tn+fp+fn+tp)
        
        if noise_level not in noise_levels:
            noise_levels[noise_level] = []
        noise_levels[noise_level].append(avg_cost_model)

        selected_model['fpr_test'] = float(fp/(fp+tn))
        selected_model['fnr_test'] = float(fn/(fn+tp))
        selected_model['prev_test'] = float(y_test.mean())
        selected_model['cost_test'] = float(avg_cost_model)

        tn, fp, fn, tp = confusion_matrix(y_test, np.ones(len(y_test))).ravel()
        
# Print the average misclassification cost per noise level
for noise_level, costs in noise_levels.items():
    avg_cost = np.mean(costs)
    print(f"Noise Level {noise_level}: Average Misclassification Cost = {avg_cost:.5f}")
    

os.makedirs(f'../../L2D/classifier_h/selected_model/', exist_ok=True)

with open(f'../../L2D/classifier_h/selected_model/best_model.pickle', 'wb') as outfile:
    pickle.dump(model, outfile)

with open(f'../../L2D/classifier_h/selected_model/model_properties.yaml', 'w') as outfile:
    yaml.dump(selected_model, outfile)




# Calibration and ROC results 

# Original test set
probs_test = pd.Series(output(X_test, model, init_score), index=X_test.index)

e_c = y_val.replace([0,1], [data_cfg['lambda'],1]).mean()
reb_1 = (y_val.mean()/e_c)
reb_0 = ((1-y_val).mean()*data_cfg['lambda']/e_c)
n_min = len(y_test.loc[y_test==0])
n_max = int(n_min*reb_1/reb_0)
oversampled = pd.concat([y_test.loc[y_test == 0], y_test.loc[y_test == 1].sample(replace=True, n = n_max, random_state=42)]).index
outcomes = y_test.loc[oversampled]
prob_true, prob_pred = calibration_curve(y_true = outcomes, y_prob = probs_test.loc[oversampled], strategy='quantile', n_bins = 10)

print("Calibration on the test set: ", np.mean(np.abs(prob_true - prob_pred))*100)

# Calculate the ROC-AUC for the original test set
test_roc_auc = roc_auc_score(y_true = outcomes, y_score = probs_test.loc[oversampled])
print(f"Test ROC-AUC: {test_roc_auc:.5f}")

# Noisy test set
for file_name in os.listdir('../../drift_alert_data'):
    if file_name.endswith('.parquet'):
        match = re.search(r"_(\d+\.\d+_\d+\.\d+)_seed_(\d+)\.parquet", file_name)
        end_name = f"{match.group(1)}_seed{match.group(2)}"

        test_noisy = pd.read_parquet(f'../../drift_alert_data/{file_name}')

        X_test_noisy = test_noisy.drop(columns = ["month",'model_score', "fraud_bool"])
        y_test_noisy = test_noisy["fraud_bool"]

        probs_test_noisy = pd.Series(output(X_test_noisy, model, init_score), index=X_test_noisy.index)

        n_min = len(y_test_noisy.loc[y_test_noisy==0])
        n_max = int(n_min*reb_1/reb_0)
        oversampled = pd.concat([y_test_noisy.loc[y_test_noisy == 0], y_test_noisy.loc[y_test_noisy == 1].sample(replace=True, n = n_max, random_state=42)]).index
        outcomes = y_test_noisy.loc[oversampled]
        prob_true, prob_pred = calibration_curve(y_true = outcomes, y_prob = probs_test_noisy.loc[oversampled], strategy='quantile', n_bins = 10)

        print(f"Calibration on the noisy test set {end_name}: ", np.mean(np.abs(prob_true - prob_pred))*100)

        # Calculate the ROC-AUC for the noisy test set
        test_roc_auc = roc_auc_score(y_true = outcomes, y_score = probs_test_noisy.loc[oversampled])
        print(f"Noisy Test ROC-AUC {end_name}: {test_roc_auc:.5f}")
