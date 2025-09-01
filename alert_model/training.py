import pandas as pd
import yaml
import numpy as np
import hpo
import os
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Function to ensure categorical columns have the correct categories
def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

# Function to calculate the true positive rate at a given false positive rate threshold
def fpr_thresh(y_true, y_pred, fpr):
    results = pd.DataFrame()
    results["true"] = y_true
    results["score"] = y_pred
    temp = results.sort_values(by="score", ascending=False)

    N = (temp["true"] == 0).sum()
    FP = round(fpr * N)
    aux = temp[temp["true"] == 0]
    threshold = aux.iloc[FP - 1, 1]
    y_pred = np.where(results["score"] >= threshold, 1, 0)
    tpr = metrics.recall_score(y_true, y_pred)

    return tpr, threshold

# Load configuration file
with open('../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

# Data loading
data = pd.read_csv('../data/Base.csv')
LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

# Preprocess data
data.sort_values(by = 'month', inplace = True)
data.reset_index(inplace=True)
data.drop(columns = 'index', inplace = True)
data.index.rename('case_id', inplace=True)
data.loc[:,data_cfg['data_cols']['categorical']] = data.loc[:,data_cfg['data_cols']['categorical']].astype('category')
data = cat_checker(data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

# Split data into training, validation, and deployment sets
train = data.loc[(data["month"] < 3)].drop(columns="month")
ml_val = data.loc[(data["month"] == 3)].drop(columns="month")
deployment = data.loc[(data["month"] > 2)].drop(columns="month")

X_train = train.drop(columns = 'fraud_bool')
y_train = train['fraud_bool']
X_val = ml_val.drop(columns = 'fraud_bool') 
y_val = ml_val['fraud_bool']
X_dep = deployment.drop(columns = 'fraud_bool')
y_dep = deployment['fraud_bool']

# Hyperparameter optimization if the best model does not exist
if not (os.path.exists('../alert_model/best_model.pickle')):
    opt = hpo.HPO(X_train,X_val,y_train,y_val, method = 'TPE', path = f"../alert_model")
    opt.initialize_optimizer(CATEGORICAL_COLS, 25)

# Load the best model
with open('../alert_model/best_model.pickle', 'rb') as infile:
    model = pickle.load(infile)

# Predict probabilities on the validation set
y_pred = model.predict_proba(X_val)
y_pred = y_pred[:,1]

# Calculate the true positive rate at 5% false positive rate
roc_curve_clf = dict()
rec_at_5, thresh = fpr_thresh(y_val, y_pred, 0.05)
print(rec_at_5, thresh)

# Plot ROC Curve for deployment and validation sets
y_dep_pred = model.predict_proba(X_dep)
y_dep_pred = y_dep_pred[:,1]
fpr, tpr, thresholds = roc_curve(y_dep, y_dep_pred)
roc_auc = auc(fpr, tpr)
fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_pred)
roc_auc_val = auc(fpr_val, tpr_val)

os.makedirs('../results/figs/alert_model', exist_ok=True)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Deployment ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_val, tpr_val, color='green', lw=2, linestyle='--', label='Validation ROC curve (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('ROC curve - Alert Model', fontsize = 18)
plt.legend(loc="lower right", fontsize = 14)
plt.savefig(f'../results/figs/alert_model/ROC_AUC_curve.pdf', format='pdf', dpi=300)
plt.clf()

# Save deployment data with model scores
os.makedirs('../alert_data/', exist_ok = True)
deployment['model_score'] = model.predict_proba(deployment.drop(columns = 'fraud_bool'))[:,1]
deployment.to_parquet('../alert_data/BAF_alert_model_score.parquet')

# Calculate recall on deployment split
deployment_y_pred = np.where(deployment['model_score'] >= thresh, 1, 0)
deployment_recall = metrics.recall_score(deployment['fraud_bool'], deployment_y_pred)
print(f"Recall on deployment split: {deployment_recall}")

# Calculate FPR on the deployment split
deployment_y_pred = np.where(deployment['model_score'] >= thresh, 1, 0)
conf_matrix = confusion_matrix(deployment['fraud_bool'], deployment_y_pred)
fp = conf_matrix[0, 1]  # False Positives
tn = conf_matrix[0, 0]  # True Negatives
fpr_deployment = fp / (fp + tn)

# Print the FPR for deployment split
print(f"False Positive Rate on deployment split: {fpr_deployment}")

model_properties = {'fpr':0.05,
                    'fnr': 1 - rec_at_5,
                    'threshold': thresh
                    }

file_to_store = open("../alert_model/model_properties.pickle", "wb")
pickle.dump(model_properties, file_to_store)
file_to_store.close()

# Print model parameters
print(model.get_params()) 

