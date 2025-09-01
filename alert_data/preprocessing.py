import yaml
import pandas as pd
from sklearn import metrics
import numpy as np
import os

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

BAF = pd.read_csv('../data/Base.csv')

BAF.sort_values(by = 'month', inplace = True)
BAF.reset_index(inplace=True)
BAF.drop(columns = 'index', inplace = True)
BAF.index.rename('case_id', inplace=True)

data_cfg_path = '../data/data_config.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

BAF.loc[:,data_cfg['data_cols']['categorical']] = BAF.loc[:,data_cfg['data_cols']['categorical']].astype('category')

if not os.path.isfile('../alert_model/best_model.pickle'):
    print('The Alert Model is not Trained!')
else:
    BAF_dep = pd.read_parquet('../alert_data/BAF_alert_model_score.parquet')
    BAF_dep["month"] = BAF.loc[BAF_dep.index,"month"]

    BAF_val = BAF_dep.loc[BAF_dep['month'] == 3]
    tpr, t = fpr_thresh(BAF_val['fraud_bool'], BAF_val['model_score'], 0.05)
    print(f"TPR: {tpr}; T: {t}")
    alerts_5 = BAF_dep.loc[BAF_dep['model_score'] > t]

    if not os.path.isfile('../alert_data/alerts.parquet'):
        alerts_5.to_parquet('../alert_data/alerts.parquet')

print(alerts_5['fraud_bool'].mean())
print(len(alerts_5['fraud_bool']))