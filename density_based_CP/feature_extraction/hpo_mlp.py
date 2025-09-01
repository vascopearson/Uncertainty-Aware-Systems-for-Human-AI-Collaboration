import pandas as pd
import yaml
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import optuna

with open('../../data/data_config.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    return new_data

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_layer_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layer_sizes[-1], 1))  # Output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels, weights in train_loader:
        inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        weighted_loss = (loss * weights.unsqueeze(1)).mean()
        weighted_loss.backward()
        optimizer.step()
        
        running_loss += weighted_loss.item()
    return running_loss / len(train_loader)

def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels, weights in val_loader:
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            outputs = model(inputs)
            loss = nn.BCEWithLogitsLoss(reduction='none')(outputs, labels.unsqueeze(1).float())
            weighted_loss = (loss * weights.unsqueeze(1)).mean()
            running_loss += weighted_loss.item()
            probs = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid to get probabilities
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds).squeeze(), np.array(all_labels), running_loss / len(val_loader)

class HPO:

    def __init__(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        train_w,
        val_w,
        method="TPE",
        parameters=None,
        path="",
        patience=10):

        self.max_trials = int(parameters['total'])
        self.startup = int(parameters['startups'])
        self.trial_count = 0 

        self.train_w = train_w
        self.val_w = val_w

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.method = method
        self.config = parameters
        self.parameters = parameters['params']
        self.params_hist = []
        self.path = path
        self.current_best = +np.inf
        self.patience = patience

    def initialize_optimizer(self):
        self.tpe_optimization()
    
    def tpe_optimization(self):
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=self.startup)
        )

        study.set_user_attr("X_train", self.X_train)
        study.set_user_attr("X_val", self.X_val)
        study.set_user_attr("y_train", self.y_train)
        study.set_user_attr("y_val", self.y_val)

        os.makedirs(self.path, exist_ok=True)
        with open(
            "{}/config.yaml".format(self.path), "w"
        ) as fout:
            yaml.dump(self.config, fout)

        optuna.logging.set_verbosity(optuna.logging.FATAL)
        study.optimize(self.objective, n_trials=self.max_trials)

        trials_df = study.trials_dataframe()
        keys = trials_df.keys()

        with open(
            "{}/history.yaml".format(self.path), "w"
        ) as fout:
            yaml.dump(self.params_hist, fout)
        
        return

    def objective(self, trial):
        X_train = trial.study.user_attrs["X_train"]
        X_val = trial.study.user_attrs["X_val"]
        y_train = trial.study.user_attrs["y_train"]
        y_val = trial.study.user_attrs["y_val"]

        # Define the hyperparameters to optimize
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [[200, 50], [128, 50], [128, 20], [128, 128, 50], [128, 128, 20]])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        num_epochs = trial.suggest_int("num_epochs", 10, 100)
        batch_size = trial.suggest_int("batch_size", 10, 128, log=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_size = X_train.shape[1]
        model = MLP(input_size, hidden_layer_sizes, dropout_rate).to(device)

        w_train = y_train.replace([0, 1], [data_cfg['lambda'], 1])
        w_val = y_val.replace([0, 1], [data_cfg['lambda'], 1])

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                                torch.tensor(y_train.values, dtype=torch.long),
                                                torch.tensor(w_train.values, dtype=torch.float32)),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
                                              torch.tensor(y_val.values, dtype=torch.long),
                                              torch.tensor(w_val.values, dtype=torch.float32)),
                                batch_size=len(X_val), shuffle=False)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model = None

        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            preds, labels, val_loss = evaluate_model(model, val_loader, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(best_val_loss)
                best_model = model
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        if best_model is None:
            best_model = model

        preds, labels, final_val_loss = evaluate_model(best_model, val_loader, device)
        preds = (preds >= 0.5).astype(int)
        ll = final_val_loss

        param_dict = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            'll': float(ll)
        }
        print(f"Trial: {trial}")
        print(f"Parameters: {param_dict}")

        self.params_hist.append(param_dict)

        if ll < self.current_best:
            self.current_best = ll
            os.makedirs(self.path, exist_ok=True)
            torch.save(model, os.path.join(self.path, 'best_model.pth'))
            with open("{}/best_config.yaml".format(self.path), "w") as fout:
                yaml.dump([param_dict], fout)

        return ll  
