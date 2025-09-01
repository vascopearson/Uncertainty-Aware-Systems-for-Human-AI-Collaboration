import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import numpy as np

from density_softmax_h import RealNVP

class RealNVPHPO:
    def __init__(self, latent_features_train, latent_features_val, method="TPE", parameters=None, path="", patience=10):
        self.max_trials = int(parameters['total'])
        self.startup = int(parameters['startups'])
        self.trial_count = 0 
        self.latent_features_train = latent_features_train
        self.latent_features_val = latent_features_val
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
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=self.startup))
        study.set_user_attr("latent_features_train", self.latent_features_train)
        study.set_user_attr("latent_features_val", self.latent_features_val)

        os.makedirs(self.path, exist_ok=True)
        with open(f"{self.path}/config.yaml", "w") as fout:
            yaml.dump(self.config, fout)

        optuna.logging.set_verbosity(optuna.logging.FATAL)
        study.optimize(self.objective, n_trials=self.max_trials)

        trials_df = study.trials_dataframe()
        with open(f"{self.path}/history.yaml", "w") as fout:
            yaml.dump(self.params_hist, fout)

        return

    def objective(self, trial):
        print(f"Starting trial {trial.number}...")
        latent_features_train = trial.study.user_attrs["latent_features_train"]
        latent_features_val = trial.study.user_attrs["latent_features_val"]

        # Define hyperparameters to optimize
        num_coupling_layers = trial.suggest_int("num_coupling_layers", 2, 10)
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=32)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 10, 128, log=True)
        num_epochs = trial.suggest_int("num_epochs", 50, 300)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        realnvp_model = RealNVP(num_coupling_layers=num_coupling_layers, 
                                input_dim=latent_features_train.shape[1], 
                                hidden_dim=hidden_dim).to(device)

        optimizer = optim.Adam(realnvp_model.parameters(), lr=learning_rate)
        train_loader = DataLoader(TensorDataset(latent_features_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(latent_features_val), batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            realnvp_model.train()
            running_loss = 0.0

            for features in train_loader:
                features = features[0].to(device)
                optimizer.zero_grad()
                loss = realnvp_model.log_loss(features)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Training loss: {running_loss / len(train_loader):.4f}")

            val_loss = self.evaluate_model(realnvp_model, val_loader, device)
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                os.makedirs(self.path, exist_ok=True)
                torch.save(realnvp_model, os.path.join(self.path, 'best_realnvp_model.pth'))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        return best_val_loss

    def evaluate_model(self, model, val_loader, device):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for features in val_loader:
                features = features[0].to(device)
                loss = model.log_loss(features)
                running_loss += loss.item()
        return running_loss / len(val_loader)
