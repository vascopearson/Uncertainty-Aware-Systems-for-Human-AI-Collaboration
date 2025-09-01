import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

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


def train_single_mlp(X_train, X_val, y_train, y_val, hidden_layer_sizes, learning_rate, weight_decay, num_epochs, dropout_rate, model_path, lambda_weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_size = X_train.shape[1]
    model = MLP(input_size, hidden_layer_sizes, dropout_rate).to(device)

    # Calculate class weights
    w_train = y_train.replace([0, 1], [lambda_weight, 1])
    w_val = y_val.replace([0, 1], [lambda_weight, 1])

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                            torch.tensor(y_train.values, dtype=torch.long),
                                            torch.tensor(w_train.values, dtype=torch.float32)),
                              batch_size=30, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
                                          torch.tensor(y_val.values, dtype=torch.long),
                                          torch.tensor(w_val.values, dtype=torch.float32)),
                            batch_size=len(X_val), shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        preds, labels, val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            os.makedirs(model_path, exist_ok=True)
            torch.save(model, os.path.join(model_path, 'best_model.pth'))

    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Evaluate on the validation set to get the final metrics
    best_model = torch.load('../feature_extraction/best_model.pth')
    preds, labels, final_val_loss = evaluate_model(best_model, val_loader, device)
    preds = (preds >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print("Final Validation Loss: ", final_val_loss)
    
    model_properties = {'threshold': 0.5, 'fpr': fpr, 'fnr': fnr, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    print(model_properties)
    with open(os.path.join(model_path, 'model_properties.pickle'), "wb") as fout:
        pickle.dump(model_properties, fout)
