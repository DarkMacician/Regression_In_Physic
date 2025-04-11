import numpy as np
import pandas as pd
import torch
from numpy import log as ln
from torch import nn


def dpt(T):
    a11 = np.array([0, 0.125431, -0.167256, -0.265865, 1.59760, -1.19088, 0.264833])
    a22 = np.array([0, 0.310810, -0.171211, -0.715805, 2.48678, -1.78317, 0.394405])

    w11 = lambda T: (-1/6) * ln(T) + np.sum(
        [a11[i] * T**(-(i-1)/2)  for i in range(1, 7)]
    )
    w22 = lambda T: (-1/6) * ln(T) + ln(17/18) + np.sum(
        [a22[i] * T ** (-(i-1)/2) for i in range(1, 7)]
    )

    dw11_dt = lambda T: -(1 / (6*T)) + np.sum(
        [a11[i] * (-(i-1)/2) * T ** ((-i - 1) / 2) for i in range(1, 7)]
    )
    dw22_dt = lambda T: -(1 / (6*T)) + np.sum(
        [a22[i] * (-(i-1)/2) * T ** ((-i - 1) / 2) for i in range(1, 7)]
    )

    dw11_d2t = lambda T: (1 / (6*T**2)) + np.sum(
        [a11[i] * (-(i-1)/2) * ((-i-1)/2) * T**((-i-3)/2) for i in range(1,7)]
    )
    dw22_d2t = lambda T: (1 / (6*T**2)) + np.sum(
        [a22[i] * (-(i - 1) / 2) * ((-i - 1) / 2 )* T ** ((-i - 3) / 2) for i in range(1, 7)]
    )

    omega11 = lambda T: np.exp(w11(T))
    omega22 = lambda T: np.exp(w22(T))

    do11_dt = lambda T: dw11_dt(T) * omega11(T)
    do22_dt = lambda T: dw22_dt(T) * omega22(T)

    do11_d2t = lambda T: dw11_d2t(T) * omega11(T) + dw11_dt(T) ** 2 * omega11(T)
    do22_d2t = lambda T: dw22_d2t(T) * omega22(T) + dw22_dt(T) ** 2 * omega22(T)

    omega12 = lambda T: omega11(T) + 1/(1+2) * T * do11_dt(T)
    omega23 = lambda T: omega22(T) + 1/(2 + 2) * T * do22_dt(T)

    do12_dt = lambda T: do11_dt(T) + 1/(1+2) * (do11_dt(T) + T*do11_d2t(T))
    do23_dt = lambda T: do22_dt(T) + 1/(2 + 2) * (do22_dt(T) + T * do22_d2t(T))

    omega13 = lambda T:  omega12(T) + 1/(2+2) * T * (do12_dt(T))
    omega24 = lambda T: omega23(T) + 1 / (3 + 2) * T * (do23_dt(T))

    A = lambda T: omega22(T) / omega11(T)
    B = lambda T: (5*omega12(T) - 4*omega13(T))/omega11(T)
    C = lambda T: omega12(T) / omega11(T)

    Delta = lambda T: (6*C(T) - 5)**2/(55 - 12*B(T) + 16*A(T))

    fdp = lambda T: 1/(1 - Delta(T))

    return (3/8)*np.sqrt(T/np.pi)*(fdp(T))/(omega11(T))

def encode(df, col1, col2, col3):
  return (df[col1] * df[col2]) / np.sqrt(df[col3])

def decode(col1, col2, col3):
  return (col1 * np.sqrt(col3) / col2)

def load_models(sheet_names, model_classes):
    models = {}
    for sheet_name in sheet_names:
        for model_name, model_class in model_classes.items():
            model_path = f"best_{model_name}_{sheet_name}.pth"
            model = model_class()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            models[f"{model_name}_{sheet_name}"] = model
            print(f"Loaded {model_name} for {sheet_name}")
    return models

def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs=100000):
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    return model, best_loss


# MRE Loss Class
class MRELoss(nn.Module):
    def __init__(self):
        super(MRELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_true - y_pred) / (y_true + 1e-8))

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.layers(x)


# CNN Model for Regression
class CNNRegressor(nn.Module):
    def __init__(self, input_channels):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Đổi thứ tự để CNN nhận input_channels đúng
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.mean(x, dim=2)  # Global Average Pooling
        return self.fc(x)


# RNN Model for Regression
class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))


def process_and_save_splits(file_path, output_file):
    data_sheets = pd.read_excel(file_path, sheet_name=None)
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, df in data_sheets.items():
            print(f"Processing sheet: {sheet_name}")

            X_full = df[["rho*", 'T*']]
            y = df['D*']

            test_mask = (df['rho*'] > 0) & (df['rho*'] < 0.1)
            X_test = X_full[test_mask]
            y_test = y[test_mask]

            X_train = X_full[~test_mask]
            y_train = y[~test_mask]

            train_df = X_train.copy()
            train_df['D*'] = y_train
            test_df = X_test.copy()
            test_df['D*'] = y_test

            train_df.to_excel(writer, sheet_name=f"{sheet_name}_train", index=False)
            test_df.to_excel(writer, sheet_name=f"{sheet_name}_test", index=False)
            print(f"Saved train and test splits for {sheet_name}")