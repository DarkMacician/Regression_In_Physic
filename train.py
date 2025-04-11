import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import optim

from model import MLP, CNNRegressor, RNNRegressor, MRELoss


# Early Stopping Class
def train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, num_epochs=70000):
    best_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # Giảm learning rate sau mỗi 10,000 epoch
        if (epoch + 1) % 10000 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.85  # Giảm 15%
                print(f"Epoch {epoch + 1}: Learning rate giảm còn {param_group['lr']}")

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_test)
            val_loss = criterion(y_val_pred, y_test)

        # Lưu model tốt nhất
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Train Loss={loss:.4f}, Val Loss={val_loss:.4f}")

    return model, best_loss


# Lưu tập test
def save_test_set(X_test, y_test, sheet_name, model_name):
    os.makedirs("test_sets", exist_ok=True)
    test_data = pd.DataFrame(X_test.numpy(), columns=["rho*", "T*"])
    test_data["y"] = y_test.numpy()
    test_data.to_csv(f"test_sets/{model_name}_{sheet_name}_test_0to01.csv", index=False)

def train_randomly(file_path):
    data_sheets = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in data_sheets.items():
        print(f"Training on sheet: {sheet_name}")

        X_full = df[["rho*", 'T*', "D*"]]
        y = df['D*rho*/sqrt(T)'] if sheet_name == "FE" else (df['D*rho'] if sheet_name == "DA" else df['D*'])

        X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
        X_train, X_test = X_train_full[["rho*", 'T*']], X_test_full[["rho*", 'T*']]

        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        X_train_processed = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_processed = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_processed = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
        y_test_processed = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)

        models = [
            (MLP(X_train_processed.shape[1], 64), "MLP", X_train_processed, X_test_processed),
            (CNNRegressor(2), "CNNRegressor", X_train_processed.unsqueeze(1), X_test_processed.unsqueeze(1)),
            (RNNRegressor(X_train_processed.shape[1], 32), "RNNRegressor", X_train_processed.unsqueeze(1), X_test_processed.unsqueeze(1)),
        ]

        for model, model_name, X_train_input, X_test_input in models:
            print(f"Training {model_name} on {sheet_name}...")
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = MRELoss()
            trained_model, best_loss = train_model(model, criterion, optimizer, X_train_input, y_train_processed, X_test_input, y_test_processed)

            torch.save(trained_model.state_dict(), f"weight/randomly/best_{model_name}_{sheet_name}_randomly.pth")
            save_test_set(X_test_processed, y_test_processed, sheet_name, model_name)
            print(f"Best {model_name} loss on {sheet_name}: {best_loss:.4f}\n")


# Huấn luyện với test set có điều kiện rho* trong (0, 0.1)
def train(file_path):
    data_sheets = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in data_sheets.items():
        print(f"Training on sheet: {sheet_name}")

        X_full = df[["rho*", 'T*']]
        y = df['D*'] * df['rho*'] / df['T*'].apply(lambda x: x ** 0.5) if sheet_name == "FE" else df['D*']

        # Chia tập test với điều kiện 0 < rho* < 0.1
        test_mask = (df['rho*'] > 0) & (df['rho*'] < 0.1)
        X_test, y_test = X_full[test_mask], y[test_mask]
        X_train, y_train = X_full[~test_mask], y[~test_mask]

        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        X_train_processed = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_processed = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_processed = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
        y_test_processed = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)

        models = [
            (MLP(X_train_processed.shape[1], 64), "MLP", X_train_processed, X_test_processed),
            (CNNRegressor(2), "CNNRegressor", X_train_processed.unsqueeze(1), X_test_processed.unsqueeze(1)),
            (RNNRegressor(X_train_processed.shape[1], 32), "RNNRegressor", X_train_processed.unsqueeze(1), X_test_processed.unsqueeze(1)),
        ]

        for model, model_name, X_train_input, X_test_input in models:
            print(f"Training {model_name} on {sheet_name}...")
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = MRELoss()
            trained_model, best_loss = train_model(model, criterion, optimizer, X_train_input, y_train_processed, X_test_input, y_test_processed)

            torch.save(trained_model.state_dict(), f"weight/from_0_to_01/best_{model_name}_{sheet_name}.pth")
            save_test_set(X_test_processed, y_test_processed, sheet_name, model_name)
            print(f"Best {model_name} loss on {sheet_name}: {best_loss:.4f}\n")


# Chạy huấn luyện
train_randomly('D:/Physic/fluid.xlsx')