import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import MLP, CNNRegressor, RNNRegressor


def evaluate_models():
    results = []

    os.makedirs("plots", exist_ok=True)  # Create directory for saving plots

    for file in os.listdir("test_sets"):
        if file.endswith("_test.csv"):
            model_name, sheet_name = file.replace("_test.csv", "").rsplit("_", 1)
            test_file_path = f"test_sets/{file}"

            test_data = pd.read_csv(test_file_path)
            X_test = torch.tensor(test_data[["rho*", "T*"]].values, dtype=torch.float32)
            y_test = test_data["y"].values

            model_class = MLP if "MLP" in model_name else RNNRegressor if "RNNRegressor" in model_name else CNNRegressor
            if model_class == CNNRegressor:
                model = model_class(2)  # CNN only takes one parameter
            else:
                model = model_class(X_test.shape[1], 64 if "MLP" in model_name else 32)

            model.load_state_dict(torch.load(f"weight/randomly/best_{model_name}_{sheet_name}_randomly.pth"))
            model.eval()

            with torch.no_grad():
                y_pred = model(
                    X_test.unsqueeze(1) if "CNN" in model_name or "RNN" in model_name else X_test
                ).squeeze().numpy()

            # Save predictions to test set
            test_data["y_pred"] = y_pred
            test_data["mre"] = abs(test_data["y"] - test_data["y_pred"]) / abs(test_data["y"])*100
            test_data.to_csv(test_file_path, index=False)

            # Compute evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mre = (abs(y_test - y_pred) / abs(y_test)).mean()

            results.append({"Model": model_name, "Sheet": sheet_name, "MSE": mse, "MAE": mae, "MRE": mre})

            # Generate scatter plot for this test set
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                test_data["rho*"], test_data["mre"], c=test_data["T*"], cmap="viridis", alpha=0.75
            )
            plt.colorbar(scatter, label="T*")
            plt.xlabel("rho*")
            plt.ylabel("MRE")
            plt.title(f"Scatter Plot of MRE ({model_name}, {sheet_name})")
            plt.grid(True)

            plot_path = f"plots/scatter_{model_name}_{sheet_name}_randomly.png"
            plt.savefig(plot_path)  # Save figure
            plt.close()

    # Save evaluation results
    results_df = pd.DataFrame(results)
    results_df.to_csv("randomly.csv", index=False)

    print("Evaluation complete. Results saved, and 9 scatter plots generated.")

evaluate_models()