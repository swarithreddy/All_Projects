import os
import pickle
from typing import Tuple

import matplotlib

# Use a non-interactive backend so that plots can be saved without opening a window
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def get_project_paths() -> Tuple[str, str, str]:
    """
    Returns:
        data_path: Absolute path to the CSV dataset.
        model_path: Absolute path where the trained model will be saved.
        plots_dir: Absolute path to the directory where plots will be stored.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)

    data_path = os.path.join(project_root, "data", "house_data.csv")
    model_path = os.path.join(project_root, "models", "linear_model.pkl")
    plots_dir = os.path.join(project_root, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    return data_path, model_path, plots_dir


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the dataset from the given CSV path and separate features and target.

    Features (X): area, bedrooms, age
    Target (y): price
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    feature_cols = ["area", "bedrooms", "age"]
    target_col = "price"

    X = df[feature_cols]
    y = df[target_col]

    return X, y


def train_and_evaluate_model(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[LinearRegression, float, float, np.ndarray]:
    """
    Split the data, train the Linear Regression model, and evaluate it.

    Returns:
        model: Trained LinearRegression model
        mse: Mean Squared Error on test set
        r2: R^2 score on test set
        y_pred_test: Predicted values on the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print("Model training complete.")
    print("Model evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Print learned coefficients as a simple "feature importance" explanation
    feature_names = ["area", "bedrooms", "age"]
    print("\nLearned coefficients (feature importance):")
    for name, coef in zip(feature_names, model.coef_):
        print(f" - {name}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    return model, mse, r2, y_pred_test


def save_model(model: LinearRegression, model_path: str) -> None:
    """Save the trained model to disk using pickle."""
    print(f"Saving model to {model_path} ...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("Model saved successfully.")


def plot_actual_vs_predicted(
    y_test: pd.Series, y_pred: np.ndarray, output_path: str
) -> None:
    """
    Save a scatter plot comparing actual vs predicted prices.
    This is a simple visual way to see model performance.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.7, label="Predicted vs Actual")
    min_price = min(y_test.min(), y_pred.min())
    max_price = max(y_test.max(), y_pred.max())
    plt.plot([min_price, max_price], [min_price, max_price], "r--", label="Ideal line")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(
    model: LinearRegression, feature_names: list, output_path: str
) -> None:
    """
    Save a bar chart of the absolute value of coefficients as a simple
    feature importance explanation.
    """
    coefficients = np.abs(model.coef_)

    plt.figure(figsize=(6, 4))
    plt.bar(feature_names, coefficients, color="green")
    plt.xlabel("Feature")
    plt.ylabel("Importance (|coefficient|)")
    plt.title("Feature Importance (Linear Regression Coefficients)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    data_path, model_path, plots_dir = get_project_paths()

    if not os.path.exists(data_path):
        print(f"Dataset not found at: {data_path}")
        print("Make sure 'data/house_data.csv' exists before training.")
        return

    try:
        X, y = load_dataset(data_path)
        model, mse, r2, y_pred_test = train_and_evaluate_model(X, y)

        save_model(model, model_path)

        # Optional visualizations saved to disk
        print("\nSaving performance plots (no GUI will be opened)...")
        perf_plot_path = os.path.join(plots_dir, "actual_vs_predicted.png")
        feature_plot_path = os.path.join(plots_dir, "feature_importance.png")

        # For the performance plot, we need the y_test used in training.
        # Recompute split only for plotting purpose.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        y_pred_test = model.predict(X_test)

        plot_actual_vs_predicted(y_test, y_pred_test, perf_plot_path)
        plot_feature_importance(
            model, ["area", "bedrooms", "age"], feature_plot_path
        )

        print(f" - Actual vs Predicted plot saved to: {perf_plot_path}")
        print(f" - Feature importance plot saved to: {feature_plot_path}")

    except Exception as e:
        print("An error occurred during training:")
        print(str(e))


if __name__ == "__main__":
    main()

