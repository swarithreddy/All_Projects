import os
import pickle
from typing import Optional

import numpy as np


def get_project_model_path() -> str:
    """
    Returns the absolute path to the saved linear regression model.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    model_path = os.path.join(project_root, "models", "linear_model.pkl")
    return model_path


def load_model(model_path: str) -> Optional[object]:
    """
    Load the trained model from disk.
    """
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        print("Please run 'python src/train_model.py' first to train and save the model.")
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print("Failed to load the model:")
        print(str(e))
        return None


def get_positive_float(prompt: str) -> float:
    """
    Prompt the user for a positive numeric value.
    Re-prompts until a valid value is entered.
    """
    while True:
        user_input = input(prompt)
        try:
            value = float(user_input)
            if value <= 0:
                print("Please enter a positive number greater than 0.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value (e.g., 1200).")


def get_non_negative_int(prompt: str) -> int:
    """
    Prompt the user for a non-negative integer (e.g., number of bedrooms or age).
    Re-prompts until a valid value is entered.
    """
    while True:
        user_input = input(prompt)
        try:
            value = int(user_input)
            if value < 0:
                print("Please enter a non-negative integer (0 or greater).")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter an integer number (e.g., 3).")


def make_prediction(model: object) -> None:
    """
    Interactively collect input from the user and print the predicted house price.
    """
    print("\nPlease enter house details for price prediction.")
    print("------------------------------------------------")

    # Collect validated user inputs
    area = get_positive_float("Enter house area (in square feet): ")
    bedrooms = get_non_negative_int("Enter number of bedrooms: ")
    age = get_non_negative_int("Enter house age (in years): ")

    # Prepare features for prediction (shape: [1, n_features])
    features = np.array([[area, bedrooms, age]])

    try:
        predicted_price = model.predict(features)[0]
        # Round and format as currency-like output
        predicted_price_rounded = int(round(predicted_price))
        print("\nPrediction complete.")
        print(f"Predicted House Price: ${predicted_price_rounded:,.0f}")
    except Exception as e:
        print("An error occurred while making the prediction:")
        print(str(e))


def main() -> None:
    print("Loading trained Linear Regression model...")
    model_path = get_project_model_path()
    model = load_model(model_path)

    if model is None:
        return

    try:
        make_prediction(model)
    except KeyboardInterrupt:
        print("\nPrediction cancelled by user.")


if __name__ == "__main__":
    main()

