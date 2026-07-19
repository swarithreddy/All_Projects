## Linear Regression House Price Prediction (Terminal-Based ML Project)

This project demonstrates a complete **Machine Learning workflow using Linear Regression** entirely from the **terminal**.  
It is designed to be suitable for a **college Machine Learning / Data Mining project**.

The workflow includes:

- **Loading data** from CSV using `pandas`
- **Training a Linear Regression model** using `scikit-learn`
- **Evaluating** the model (MSE and R² score)
- **Saving** the trained model as a `.pkl` file using `pickle`
- **Loading** the saved model in a separate script
- **Accepting user input from the terminal** and **predicting house price**
- Optional: **Saving plots** for model performance and feature importance using `matplotlib` (no GUI windows opened)

---

## Folder Structure

Inside `linear_regression_project/`:

- **data**
  - `house_data.csv` – sample dataset for training (area, bedrooms, age, price)
- **models**
  - `linear_model.pkl` – trained Linear Regression model (created/overwritten by training script)
- **src**
  - `train_model.py` – trains and evaluates the model, saves it as `.pkl`, and creates plots
  - `predict.py` – loads the saved model, takes user input from terminal, and predicts price
- **plots** (created automatically)
  - `actual_vs_predicted.png` – visualization of model performance
  - `feature_importance.png` – simple feature importance bar chart
- `requirements.txt` – Python dependencies
- `README.md` – project documentation

---

## Dataset Description

File: `data/house_data.csv`

Columns:

- **area** – house area in square feet
- **bedrooms** – number of bedrooms
- **age** – age of the house in years
- **price** – house price (target variable)

Sample rows:

```text
area,bedrooms,age,price
1000,2,10,300000
1200,3,5,400000
1500,3,8,450000
1800,4,6,500000
2000,4,3,600000
2200,5,2,650000
2500,5,1,700000
```

---

## Installation & Setup

### 1. Navigate to the project folder

In your terminal (PowerShell or Command Prompt):

```bash
cd "C:\Users\swarith reddy\OneDrive\Desktop\New folder\dwdm_project\linear_regression_project"
```

### 2. (Optional but recommended) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib`

---

## Running the Full ML Workflow from Terminal

### Step 1: Train the Model

Run:

```bash
python src/train_model.py
```

Expected terminal output (example):

```text
Loading dataset...
Training model...
Model training complete.
Model evaluation:
Mean Squared Error (MSE): 120000.00
R² Score: 0.9200

Learned coefficients (feature importance):
 - area: 200.1234
 - bedrooms: 15000.5678
 - age: -3000.4321
Intercept: 10000.0000

Saving model to C:\...\linear_regression_project\models\linear_model.pkl ...
Model saved successfully.

Saving performance plots (no GUI will be opened)...
 - Actual vs Predicted plot saved to: C:\...\linear_regression_project\plots\actual_vs_predicted.png
 - Feature importance plot saved to: C:\...\linear_regression_project\plots\feature_importance.png
```

After this:

- Trained model is saved at `models/linear_model.pkl`
- Plots are saved in the `plots/` directory

---

### Step 2: Predict Using the Saved Model

Run:

```bash
python src/predict.py
```

Example terminal interaction:

```text
Loading trained Linear Regression model...

Please enter house details for price prediction.
------------------------------------------------
Enter house area (in square feet): 1700
Enter number of bedrooms: 3
Enter house age (in years): 5

Prediction complete.
Predicted House Price: $480,000
```

---

## Script Details

### `src/train_model.py`

- **Loads dataset** from `data/house_data.csv` using `pandas`
- Separates:
  - **Features (X)**: `area`, `bedrooms`, `age`
  - **Target (y)**: `price`
- Splits data into train and test sets using `train_test_split`
- Trains a **Linear Regression** model (`sklearn.linear_model.LinearRegression`)
- Evaluates the model using:
  - **Mean Squared Error (MSE)**
  - **R² Score**
- Prints evaluation metrics and learned coefficients (feature importance explanation)
- **Saves** the model as `models/linear_model.pkl` using `pickle`
- Optionally:
  - Saves **Actual vs Predicted** plot: `plots/actual_vs_predicted.png`
  - Saves **Feature Importance** bar chart: `plots/feature_importance.png`

### `src/predict.py`

- **Loads** the saved model from `models/linear_model.pkl`
- Accepts user input from terminal:
  - House area (square feet)
  - Number of bedrooms
  - House age (years)
- Performs **input validation**:
  - Rejects letters where numbers are expected
  - Rejects negative values
  - Asks the user again on invalid input
- Converts inputs into the correct feature format (NumPy array)
- Uses the loaded model to **predict house price**
- Prints result clearly, e.g.:

```text
Predicted House Price: $520,000
```

---

## Error Handling & Validation

- If the **model file** does not exist:
  - `predict.py` prints a clear message and asks you to run `train_model.py` first.
- If the **dataset** is missing:
  - `train_model.py` warns that `data/house_data.csv` was not found.
- **User input errors**:
  - Letters instead of numbers → user is re-prompted with a helpful message.
  - Negative or zero values for area → rejected and re-prompted.
  - Negative integers for bedrooms/age → rejected and re-prompted.

---

## How to Describe This as a College Project

In a report or presentation, you can highlight:

- **Goal**: Predict house prices using Linear Regression.
- **Tools**: Python, `pandas`, `numpy`, `scikit-learn`, `pickle`, `matplotlib`.
- **Workflow**:
  - Data loading and preprocessing
  - Train–test splitting
  - Model training (Linear Regression)
  - Model evaluation (MSE, R²)
  - Model persistence (saving/loading with `.pkl`)
  - Terminal-based user interaction for prediction
- **Advantages**:
  - Simple, explainable model
  - Easy to extend with more features or data
  - No GUI needed – fully terminal-based, portable

---

## Extending the Project (Optional Ideas)

- Add more features (e.g., location score, number of bathrooms).
- Use a larger real-world dataset.
- Compare Linear Regression with other models (e.g., Decision Tree, Random Forest).
- Add logging to a file for all predictions made.
- Wrap the scripts in a simple CLI menu.

