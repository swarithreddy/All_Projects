# IITKML Data Science Exercises

A collection of machine-learning coursework and dataset experiments. The two top-level guided scripts compare classification models using scikit-learn and generate plots and evaluation metrics.

> The filename `BrainTumorData.csv` is misleading: its columns and labels match the Wisconsin diagnostic breast-cancer dataset (malignant/benign), not a brain-tumor dataset. These scripts are educational exercises, not medical tools.

## Included Exercises

| File | Purpose |
|------|---------|
| `guided_project_2.py` | Compares nine classifiers across Standard, MinMax, and Robust scaling with 10-fold cross-validation, then evaluates selected scaled SVM and logistic-regression models on a holdout split. |
| `guided_project_brain tumer prediction.py` | Compares nine baseline classifiers and standardized pipelines, then evaluates a standardized SVM. |

Both scripts currently load `BrainTumorData.csv` from the working directory. Other data files in this folder, including `diabetes.csv`, `KMeansData.csv`, and the smoker dataset, are not used by these two scripts.

## Technology Stack

- Python 3
- pandas and NumPy
- scikit-learn
- Matplotlib

## Installation

Create and activate a virtual environment, then install the dependencies used by the scripts:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install pandas numpy scikit-learn matplotlib
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install pandas numpy scikit-learn matplotlib
```

## Run

Run from the `IITKML` directory so the relative CSV path resolves:

```bash
python guided_project_2.py
python "guided_project_brain tumer prediction.py"
```

The scripts print dataset summaries and model metrics and call Matplotlib plotting functions. A graphical display may be needed to view plots.

## Data and Outputs

- The scripts expect `BrainTumorData.csv` in the project root, with a `diagnosis` column containing `M` and `B` labels and the feature columns used by the source.
- The holdout split uses 33% test data and `random_state=21`.
- Cross-validation uses 10 shuffled folds. Some estimators do not specify a random seed, so results may vary between runs.
- `scaler.pkl` is present in the repository but is not loaded by these two scripts.

## Limitations

- There is no dependency manifest or automated test suite in this project folder.
- The scripts are exploratory notebooks-in-script form: they display plots and do not expose a reusable prediction API.
- Diagnostic model metrics are not evidence of clinical validity. Do not use these models for diagnosis or treatment decisions.
