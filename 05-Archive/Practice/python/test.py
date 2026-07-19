from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Step 1: Load the 'adult' dataset from OpenML
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame
print(df.head())
print(df.shape)
print(df.columns)

# Step 2: Define features (X) and target (y)
X = df.drop('class', axis=1)
y = df['class']

# Step 3: Identify categorical columns
cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

# Step 4: Create preprocessing + model pipeline
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000))
])

# Step 5: Perform 500-fold cross-validation using all CPU cores
scores = cross_val_score(model, X, y, cv=50, scoring='accuracy', n_jobs=-1)

# Step 6: Print fold accuracies and average
for i, score in enumerate(scores, 1):
    print(f"Fold {i} Accuracy: {score:.4f}")

print(f"\nAverage Accuracy over 500 folds: {np.mean(scores):.4f}")
