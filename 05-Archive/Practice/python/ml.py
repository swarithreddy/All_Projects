import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset from local path
file_path = r"C:\Users\swarith reddy\Downloads\diabetes.csv"  # Use raw string to avoid escape issues
data = pd.read_csv(file_path)

# Step 2: Split into features (X) and target (y)
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column is target

# Step 3: Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Logistic Regression with 5-fold cross-validation
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Step 5: Print results
for i, score in enumerate(scores, 1):
    print(f"Fold {i} Accuracy: {score:.4f}")

print(f"\nAverage Accuracy: {scores.mean():.4f}")
