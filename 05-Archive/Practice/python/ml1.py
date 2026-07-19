import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

# Step 1: Load the dataset
file_path = r"C:\Users\swarith reddy\Downloads\diabetes.csv"
data = pd.read_csv(file_path)

# Step 2: Split into features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Step 3: Logistic Regression with parallelized cross-validation
model = LogisticRegression(max_iter=2000)
kf=LeaveOneOut()
# cross_val_score with n_jobs=-1 uses all CPU cores
scores = cross_val_score(model, X, y, cv=kf)

print(scores)
