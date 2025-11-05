import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv(r"C:\Users\swarith reddy\OneDrive\Desktop\IITKML\find_smokers_by_vital_signs_dataset\smoking.csv")


# Step 1: Data Cleaning
# Drop 'oral' column as it appears constant ('Y')
data = data.drop(columns=['oral'])

# Check for missing values
if data.isnull().sum().sum() > 0:
    data = data.dropna()

# Encode categorical variables
le_gender = LabelEncoder()
le_tartar = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])
data['tartar'] = le_tartar.fit_transform(data['tartar'])

# Handle outliers using IQR for numerical columns
numerical_cols = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 
                  'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic', 
                  'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride', 
                  'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 
                  'AST', 'ALT', 'Gtp']

for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Step 2: Feature Engineering
# Separate features and target
X = data.drop(columns=['smoking', 'ID'])
y = data['smoking']

# Scale numerical features
scaler = RobustScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training with Random Forest
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Step 5: Evaluation
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)