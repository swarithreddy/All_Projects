import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv(r"C:\Users\swarith reddy\OneDrive\Desktop\IITKML\find_smokers_by_vital_signs_dataset\smoking.csv")

# Step 1: Data Cleaning
# Drop 'oral' (constant) and low-importance features
data = data.drop(columns=['oral', 'Urine protein', 'hearing(left)', 'hearing(right)', 'dental caries'])

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
                  'eyesight(right)', 'systolic', 'relaxation', 'fasting blood sugar', 
                  'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 
                  'serum creatinine', 'AST', 'ALT', 'Gtp']

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

# Step 3: Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training with XGBoost
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_

# Step 6: Evaluation
y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Cross-validation score
cv_scores = cross_val_score(best_xgb, X, y, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Cross-Validation Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_xgb.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)