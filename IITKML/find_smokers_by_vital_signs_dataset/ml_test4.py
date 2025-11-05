import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest, StackingClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv(r"C:\Users\swarith reddy\OneDrive\Desktop\IITKML\find_smokers_by_vital_signs_dataset\smoking.csv")

# Step 1: Data Cleaning
# Drop low-importance and constant features
data = data.drop(columns=['oral', 'Urine protein', 'hearing(left)', 'hearing(right)', 'dental caries'], errors='ignore')

# Check for missing values
if data.isnull().sum().sum() > 0:
    data = data.dropna()

# Encode categorical variables
le_gender = LabelEncoder()
le_tartar = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])
data['tartar'] = le_tartar.fit_transform(data['tartar'])

# Step 2: Feature Engineering
# Add BMI, lipid ratio, liver index, and interaction terms
data['BMI'] = data['weight(kg)'] / (data['height(cm)'] / 100) ** 2
data['lipid_ratio'] = data['Cholesterol'] / data['HDL']
data['liver_index'] = data['Gtp'] + data['AST'] + data['ALT']
data['gender_hemoglobin'] = data['gender'] * data['hemoglobin']
data['BMI_Gtp'] = data['BMI'] * data['Gtp']

# Update numerical columns
numerical_cols = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 
                  'eyesight(right)', 'systolic', 'relaxation', 'fasting blood sugar', 
                  'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 
                  'serum creatinine', 'AST', 'ALT', 'Gtp', 'BMI', 'lipid_ratio', 
                  'liver_index', 'gender_hemoglobin', 'BMI_Gtp']

# Handle outliers using IQR
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Step 3: Anomaly Detection with IsolationForest
X_temp = data.drop(columns=['smoking', 'ID'])
y_temp = data['smoking']
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(X_temp)
data = data[outliers == 1]
X_temp = X_temp[outliers == 1]
y_temp = y_temp[outliers == 1]

# Step 4: Polynomial Features for Top Features
top_features = ['hemoglobin', 'Gtp', 'BMI']
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X_temp[top_features])
poly_feature_names = [f"poly_{f}" for f in poly.get_feature_names_out(top_features)]
X_temp[poly_feature_names] = poly_features
numerical_cols.extend(poly_feature_names)

# Step 5: Feature Selection with SelectKBest
X = X_temp
y = y_temp
selector = SelectKBest(score_func=f_classif, k=15)
X = selector.fit_transform(X, y)
selected_features = X_temp.columns[selector.get_support()].tolist()
print(f"Selected Features: {selected_features}")

# Scale numerical features
scaler = RobustScaler()
X = scaler.fit_transform(X)

# Step 6: Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Define Base Models for Stacking
# XGBoost
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.3]
}
xgb_search = GridSearchCV(xgb, xgb_grid, cv=3, scoring='accuracy', n_jobs=-1)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

# CatBoost
cat = CatBoostClassifier(random_state=42, verbose=0)
cat_grid = {
    'iterations': [100, 200],
    'depth': [6, 10],
    'learning_rate': [0.1, 0.3]
}
cat_search = GridSearchCV(cat, cat_grid, cv=3, scoring='accuracy', n_jobs=-1)
cat_search.fit(X_train, y_train)
best_cat = cat_search.best_estimator_

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
rf_search = GridSearchCV(rf, rf_grid, cv=3, scoring='accuracy', n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

# Step 9: Stacking Ensemble
estimators = [
    ('xgb', best_xgb),
    ('cat', best_cat),
    ('rf', best_rf)
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)
stacking_clf.fit(X_train, y_train)

# Step 10: Save All Objects in One Pickle File
pipeline = {
    'model': stacking_clf,
    'le_gender': le_gender,
    'le_tartar': le_tartar,
    'iso_forest': iso_forest,
    'poly_transformer': poly,
    'selector': selector,
    'scaler': scaler,
    'smote': smote
}
with open('smoking_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Step 11: Evaluation
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Cross-validation score
cv_scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"Stacking Ensemble Results:")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Cross-Validation Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

# Feature Importance (from XGBoost as representative)
best_xgb.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_xgb.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nFeature Importance (from XGBoost):")
print(feature_importance)

# Step 12: Verify Model Loading
with open('smoking_pipeline1.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)
loaded_model = loaded_pipeline['model']
loaded_pred = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_pred)
print(f"\nLoaded Model Accuracy: {loaded_accuracy:.4f}")