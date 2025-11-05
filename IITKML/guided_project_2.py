import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Display settings
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

# Load dataset
data = pd.read_csv('BrainTumorData.csv', index_col=False)
print("\nSample BrainTumor dataset head(5):\n", data.head(5))
print("\nShape of the dataset:", data.shape)
print("\nDataset description:\n", data.describe())

# Map diagnosis column (M = 1, B = 0)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
print("\nAfter mapping diagnosis column:\n", data[['id', 'diagnosis']].head())

# Visualize diagnosis distribution
plt.hist(data['diagnosis'])
plt.title('Diagnosis (M=1, B=0)')
plt.show()

# Set 'id' as index
data = data.set_index('id')

# Drop unwanted column if exists
if 'Unnamed: 32' in data.columns:
    del data['Unnamed: 32']

# Count of diagnosis values
print("\nDiagnosis class distribution:\n", data.groupby('diagnosis').size())

# Density plots
data.plot(kind='density', subplots=True, layout=(5, 7), sharex=False, legend=False, fontsize=1)
plt.tight_layout()
plt.show()

# Correlation heatmap
from matplotlib import cm as cm
fig = plt.figure()
cax = fig.add_subplot(111).imshow(data.corr(), interpolation="none")
plt.title('Cancer Attributes Correlation')
plt.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
plt.grid(True)
plt.show()

# Prepare input and output
Y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=21)

# Define models
models = [
    ('LogReg', LogisticRegression(max_iter=1000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('CART', DecisionTreeClassifier()),
    ('SVM', SVC()),
    ('NB', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('RF', RandomForestClassifier(random_state=21)),
    ('AdaBoost', AdaBoostClassifier(random_state=21)),
    ('GradBoost', GradientBoostingClassifier(random_state=21))
]

# Define scalers
scalers = [
    ('Standard', StandardScaler()),
    ('MinMax', MinMaxScaler()),
    ('Robust', RobustScaler())
]

# Create pipelines for all model-scaler combinations
pipelines = []
for scaler_name, scaler in scalers:
    for model_name, model in models:
        pipelines.append((f"{scaler_name}_{model_name}", Pipeline([(scaler_name, scaler), (model_name, model)])))

# Define evaluation metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# Evaluate models
num_folds = 10
results = {metric: [] for metric in scoring}
names = []

print("\nModel evaluation results with different scalers:")
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
for name, model in pipelines:
    start = time.time()
    cv_results = cross_validate(model, X_train, Y_train, cv=kfold, scoring=scoring, return_train_score=False)
    end = time.time()
    for metric in scoring:
        results[metric].append(cv_results[f'test_{metric}'])
    names.append(name)
    print(f"\n{name:<20}:")
    print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.6f} ({cv_results['test_accuracy'].std():.6f})")
    print(f"  Precision: {cv_results['test_precision'].mean():.6f} ({cv_results['test_precision'].std():.6f})")
    print(f"  Recall:    {cv_results['test_recall'].mean():.6f} ({cv_results['test_recall'].std():.6f})")
    print(f"  F1-Score:  {cv_results['test_f1'].mean():.6f} ({cv_results['test_f1'].std():.6f})")
    print(f"  Run time:  {end-start:.6f} seconds")

# Visualize performance for each metric
for metric in scoring:
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f'{metric.capitalize()} Comparison Across Models and Scalers')
    ax = fig.add_subplot(111)
    plt.boxplot(results[metric])
    ax.set_xticklabels(names, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Final model using SVM on scaled data
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# ----- StandardScaler + SVM -----
print("\n----- StandardScaler + SVM -----")
scaler_std = StandardScaler().fit(X_train)
X_train_std = scaler_std.transform(X_train)
X_test_std = scaler_std.transform(X_test)

model_std_svm = SVC()
start = time.time()
model_std_svm.fit(X_train_std, Y_train)
end = time.time()
print(f"SVM Training Completed. Run Time: {end - start:.6f} seconds")

pred_std_svm = model_std_svm.predict(X_test_std)
print("All predictions done successfully using SVM.")
print(f"Accuracy Score: {accuracy_score(Y_test, pred_std_svm):.6f}")
print("Confusion Matrix:\n", confusion_matrix(Y_test, pred_std_svm))
print("Classification Report:\n", classification_report(Y_test, pred_std_svm))

# ----- RobustScaler + SVM -----
print("\n----- RobustScaler + SVM -----")
scaler_rob = RobustScaler().fit(X_train)
X_train_rob = scaler_rob.transform(X_train)
X_test_rob = scaler_rob.transform(X_test)

model_rob_svm = SVC()
start = time.time()
model_rob_svm.fit(X_train_rob, Y_train)
end = time.time()
print(f"SVM Training Completed. Run Time: {end - start:.6f} seconds")

pred_rob_svm = model_rob_svm.predict(X_test_rob)
print("All predictions done successfully using SVM.")
print(f"Accuracy Score: {accuracy_score(Y_test, pred_rob_svm):.6f}")
print("Confusion Matrix:\n", confusion_matrix(Y_test, pred_rob_svm))
print("Classification Report:\n", classification_report(Y_test, pred_rob_svm))

# ----- StandardScaler + Logistic Regression -----
print("\n----- StandardScaler + Logistic Regression -----")
model_std_logreg = LogisticRegression(max_iter=1000)
start = time.time()
model_std_logreg.fit(X_train_std, Y_train)
end = time.time()
print(f"LogReg Training Completed. Run Time: {end - start:.6f} seconds")

pred_std_logreg = model_std_logreg.predict(X_test_std)
print("All predictions done successfully using LogReg.")
print(f"Accuracy Score: {accuracy_score(Y_test, pred_std_logreg):.6f}")
print("Confusion Matrix:\n", confusion_matrix(Y_test, pred_std_logreg))
print("Classification Report:\n", classification_report(Y_test, pred_std_logreg))

# ----- RobustScaler + Logistic Regression -----
print("\n----- RobustScaler + Logistic Regression -----")
model_rob_logreg = LogisticRegression(max_iter=1000)
start = time.time()
model_rob_logreg.fit(X_train_rob, Y_train)
end = time.time()
print(f"LogReg Training Completed. Run Time: {end - start:.6f} seconds")

pred_rob_logreg = model_rob_logreg.predict(X_test_rob)
print("All predictions done successfully using LogReg.")
print(f"Accuracy Score: {accuracy_score(Y_test, pred_rob_logreg):.6f}")
print("Confusion Matrix:\n", confusion_matrix(Y_test, pred_rob_logreg))
print("Classification Report:\n", classification_report(Y_test, pred_rob_logreg))

