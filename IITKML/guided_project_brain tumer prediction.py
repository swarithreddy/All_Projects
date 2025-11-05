import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

# Baseline algorithms
models_list = [
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

num_folds = 10
names = []
results = []

print("\nBaseline model results:")
for name, model in models_list:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
    start_Time = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    end_Time = time.time()
    results.append(cv_results)
    names.append(name)
    print(f"{name:<12}: {cv_results.mean():.6f} ({cv_results.std():.6f}) (run time: {end_Time-start_Time:.6f})")

# Performance comparison
fig = plt.figure()
fig.suptitle('Baseline Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Standardized models using pipeline
pipelines = [
    ('ScaledLogReg', Pipeline([('Scaler', StandardScaler()), ('LogReg', LogisticRegression(max_iter=1000))])),
    ('ScaledLDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])),
    ('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])),
    ('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])),
    ('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])),
    ('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])),
    ('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestClassifier(random_state=21))])),
    ('ScaledAdaBoost', Pipeline([('Scaler', StandardScaler()), ('AdaBoost', AdaBoostClassifier(random_state=21))])),
    ('ScaledGradBoost', Pipeline([('Scaler', StandardScaler()), ('GradBoost', GradientBoostingClassifier(random_state=21))]))
]

results = []
names = []
print("\nAccuracies of algorithms after standardizing dataset:")
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
for name, model in pipelines:
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print(f"{name:<15}: {cv_results.mean():.6f} ({cv_results.std():.6f}) (run time: {end-start:.6f})")

# Performance comparison after scaling
fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# Final model using SVM on scaled data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC()
start = time.time()
model.fit(X_train_scaled, Y_train)
end = time.time()
print(f"\nSVM Training Completed. Run Time: {end - start:.6f} seconds")

# Predictions and Evaluation
predictions = model.predict(X_test_scaled)
print("\nAll predictions done successfully using SVM.")
print(f"\nAccuracy Score: {accuracy_score(Y_test, predictions):.6f}")
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, predictions))
print("\nClassification Report:\n", classification_report(Y_test, predictions))
