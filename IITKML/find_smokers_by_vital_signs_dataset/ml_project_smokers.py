import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import time

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import time
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler  # Added StandardScaler

# ---------------- Load and Preprocess ----------------
df = pd.read_csv(r"C:\Users\swarith reddy\OneDrive\Desktop\IITKML\find_smokers_by_vital_signs_dataset\smoking.csv")
df.set_index('ID', inplace=True)
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['oral'] = df['oral'].map({'Y': 1, 'N': 0})
df['tartar'] = df['tartar'].map({'Y': 1, 'N': 0})

# # ---------------- Select Valid Numeric Columns ----------------
# numeric_df = df.select_dtypes(include='number')
# numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]  # remove constant columns

# # ---------------- 1. Histogram ----------------
# numeric_df.plot(kind='hist', subplots=True, layout=(5, 7), bins=30, edgecolor='black',
#                 figsize=(18, 12), legend=False, sharex=False, sharey=False, fontsize=6)
# plt.suptitle('Histograms of Numeric Features', fontsize=16)
# plt.tight_layout()
# plt.show()

# # ---------------- 2. Density (KDE) ----------------
# numeric_df.plot(kind='density', subplots=True, layout=(5, 7), sharex=False,
#                 legend=False, fontsize=6, figsize=(18, 12))
# plt.suptitle('Density (KDE) Plots of Numeric Features', fontsize=16)
# plt.tight_layout()
# plt.show()

# # ---------------- 3. Box Plots ----------------
# numeric_df.plot(kind='box', subplots=True, layout=(5, 7), sharex=False, sharey=False,
#                 fontsize=6, figsize=(18, 12))
# plt.suptitle('Box Plots of Numeric Features', fontsize=16)
# plt.tight_layout()
# plt.show()

# # ---------------- 4. Correlation Heatmap ----------------
# plt.figure(figsize=(14, 10))
# sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap of Numeric Features')
# plt.tight_layout()
# plt.show()

# # ---------------- 5. Scatter Matrix ----------------
# scatter_features = ['age', 'waist(cm)', 'Cholesterol', 'HDL', 'LDL', 'smoking']
# scatter_df = df[scatter_features]

# plt.figure(figsize=(10, 10))
# scatter_matrix(scatter_df, figsize=(10, 10), diagonal='kde',
#                color=['red' if s == 1 else 'blue' for s in df['smoking']])
# plt.suptitle('Scatter Matrix of Selected Features (Colored by Smoking)', y=1.02)
# plt.show()

# # ---------------- 6.1 ML Modeling ----------------
# data = df.select_dtypes(include='number')

# # Correlation Heatmap (again, with imshow version)
# plt.figure(figsize=(12, 10))
# cax = plt.imshow(data.corr(), interpolation="none", cmap='coolwarm')
# plt.title('Feature Correlation (Smoking Dataset)')
# plt.colorbar(cax)
# plt.grid(False)
# plt.xticks(range(len(data.columns)), data.columns, rotation=90, fontsize=6)
# plt.yticks(range(len(data.columns)), data.columns, fontsize=6)
# plt.tight_layout()
# plt.show()

# # Prepare Input and Output
# Y = data['smoking'].values
# X = data.drop('smoking', axis=1).values
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=21)

# # ML Models List
# models_list = [
#     ('LogReg', LogisticRegression(max_iter=1000)),
#     ('LDA', LinearDiscriminantAnalysis()),
#     ('CART', DecisionTreeClassifier()),
#     ('SVM', SVC()),
#     ('NB', GaussianNB()),
#     ('KNN', KNeighborsClassifier()),
#     ('RF', RandomForestClassifier(random_state=21)),
#     ('AdaBoost', AdaBoostClassifier(random_state=21)),
#     ('GradBoost', GradientBoostingClassifier(random_state=21))
# ]

# # Model Evaluation
# num_folds = 10
# results = []
# names = []

# print("\nBaseline model results:")
# for name, model in models_list:
#     kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
#     start_time = time.time()
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#     end_time = time.time()
#     results.append(cv_results)
#     names.append(name)
#     print(f"{name:<12}: {cv_results.mean():.6f} ({cv_results.std():.6f}) (run time: {end_time - start_time:.6f} sec)")

# # ---------------- 7. Performance Comparison Plot ----------------
# plt.figure(figsize=(10, 6))
# plt.boxplot(results)
# plt.title('Model Accuracy Comparison (Cross-Validation)')
# plt.xlabel('Algorithm')
# plt.ylabel('Accuracy')
# plt.xticks(ticks=range(1, len(names)+1), labels=names)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# ---------------- 6.2 ML Modeling ----------------
data = df.select_dtypes(include='number')

# Correlation Heatmap (again, with imshow version)
plt.figure(figsize=(12, 10))
cax = plt.imshow(data.corr(), interpolation="none", cmap='coolwarm')
plt.title('Feature Correlation (Smoking Dataset)')
plt.colorbar(cax)
plt.grid(False)
plt.xticks(range(len(data.columns)), data.columns, rotation=90, fontsize=6)
plt.yticks(range(len(data.columns)), data.columns, fontsize=6)
plt.tight_layout()
plt.show()

# Prepare Input and Output
Y = data['smoking'].values
X = data.drop('smoking', axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=21)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ML Models List
models_list = [
    ('LogReg', LogisticRegression(max_iter=1000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('RF', RandomForestClassifier(random_state=21)),
    ('AdaBoost', AdaBoostClassifier(random_state=21)),
    ('GradBoost', GradientBoostingClassifier(random_state=21))
]

# Model Evaluation with Scaled Data
num_folds = 10
results = []
names = []

print("\nBaseline model results (with StandardScaler):")
for name, model in models_list:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
    start_time = time.time()
    cv_results = cross_val_score(model, X_train_scaled, Y_train, cv=kfold, scoring='accuracy')
    end_time = time.time()
    results.append(cv_results)
    names.append(name)
    print(f"{name:<12}: {cv_results.mean():.6f} ({cv_results.std():.6f}) (run time: {end_time - start_time:.6f} sec)")

# ---------------- 7. Performance Comparison Plot ----------------
plt.figure(figsize=(10, 6))
plt.boxplot(results)
plt.title('Model Accuracy Comparison (Cross-Validation with StandardScaler)')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.xticks(ticks=range(1, len(names)+1), labels=names)
plt.grid(True)
plt.tight_layout()
plt.show()