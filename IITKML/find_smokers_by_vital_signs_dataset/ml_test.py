import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import time
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer, RobustScaler
from sklearn.decomposition import PCA

# ---------------- Load and Preprocess ----------------
df = pd.read_csv(r"C:\Users\swarith reddy\OneDrive\Desktop\IITKML\find_smokers_by_vital_signs_dataset\smoking.csv")
df.set_index('ID', inplace=True)
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['oral'] = df['oral'].map({'Y': 1, 'N': 0})
df['tartar'] = df['tartar'].map({'Y': 1, 'N': 0})

# ---------------- Select Valid Numeric Columns ----------------
numeric_df = df.select_dtypes(include='number')
numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]  # remove constant columns

# ---------------- 1. Histogram ----------------
numeric_df.plot(kind='hist', subplots=True, layout=(5, 7), bins=30, edgecolor='black',
                figsize=(18, 12), legend=False, sharex=False, sharey=False, fontsize=6)
plt.suptitle('Histograms of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()

# ---------------- 2. Density (KDE) ----------------
numeric_df.plot(kind='density', subplots=True, layout=(5, 7), sharex=False,
                legend=False, fontsize=6, figsize=(18, 12))
plt.suptitle('Density (KDE) Plots of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()

# ---------------- 3. Box Plots ----------------
numeric_df.plot(kind='box', subplots=True, layout=(5, 7), sharex=False, sharey=False,
                fontsize=6, figsize=(18, 12))
plt.suptitle('Box Plots of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()

# ---------------- 4. Correlation Heatmap ----------------
plt.figure(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()

# ---------------- 5. Scatter Matrix ----------------
scatter_features = ['age', 'waist(cm)', 'Cholesterol', 'HDL', 'LDL', 'smoking']
scatter_df = df[scatter_features]
plt.figure(figsize=(10, 10))
scatter_matrix(scatter_df, figsize=(10, 10), diagonal='kde',
               color=['red' if s == 1 else 'blue' for s in df['smoking']])
plt.suptitle('Scatter Matrix of Selected Features (Colored by Smoking)', y=1.02)
plt.show()

# ---------------- 6. ML Modeling ----------------
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

# Define Preprocessing Techniques
preprocessors = {
    'No Scaling': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'Normalizer': Normalizer(),
    'Binarizer': Binarizer(threshold=0.0),  # Threshold at 0 (converts non-zero to 1)
    'RobustScaler': RobustScaler(),
    'PCA_StandardScaler': [StandardScaler(), PCA(n_components=0.95)]  # Retain 95% variance
}

# ML Models List
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

# Model Evaluation Across Preprocessors
num_folds = 10
results_dict = {name: [] for name, _ in models_list}
names = [name for name, _ in models_list]
preprocess_names = list(preprocessors.keys())
mean_scores = {preproc: [] for preproc in preprocess_names}

print("\nModel results across preprocessing techniques:")
for preproc_name, preprocessor in preprocessors.items():
    print(f"\n=== {preproc_name} ===")
    # Prepare data based on preprocessor
    if preproc_name == 'No Scaling':
        X_train_preproc = X_train
        X_test_preproc = X_test
    elif preproc_name == 'PCA_StandardScaler':
        scaler, pca = preprocessor
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_preproc = pca.fit_transform(X_train_scaled)
        X_test_preproc = pca.transform(X_test_scaled)
        print(f"PCA components used: {pca.n_components_}")
    else:
        X_train_preproc = preprocessor.fit_transform(X_train)
        X_test_preproc = preprocessor.transform(X_test)

    # Evaluate each model
    for name, model in models_list:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
        start_time = time.time()
        cv_results = cross_val_score(model, X_train_preproc, Y_train, cv=kfold, scoring='accuracy')
        end_time = time.time()
        results_dict[name].append(cv_results)
        mean_scores[preproc_name].append(cv_results.mean())
        print(f"{name:<12}: {cv_results.mean():.6f} ({cv_results.std():.6f}) (run time: {end_time - start_time:.6f} sec)")

# ---------------- 7. Performance Comparison Plot ----------------
plt.figure(figsize=(12, 8))
for i, name in enumerate(names):
    scores = [results_dict[name][j].mean() for j in range(len(preprocessors))]
    plt.plot(preprocess_names, scores, marker='o', label=name)
plt.title('Model Accuracy Comparison Across Preprocessing Techniques')
plt.xlabel('Preprocessing Technique')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Box Plot for Detailed Comparison
plt.figure(figsize=(14, 8))
plt.boxplot([results_dict[name][i] for name in names for i in range(len(preprocessors))],
            labels=[f"{name}\n{preproc}" for name in names for preproc in preprocess_names],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='black'),
            medianprops=dict(color='red'))
plt.title('Model Accuracy Distribution Across Preprocessing Techniques')
plt.xlabel('Algorithm and Preprocessor')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()