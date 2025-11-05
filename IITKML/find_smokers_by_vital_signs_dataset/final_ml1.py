import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and Preprocess
df = pd.read_csv(r"C:\Users\swarith reddy\OneDrive\Desktop\IITKML\find_smokers_by_vital_signs_dataset\smoking.csv")
df.set_index('ID', inplace=True)
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['oral'] = df['oral'].map({'Y': 1, 'N': 0})
df['tartar'] = df['tartar'].map({'Y': 1, 'N': 0})

# Select numeric data
data = df.select_dtypes(include='number')

# Prepare Input and Output
Y = data['smoking'].values
X = data.drop('smoking', axis=1).values

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=21)

# Initialize and apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForestClassifier
rf_model = RandomForestClassifier(random_state=21)
rf_model.fit(X_train_scaled, Y_train)

# Save the model and scaler to pickle files
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("RandomForestClassifier model and StandardScaler saved as 'rf_model.pkl' and 'scaler.pkl'")