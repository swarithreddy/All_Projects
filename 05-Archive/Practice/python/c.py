import pandas as pd

# Load the Excel file
file_path = r"C:\Users\swarith reddy\Downloads\5. CSE III-I(R22)OE ALLOTED.xls"

# Try reading with xlrd, fallback to openpyxl if needed
try:
    df = pd.read_excel(file_path, engine='xlrd')
except:
    df = pd.read_excel(file_path, engine='openpyxl')

# Check if 'OE' column exists
if 'OE' not in df.columns:
    print("Column 'OE' not found in the Excel file.")
else:
    # Drop NaNs
    subject_series = df['OE'].dropna()

    # Remove section suffixes like -A, -B, etc. using regex
    cleaned_subjects = subject_series.str.replace(r'-[A-Z]$', '', regex=True)

    # Count frequencies
    subject_counts = cleaned_subjects.value_counts()

    print(subject_counts)
