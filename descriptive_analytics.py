import pandas as pd

df = pd.read_csv('hr_analytics.csv')

df_no_duplicates_last = df.drop_duplicates(keep='last')

print("Basic Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nExample Values:")
for column in df.columns:
    unique_values = df[column].unique()
    print(f"{column}: {unique_values[:5]}")

