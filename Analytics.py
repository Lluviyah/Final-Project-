import pandas as pd


filename = "hr_analytics.csv"
df = pd.read_csv('hr_analytics.csv')
print(df.columns)
print(df['time_spend_company'].describe())