import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#If any of this libraries is missing from your computer. Please install them using pip.
df = pd.read_csv('hr_analytics.csv') 
df = df.drop_duplicates(keep='last') 

df = df.drop(columns='sales') 
df = pd.get_dummies(df,dtype=int, drop_first=True) 
df = df.drop_duplicates(keep='last') 

 

df[['average_montly_hours','time_spend_company','number_project']]= (df[['average_montly_hours','time_spend_company','number_project']]-df[['average_montly_hours','time_spend_company','number_project']].min())/(df[['average_montly_hours','time_spend_company','number_project']].max()-df[['average_montly_hours','time_spend_company','number_project']].min()) 
 

y = df['satisfaction_level']
x = df[['last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','left','promotion_last_5years','salary_low','salary_medium']]

x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
print(model.summary())
