import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm_api
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler




df = pd.read_csv('hr_analytics.csv')
df = df.drop_duplicates(keep='last')
df = df.drop(columns='sales')

df = pd.get_dummies(df,dtype=int,drop_first=True)

# Scaling the values using Min-Max Method:
df[['average_montly_hours','time_spend_company','number_project']]= (df[['average_montly_hours','time_spend_company','number_project']]-df[['average_montly_hours','time_spend_company','number_project']].min())/(df[['average_montly_hours','time_spend_company','number_project']].max()-df[['average_montly_hours','time_spend_company','number_project']].min())
# Specifying the Y var

Y = df['left']
# Setting up train and test
x_train, x_test, y_train, y_test = train_test_split(df.drop('left', axis=1), Y, test_size=0.3, random_state=1)
# Outputting as csv for viewing purposes
df.to_csv('dummy.csv')
# Create model
model = sm.Logit(y_train, sm.add_constant(x_train)).fit()
print(model.summary())
# Get predictions
y_pred_prob = model.predict(sm.add_constant(x_test))
# Make pedictions binary
y_pred_binary = (y_pred_prob >= 0.5).astype(int)
# Calculate accuracy and precision
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1:", f1)


