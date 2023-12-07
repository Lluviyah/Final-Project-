import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
df = pd.read_csv('hr_analytics.csv')
Y = df['left']
x_train, x_test, y_train, y_test = train_test_split(df.drop('left', axis=1), Y, test_size=0.3, random_state=1)
df = pd.get_dummies(df,dtype=int)
df.to_csv('dummy.csv')
print(df)

models = {
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=4),
    'Random Forest': RandomForestClassifier(random_state=0),
    'SVM': SVC(gamma='auto'),
    'Binary Logistic Regression': LogisticRegression(solver='liblinear', random_state=1),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=6)
}

for name, model in models.items():
    print(f"{name}: ")
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = accuracy_score(y_test, y_hat)
    print(f"Accuracy: {acc}")
    print(f"F1: {f1_score(y_test, y_hat)}")