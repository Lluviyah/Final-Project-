import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('hr_analytics.csv')
# Box plot for average monthly hours
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['average_montly_hours'])
plt.title('Box Plot - Average Monthly Hours')
plt.show()

# Box plot for number of projects
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['number_project'])
plt.title('Box Plot - Number of Projects')
plt.show()