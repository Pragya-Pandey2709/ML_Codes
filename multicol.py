
#Multicollinearity In Linear Regression
import pandas as pd
import statsmodels.api as sm
df_adv = pd.read_csv('data/Advertising.csv', index_col=0)
X = df_adv[['TV', 'radio','newspaper']]
y = df_adv['sales']
df_adv.head()

X = sm.add_constant(X)
X


## fit a OLS model with intercept on TV and Radio

model= sm.OLS(y, X).fit()
model.summary()




import matplotlib.pyplot as plt
X.iloc[:,1:].corr()

df_salary = pd.read_csv('data/Salary_Data.csv')
df_salary.head()

 
X = df_salary[['YearsExperience', 'Age']]
y = df_salary['Salary']
## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
model= sm.OLS(y, X).fit()

model.summary()




X.iloc[:,1:].corr()
#YearsExperience	Age
#YearsExperience	1.000000	0.987258
#Age	0.987258	1.000000
 
