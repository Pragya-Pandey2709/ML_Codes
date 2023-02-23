
Ridge and LAsso Regression implementation
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=load_boston()
df

dataset = pd.DataFrame(df.data)
print(dataset.head())

dataset.columns=df.feature_names
dataset.head()
#CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
#0	0.00632	18.0	2.31	0.0	0.538	6.575	65.2	4.0900	1.0	296.0	15.3	396.90	4.98
#1	0.02731	0.0	7.07	0.0	0.469	6.421	78.9	4.9671	2.0	242.0	17.8	396.90	9.14
#2	0.02729	0.0	7.07	0.0	0.469	7.185	61.1	4.9671	2.0	242.0	17.8	392.83	4.03
#3	0.03237	0.0	2.18	0.0	0.458	6.998	45.8	6.0622	3.0	222.0	18.7	394.63	2.94
#4	0.06905	0.0	2.18	0.0	0.458	7.147	54.2	6.0622	3.0	222.0	18.7	396.90	5.33
df.target.shape

dataset["Price"]=df.target
dataset.head()

X=dataset.iloc[:,:-1] ## independent features
y=dataset.iloc[:,-1] ## dependent features
#Linear Regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,X,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)

#-37.131807467699204


#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)

#GridSearchCV(cv=5, error_score='raise-deprecating',
       #estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   #normalize=False, random_state=None, solver='auto', tol=0.001),
       #fit_params=None, iid='warn', n_jobs=None,
       #param_grid={'alpha': [1e-15, 1e-10, 1e-08, 0.001, 0.01, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]},
       #pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       #='neg_mean_squared_error', verbose=0)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

#-29.871945115432595


#Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

#-35.491283263627096

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
prediction_lasso=lasso_regressor.predict(X_test)
prediction_ridge=ridge_regressor.predict(X_test)
import seaborn as sns

sns.distplot(y_test-prediction_lasso)


import seaborn as sns

sns.distplot(y_test-prediction_ridge)
<matplotlib.axes._subplots.AxesSubplot at 0x1b4bf777240>

 
