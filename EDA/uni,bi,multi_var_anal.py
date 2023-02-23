
#Univariate,Bivariate and MultiVariate Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
df.head()
#sepal_length	sepal_width	petal_length	petal_width	species
#0	5.1	3.5	1.4	0.2	setosa
#1	4.9	3.0	1.4	0.2	setosa
#2	4.7	3.2	1.3	0.2	setosa
#3	4.6	3.1	1.5	0.2	setosa
#4	5.0	3.6	1.4	0.2	setosa
df.shape
#(150, 5)
#Univariate Analysis
df_setosa=df.loc[df['species']=='setosa']
df_virginica=df.loc[df['species']=='virginica']
df_versicolor=df.loc[df['species']=='versicolor']
plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'o')
plt.xlabel('Petal length')
plt.show()

#Bivariate Analysis
sns.FacetGrid(df,hue="species",size=5).map(plt.scatter,"petal_length","sepal_width").add_legend();
plt.show()


#Multivariate Analysis
sns.pairplot(df,hue="species",size=3)

 
