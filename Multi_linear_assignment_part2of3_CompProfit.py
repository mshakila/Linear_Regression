# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:27:31 2020

@author: Admin
"""

#  # ASSIGNMENT MULTI LINEAR REGRESSION  - computer dataset

###### 1. Business problem: To predict the price of computer

###### 2. Dataset acquisition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

computer = pd.read_csv("file:///E:/EXCELR/Assignments/Multi linear regression_Assignment/Computer_Data.csv")
computer.shape # (6259, 11)
computer.columns
computer.drop(['Unnamed: 0'],axis=1,inplace=True)
#'price', 'speed', 'hd', 'ram', 'screen', 'cd', 'multi','premium','ads', 'trend'],
computer.shape
computer.head(3)
computer.dtypes
# I business moment - measures of central tendency
# finding mean
np.mean(computer)
'''
price     2219.576610
speed       52.011024
hd         416.601694
ram          8.286947
screen      14.608723
ads        221.301007
trend       15.926985
'''
# PRICE variable
np.median(computer.price)
import statistics as stat
stat.mode(computer.price)
np.var(computer.price)
np.std(computer.price)
np.max(computer.price) - np.min(computer.price)
from scipy.stats import kurtosis, skew
kurtosis(computer.price)
skew(computer.price)

# visualizations
%matplotlib qt
plt.hist(computer.price)
plt.boxplot(computer.price,0,'ro',0)
# normality
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import anderson, shapiro, normaltest
%matplotlib inline
qqplot(computer.price,line='s')
anderson(computer.price)
shapiro(computer.price)
normaltest(computer.price)
normaltest(computer.speed)

# similar functions to be done for other variables, already done in r-studio
# finding missing values
computer.isnull().sum()

# converting to 
# correlation matrix
computer.corr()
%matplotlib qt
import seaborn as sns
sns.pairplot(computer)

computer.dtypes
from collections import Counter # freq of col
Counter(computer.speed)
Counter(computer.ram)
Counter(computer.screen)
# all above to be converted to string(or object)

# converting to string
computer['speed']=computer['speed'].apply(str)
computer['ram'] = computer['ram'].apply(str)
computer['screen'] = computer['screen'].apply(str)
computer.dtypes

computer1 = computer.loc[:,['price','hd','ads','trend']]
computer1.dtypes

# DETECTING AND REMOVING OUTLIERS
from scipy import stats
z = np.abs(stats.zscore(computer1))
threshold =3
np.where(z>3) # there are many outliers
# removing outliers
computer1 = computer1[(z<3).all(axis=1)]
6259 - 6122 # removed 137 outliers

computer2 = computer[(z<3).all(axis=1)]

# checking multicollinearity - calculating vif factors
from statsmodels.stats.outliers_influence import variance_inflation_factor
computer1.columns
X = computer1.iloc[:,[1,2,3]]
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif
# vif is < 10, so no multicollinearity btw predictors

# correlation 
computer1.corr()
# correlation is not high 

####################### Running Linear Regression
import statsmodels.formula.api as smf 
computer2.columns
model = smf.ols('price ~ speed+hd+ ram + screen + cd+multi+premium+ads+trend',data=computer2).fit()
model.summary()
# R2 is 0.804, adj-R2 is 0.803, F-stat signif, all coeff are signifi

# added variable plot 
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model)
# this plot partial regression : the correlation btw price and predictors when remove effect of other variables

# check if there are any influential values using influence plot
sm.graphics.influence_plot(model)
# influence = model.get_influence()

# RMSE of residuals
np.sqrt(np.mean(model.resid**2)) # 244

# checking normality of residuals
stats.anderson(model.resid) # residuals are normal

# checking auto-correlation of residuals
from statsmodels.stats import diagnostic as diag
diag.acorr_ljungbox(model.resid, lags=1)
# pvalue is <0.05, so autocorrelation is present

# checking heteroscedasticity
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F-statistic','p-value']
gold_test = sms.het_goldfeldquandt(model.resid,model.model.exog)
lzip(name,gold_test)
# ('F-stat', 0.6722696289421596), ('p-value', 0.9999999999999999)], 
# go with null, residuals are homoscedastic: constant variance

pred_price = model.predict(computer2.drop(['price'],axis=1))
pred_price[0:4]
computer2.price[0:4]
model.resid[0:4]
1499-1787

# except for autocorrelation all assumptions are satisfied

# splitting data
from sklearn.model_selection import train_test_split
train,test = train_test_split(computer2,test_size=0.30, random_state=100)
 1837/6122
# Now the data is divided into independent (x) and dependent variables (y)
computer2.columns
x = computer2.iloc[:,1:10]
y = computer2.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

model1 = smf.ols('price ~ speed+hd+ ram + screen + cd+multi+premium+ads+trend',data=train).fit()
model1.summary() # R2 is 0.803

sm.graphics.influence_plot(model1)
np.sqrt(np.mean(model1.resid**2)) # RMSE 245
stats.anderson(model1.resid) # residuals are normal
diag.acorr_ljungbox(model1.resid, lags=1) # NO autocorrelation 
gold_test1 = sms.het_goldfeldquandt(model1.resid,model1.model.exog)
lzip(name,gold_test1) # residuals have constant variance

pred_test = model1.predict(x_test)
test_resid = y_test - pred_test
np.sqrt(np.mean(test_resid**2)) # RMSE 244

'''
CONCLUSIONS

We have to predict the price of computers given many features. since price is
continuous variable , we are using multi linear regression technique.

Two models are built: one using all records and second model is built using 
only train dataset ( and using test data for validation).

RMSE is almost same for all. Second model is better as all assumptions related 
to observarions (random), predictors (no multicollinearity) and residuals
(normal, homoscedasticity, no autocorrelation) are satisfied.
'''
