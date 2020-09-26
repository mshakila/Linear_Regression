# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 02:09:56 2020

@author: Admin
"""

# ASSIGNMENT MULTI LINEAR REGRESSION - ToyotaCorolla dataset

#######   Business Problem : To predict price of Toyota Corolla car  ############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#######   Dataset acquisition  #############################

corolla_raw =pd.read_csv("file:///E:/EXCELR/Assignments/Multi linear regression_Assignment/ToyotaCorolla.csv",  encoding ='cp1252')

corolla_raw.columns # there are 38 variables but we need only a few to predict
# price as per business problem
# we need following variables
#("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")

corolla_raw1 = corolla_raw.loc[:,["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
corolla_raw1.rename(columns={'Age_08_04': 'Age'},inplace=True)
corolla_raw1.shape
# the dataset has 1436 records and 9 variables
corolla_raw1.columns
# 'Price', 'Age', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax','Weight'

################# Exploraroty Data Analysis ###############
corolla_raw1.head()
corolla_raw1.dtypes
# showing all variables  as integers
from collections import Counter
Counter(corolla_raw1.HP)
Counter(corolla_raw1.cc)
Counter(corolla_raw1.Doors)
Counter(corolla_raw1.Gears)
Counter(corolla_raw1.Quarterly_Tax)

# all above columns are categorical or discrete

# since some are of low frequencies let us merge them (make bins) and then convert to string

# merging categories in HP
# HP has initially 12 levels, reducing to 3 levels
corolla = corolla_raw1.copy()
def cats_hp(x):
    if x <=86:
        return 86
    elif x <= 107:
        return 97
    else:
        return 110

corolla['HP']=corolla['HP'].apply(cats_hp)
Counter(corolla.HP)

# merging categories in cc
corolla.cc[corolla.cc == 16000] = 1600
Counter(corolla.cc)
def cats_cc(x):
    if x <= 1300:
        return 1300
    elif x <= 1599:
        return 1400
    elif x <= 1600:
        return 1600
    else:
        return 2000
    
corolla['cc']=corolla['cc'].apply(cats_cc)
Counter(corolla.cc)

# merging categories in Doors
def cats_doors(x):
    if x <=3:
        return 3
    elif x <= 4:
        return 4
    else:
        return 5

corolla['Doors']=corolla['Doors'].apply(cats_doors)
Counter(corolla.Doors)


# merging categories in Gears
def cats_gears(x):
    if x <=5:
        return 5
    else:
        return 6

corolla['Gears']=corolla['Gears'].apply(cats_gears)
Counter(corolla.Gears)   
    
    
# merging categories in Quarterly_Tax
def cats_tax(x):
    if x <= 72:
        return 69
    elif x <= 85:
        return 85
    else:
        return 185

corolla['Quarterly_Tax']=corolla['Quarterly_Tax'].apply(cats_tax)
Counter(corolla.Quarterly_Tax) 

# converting the above 5 variables to strings/objects
corolla_new = corolla.copy()
corolla_new['HP'] = corolla_new['HP'].apply(str)   
corolla_new['cc'] = corolla_new['cc'].apply(str)   
corolla_new['Doors'] = corolla_new['Doors'].apply(str)   
corolla_new['Gears'] = corolla_new['Gears'].apply(str)   
corolla_new['Quarterly_Tax'] = corolla_new['Quarterly_Tax'].apply(str)   
corolla_new.dtypes
corolla.dtypes

########################## Business moment decisions of  variables
corolla_new[['Price','Age','KM','Weight']].mean()
corolla_new[['Price','Age','KM','Weight']].median()
corolla_new.mode()
corolla_new.var()
corolla_new.std()
corolla_new.iloc[:,[0,1,2,8]].kurt()
corolla_new.iloc[:,[0,1,2,8]].skew()

# visualization of variables
plt.hist(corolla_new.Price) # right skewed
plt.boxplot(corolla_new.Price,0,'ro',0)
plt.boxplot(corolla_new.KM,0,'go',0);plt.title('Boxplot of KM (distance)')
corolla_new.Weight.plot(kind='area')
corolla_new.Age.plot(kind='bar');plt.title('Barplot of Age')
corolla_new.Gears.value_counts().plot(kind='pie') # most cars have 5 gears.
corolla_new.HP.value_counts().plot(kind='pie') # more than 50% cars have HP of around 110
pd.crosstab(corolla_new.HP,corolla_new.Gears).plot(kind='bar') # cars with 6 gears are mainly present with 110 HP.  
sns.boxplot(x='Gears',y='Price',data=corolla_new)
sns.boxplot(x='HP',y='Price',data=corolla_new)

# Normality
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import anderson,normaltest,
%matplotlib inline
qqplot(corolla_new.Price,line='s')
anderson(corolla_new.Price) # stat is 63, pvalue <0.01, data is not normal
qqplot(corolla_new.Age,line='s') 
anderson(corolla_new.Age) # stat is 30 pvalue is <0.01, data is not normal
qqplot(corolla_new.KM,line='s') 
anderson(corolla_new.KM) # stat is 15 pvalue is <0.01, data is not normal
qqplot(corolla_new.Weight,line='s') 
anderson(corolla_new.Weight) # stat is 50 pvalue is <0.01, data is not normal
normaltest(corolla_new.Weight) # stat is 1041 pvalue is 0.00, data is not normal

normaltest(np.log(corolla_new.Price)) # log,sqrt, sqr,1/sqr, 1/x, x+x2
normaltest(corolla_new.Weight + corolla_new.Weight**2)

################ Model Building ################################# 
# linearity
# the predictors should be linearly related to the target. can be found
# by using correlation matrix
# scatter plot matrix for continuous variables
%matplotlib qt
sns.pairplot(corolla_new.iloc[:,[0,1,2,8]])
corolla_new.corr() # age and price have strong negative corr. 
# KM (negative) and weight (positive) have weak corr with Price

import statsmodels.formula.api as smf  # used to build linear regression model


##################### model using all variables ############
model1 = smf.ols('Price ~ Age + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data=corolla_new).fit()
model1.summary()
# R2 is 0.871, adj-R2 is 0.870, F-stat is signif, all coefficients are signif except Q_tax-85

# checking for influential values
import statsmodels.api as sm
sm.graphics.influence_plot(model1)
# index 221 AND 960 is showing high influence so we can exclude that entire row
corolla2 = corolla_new.drop(corolla_new.index[[221,960]],axis=0)

np.sqrt(np.mean(model1.resid**2)) # RMSE 1303.595
np.mean(model1.resid) # mean of errors is zero

############# model after removing 2 influential obserations
model2 = smf.ols('Price ~ Age + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data=corolla2).fit()
model2.summary()
# R2 is 0.887, adj-R2 is 0.886, F-stat is signif, all coefficients are signif except Q_tax-85

np.sqrt(np.mean(model2.resid**2)) # RMSE 1220.526
np.mean(model2.resid) # mean of errors is zero

############## model after removing 3 influentl observations
corolla1 = corolla_new.drop(corolla_new.index[[601,960,221]])
model3 = smf.ols('Price ~ Age + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data=corolla1).fit()
model3.summary()
# R2 os 0.891 , but quarterly_tax is not significant

np.sqrt(np.mean(model3.resid**2)) # RMSE 1195.469
np.mean(model3.resid) # mean of errors is zero

############## model using train and test dataset
# splitting corolla1 into train and test data
from sklearn.model_selection import train_test_split
corolla_train, corolla_test = train_test_split(corolla2, test_size=0.3,random_state=123)


# preparing model on train data
model_train = smf.ols('Price ~ Age + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data=corolla_train).fit()
model_train.summary()
# R2 is 0.887, adj-R2 is 0.886, F-stat is signif, all coefficients are signif

# prediction on train data
train_pred=model_train.predict(corolla_train)
train_resid = train_pred - corolla_train.Price
np.sqrt(np.mean(train_resid**2)) # RMSE 1208.016

# test data prediction
test_pred = model_train.predict(corolla_test)
test_resid = test_pred - corolla_test.Price
np.sqrt(np.mean(test_resid**2)) # RMSE 1260.279

# RMSE for both train and test residuals is close, so our model is just right, neither overfitter or underfitted

# added variable plot
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model_train)

######################## checking Regression assumptions

###################### Linearity in parameters
%matplotlib inline
plt.scatter(corolla_train.Price,train_pred,c='r')
plt.scatter(corolla_test.Price, test_pred,c='b')
# both plots show strong positive linear relationship between Y and fitted values

########## normality of residuals
plt.hist(model_train.resid_pearson)

import pylab
import scipy.stats as st
st.probplot(model_train.resid_pearson, dist='norm',plot=pylab) 
# residuals are normally distributed

###### Homoscedasticity
# residual and fitted
%matplotlib qt
plt.scatter(train_pred,model_train.resid_pearson,c='b'),plt.axhline(y=0,color='red')
# most residuals are randomly scattered, showing no pattern, constant variance,
# showing homoscedasicity

########### Multicollinearity
# calculating VIF's values of independent variables
corolla_train.dtypes
rsq_age = smf.ols('Age~KM+Weight',data=corolla_train).fit().rsquared  
vif_age = 1/(1-rsq_age) # 16.33

rsq_wt = smf.ols('Weight~ Age+KM',data=corolla_train).fit().rsquared  
vif_wt = 1/(1-rsq_wt) # 564.98

rsq_km = smf.ols('KM~Age + Weight',data=corolla_train).fit().rsquared  
vif_km = 1/(1-rsq_km) #  564.84

           # Storing vif values in a data frame
d1 = {'Variables':['Age','Weight','KM'],'VIF':[vif_age,vif_wt,vif_km]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# all continuous variables have vif less than 2, so no multicollinearity problem

############ checking auto-correlation of residuals
from statsmodels.stats import diagnostic as diag
diag.acorr_ljungbox(model_train.resid_pearson, lags=1) 
# pvalue is >0.05, so no autocorrelation 

# All assumptions of regression are satisfied

# The train dataset has given the best model.

'''
CONCLUSIONS

We have to predict the profit of Toyota Corolla car. Since Price is continuous
and many predictors are given, we have used multi linear regression technique.
 
EDA has been done to understand the variables. Some continuous predictors were 
having only few levels, so these have been converted to bins and used as 
factors for analysis.

Many models are built. In the model with highest R-sqr and least RMSE, all 
regression coefficients were not significant. The dataset was split into train and 
test. Model was built USING train and it was validated using test data. This model
has second highest R-sqr value and second least RMSE. But all its parameters were 
significant. So, this model was chosen to predict Price for test data.

All assumptions of regression were checked: Linearity of parameters, for
residuals - normality, autocorrleation were checked. MUlticollinearity of predictors
was checked. 

All assumptions were satisfied.


'''








