# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 01:20:13 2019

@author: Admin
"""

# MULTI LINEAR REGRESSION ASSIGNMENT
# Q1.. Prepare a prediction model for profit of 50_startups data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

startups = pd.read_csv("file:///E:/EXCELR/Assignments/Multi linear regression_Assignment/50_Startups.csv")

startups.columns
# 'R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit'
startups.columns = ["RD","Admin","Marketing","State","Profit"] # Replacing column names with easier ones
# 'RD', 'Admin', 'Marketing', 'State', 'Profit'
startups.head()
startups.tail()
startups.shape # (50, 5)

# if large dataset or not
from scipy.stats import kurtosis, skew
10 * kurtosis(startups.RD) # -8.050
# so it is a large dataset

# I business moment - measures of central tendency
# finding mean
np.mean(startups)
 # RD            73721.6156
#Admin        121344.6396
# Marketing    211025.0978
# Profit       112012.6392

# finding median
np.median(startups.RD) # 73051
np.median(startups.Admin) # 122699
np.median(startups.Marketing) # 212716
np.median(startups.Profit) # 107978

# finding mode
import statistics
statistics.mode(startups.RD) # 0.0
statistics.mode(startups.Admin) # no unique mode
statistics.mode(startups.Marketing) # 0.0
statistics.mode(startups.Profit) # no unique mode

# II business moment - measures of dispersion
np.var(startups.RD)
np.var(startups.Admin)
np.var(startups.Marketing)
np.var(startups.Profit)

np.std(startups.RD)
np.std(startups.Admin)
np.std(startups.Marketing)
np.std(startups.Profit)

np.max(startups.RD) - np.min(startups.RD)
np.max(startups.Admin) - np.min(startups.Admin)
np.max(startups.Marketing) - np.min(startups.Marketing)
np.max(startups.Profit) - np.min(startups.Profit)

# 3rd and 4th business moments
kurtosis(startups.RD)
skew(startups.RD)
kurtosis(startups.Admin)
skew(startups.Admin)
kurtosis(startups.Marketing)
skew(startups.Marketing)
kurtosis(startups.Profit)
skew(startups.Profit)

# visualizations
plt.hist(startups.RD)
plt.boxplot(startups.RD)

plt.hist(startups.Admin)
plt.boxplot(startups.Admin,0,'rs',0)

# check normality
from statsmodels.graphics.gofplots import qqplot
# from statsmodels.graphics.gofplots import qqline
from scipy.stats import shapiro,anderson,normaltest

%matplotlib inline
qqplot(startups.RD, line='s')
qqplot(startups.Admin, line='s')
shapiro(startups.Admin) # 0.23660 , 
# null hyp is data is normal, since p>0.05, cannot reject null, so Admin data is normal
anderson(startups.Marketing) # 0.736,data is normal
normaltest(startups.Profit) # pvalue=0.991, normal
 # all variables are normal
 
# correlation matrix
startups.corr()

# scatter plot
import seaborn as sns
%matplotlib qt
sns.pairplot(startups)
%matplotlib inline
plt.plot(startups.RD,startups.Profit,'go') # 2 variable scatter plot

# creating dumming variables for State variable
state_dummies = pd.DataFrame(pd.get_dummies(startups['State']))
startups1 = pd.concat([startups,state_dummies],axis=1)
startups1 = startups1.drop(['State'],axis=1)
startups1.rename(columns={'New York': 'NewYork'},inplace =True)

# finding missing values
startups1.isnull().sum() # no missing values in dataset

# DETECTING AND REMOVING OUTLIERS
from scipy import stats
z = np.abs(stats.zscore(startups1))
# lets define a threshold of outlier
threshold = 3
np.where(z>3) # gives outlier position
# there are no outliers
 
# if there were outliers then 
 # startups2=startups1[(z<3).all(axis=1)]

# checking multicollinearity - calculating vif factors
 
# For each X, calculate VIF and save in dataframe
from statsmodels.stats.outliers_influence import variance_inflation_factor
startups1.columns
X = startups1.iloc[:,[0,1,2,4,5]]
vif = pd.DataFrame()
vif["features"] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif # inspect vif factors
# all values < 10, so no multicollinearity

# correlation
startups1.corr()
startups1.columns
# 'RD', 'Admin', 'Marketing', 'State', 'Profit', 'California', 'Florida','NewYork'

############## Model building using all variables ##############
import statsmodels.formula.api as smf

model = smf.ols('Profit ~ RD+Admin+Marketing+California+Florida',data=startups1).fit()
model.params
model.summary()
# only intercept and RD are significant
# R2 is 0.951 and adjR2 is 0.945
# pvalue of Fstats suggests that overall model is good
# only RD and intercept are significant
# RD and marketing high corr 0.724

################ build model using only RD ###################
model_rd = smf.ols('Profit ~ RD',data=startups1).fit()
model_rd.summary()
# RD alone is significant
model_mark = smf.ols('Profit ~ Marketing', data=startups1).fit()
model_mark.summary()
# marketing separately is significant

################# model building only admin  ############################
model_admin = smf.ols('Profit ~ Admin',data=startups1).fit()
model_admin.summary() # still admin not sig

######################## model using log(marketing)  ########################
startups1['log_marketing']=np.log(startups1.Marketing)
# removing 3 records which show marketing values as 0. 
startups2 = startups1.drop([startups1.index[19],startups1.index[47],startups1.index[48]])
model_logMark = smf.ols('Profit~RD+Admin+log_marketing+California+Florida',data=startups2).fit()
model_logMark.params
model_logMark.summary()
# still marketing not significant

####################### model using log(admin)  ######################
startups1['log_admin']=np.log(startups1.Admin)
model_logAdmin = smf.ols('Profit~RD+log_admin+Marketing+California+Florida',data=startups1).fit()
model_logAdmin.params
model_logAdmin.summary()
# still Admin not significant

#################### model using admin square
startups1['sq_admin']=startups1.Admin**2
4**2
model_sqAdmin = smf.ols('Profit~RD+sq_admin+Marketing+California+Florida',data=startups1).fit()
model_sqAdmin.summary()
# still admin not sig, only RD significant

# added-variable plot
import statsmodels.api as sm
%matplotlib qt
sm.graphics.plot_partregress_grid(model)

# admin variable showing very less linearity among continuous variables.

######################## let us remove admin and run the model
model_noAdmin = smf.ols('Profit ~ RD+Marketing+California+Florida',data=startups1).fit()
model_noAdmin.summary()
# still only RD significant

################# let us remove admin and run the model
model_noMark = smf.ols('Profit ~ RD+Admin+California+Florida',data=startups1).fit()
model_noMark.summary()
# still only RD significant

############# let us use only continuous variables and run the model
startups1.dtypes
model2 = smf.ols('Profit ~ RD+Admin+Marketing',data=startups1).fit()
model2.summary()
# still only RD significant

##################### model using standardized var  #################################
# standardize = lambda x: (x-x.mean()) / x.std()
# here we are normalizing the variables
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
startups3 = pd.DataFrame(scaler.fit_transform(startups[["RD","Admin","Marketing","Profit"]]))
startups3.columns = ["RD_std","Admin_std","Marketing_std","Profit_std"]
startups4 = pd.concat([startups1,startups3],axis = 1)
np.min(startups3.RD_std) # min is 0
np.max(startups3.RD_std) # max is 1
model_std = smf.ols('Profit_std~RD_std+Admin_std+Marketing_std+California+Florida',data=startups4).fit()
model_std.summary()
# even after standardizing only intercept and RD-std are significant
# R2 is 0.951 

############# considering first model built using all variables ###########
model = smf.ols('Profit ~ RD+Admin+Marketing+California+Florida',data=startups1).fit()
# dropping 
model.summary()
# Confidence values 99%
print(model.conf_int(0.01)) # 99% confidence level

# Predicted values of Profit 
X.columns
Y
profit_pred = model.predict(X)

import statsmodels.api as sm
%matplotlib qt
# added variable plot for final model
sm.graphics.plot_partregress_grid(model)

import statsmodels.api as sm
sm.graphics.influence_plot(model) # 49, 48, 45, 46 
# influence = model.get_influence()
# index 49,48,45,46 are showing high influence, but since have only 50 records
# we are not deleting any records
# Studentized Residuals = Residual/standard deviation of residuals

# RMSE of residuals
np.sqrt(np.mean(model.resid**2)) # 8854

# checking normality of residuals
stats.anderson(model.resid) # residuals are normal

# checking auto-correlation of residuals
from statsmodels.stats import diagnostic as diag
diag.acorr_ljungbox(model.resid, lags=1)
# pvalue is >0.05, so autocorrelation is absent

# checking heteroscedasticity
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F-statistic','p-value']
gold_test = sms.het_goldfeldquandt(model.resid,model.model.exog)
lzip(name,gold_test)
# [('F-statistic', 1.6512780672760905), ('p-value', 0.1415638365808888)]
# go with null, residuals are homoscedastic: constant variance

#  all assumptions are satisfied


'''
CONCLUSIONS

We have to predict the profit of 50 startup firms. Since profit is continuous
 and many predictors are given, we have used multi linear regression technique.
 
Many models are built. Since state-variable is categorical, it is converted
to dummy variables and used for building the model. 

By including more records and more relevant variables we can build a better model. 

'''



