# MULTI LINEAR REGRESSION ASSIGNMENT - startups dataset
# Q1.. Prepare a prediction model for profit of 50_startups data

startups <- read.csv("file:///E:/EXCELR/Assignments/Multi linear regression_Assignment/50_Startups.csv")
names(startups)
names(startups) <- c('RD','Admin','Marketing','State','Profit')
attach(startups)
dim(startups)
head(startups)

str(startups)
library(Hmisc)
describe(startups)

# large dataset or not
library(moments)
10 * kurtosis(Profit) # 28. there are 50 records in dataset. so it is relatively large
10*kurtosis(Admin)
library(e1071)
10 * kurtosis(Profit)

############ business moment decisions 

# Profit variable - this is the target variable
mean(Profit) # 112012.6
median(Profit)# 107978.2
mlv(Profit, method = 'mfv') # no mode
var(Profit) # 1624588173
sd(Profit) # 40306.18
range(Profit) # 14681.4 192261.8 = 177580.4
192261.8 - 14681.4
skewness(Profit) # 0.02191219, normal
kurtosis(Profit) # 2.824704, normal

# RD variable
mean(RD) # 73721.62
median(RD) # 73051.08
library(modeest)
mlv(RD, method = 'mfv') # 0 (means zero is the value with highest frequency )
var(RD) # 2107017150
sd(RD) # 45902.26
range(RD) # 0.0 165349.2 = 165349.2
165349.2 - 0.0
skewness(RD) # 0.1542932, normal
kurtosis(RD) # 2.194932, slightly flattened (for normal, kurtosis=3)

# Admin variable
mean(Admin) # 122699.8
median(Admin) # 122699.8
mlv(Admin, method = 'mfv') # no mode
var(Admin) # 784997271
sd(Admin) # 28017.8
range(Admin) # 51283.14 - 182645.56 = 131362.4
182645.56 - 51283.14
skewness(Admin) # -0.4600745, normal
kurtosis(Admin) # 3.085538, normal

 
# Marketing variable
 mean(Marketing) # 211025.1
 median(Marketing) # 212716.2
 mlv(Marketing, method = 'mfv') # 0
 var(Marketing) # 14954920097
 sd(Marketing) # 122290.3
 range(Marketing) # 0.0 471784.1 = 471784.1
 skewness(Marketing) # -0.04372111, normal
 kurtosis(Marketing) # 2.275967


########## visualizations
 
 # Profit expenses
 hist(Profit,probability = T)
 lines(density(Profit))
 boxplot(Profit,horizontal = TRUE,main='Boxplot of Profit')  
 qqnorm(Profit,main='Normal qq plot of Profit')
 qqline(Profit)
 shapiro.test(Profit) # p-value = 0.7666, normal
 ad.test(Profit) #  p-value = 0.7616, normal
 # outlier detection
 box_Profit=boxplot(Profit,horizontal = TRUE,main='Boxplot of Profit')  
 box_Profit$out # numeric(0), no outliers
 
# R and D expenses
hist(RD,probability = T)
lines(density(RD))
boxplot(RD,horizontal = TRUE,main='Boxplot of RD')  
qqnorm(RD,main='normal qq plot of RD')
qqline(RD)
shapiro.test(RD) # p-value = 0.1801, normal
library(nortest)
ad.test(RD) #  p-value = 0.4485, normal
# outlier detection
box_RD=boxplot(RD,horizontal = TRUE,main='Boxplot of RD')  
box_RD$out # numeric(0), no outliers

# Admin expenses
hist(Admin,probability = T)
lines(density(Admin))
boxplot(Admin,horizontal = TRUE,main='Boxplot of Admin')  
qqnorm(Admin,main='Normal qq plot of Admin')
qqline(Admin)
shapiro.test(Admin) # p-value = 0.2366, normal
ad.test(Admin) #  p-value = 0.2366, normal
# outlier detection
box_Admin=boxplot(Admin,horizontal = TRUE,main='Boxplot of Admin')  
box_Admin$out # numeric(0), no outliers

# Marketing expenses
hist(Marketing,probability = T)
lines(density(Marketing))
boxplot(Marketing,horizontal = TRUE,main='Boxplot of Marketing')  
qqnorm(Marketing,main='Normal qq plot of Marketing')
qqline(Marketing)
shapiro.test(Marketing) # p-value = 0.3451, normal
ad.test(Marketing) #  p-value = 0.5275, normal
# outlier detection
box_Marketing=boxplot(Marketing,horizontal = TRUE,main='Boxplot of Marketing')  
box_Marketing$out # numeric(0), no outliers


# finding missing values in dataset
startups[!complete.cases(startups),] # no missing values

# State variable
table(State)
# dummy variable creation for State
library(psych)
state_dummy <- dummy.code(State, group = NULL)
startups <- as.data.frame(cbind(startups,state_dummy))
# 3 dummy variables are created:  California, Florida and New York
head(startups,3)
names(startups)[8] <- 'New_York'
# dropping State variable
startups <- startups[,-4]

summary(startups)
str(startups)
############ Model building  #######################

# There should exist linear association btw target and predictors known as correlation
# Find the correlation b/n Output (profit) & predictors-Scatter plot
pairs.panels(startups[,c(1,2,3,4)])
# pairs(startups[,c(1,2,3,4)])
# plot(startups[,c(1,2,3,4)])
#  Correlation Coefficient matrix - Strength & Direction of Correlation
cor(startups[,c(1,2,3,4)])

### Partial Correlation matrix - Pure Correlation  b/n the varibles
#install.packages("corpcor")
library(corpcor)
cor2pcor(cor(startups[,c(1,2,3,4)]))

##### Linear Model #############
# using all variables as is for analysis 

reg.profit <- lm(Profit ~ .,data = startups)
summary(reg.profit)
# find that New_York is NA , also (n-1) dummy variables
# to be used in analysis, dropping New_York
reg.profit <- lm(Profit ~ . - New_York,data = startups)
summary(reg.profit) # 0.9508,
reg.profit$coefficients
sqrt(mean(reg.profit$residuals^2)) # RMSE is 8854
# multicollinearity check using vif
library(car)
vif(reg.profit)
#      RD      Admin  Marketing California    Florida 
# 2.495511   1.177766   2.416797   1.335061   1.361299
# since all are less than 10, suggests no multicollinearity
sqrt(vif(reg.profit)) # sqrt for each less than 2, hence no collinearity among predictors


#############################################
# II model using standardized data

names(startups)
startups_std <-  data.frame(scale(startups[,c(1,2,3,4)]))
summary(startups_std)
startups_std <- data.frame(cbind(startups_std,startups[,c(5,6,7)]))
head(startups_std)

reg.profit.std <- lm(Profit ~.-New_York, data = startups_std)
summary(reg.profit.std) # R2 is 0.9508
sqrt(mean(reg.profit.std$residuals^2)) # 0.2196874

vif(reg.profit.std)
#         RD      Admin  Marketing California    Florida 
#   2.495511   1.177766   2.416797   1.335061   1.361299

##########################################################
# added variable plots
library(psych)
avPlots(reg.profit,id.n=2,id.cex=0.7)

######################################
# model by removing admin variable
names(startups)
reg.profit.noadmin <- lm(Profit ~ RD+Marketing+California+Florida, data = startups)
summary(reg.profit.noadmin)

########################################
# model remove marketing variable

reg.profit.nomarketing <- lm(Profit ~ RD+Admin+California+Florida, data = startups)
summary(reg.profit.nomarketing)
# but we are aware that marketing is very important for 
# increasing sales,so cannot delete this variable
###########################################
#RMSE

# Deletion Diagnostics for identifying influential observations
influence.measures(reg.profit)
#     7,      47,       50
#  0.2119, 0.2654,  0.1015
library(car)
## plotting Influential measures 
windows()
influenceIndexPlot(reg.profit,id.n=3) # index plots for infuence measures, 49,50
influencePlot(reg.profit,id.n=3) # A user friendly representation of the above, 49,50

startups[50,]
# RD    Admin     Marketing  Profit California Florida New_York
#  50  0 116983.8  45173.06 14681.4          1       0        0

summary(startups)

######### deleting 50th record and running model #######
# when compare corr , it has slightly improved by removing 50th record
cor(startups[-50,c(1,2,3,4)])
# corr without removing 50th record
cor(startups[,c(1,2,3,4)])

final_model <- lm(Profit ~.-New_York, data=startups[-50,])
summary(final_model)
sqrt(mean(final_model$residuals^2)) # 7383.181
# remove 47, R2 = 0.949, intercept,RD,marketing are sig
# remove 50, R2 = 0.9618, intercept,RD,marketing are sig

pred_interval <- predict(final_model,interval = 'predict')
pred <- predict(final_model)
pred <- data.frame(pred)
head(pred)

startups_pred <- data.frame(cbind(startups[-50,],pred))

# Residuals
sum(final_model$residuals) # -1.449507e-12
mean(final_model$residuals) # -2.967241e-14
sqrt(mean(final_model$residuals^2)) # RMSE is 7383.181
sqrt(mean(reg.profit$residuals^2)) # RMSE is 8854.761

# normality of residuals
shapiro.test(final_model$residuals) # p-value = 0.2833, normal
# qq plot given in diagnostic plots

# Diagnostic Plots
library(car)
plot(final_model)# Residual Plots, QQ-Plos, Std. Residuals vs Fitted, Cook's distance

qqPlot(final_model, id.n=5) # QQ plots of studentized residuals, helps identify outliers

Profit_pred <- data.frame(cbind(startups[-50,],pred,final_model$residuals))
head(Profit_pred)
dim(Profit_pred) # 49 9
