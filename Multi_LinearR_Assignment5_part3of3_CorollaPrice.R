# ASSIGNMENT MULTI LINEAR REGRESSION - ToyotaCorolla dataset

#######  1. Business Problem : To predict price of Toyota Corolla car

#######  2. Dataset acquisition 
corolla_raw <- read.csv("file:///E:/EXCELR/Assignments/Multi linear regression_Assignment/ToyotaCorolla.csv")

names(corolla_raw) # there are 38 variables but we need only a few to predict
# price as per business problem
# we need following variables
#("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")


corolla <- corolla_raw[,c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")] 
names(corolla)
names(corolla)[2] <- 'Age'
dim(corolla) #  1436 records,   9 variables, Price is target

#######  2. Exploratory Data Analysis
str(corolla) # all vars given as integers
head(corolla)
attach(corolla)

table(corolla$HP) # there are 12 levels, hence convert to factor
#  69  71  72  73  86  90  97  98 107 110 116 192 
#  34   1  73   1 249  36 164   2  21 835   9  11

table(corolla$cc) # has 13 levels, hence convert to factor
#  1300  1332  1398  1400  1587  1598  1600  1800  1900  1975  1995  2000 16000 
#  248     2     2   164     4     4   845    14    30     1     2   119     1
# we see that there is one record showing 16000 whereas next highest is just 2000,
# there are many under 1600 level, this must be wrongly typed, change it to 1600
corolla$cc[corolla$cc == 16000] <- 1600 # now 12 levels

table(corolla$Doors) # there are 4 levels, hence convert to factor
#   2   3   4   5 
#   2 622 138 674 

table(corolla$Gears) #  there are 4 levels, hence convert to factor
#    3    4    5    6 
#    2    1 1390   43

table(corolla$Quarterly_Tax) #  there are 13 levels, hence convert to factor
#  19  40  64  69  72  85 100 163 185 197 210 234 283 
#   72   1  18 559   3 613  19   1  96  14  18  19   3 

######## converting HP,cc, doors, gears,quarterly_tax - all to factors
cols <- c('HP','cc', 'Doors', 'Gears','Quarterly_Tax')
corolla1 <- corolla
corolla1[,cols] <- data.frame(apply(corolla1[cols], 2, as.factor))
str(corolla1)
attach(corolla1)
summary(corolla1) # gives details of mean, median,min,max,1st quartile and 3rd quartile

############### reducing levels of factors having more than 4 levels ######
# we have seen that some categorical vars have many levels (12,13), 
# this will make analysis difficult. By reducing to 4 levels, easy interpretation.
# also some have just about 4 records in a single level, can merge the same

library(forcats)
corolla_new <- corolla1
# cc has initially 12 levels, reducing to 4 levels
table(corolla_new$cc)
levels(corolla_new$cc)
corolla_new$cc <- fct_collapse(corolla_new$cc, '1400'=c('1332','1398','1400','1587','1598'),
                               '2000'=c('1800','1900','1975','1995','2000'))
# HP has initially 12 levels, reducing to 4 levels
table(corolla_new$HP)
corolla_new$HP <- fct_collapse(corolla_new$HP, '69'=c('69','71','72','73'),
                               '97'=c('90','97','98'),
                               '110'=c('107','110','116','192'))
levels(corolla_new$HP)
# doors has just 2 records in 2doors, hence merge it
table(corolla_new$Doors)
corolla_new$Doors <- fct_collapse(corolla_new$Doors, '3'=c('2','3'))
# gears has just 3 records in 3 and 4 gears, hence merge the same
table(corolla_new$Gears)
corolla_new$Gears <- fct_collapse(corolla_new$Gears, '5'=c('3','4','5'))
# Quarterly_Tax has 13 levels, reduce to 4 levels
table(corolla_new$Quarterly_Tax)
corolla_new$Quarterly_Tax <- fct_collapse(corolla_new$Quarterly_Tax,
                                          '19'=c('19','40','64'),
                                          '69'=c('69','72'), '85'='85',
                                          '185'=c('100','163','185','197','210','234','283'))

table(corolla_new$HP)
table(corolla_new$cc)
table(corolla_new$Doors)
table(corolla_new$Gears)
table(corolla_new$Quarterly_Tax)

########## Business moment decisions for continuous var #############

# price variable - gives price of the car
mean(Price) # 10730.82
median(Price) # 9900
library(modeest)
mlv(Price,method='mfv') # 8950
which(Price %in% 8950) # 109 records have price of 8950
var(Price) # 13154872
sd(Price) # 3626.965
range(Price) # 4350 32500 = 28150
4350 - 32500
library(moments)
skewness(Price) # 1.700327
kurtosis(Price) # 6.720604

qqnorm(Price)
qqline(Price)
shapiro.test(Price)
library(nortest)
ad.test(Price)

hist(Price,probability = T)
lines(density(Price))
box_price <- boxplot(Price,horizontal = T)
box_price$out # there are 110 outliers in Price
min(box_price$out) # 17250
max(box_price$out) # 32500
# data is right skewed , not normal, many outliers

# Age variable ( gives age of car (in months) as on august,2004)
mean(Age) # 55.94708
median(Age) # 61
mlv(Age,method='mfv') # 68
var(Age) # 345.9596
sd(Age) # 18.6
range(Age) # 1 80 = 79

# normality checking
skewness(Age) # -0.8249756
kurtosis(Age) # 2.919459
qqnorm(Age)
qqline(Age)
# not normal 

hist(Age,probability = T)
lines(density(Age))
box_age <- boxplot(Age,horizontal = T)
box_age$out # there are 7 outliers in Age
# data is left skewed , not normal, a few outliers

# KM variable - indicates how much distance (in kms) the car has travelled
mean(KM) # 68533.26
median(KM) # 63389.
mlv(KM,method='mfv') # 36000
var(KM) # 1406733707
sd(KM) # 37506.45
range(KM) # 1 243000 = 242999

# normality checking
skewness(KM) # 1.013791
kurtosis(KM) # 4.67502
qqnorm(KM)
qqline(KM)
# not normal 

hist(KM,probability = T)
lines(density(KM))
box_km <- boxplot(KM,horizontal = T)
box_km$out # there are 49 outliers in KM
# data is right skewed , not normal, many outliers

# Weight variable - indicates weight of the car 
mean(Weight) # 1072.46
median(Weight) # 1070
mlv(Weight,method='mfv') # 1075
var(Weight) # 2771.088
sd(Weight) # 52.64112
range(Weight) # 1000 1615 = 615

# normality checking
skewness(Weight) # 3.102148
kurtosis(Weight) # 22.29137
qqnorm(Weight)
qqline(Weight)
# not normal 

hist(Weight,probability = T)
lines(density(Weight))
box_wt <- boxplot(Weight,horizontal = T)
box_wt$out # there are 66 outliers in Weight
# data is right skewed , not normal, many outliers

# categorical variables visualization
str(corolla_new)
summary(corolla_new)
names(corolla_new)
attach(corolla_new)
# "HP" "cc"  "Doors" "Gears"  "Quarterly_Tax" 
barplot(table(HP)) # most cars have 110 hp
barplot(table(cc)) # highest number of cars for 1600cc
barplot(table(Doors)) # most cars have 3 or 5 doors
barplot(table(Gears)) # most cars have 5 gears (apprx 1400)
barplot(table(Quarterly_Tax)) # more records in 69 and 85 tax payments

# stacked barplot 
barplot(table(Gears,HP),xlab='HP',col=c('red','green'),beside = T,main="Stacked Barplot of Gears with HP")
legend('topright',c('5 gears','6 gears'),fill=c('red','green'))
#  shows that 6 gears are only available for 
# 110 HP i.e, '110'=c('107','110','116','192'))

# finding missing values
corolla_new[!complete.cases(corolla_new),]  # no missing values

# we find that price, age, km, weight are in different units hence 
# need to standardize or normalize data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
data_norm <- data.frame(lapply(corolla_new[,c(1,2,3,9)],normalize))
names(data_norm) <- c("Price_std",'Age_std','KM_std','Weight_std')
names(data_norm)
corolla_norm <- cbind(subset(corolla_new,select=-c(Price,Age,KM,Weight)),data_norm)
str(corolla_norm)

############## correlation ########################
# for regression, predictors should be correlated with target 
str(corolla_new)
library(psych)
pairs.panels(corolla_new[,c(1,2,3,9)])
cor(corolla_new[,c(1,2,3,9)])
# price and age have strong negative correlation - as age of car increases, price decreases
# price and KM have moderate negative corr - as KM travelled increases, price decreases
# price and Weight have moderate positive corr - as Weight travelled increases, price also increases

### Partial Correlation matrix - Pure Correlation  b/n the varibles
library(corpcor)
cor2pcor(cor(corolla_new[,c(1,2,3,9)]))
# the cor between predictors is weak except for KM and Weight: here moderate correlation
# may not create multicollinearity problem, lets check after building model using vif

######## 4.& 5. Model building and Evaluation################
# we have to predict price which is continuous variable. 
# we will use multi linear regression tehcnique using ordinary least
# squares method which finds the best fit line having least errors

# let us build multi linear regression model using corolla_new dataset - has continuous and categorical var
str(corolla_new)
reg.price <- lm(Price ~ . , data = corolla_new)
summary(reg.price)
sqrt(mean(reg.price$residuals^2)) # RMSE is 1263.851
# R2 is 0.8785 and adj-R2 is 0.8772, F-stat is significant, all numeric vars sig 
# and many dummy ( categor) vars are significant

################ Multicollinearity check #######################
# let us now look if there exists multicollinearity among predictors
vif(reg.price) # variance inflation factors, if >10 then problem
sqrt(vif(reg.price)) # if this is >2, then problem.
# all values are less, so there is no collinearity 

############## model using only numeric variables ##############
reg.price1 <- lm (Price ~ Age+KM+Weight, data = corolla_new)
summary(reg.price1) 
sqrt(mean(reg.price1$residuals^2)) # RMSE is 1413.074

# R2 is 0.8481 (less than previous model), F-stat is significant, all coeff sig
# previous model is better

############## model using numeric variables Age,KM,Weight,Quarterly_Tax ##############
str(corolla)
reg.price2 <- lm (Price ~ Age+KM+Weight+Quarterly_Tax, data = corolla)
summary(reg.price2)
# R2 is 0.8484, F-stat signif, all vars signif except Quarterly_Tax, 
# so QTax as integer is not useful for analysis

############## model using normalized vars ################
str(corolla_norm)
reg.price3 <- lm(Price_std ~ ., data = corolla_norm)
summary(reg.price3)
# R2 is 0.8785, same result as non-normalized data, so normalization has no extra effect on the model

################### model using sqrt of Price #################
reg.price4 <- lm(sqrt(Price) ~ . , data = corolla_new)
summary(reg.price4)
# R2 is 0.8743, F-stat signif, all numeric vars signif, many factors signif
# but R2 is less than 0.8785 of reg.price1 model

################## model using sqr-roots of numeric predictors ######### 
sqrt_Age <- sqrt(corolla_new$Age)
sqrt_KM <- sqrt(corolla_new$KM)
sqrt_Weight <- sqrt(corolla_new$Weight)
corolla2 <- cbind(subset(corolla_new,select=-c(Age,KM,Weight)),sqrt_Age,sqrt_KM,sqrt_Weight)
str(corolla2)

reg.price5 <- lm(Price ~ . , data = corolla2)
summary(reg.price5)
# R2 is 0.8882 and adj-R2 is 0.887, F-stat signif, many variables are signif

# Doors is not significant, so dropping it and running the model
reg.price6 <- lm(Price ~ .-Doors, data=corolla2)
summary(reg.price6) # R2 is 0.8881, slightly less, so not dropping Doors

############## model using sqrt of price and sqrt of numeric predictors ##########
reg.price7 <- lm(sqrt(Price) ~ . , data = corolla2)
summary(reg.price7)
# R2 is 0.8712 less than previous model (0.8882)

######################### model using  35 vars (out of 38) in dataset

toyota_raw <- read.csv("file:///E:/EXCELR/Assignments/Multi linear regression_Assignment/ToyotaCorolla.csv")
names(toyota_raw)
toyota <- toyota_raw[,-c(1,5,6)] # removing id, duplicate cols of month and year which are reflected in Age
attach(toyota)
reg.toyota <- lm(Price ~ ., data = toyota)
summary(reg.toyota) 
# R2 is 0.96, very high, but many are not significant and interpretation of results in difficult
reg.toyota$coefficients

#########################
# R2 0.8882, for reg.price5 (using sqrt of numeric predictors) is highest,
# use this as final model, dataset is corolla2
str(corolla2)
pairs.panels(corolla2[,c(1,7,8,9)])
cor(corolla2[,c(1,7,8,9)])
cor(corolla_new[,c(1,2,3,9)])
# correlation has increased in corolla2 dataset


reg.price5 <- lm(Price ~ . , data = corolla2)
summary(reg.price5)
# R2 is 0.8882 and adj-R2 is 0.887, F-stat signif, many variables are signif

############## Influential Observations
# Deletion Diagnostics for identifying influential observations
# influential records will effect the analysis results. lets check them

library(car)
influence.measures(reg.price5)
influenceIndexPlot(reg.price5,id.n=3) # 222, 524
influencePlot(reg.price5,id.n=3) # 222, 961, 524

# added variable plots
avPlots(reg.price5)

corolla2[222,] # seeing values of 222nd index 
#     Price  HP   cc Doors Gears Quarterly_Tax sqrt_Age  sqrt_KM sqrt_Weight
# 222 12450 110 1600     5     5           185  6.63325 272.3454    40.18706

# let us remove these influential obsevn and see if we can build better model
reg.price8 <- lm(Price ~ ., data=corolla2[-c(222,524,961),])
summary(reg.price8)
# remove all 3, R2 is 0.8972
# -222, 0.8914
# -524, 0.891
# -961, 0.8898
# when remove all 3 influence obsvn, R2 is 0.8972, highest so far

############ outlier assessment ##########
outlierTest(reg.price5)
# the influence records also appear as outliers, so its good that we removed them
qqPlot(reg.price5) # qqplot of studentized residuals shows 222 and 524
leveragePlots(reg.price5) # leverage plots also indicate the same

######## final model ################
# let us use corolla2 dataset sans records with indices 222, 524 and 216
corolla3 <- corolla2[-c(222,524,961),]

# let us split the corolla3 dataset into 70:30 ratio
library(caTools)
set.seed(100)
sample <- sample.split(corolla3$Price,SplitRatio=0.70)
train <- subset(corolla3,sample==T)
test <- subset(corolla3, sample==FALSE)
411/1433

final_model <- lm(Price ~ . , data = train)
summary(final_model)
# R2 is 0.9054 and adj-R2 is 0.904, more than reg.price5(0.8882)

# predicting price for test data using final model
pred_interval <- predict(final_model,newdata = test,interval = 'predict') # prediction interval
pred_price <- predict(final_model,newdata = test)
pred_price <- data.frame(pred_price)

corolla3_pred <- cbind(test,pred_price)
str(corolla3_pred)

################# Residuals ###############
test_residuals <- corolla3_pred$Price - corolla3_pred$pred_price
sum(test_residuals) # -8989.348
mean(test_residuals) # -21.87189
sqrt(mean(test_residuals^2)) # 1234.902 RMSE

# for training data residuals
sum(final_model$residuals) # 0
mean(final_model$residuals) # 0
sqrt(mean(final_model$residuals^2)) # RMSE is 1141.127

# normality of residuals
skewness(test_residuals) # -1.048036
kurtosis(test_residuals) # 9.506295
qqnorm(test_residuals)
qqline(test_residuals)

qqPlot(final_model) # qqplot of studentized residual
library(MASS)
stud_resd <- studres(final_model)
hist(stud_resd,freq = FALSE)
xfit <- seq(min(stud_resd),max(stud_resd),length=40)
yfit <- dnorm(xfit)
lines(xfit,yfit)
# residuals are normally distributed 

# Homoscedasticity 
ncvTest(final_model) # non constant error variance test, null hyp is variance is constant
# pvalue = 0, reject null, residuals follow heteroscedasticity
spreadLevelPlot(final_model)

# diagnostic plots
plot(final_model)
####### Residual vs fitted values 
# donot show any pattern, # but red line is slightly tilted indicating
# there is room for improving the model
########### standardized residuals vs fitted (scale-location)
# residuals not showing any pattern
########## Residuals vs Leverage 
# helps to identify influential data points  1059,17,142

########### evaluate linearity of residuals
crPlots(final_model) # component + residual plots or called as partial residual plots
ceresPlots(final_model)

# independence of errors - test autocorrelation of errors
durbinWatsonTest(final_model) # test null hypothesis that autocorrelation is zero
# pvalue is 0.002, reject null, there is autocorrelation of errors, hence chance for improvement

# ASSESSMENT OF THE LINEAR MODEL ASSUMPTIONS using global test
library(gvlma)
gvmodel <- gvlma(final_model)
summary(gvmodel)  
# assumptions of skewness and link function are acceptable
# 
# (link function helps to form linear eqn even when Y and Xs are not linearly related)

# Find outliers
qqPlot(final_model,id.n=5) # QQ plots of studentized residuals, helps identify outliers
#  148 1059 108  752 

# removing only outliers
final_model1 <- lm(Price ~ ., data=train[-c(148, 1059, 108 , 752 ),])
summary(final_model1)
# R2 0.9079

# removing only influential observations
final_model3 <- lm(Price ~ ., data=train[-c(1059,17,142),])
summary(final_model3)
# R2 0.9055

# removing both outliers and influential observations
final_model2 <- lm(Price ~ ., data=train[-c(148, 1059, 108 , 752,1059,17,142),])
summary(final_model2)
# R2 0.9081

# slight increase in R2 as compared to final_model which was 
# built without removing these outliers and /or influential 
# observations. Hence will go with final_model itself.

corolla3_pred['test_residuals'] <- test_residuals
head(corolla3_pred)


# Conclusions

# As per the business problem we have predicted price of Toyota Corolla cars. we have 4 continuous and 5
# categorical variables. We have converted categorical input variable into dummy variables and used for 
# further analysis. There are no missing values. There were a few outliers and influential observations, 
# which were deleted to get a better model.
#
# We are using multiple linear regreession technique to predict Price. We have built different models. 
# The different models are compared based on R-sqr, adjR-sqr, significance of inputs, F-stat values.
# 
# Final model has the highest R-sqr and less RMSE. The residual analysis indicates that this is a good
# model. 

# There is limitation of time and also dataset is smaller (need more records).

