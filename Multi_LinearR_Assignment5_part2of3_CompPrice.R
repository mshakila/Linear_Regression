# # ASSIGNMENT MULTI LINEAR REGRESSION  - computer dataset

###### 1. Business problem: To predict the price of computer

###### 2. Dataset acquisition
computer <- read.csv("file:///E:/EXCELR/Assignments/Multi linear regression_Assignment/Computer_Data.csv")
names(computer) # X is first column shows indices, so remove it
computer <- computer[,-1]
# "price"   "speed"   "hd"      "ram"     "screen"  "cd"      "multi"  
# "premium" "ads"     "trend"
dim(computer)
# 6259 records   10 variables
head(computer)
str(computer)
# cd, multi and premium are factors; others integers
summary(computer)
attach(computer)
levels(cd)
table(cd) # no 3351   yes 2908
table(multi) # no 5386    yes 873
table(premium) # no 612   yes 5647
table(speed) # has 5 levels, but int so convert to factor
#   25   33   50   66   75  100 
#  566 2033  994 2028  122  516
table(ram) # has 6 levels, but int so convert to factor
#    2    4    8   16   24   32 
#  394 2236 2320  996  297   16 
table(screen) # # has 3 levels, but int so convert to factor
#   14   15   17 
#  3661 1992  606
summary(computer)

# converting speed, ram and screen to factor
cols <- c("speed","ram","screen")
computer1 <- computer
computer1[,cols] <- data.frame(apply(computer1[cols], 2, as.factor))
str(computer1)
summary(computer1)

# large dataset or not
library(moments)
10 * kurtosis(computer$price) # 38 records needed, but have 6000+, hence large dataset

######## 3. Exploratory Data Analysis
# Business moments decisions
str(computer1)

# price variable - target variable
mean(price) # 2219.577
median(price) # 2144
library(modeest)
mlv(price,method='mfv') # 1999
var(price) # 337333
sd(price) # 580
range(price) # 949 5399 
skewness(price) # 0.7115542 # skewness of 0 to +_0.5 is normal
kurtosis(price) # 3.728875 # kurtosis of 3 is normal

# hd variable
mean(hd) # 416.6017
median(hd) # 340
mlv(hd,method='mfv') # 340
var(hd) # 66847
sd(hd) # 258
range(hd) # 80 2100
skewness(hd) # 1.377689
kurtosis(hd) # 5.449539

# ads variable
mean(ads) # 221.301
median(ads) # 246
mlv(ads,method='mfv') # 339
var(ads) # 5600.32
sd(ads) # 74.83528
range(ads) #   39 339
339-39 # 300
skewness(ads) # -0.5531955
kurtosis(ads) # 2.459629

# trend variable
mean(trend) # 15.92699
median(trend) # 16
mlv(trend,method='mfv') # 17
var(trend) # 61.99962
sd(trend) # 7.873984
range(trend) # 1 35
35-1 # 34
skewness(trend) # 0.2366127
kurtosis(trend) # 2.325446

########## visualizations #############
# price, hd, ads, int

############# price variable
hist(price,probability = T)
lines(density(price))
# outlier detection
boxplot(price,horizontal = TRUE) # shows many outliers
box_price <- boxplot(price)
box_price$out # there are 75 records having extreme values, cannot delete all
min(box_price$out) # 3799
max(box_price$out) # 5399
# normality
qqnorm(price)
qqline(price) # data does not look normal
qqnorm(sqrt(price))
qqline(sqrt(price))
shapiro.test(price) # sample size must be between 3 and 5000
library(nortest)
ad.test(price) # p-value < 2.2e-16
ad.test(sqrt(price))
library(moments)
skewness(price) # 0.7115542
kurtosis(price) # 3.728875
skewness(sqrt(price)) # 0.3547703
kurtosis(sqrt(price)) # 2.927888
# by looking at skewness, kurtosis, qqnorm and qqline, data looks normal
# also when take squareroot of price, more normal

# Normality tests (shapiro, anderson-darling,etc) will always reveal 
# non-normality as the sample size grows. we have seen shapiro test 
# can be used for only 3 to 5000 observations.
# Hence visualizations like qqplot are better suited to check for normality
# Here in qqplot, small deviations will have smaller impact on results

################ hd variable
hist(hd,probability = T)
lines(density(price))
boxplot(hd,horizontal = T)
# outlier detection
box_hd <- boxplot(hd)
box_hd$out # all records (489 records) having 1000gb hd capacity are coming under outliers
min(box_hd$out) # 1000
max(box_hd$out) # 2100
range(hd) # 80 2100
# normality
qqnorm(hd)
qqline(hd) # data does not look normal
ad.test(hd) # p-value < 2.2e-16
skewness(hd) # 1.377689
kurtosis(hd) # 5.449539
# data is right skewed and not normal

############# ads variable
hist(ads,probability = T)
lines(density(ads))
boxplot(ads,horizontal = T)
# outlier detection
box_ads <- boxplot(ads)
box_ads$out # no outliers
# normality
qqnorm(ads)
qqline(ads)  # data looks normal
ad.test(ads) # p-value < 2.2e-16
skewness(ads) # -0.5531955
kurtosis(ads) # 2.459629 
# data is slightly left skewed but overall can consider as normal data

########### trend variable
hist(trend,probability = T)
lines(density(trend))
boxplot(trend,horizontal = T)
# outlier detection
box_trend <- boxplot(trend)
box_trend$out # no outliers
# normality
qqnorm(trend)
qqline(trend) # data looks normal
ad.test(trend) # p-value < 2.2e-16
skewness(trend) # 0.2366127
kurtosis(trend) # 2.325446
# data is normally distributed

# categorical variables visualization
str(computer1)
barplot(table(computer1$speed)) # more number of computers in 33 and 66
barplot(table(computer1$ram)) # more in 4 and 9
barplot(table(computer1$screen)) # more in 14" screen
barplot(table(computer1$cd)) # almost same number with and without cd
barplot(table(computer1$multi)) # more than 5000 donot have multi
barplot(table(computer1$premium)) # more than 5000 have premium

# stacked barplot
barplot(table(computer1$premium,computer1$screen),xlab='screen',
        col=c('red','green'),beside = TRUE)
legend('topright',c('no_premium','yes_premium'),fill=c('red','green'))
# in computers with 14" screen, about 500 do not have premium, more than 3000 have premium

# finding missing values
computer1[!complete.cases(computer1),]
# no missing values

############## correlation ########################
# finding correlation which is an important prerequisite for regression
# corr btw output (profit) and predictors - scatter plot
# pairs(computer1[,c(1,3,9,10)])
str(computer1)
library(psych)
pairs.panels(computer1[,c(1,3,9,10)])
# correlation coeff matrix
cor(computer1[,c(1,3,9,10)])
# the correlation is very less

######## 4.& 5. Model building and Evaluation################
# we have to predict price which is continuous variable. 
# we will use multi linear regression tehcnique using ordinary least
# squares method which finds the best fit line having least errors

# let us build multi linear regression model
# using all variables as is for analysis (4 cols not converted to factors)
reg.price <- lm(price ~ ., data=computer)
summary(reg.price)
# R2 0.7756 , adj R2 0.7752, Fstat sig, all coeff highly sig
sqrt(mean(reg.price$residuals^2)) # RMSE is 275.1298

############### Multicollinearity check ###########
vif(reg.price)
sqrt(vif(reg.price))
# no collinearity present

############ model using relevant factor variables
# using price,hd,ads,trend as numeric and remaining as factors
reg.price1 <- lm(price ~ ., data=computer1)
summary(reg.price1)
# R2 0.8029 , adj R2 0.8024, Fstat sig, all coeff highly sig
sqrt(mean(reg.price1$residuals^2)) # RMSE is 257.8173
# R2 higher and RMSE lesser, better than previous model

########## model: removing outliers in price ###############
names(computer1)
box_price <- boxplot(computer1$price,horizontal = T)
box_price$out
min(box_price$out) # 3799
max(box_price$out) # 5399
range(price)
# removing 75 outliers of price
computer3 <- computer1[!rowSums(computer1[1] >= 3799),]
dim(computer3) # 6184 records (removed 75 records)
6184+75
qqnorm(computer3$price)
qqline(computer3$price)
skewness(computer3$price) # 0.413598 
# with outliers it was 0.7115542 (slightly skewed). Now data is normal
kurtosis(computer3$price) # 2.59441
# with outliers it was 3.728875 (slight peak), now almost normal

# model using computer3 dataset, (removing outliers)
reg.price2 <- lm(price ~ ., data = computer3)
summary(reg.price2)
sqrt(mean(reg.price2$residuals^2)) # RMSE is 238

# R2 0.807 , adj R2 0.8065, Fstat sig, all coeff highly sig
# previously R2 was 0.8029 without removing outliers, now improved, RMSE is also less
# this is a better model than previous ones

##################### model: removing outliers in hd ###############
# as per boxplot of hd (box_hd), there are 489 outliers
# removing 489 outliers of hd
computer4 <- computer1[!rowSums(computer1[3]>=1000),]
5770 - 6259
qqnorm(computer4$hd)
qqline(computer4$hd) # does not look normal

# building model using computer4 data
reg.price3 <- lm(price ~ ., data=computer4)
summary(reg.price3)
# R2 is 0.7844, it has reduced from previous model, so cannot remove these 489 records

############## model using squareroot of price #############
reg.price4 <- lm(sqrt(price) ~ ., data=computer3)
summary(reg.price4)
# R2 0.8126 , adj R2 0.812, Fstat sig, all coeff highly sig
qqnorm(sqrt(computer3$price))
qqline(sqrt(computer3$price))
skewness((sqrt(computer3$price))) # 0.159687
kurtosis((sqrt(computer3$price))) # 2.401478
skewness(((computer3$price))) # 0.413598
kurtosis(((computer3$price))) # 2.59441
# sqroot of price is better than only price


############### model using sqrt(price) and sqrt(hd)
sqrt_hd <- sqrt(computer3$hd) 
computer5 <- cbind(subset(computer3,select = -hd),sqrt_hd)
names(computer5)
dim(computer5)
skewness(computer5$sqrt_hd) # 0.5862295, without sqrt it was 1.377689
kurtosis(computer5$sqrt_hd) # 3.138409, without sqrt it was 5.449539
skewness(computer$hd)
kurtosis(computer$hd)
# now with sqrt,hd is normal

# using sqrt of price and sqrt of hd in model
reg.price5 <- lm(sqrt(price) ~ ., data=computer5)
summary(reg.price5)
sqrt(mean(reg.price5$residuals^2)) # RMSE is 2.448346
# R2 0.8197 , adj R2 0.8192, Fstat sig, all coeff highly sig
# this has the highest R2,and least RMSE, we can use this for final model

#### final model #####
# first let us split the data into 70:30 ratio using computer5 dataset
library(caTools)
set.seed(123)
sample = sample.split(computer5$price,SplitRatio = 0.70)
train <- subset(computer5,sample==TRUE)
test <- subset(computer5,sample==FALSE)
1829/6184
4355/6184
final_model <- lm(sqrt(price) ~ ., data=train)
summary(final_model)
# same reg.price5, R2 0.8211 , adj R2 0.8204, Fstat sig, all coeff highly sig

# predicting price for test data using final model
pred_interval <- predict(final_model,newdata = test,interval='predict')
pred <- predict(final_model,newdata = test)
pred <- data.frame(pred)

computer5_pred <- cbind(test,pred)
computer5_pred['sqrt_price'] <- sqrt(computer5_pred$price)

# Residuals
test_residuals <- (computer5_pred$sqrt_price - computer5_pred$pred)
sum(test_residuals) # -96
mean(test_residuals) # -0.052
sqrt(mean(test_residuals^2)) # RMSE is 2.462

# for training data residuals
sum(final_model$residuals) # 0
mean(final_model$residuals) # 0
sqrt(mean(final_model$residuals^2)) # RMSE is 2.444


# normality of residuals
skewness(test_residuals) # 0.3874594
kurtosis(test_residuals) # 3.936103
qqnorm(test_residuals)
qqline(test_residuals)
# residuals are normally distributed 

# diagnostic plots
library(car)
plot(final_model)
####### Residual vs fitted values 
# donot show any pattern, # but red line is slightly tilted indicating
# there is room for improving the model
########### standardized residuals vs fitted (scale-location)
# showing 3 outliers 310,208,1806
########## Residuals vs Leverage 
# helps to identify influential data points  4847,80,86

qqPlot(final_model, id.n=5) # QQ plots of studentized residuals, helps identify outliers
# 208 310 140 214

# removing only outliers
final_model1 <- lm(sqrt(price) ~ ., data=train[-c(310,208,140,214),])
summary(final_model1)
# R2 0.8223

# removing only influential observations
final_model3 <- lm(sqrt(price) ~ ., data=train[-c(4847,80,86),])
summary(final_model3)
# R2 0.8212

# removing both outliers and influential observations
final_model2 <- lm(sqrt(price) ~ ., data=train[-c(310,208,140,214,4847,80,86),])
summary(final_model2)
sqrt(mean(final_model2$residuals^2)) # 
# R2 0.8224 and RMSE is 2.433

# very slight increase in R2 as compared to final_model which was 
# built without removing these outliers and /or influential 
# observations. Hence will go with final_model

computer5_pred['test_residuals'] <- test_residuals
head(computer5_pred)
dim(computer5_pred)

# Conclusions
# 
# As per business problem we have predicted Price of computer using continuous 
# and categorical variables. All continuous variables were normally distributed. There were no missing values.
# No collinearity present among predictors.
#
# We are using multiple linear regreession technique to predict Price. We have built different models. 
# The different models are compared based on R-sqr, adjR-sqr, significance of inputs, F-stat values.
# 
# Final model has the highest R-sqr and less RMSE. The residual analysis indicates that this is a good
# model. 

# There is limitation of time and also need more variables to better the prediction.
# 

