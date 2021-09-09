setwd('~/Desktop/andlab/code')
library('caret')

df <- read.csv('cleandata.csv')
odf<- df[(df$aggressive_sumscore  <= (-2.020650971)) | (df$aggressive_sumscore  >= (0.2168869)), ]

df <- subset(odf, select=-c(X, prosocial_child, prosocial_parent, interview_date, interview_age, subjectkey))

reg <- lm(df$aggressive_sumscore ~ ., data=df)
#summary(reg)

newdf <- data.frame('y' = df$aggressive_sumscore)
newdf$fitted <- reg$fitted.values

new_reg <- lm(newdf$fitted ~ newdf$y, data=newdf)

pred_interval <- predict(new_reg, newdata=newdf, interval="confidence",
                         level = 0.95)

#col= ifelse(newdf$y <= -2.020, "chocolate2", 'darkblue'),
plot(newdf$y, newdf$fitted,  cex=0.6, pch =21, bg = ifelse(newdf$y <= -2.020, rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)) ,
     col= ifelse(newdf$y <= -2.020,rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)), xlab="Observed Aggression Behaviors", ylab="Predicted Aggression Behaviors", main="Regression")
abline(new_reg, col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.9))
lines(newdf$y, pred_interval[,2], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)
lines(newdf$y, pred_interval[,3], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)

#--------------------
#brain plot
odf <- read.csv('./brain_cb.csv')
odf <- odf[complete.cases(odf), ]        
odf<- odf[(odf$aggressive_sumscore  <= (-2.02065)) | (odf$aggressive_sumscore  >= (0.2168869)), ]

df <- subset(odf, select=-c(subjectkey, prosocial_child, prosocial_parent))

reg <- lm(df$aggressive_sumscore ~ ., data=df)
#summary(reg)

newdf <- data.frame('y' = df$aggressive_sumscore)
newdf$fitted <- reg$fitted.values

new_reg <- lm(newdf$fitted ~ newdf$y, data=newdf)

pred_interval <- predict(new_reg, newdata=newdf, interval="confidence",
                         level = 0.95)

#col= ifelse(newdf$y <= -2.020, "chocolate2", 'darkblue'),
plot(newdf$y, newdf$fitted,  cex=0.6, pch =21, bg = ifelse(newdf$y <= -2.020, rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)) ,
     col= ifelse(newdf$y <= -2.020,rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)), xlab="Observed Aggression Behaviors", ylab="Predicted Aggression Behaviors", main="Regression")
abline(new_reg, col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.9))
lines(newdf$y, pred_interval[,2], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)
lines(newdf$y, pred_interval[,3], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)



#--------------------
#brain + EPF plot
cdf <- read.csv('cleandata.csv')
bdf <- read.csv('brain_cb.csv')
df <- merge(x = cdf, y = bdf, by = "subjectkey", all.x = TRUE)

df <- df[complete.cases(df), ]        
df<- df[(df$aggressive_sumscore.x  <= (-2.020650971)) | (df$aggressive_sumscore.x  >= (0.2168869)), ]

df <- subset(df, select=-c(X, prosocial_parent.x, prosocial_child.x,interview_date,  subjectkey, interview_age, aggressive_sumscore.y, prosocial_child.y, prosocial_parent.y))

reg <- lm(df$aggressive_sumscore.x ~ ., data=df)
#summary(reg)

newdf <- data.frame('y' = df$aggressive_sumscore.x)
newdf$fitted <- reg$fitted.values

new_reg <- lm(newdf$fitted ~ newdf$y, data=newdf)

pred_interval <- predict(new_reg, newdata=newdf, interval="confidence",
                         level = 0.95)

#col= ifelse(newdf$y <= -2.020, "chocolate2", 'darkblue'),
plot(newdf$y, newdf$fitted,  cex=0.6, pch =21, bg = ifelse(newdf$y <= -2.020, rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)) ,
     col= ifelse(newdf$y <= -2.020,rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)), xlab="Observed Aggression Behaviors", ylab="Predicted Aggression Behaviors", main="Regression")
abline(new_reg, col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.9))
lines(newdf$y, pred_interval[,2], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)
lines(newdf$y, pred_interval[,3], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)




