setwd('~/Desktop/andlab/code')
library('caret')

prosocial_preprocessing <- function(df)
{
  df <- df[complete.cases(df), ]      
  df<- df[(df$prosocial_parent  <= (3)) | (df$prosocial_parent  == 6), ]
  df<- df[(df$prosocial_child <= (3)) | (df$prosocial_child  == 6), ]
  df$prosocial_sum <- df$prosocial_parent+df$prosocial_child
  df <- df[df$prosocial_sum != 9, ]
  df <- df[df$prosocial_sum != 8, ]
  df <- df[df$prosocial_sum != 7, ]
  cut <- df[df$prosocial_sum == 6, ]
  cut <- cut[cut$prosocial_child != 3, ]
  df <- subset(df, !(subjectkey %in% cut$subjectkey))
  return(df)
}

df<-prosocial_preprocessing(read.csv('cleandata.csv'))
df <- subset(df, select=-c(X, aggressive_sumscore, prosocial_child, prosocial_parent, interview_date, interview_age, subjectkey))

reg <- lm(df$prosocial_sum ~ ., data=df)
#summary(reg)

newdf <- data.frame('y' = df$prosocial_sum)
newdf$fitted <- reg$fitted.values

new_reg <- lm(newdf$fitted ~ newdf$y, data=newdf)

pred_interval <- predict(new_reg, newdata=newdf, interval="confidence",
                         level = 0.95)

#col= ifelse(newdf$y <= -2.020, "chocolate2", 'darkblue'),
plot(newdf$y, newdf$fitted,  cex=0.6, pch =21, bg = ifelse(newdf$y <= -2.020, rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)) ,
     col= ifelse(newdf$y <= -2.020,rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)), xlab="Observed Prosocial Behaviors", ylab="Predicted Prosocial Behaviors", main="Regression")
abline(new_reg, col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.9))
lines(newdf$y, pred_interval[,2], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)
lines(newdf$y, pred_interval[,3], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)

#--------------------
#brain plot
df<-prosocial_preprocessing(read.csv('brain_cb.csv'))
df <- subset(df, select=-c(prosocial_child, prosocial_parent, subjectkey))
reg <- lm(df$prosocial_sum ~ ., data=df)
#summary(reg)

newdf <- data.frame('y' = df$prosocial_sum)
newdf$fitted <- reg$fitted.values

new_reg <- lm(newdf$fitted ~ newdf$y, data=newdf)

pred_interval <- predict(new_reg, newdata=newdf, interval="confidence",
                         level = 0.95)

#col= ifelse(newdf$y <= -2.020, "chocolate2", 'darkblue'),
plot(newdf$y, newdf$fitted,  cex=0.6, pch =21, bg = ifelse(newdf$y <= -2.020, rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)) ,
     col= ifelse(newdf$y <= -2.020,rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)), xlab="Observed Prosocial Behaviors", ylab="Predicted Prosocial Behaviors", main="Brain")
abline(new_reg, col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.9))
lines(newdf$y, pred_interval[,2], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)
lines(newdf$y, pred_interval[,3], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)



#--------------------
#brain + EPF plot
df <- merge(x = read.csv('cleandata.csv'), y = read.csv('brain_cb.csv'), by = "subjectkey", all.x = TRUE)
df <- df[complete.cases(df), ]        

df<- df[(df$prosocial_parent.x  <= (3)) | (df$prosocial_parent.x  == 6), ]
df<- df[(df$prosocial_child.x <= (3)) | (df$prosocial_child.x  == 6), ]
df$prosocial_sum <- df$prosocial_parent.x+df$prosocial_child.x
df <- df[df$prosocial_sum != 9, ]
df <- df[df$prosocial_sum != 8, ]
df <- df[df$prosocial_sum != 7, ]
cut <- df[df$prosocial_sum == 6, ]
cut <- cut[cut$prosocial_child.x != 3, ]
df <- subset(df, !(subjectkey %in% cut$subjectkey))

df <- subset(df, select=-c(X, prosocial_parent.x, prosocial_child.x, interview_date,subjectkey, aggressive_sumscore.x, interview_age, aggressive_sumscore.y, prosocial_child.y, prosocial_parent.y))
reg <- lm(df$prosocial_sum ~ ., data=df)
#summary(reg)

newdf <- data.frame('y' = df$prosocial_sum)
newdf$fitted <- reg$fitted.values

new_reg <- lm(newdf$fitted ~ newdf$y, data=newdf)

pred_interval <- predict(new_reg, newdata=newdf, interval="confidence",
                         level = 0.95)

#col= ifelse(newdf$y <= -2.020, "chocolate2", 'darkblue'),
plot(newdf$y, newdf$fitted,  cex=0.6, pch =21, bg = ifelse(newdf$y <= -2.020, rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)) ,
     col= ifelse(newdf$y <= -2.020,rgb(red = 0.6, green = 0.75, blue = 0.85, alpha = 0.05), rgb(red = 0.24, green = 0.35, blue = 0.50, alpha = 0.2)), xlab="Observed Prosocial Behaviors", ylab="Predicted Prosocial Behaviors", main="Brain + Family")
abline(new_reg, col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.9))
lines(newdf$y, pred_interval[,2], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)
lines(newdf$y, pred_interval[,3], col=rgb(red = 0.93, green = 0.42, blue = 0.30, alpha = 0.2), lty=1)




