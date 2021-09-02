
process <- function(df, reg, upr, plot=FALSE){
  
  newdf <- data.frame('y' = df$aggressive_sumscore)
  newdf$fitted <- reg$fitted.values
  
  newdf$class <- ifelse(newdf$y <= (-2.020650971), 0, ifelse(newdf$y>(upr),2,1))
  #change == -2.020650971 to change <= -2.020650971 to solve prediction for 0 is 0
  newdf$pred_class <-ifelse(newdf$fitted <= (-2.020650971), 0, ifelse(newdf$fitted>(upr),2,1))
  
  print(mean(newdf$pred_class == newdf$class))
  #cm <- confusionMatrix(factor(newdf$pred_class, levels = 0:2), factor(newdf$class, levels = 0:2) )
  cm <- confusionMatrix(table(newdf$class,newdf$pred_class))
  print(cm)
  if(plot)
  {
    par(mfrow = c(2, 2))
    plot(reg)
  }
}

df <- subset(read.csv('./cleandata.csv'), select=-c(X, prosocial_child, prosocial_parent, interview_date, interview_age, subjectkey))
df <- df[outliers(df$aggressive_sumscore), ]
reg <- lm(df$aggressive_sumscore ~ ., data=df)
summary(reg)
process(df, reg, -0.73254990)