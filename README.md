# Data
##
alldata.csv - handselected raw data

clean entries with 777 and 999

1. EPF ~ Aggression
2. Brain ~ Aggression
3. EPF + Brain ~ Aggression

1. EPF ~ Prosocial
2. Brain ~ Prosocial
3. EPF + Brain ~ Prosocial

Y(label) = (:,2)aggression; (:,3) prosocail_child; (:,4) prosocial_parent. 
X = (:,[5:31]) factor_Parent; (:,[32:47]) factor_Family; (:,[48:52]) factor_Neighbor

Test
all columns as predictor, rows with 777 & 999 removed R & R-adjusted
only columns with high correlation, rows with 777 & 999 removed R & R-adjusted

grouped columns to calculate predicted value and compare for accuracy