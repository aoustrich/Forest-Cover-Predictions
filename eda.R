library(dplyr)
library(ggplot2)
library(vroom)

train <- vroom("data/train.csv")
test <- vroom("data/test.csv")
train$Cover_Type <- as.factor(train$Cover_Type)


# Balanced Training Data?
table(train$Cover_Type)

# Histogram of Aspect
hist(train$Aspect, 
     xlab = "Aspect", 
     ylab = "Frequency", 
     main = "Histogram of Aspect")

# Histogram of Slope
hist(train$Slope, 
     xlab = "Slope", 
     ylab = "Frequency", 
     main = "Histogram of Slope")

# Boxplot of Elevation by Cover Type
boxplot(Elevation ~ Cover_Type, 
        data = train,
        xlab = "Cover Type", 
        ylab = "Elevation",
        main = "Boxplot of Elevation by Cover Type")

# Boxplot of Horizontal Distance to Hydrology by Cover Type
boxplot(Horizontal_Distance_To_Hydrology ~ Cover_Type, 
        data = train,
        xlab = "Cover Type", 
        ylab = "Horizontal Distance to Hydrology",
        main = "Boxplot of Horizontal Distance to Hydrology by Cover Type")

# Boxplot of Vertical Distance to Hydrology by Cover Type
boxplot(Vertical_Distance_To_Hydrology ~ Cover_Type, 
        data = train,
        xlab = "Cover Type", 
        ylab = "Vertical Distance to Hydrology",
        main = "Boxplot of Vertical Distance to Hydrology by Cover Type")

# Boxplot of Horizontal Distance to Roadways by Cover Type
boxplot(Horizontal_Distance_To_Roadways ~ Cover_Type, 
        data = train,
        xlab = "Cover Type", 
        ylab = "Horizontal Distance to Roadways",
        main = "Boxplot of Horizontal Distance to Roadways by Cover Type")

# Boxplot of Horizontal Distance to Firepoints by Cover Type
boxplot(Horizontal_Distance_To_Fire_Points ~ Cover_Type, 
        data = train,
        xlab = "Cover Type", 
        ylab = "Horizontal Distance to Fire Points",
        main = "Boxplot of Horizontal Distance to Fire Points by Cover Type")

# Boxplot of Hillshade at 9am by Cover Type
boxplot(Hillshade_9am ~ Cover_Type, 
        data = train,
        xlab = "Cover Type", 
        ylab = "Hillshade at 9am",
        main = "Boxplot of Hillshade at 9am by Cover Type")

# Boxplot of Hillshade at Noon by Cover Type
boxplot(Hillshade_Noon ~ Cover_Type, 
        data = train,
        xlab = "Cover Type", 
        ylab = "Hillshade at Noon",
        main = "Boxplot of Hillshade at Noon by Cover Type")

# Boxplot of Hillshade at 3pm by Cover Type
boxplot(Hillshade_3pm ~ Cover_Type, 
        data = train,
        xlab = "Cover Type", 
        ylab = "Hillshade at 3pm",
        main = "Boxplot of Hillshade at 3pm by Cover Type")

