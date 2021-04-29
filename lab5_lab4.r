rm(list=ls())
data(Titanic)
str(Titanic)
library(FNN)
x = runif(1000,0,pi)
e = rnorm(1000,0,0.1)
y = sin(x)+e

grid2=data.frame(x)
knn1 = knn.reg(train = x, test = grid2, y = y, k = 1)
knn10 = knn.reg(train = x, test = grid2, y = y, k = 10)
knn1000 = knn.reg(train = x, test = grid2, y = y, k = 1000)
plot(x,y,type = 'n')
points(x,y,pch=16,col='black')
I = order(grid2$x)
lines(grid2$x[I],knn1$pred[I],lwd=2,col=5)
lines(grid2$x[I],knn100$pred[I],lwd=2,col=6)
lines(grid2$x[I],knn10$pred[I],lwd=2,col=2)
coe=coef(lm(y~x+I(x^2)))
lines(x[ORD],coe[1]+coe[2]*x[ORD]+coe[3]*x[ORD]^2,
      lwd=2,col=4)

apply(Titanic,c(3),sum)
EPI_data <- read.csv("EPI_data.csv")

x=EPI_data$EPI
x[is.na(x)]=0
x
y=EPI_data$Population07
y[is.na(y)]=0
y
S=cbind(x,y)



out.dist=dist(S,method="euclidean") 
out.hclust=hclust(out.dist,method="complete")
plot(out.hclust) 


library(ISLR)
library(MASS)
library(boot)
set.seed(1)


#help("sample")
train = sample(392,196)

lm.fit <- lm(mpg~horsepower, data = Auto, subset = train)

attach(Auto)
mean((mpg-predict(lm.fit,Auto))[-train]^2)
lm.fit2 <- lm(mpg~poly(horsepower,2), data = Auto, subset = train) # Quadratic 
mean((mpg-predict(lm.fit2,Auto))[-train]^2) 

lm.fit3 <- lm(mpg~poly(horsepower,3), data = Auto, subset = train) # Cubic
mean((mpg-predict(lm.fit3,Auto))[-train]^2)

set.seed(2)
train = sample(392,196)
lm.fit <- lm(mpg~horsepower, data = Auto, subset = train)
mean((mpg-predict(lm.fit,Auto))[-train]^2)

lm.fit2 <- lm(mpg~poly(horsepower,2), data = Auto, subset = train) # Quadratic 
mean((mpg-predict(lm.fit2,Auto))[-train]^2) 

lm.fit3 <- lm(mpg~poly(horsepower,3), data = Auto, subset = train) # Cubic
mean((mpg-predict(lm.fit3,Auto))[-train]^2)

#install.packages('randomForest')
library(randomForest)
head(Titanic)
str(Titanic)
set.seed(100)
train<-sample(nrow(Titanic),0.7*nrow(Titanic),replace=FALSE)
train
model1<-randomForest(Survived ~ .,data=Titanic,importance=TRUE)
model1
model2<-randomForest(Survived ~ .,data=Titanic, ntree=500, mtry=6,importance=TRUE)
model2


#lab4 25th, March
wine_data <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", sep = ",")
# Header row is not available in the data, therefore, we need to add the variable names 
head(wine_data)


nrow(wine_data) #  178 rows

# Adding the variable names 
colnames(wine_data) <- c("Cvs", "Alcohol", 
                         "Malic_Acid", "Ash", "Alkalinity_of_Ash", 
                         "Magnesium", "Total_Phenols", "Flavanoids", "NonFlavanoid_Phenols",
                         "Proanthocyanins", "Color_Intensity", "Hue", "OD280/OD315_of_Diluted_Wine", 
                         "Proline")
head(wine_data) # Now you can see the header names.

# Using the Heatmap() function, we can check the correlations,
# In the heatmap(), the "Dark Colors" represent the "Correlated"
# In the heatmap(), the "Light Colors" represent the "Not Correlated"

#help("heatmap") 
# Now we will use the heatmap() function to show the correlation among variables.
heatmap(cor(wine_data),Rowv = NA, Colv = NA) 

# declaring the cultivar_classes using the factor() function each cultivar Cv1,Cv2 and Cv3.
cultivar_classes <- factor(wine_data$Cvs) 
cultivar_classes

# We will normalize the wine data to a common scale using scale() function so that the PCA process will not 
# overweight variables that happen to have the larger values.

#help(scale)
# We will not normalize the Cvs variable (first colume) so we exclude the Cvs column with with -1 
wine_data_PCA <- prcomp(scale(wine_data[,-1]))

# We can use the summary() function on wine_data_PCA to see the cumulative proportion that each 
# principal component (PC) contributes,
summary(wine_data_PCA)
