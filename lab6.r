data("iris")
head(iris) # inspecting the first six rows of the dataset
str(iris) # structure of the dataset
library(ggplot2)
library(e1071)

qplot(Petal.Length, Petal,Width, data=iris, color = Species)

# read the svm() documentation on RStudio by using
#the help(svm) function
help("svm")
svm_model1 <- svm(Species~., data = iris)


# summary() function
summary(svm_model1)

pred1 <- predict(svm_model1, iris)
# creating a table using the predicted one and the actual iris dataset
table1 <- table(Predicted = pred1, Actual = iris$Species)
table1
Model1_accuracyRate = sum(diag(table1))/sum(table1)
Model1_accuracyRate
Model1_MissClassificationRate = 1 - Model1_accuracyRate
Model1_MissClassificationRate