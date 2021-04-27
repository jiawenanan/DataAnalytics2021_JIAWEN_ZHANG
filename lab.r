rm(list=ls())

install.packages("ISLR")

#http://aquarius.tw.rpi.edu/html/DA/

install.packages("MASS")
library (MASS)
attach(Boston)
#?Boston
#help("Boston")
head(Boston,7)
dim(Boston)
names(Boston)
str(Boston)
nrow(Boston)
summary(Boston$crim)


library(ISLR)
data(Auto)
head(Auto,9)
names(Auto)
summary(Auto$mpg)
boxplot(Auto$mpg)
mean(Auto$weight)
help("read.csv")

days<-c('Mon','Tue','Wed','Thur','Fri','Sat','Sun')
temp<-c(28,30.5,32,31.2,29.3,27.9,26.5)
snowed<-c('T','T','T','F','F','T','T')
help("data.frame")
RPI_weather_week<-data.frame(days,temp,snowed)
head(RPI_weather_week)
str(RPI_weather_week)
summary(RPI_weather_week)
RPI_weather_week[,1]
RPI_weather_week[2,]
RPI_weather_week[,'temp']
RPI_weather_week[1:4,c('days','snowed')]
RPI_weather_week$temp
subset(RPI_weather_week,subset=snowed=='T')
sorted.snowed<-order(RPI_weather_week['snowed'])
sorted.snowed
RPI_weather_week[sorted.snowed,]
dec_temp<-order(-RPI_weather_week['temp'])
RPI_weather_week[dec_temp,]
empty_df<-data.frame()

write.csv(RPI_weather_week,file='saved_df1.csv')





EPI_data <- read.csv("EPI_data.csv")
View(EPI_data)
attach(EPI_data)
fix(EPI_data)
EPI<-PopulationDensity

##############
EPI1<-WATSUP_pt
##############

tf<-is.na(EPI)
summary(EPI)
EPI<-EPI[!tf]
fivenum(EPI,na.rm=TRUE)
stem(EPI)
hist(EPI)
hist(EPI, seq(30., 95., 1.0), prob=TRUE)
lines(density(EPI,na.rm=TRUE,bw=1.))
rug(EPI)
plot(ecdf(EPI), do.points=FALSE, verticals=TRUE)
par(pty="s")
qqnorm(EPI); qqline(EPI)
x<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for t dsn")
qqline(x)

boxplot(EPI,EPI1)
??EPI
EPI_South_Asia <- EPI[!No_surface_water]
Esouth <- EPI_South_Asia[!is.na(EPI_South_Asia)]
hist(EPI_South_Asia)
hist(EPI_South_Asia, seq(30., 95., 1.0), prob=TRUE)


install.packages("ggplot2")
library(ggplot2)


plot(ecdf(EPI_data$code),do.points=FALSE,verticals = TRUE) 
plot(ecdf(EPI_data$code),do.points=TRUE,verticals = TRUE) # points are visible on the plot.
par(pty="s")
help("qqnorm") # read the RStudio documentation for qqnorm
help("qqplot") # read the RStudio documentation for qqplot
qqnorm(EPI_data$code)
qqline(EPI_data$code) # adding the line on the Q-Q plot

plot(ecdf(EPI_data$Population07),do.points=FALSE,verticals = TRUE) 
plot(ecdf(EPI_data$Population07),do.points=TRUE,verticals = TRUE) # points are visible on the plot.
par(pty="s")
help("qqnorm") # read the RStudio documentation for qqnorm
help("qqplot") # read the RStudio documentation for qqplot
qqnorm(EPI_data$Population07)
qqline(EPI_data$Population07) # adding the line on the Q-Q plot