install.packages("MASS")
install.packages("ggplot2")
library(ggplot2)
library (MASS)

#lab0
days<-c('Mon','Tue','Wed','Thur','Fri','Sat','Sun')
temp<-c(28,30.5,32,31.2,29.3,27.9,26.5)
snowed<-c('T','T','T','F','F','T','T')
help("data.frame")
RPI_weather_week<-data.frame(days,temp,snowed)
head(RPI_weather_week)
str(RPI_weather_week)
summary(RPI_weather_week)
RPI_weather_week[,2]
RPI_weather_week[3,]
RPI_weather_week[,'temp']
RPI_weather_week[1:2,c('days','snowed')]
RPI_weather_week$temp
subset(RPI_weather_week,subset=snowed=='T')
sorted.snowed<-order(RPI_weather_week['snowed'])
sorted.snowed
RPI_weather_week[sorted.snowed,]
dec_temp<-order(-RPI_weather_week['temp'])
RPI_weather_week[dec_temp,]

write.csv(RPI_weather_week,file='lab0.csv')

#lab1 - EPI part
EPI_Data <- read.csv("/Users/zhangxingpu/Desktop/2010EPI_data.csv",header=TRUE, skip = 1)
View(EPI_Data)
plot(ecdf(EPI_Data), do.points=FALSE, verticals=TRUE)
attach(EPI_Data) 
fix(EPI_data) 
EPI
tf <- is.na(EPI) 
summary(EPI) 
EPI<-EPI[!tf]
fivenum(EPI,na.rm=TRUE)
jpeg(file="epi_stem.jpeg")
stem(EPI)
jpeg(file="epi_hist.jpeg")
hist(EPI)
hist(EPI, seq(30., 95., 1.0), prob=TRUE)
lines(density(EPI,na.rm=TRUE,bw=1.))
rug(EPI)
plot(ecdf(EPI), do.points=FALSE, verticals=TRUE)
par(pty="s")
qqnorm(EPI); qqline(EPI)
ivenum(EPI,na.rm=TRUE)
hist(EPI, seq(30., 95., 1.0), prob=TRUE)
lines(density(EPI,na.rm=TRUE,bw=1.))
plot(ecdf(EPI), do.points=FALSE, verticals=TRUE)

x<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for t dsn")
qqline(x)

Landlock
hist(Landlock)

ENVHEALTH
hist(ENVHEALTH)
y<-seq(0,100,2)
qqplot(qt(ppoints(250), df = 5), y, xlab = "Q-Q plot for t dsn")
qqline(y)

boxplot(EPI,DALY) 

summary(No_surface_water) 
summary(Desert) 
summary(High_Population_Density) 
hist(No_surface_water)
hist(Desert)
hist(High_Population_Density)

filtered_regions <- EPI[EPI_regions == "Europe"]
filtered_subregions <- EPI[GEO_subregion == "Meso America"]

#lab1 gpw3 part




GPW3_Data <- read.csv("/Users/zhangxingpu/Desktop/GPW3_GRUMP_SummaryInformation_2010.csv")
View(GPW3_Data)
plot(ecdf(GPW3_Data), do.points=FALSE, verticals=TRUE)
fix(GPW3_Data) 
Continent
tf <- is.na(Continent) 
summary(Continent) 
stem(Continent)
hist(Continent)
x<-seq(1,6,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for t dsn")
qqline(x)
filtered_UNRegion <- Continent[UNRegion == "Middle Africa"]
filtered_subregions <- Continent[LevelUsed >= 2]


EPI



