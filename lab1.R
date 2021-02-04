EPI_Data <- read.csv("/Users/zhangxingpu/Desktop/2010EPI_data.csv",header=TRUE, skip = 1)
View(EPI_Data)
plot(ecdf(EPI_Data), do.points=FALSE, verticals=TRUE)
attach(EPI_Data) 
fix(EPI_data) 
EPI
tf <- is.na(EPI) 
summary(EPI) 
ivenum(EPI,na.rm=TRUE)
jpeg(file="epi_stem.jpeg")
stem(EPI)
jpeg(file="epi_hist.jpeg")
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