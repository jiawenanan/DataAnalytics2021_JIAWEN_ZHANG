EPI_data <- read.csv("EPI_data.csv", header = TRUE)
plot(ecdf(EPI_data$EPI),do.points=FALSE,verticals = TRUE) 
plot(ecdf(EPI_data$EPI),do.points=TRUE,verticals = TRUE)
par(pty="s")
qqnorm(EPI_data$EPI)
qqline(EPI_data$EPI) 
#------------------------
x <- seq(30,95,1)
x2 <-seq(30,95,2)
x2 <-seq(30,96,2)
qqplot(qt(ppoints(250),df=5),x, xlab = "Q-Q plot")
qqline(x)
#------------------------
plot(ecdf(EPI_data$EPI),do.points=FALSE)
#------------------------
plot(ecdf(EPI_data$EPI),do.points=TRUE)
par(pty="s")
#------------------------
plot(ecdf(EPI_data$DALY),do.points=TRUE)
qqline(EPI_data$DALY)
#------------------------
plot(ecdf(EPI_data$WATER_H),do.points=TRUE)
qqline(EPI_data$WATER_H)
#------------------------
qqplot(EPI_data$EPI, EPI_data$DALY)
#------------------------
qqplot(EPI_data$EPI, EPI_data$WATER_H)
#------------------------
boxplot(EPI_data$EPI, EPI_data$ENVHEALTH, EPI_data$ECOSYSTEM,
        EPI_data$DALY, EPI_data$AIR_H, EPI_data$WATER_H,
        EPI_data$AIR_E,names = c("EPI","ENVHEALTH", "ECOSYS", "DALY", 
                  "AIR_H", "WATER_H", "AIR_E"),las=2,cex.axis=0.8)
#------------------------
multivariate <-read.csv("multivariate.csv")
head(multivariate)
mm <-lm(Homeowners ~ Immigrant)
summary(mm)
#------------------------
plot(Homeowners~Immigrant)
#------------------------
abline(mm, col=2, lwd=3)
#------------------------
abline(mm)
#------------------------
abline(mm,col=3,lwd=3) 
#------------------------
attributes(mm)
