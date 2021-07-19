library (vegan)
library(readxl)
library(tidyverse)
library(ggplot2)
library(xlsx)

#getwd()
#setwd()


##cargo los dataframes necesarios 
#attach(tax6_sinBU)
#attach(Metadatos)
tax6 <- read_excel('tax6.xlsx')
Metadatos <- read_excel('Metadatos.xlsx')
#tax6_sinBU_transpose <- as.data.frame(t(as.matrix(tax6_sinBU)))
tax6_transpose <-as.data.frame(t(as.matrix(tax6[-1])))

colnames(tax6_transpose) <- tax6[, 1]
tax6_transpose

#rownames(tax6_sinBU_transpose) <- tax6_sinBU_transpose$X
#Metadatos
#rownames(Metadatos) <- Metadatos$'Name'
#Metadatos[,-1]
#tax6_sinBU_transpose.head(10)
#colnames(tax6_sinBU_transpose)

# transformo variables ambientales

cols <- c(1, 7:9, 14:17)
A= Metadatos[,cols]
rownames(A) <- A$'Name'
A=A[,2:8]
Anorm<-log(A+1)
#attach(Metadatos)
#edit(metadatos)

dca=decorana(tax6_transpose)
dca
(plot(dca))
rda=rda(tax6_transpose~., A)
#para testar el rda
anova(rda)
RsquareAdj(rda)
vif.cca(rda)
#que variables elijo
step.forward<-ordistep(rda,pstep=10000)

#RDA solo con las variables significativas
rda_ordistep<-rda(tax6_transpose~`Simpson Index` +`Number of OTUs` ,data=Anorm)

summary(rda_ordistep)
summary(rda)

#plot 
plot(rda_ordistep,scaling=2)
plot(rda,scaling = "symmetric")
