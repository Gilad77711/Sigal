
########### Training data (rankings only, no dates):
con = url("http://www.tau.ac.il/~saharon/StatsLearn2018/train_ratings_all.dat")
X.tr = read.table (con)
con = url("http://www.tau.ac.il/~saharon/StatsLearn2018/test_ratings_all.dat")
X.test = read.table (con)
con = url("http://www.tau.ac.il/~saharon/StatsLearn2018/train_y_rating.dat")
y.tr = read.table (con)

con = url("http://www.tau.ac.il/~saharon/StatsLearn2018/train_y_date.dat")
y.da.tr = read.table (con)


con = url("http://www.tau.ac.il/~saharon/StatsLearn2018/movie_titles.txt")
titles = read.table(con,sep=",")
names (X.tr) = substr(as.character(titles[,2]),1,10)
names(X.tr)[17]<-"The Bourne1"
names(X.tr)[32]<-"Lord of th1"
names(X.tr)[62]<-"Lord of th2"
names(X.tr)[57]<-"Spider-Man1"
names(X.tr)[59]<-"Men in bla1"
names(X.tr)[94]<-"Kill Bill1"
names (X.test) = substr(as.character(titles[,2]),1,10)
names(X.test)[17]<-"The Bourne1"
names(X.test)[32]<-"Lord of th1"
names(X.test)[62]<-"Lord of th2"
names(X.test)[57]<-"Spider-Man1"
names(X.test)[59]<-"Men in bla1"
names(X.test)[94]<-"Kill Bill1"

########### Get to know our data a little:
table (y.tr) # What rankings does our target get?
apply(data.frame(X.tr[,1:14],y.tr),2,mean) # Which movies are liked?
cor(y.tr,X.tr[,1:14]) # which movies correlated with Miss Congeniality?
apply (X.tr==0, 2, sum) # how many missing?
cor (y.tr, y.da.tr) # changes with time?



########### Divide training data into training and validation
n = dim(X.tr)[1]
nva = 2000
ntr = n-nva
va.id = sample (n,nva) # choose 2000 points for validation
#trtr = data.frame (X = X.tr[-va.id,], yda=y.da.tr[-va.id,], y=y.tr[-va.id,]) # include dates
trtr = data.frame (X = X.tr[-va.id,],y=y.tr[-va.id,])
#va = data.frame (X = X.tr[va.id,], yda=y.da.tr[va.id,], y=y.tr[va.id,]) #include dates
va = data.frame (X = X.tr[va.id,],y=y.tr[va.id,])


########### baseline RMSE
sqrt(mean((va$y-mean(trtr$y))^2))

########### regression on all rankings (with missing = 0!!!!)
lin.mod = lm (y~.,data=trtr)

########### in-sample RMSE
lin.insamp = predict (lin.mod)
sqrt(mean((trtr$y-lin.insamp)^2))

########### RMSE on validation data
lin.pred = predict (lin.mod, newdata=va)
sqrt(mean((va$y-lin.pred)^2))

########### rankings can't be higher than 5!
lin.pred.cap = pmin (lin.pred,5)
sqrt(mean((va$y-lin.pred.cap)^2))



########### Output of linear models:
summary(lin.mod)
summary(lin.mod1)



#####################################Our code:

########### Multinomial Regression:
library(VGAM)

multinomial_model<-vglm(formula = y ~., data =trtr,family=cumulative(parallel=TRUE))
#Multinomial ordinal regression RMSE on validation
predva_mult<-predict(multinomial_model, va)
predva_mult_probs<-exp(predva_mult)/(1+exp(predva_mult))
P[,1]<-predva_mult_probs[,1]
P[,2]<-predva_mult_probs[,2]-predva_mult_probs[,1]
P[,3]<-predva_mult_probs[,3]-predva_mult_probs[,2]
P[,4]<-predva_mult_probs[,4]-predva_mult_probs[,3]
P[,5]<-1-predva_mult_probs[,4]
y_hat_mult<-vector()
for (i in 1:2000){
  y_hat_mult[i]<-which(P[i,]==max(P[i,]))
}
table(y_hat_mult,va$y)
sqrt(mean((va$y-y_hat_mult)^2))

##Imputation
X.tr[X.tr==0]<-NA
library("DMwR")
Full_train_data = data.frame (X = X.tr[,],y=y.tr[,])
Full_train_data_imp_knn<-knnImputation(Full_train_data[, !names(Full_train_data) %in% "y"])
anyNa(Full_train_data_imp_knn)

########### Random Forest Models- Regression and Classification
library(randomForest)
rf.mod_reg<-randomForest(formula = y ~., data =trtr , ntree = 500,      mtry = 10, importance = TRUE)
rf.mod_class<-randomForest(formula = as.factor(y) ~., data =trtr , ntree = 500,      mtry = 10, importance = TRUE)

#RF_regression RMSE on validation
predva_reg<-predict(rf.mod_reg, va)
sqrt(mean((va$y-predva_reg)^2))

#classification RMSE on validation
predva_class <- predict(rf.mod_class, va)
table(predva_class,va$y)
sqrt(mean((va$y-as.numeric(predva_class))^2))


##Regression RF is the best!!

#Feature Imporance (just to see it's matching what we saw in class)
importance(rf.mod_reg) 
varImpPlot(rf.mod_reg)

##Choosing the best number of features
error1=c()
i=5
for (i in 5:20) {
  model3 <- randomForest(y ~ ., data = trtr, ntree = 500, mtry = i, importance = TRUE)
  predValid <- predict(model3, va)
  error1[i-4] = sqrt(mean((predValid - va$y)^2))
}
error1

plot(5:20,error1)

#min at mtry=13


##Choose best number of trees
error2=c()
i=100
for (i in c(100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500)) {
  model3 <- randomForest(y ~ ., data = trtr, ntree = i, mtry = 13 , importance = TRUE)
  predValid <- predict(model3, va)
  error2[i/100] = sqrt(mean((predValid - va$y)^2))
}
plot(error2)

#RF final model:
rf.mod_fin<-randomForest(formula = y ~., data =trtr , ntree = 1100,      mtry = 13, importance = TRUE)
#RF_final RMSE
predva_fin<-predict(rf.mod_fin, va)
sqrt(mean((va$y-predva_fin)^2))

#First guess

bla<-randomForest(formula = y ~., data =Full_train_data , ntree = 1100,      mtry = 13, importance = TRUE)
y_hat_train<-predict(bla, Full_train_data[,1:99])
sqrt(mean((Full_train_data[,100]-y_hat_train)^2))
Full_test_data<-data.frame(X=X.test[,])
y_hat_test<-predict(bla,Full_test_data)
write.table(y_hat_test, "C:/Users/Gilad/OneDrive/Desktop/Data for Projects/Statistical Learning Saharon/Competition/guess1.csv", sep=",")


