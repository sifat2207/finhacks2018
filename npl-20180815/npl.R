
#library-----
library(dplyr)
library(pROC)
library(caret)
library(xgboost)


# data------
data_train=read.csv("npl_train.csv")%>%
        select(-X)
data_test=read.csv("npl_test.csv")%>%
        select(-X)

#data preparation
#cek missing data
summary(data_train)
summary(data_test)
#tidak ada data missing


#features-------
vardep="flag_kredit_macet"
var_nominal=c("kode_cabang","skor_delikuensi")
data_train=data_train%>%
        mutate_at(vars(one_of(var_nominal)),funs(as.character))%>%
        mutate_at(vars(one_of(var_nominal)),funs(as.factor))
data_test=data_test%>%
        mutate_at(vars(one_of(var_nominal)),funs(as.character))%>%
        mutate_at(vars(one_of(var_nominal)),funs(as.factor))
levels(data_test$kode_cabang)=levels(data_train$kode_cabang)
levels(data_test$skor_delikuensi)=levels(data_train$skor_delikuensi)


formu=as.formula(paste(vardep,"~."))

#model xgboost-----
threshold <- 0.5

# convert categorical factor into one-hot encoding
data_numerik=data_train%>%
        select(-kode_cabang,-skor_delikuensi,-flag_kredit_macet)
region <- model.matrix(~kode_cabang-1,data_train)
delikuensi <- model.matrix(~skor_delikuensi-1,data_train)
data_numerik=cbind(data_numerik,delikuensi,region)
data_matrix <- data.matrix(data_numerik)

train_label=data.matrix(data_train%>%select_(.dots=c(vardep)))
dtrain <- xgb.DMatrix(data = data_matrix, label=train_label )
model_xgboost <- xgboost(data = dtrain, # the data   
                         nround = 100, # max number of boosting iterations
                         objective = "binary:logistic")  # the objective function

#validitas
data_train$pred <- predict(model_xgboost, dtrain)

# Calculate the area under the ROC curve
roc.curve <- roc(data_train$flag_kredit_macet, data_train$pred, ci=T)


# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con=confusionMatrix(factor(data_train$pred>threshold), factor(data_train$flag_kredit_macet==1), positive="TRUE")



#####################################################################
#Xgboost dengan SMOTE-------
source('smote function.R')

set.seed(1234)
data_smote=data_train %>% select(-pred)
data_smote$flag_kredit_macet=as.character(data_smote$flag_kredit_macet)
data_smote$flag_kredit_macet=as.factor(data_smote$flag_kredit_macet)
data_smote <- SMOTE(flag_kredit_macet ~ ., 
                    data=data_smote,
                    perc.over = 200, perc.under=300)
data_smote$flag_kredit_macet=as.character(data_smote$flag_kredit_macet)
data_smote$flag_kredit_macet=as.numeric(data_smote$flag_kredit_macet)

# convert categorical factor into one-hot encoding
region_smote <- model.matrix(~kode_cabang-1,data_smote)
delikuensi_smote <- model.matrix(~skor_delikuensi-1,data_smote)
data_numerik_smote=data_smote%>%select(-kode_cabang,-skor_delikuensi,-flag_kredit_macet)
data_numerik_smote=cbind(data_numerik_smote,delikuensi_smote,region_smote)

data_matrix_smote <- data.matrix(data_numerik_smote)
train_label_smote=data.matrix(data_smote%>%select_(.dots=c(vardep)))
dtrain_smote <- xgb.DMatrix(data = data_matrix_smote, label=train_label_smote )
model_xgboost_smote <- xgboost(data = dtrain_smote, # the data   
                         nround = 100, # max number of boosting iterations
                         objective = "binary:logistic")  # the objective function

#validitas
data_smote$pred <- predict(model_xgboost_smote, dtrain_smote)

# Calculate the area under the ROC curve
roc.curve_smote <- roc(data_smote$flag_kredit_macet, data_smote$pred, ci=T)


# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con_smote=confusionMatrix(factor(data_smote$pred>threshold), factor(data_smote$flag_kredit_macet==1), positive="TRUE")

#####
#
data_train$pred_smote <- predict(model_xgboost_smote, dtrain)
roc.curve_train_smote <- roc(data_train$flag_kredit_macet, data_train$pred_smote, ci=T)


# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con_train_smote=confusionMatrix(factor(data_train$pred_smote>threshold), factor(data_train$flag_kredit_macet==1), positive="TRUE")


#importance variable
all_=xgb.importance(colnames(dtrain),model=model_xgboost)
smote=xgb.importance(colnames(dtrain),model=model_xgboost_smote)


#prediksi data test------
# convert categorical factor into one-hot encoding
data_numerik_test=data_test%>%
        select(-kode_cabang,-skor_delikuensi)
region_test <- model.matrix(~kode_cabang-1,data_test)
delikuensi_test <- model.matrix(~skor_delikuensi-1,data_test)
data_numerik_test=cbind(data_numerik_test,delikuensi_test,region_test)
data_matrix_test <- data.matrix(data_numerik_test)

#train_label=data.matrix(data_train%>%select_(.dots=c(vardep)))
dtest <- xgb.DMatrix(data = data_matrix_test )
data_test$pred <- predict(model_xgboost, dtest)
data_test$pred_smote <- predict(model_xgboost_smote, dtest)

#stabilitas hasil prediksi-----
#all data training
batas_percentile=round(quantile(data_train$pred, probs = seq(0.05,0.95,0.05)),8)
data_train$klas=cut(data_train$pred,breaks = c(0,batas_percentile,1), labels = c(1:20))

#data_train%>%group_by(klas)%>% summarise(min=min(pred), max=max(pred),bad=sum(flag_kredit_macet),n= n())

#smote data training
batas_percentile_smote=round(quantile(data_train$pred_smote, probs = seq(0.05,0.95,0.05)),8)
data_train$klas_smote=cut(data_train$pred_smote,breaks = c(0,batas_percentile_smote,1), labels = c(1:20))

#PSI----
library(smbinning)
data_test$klas=cut(data_test$pred,breaks = c(0,batas_percentile,1), labels = c(1:20))
data_test$klas_smote=cut(data_test$pred_smote,breaks = c(0,batas_percentile_smote,1), labels = c(1:20))

data_psi=data_train%>%
        select(klas, klas_smote)%>%
        mutate(type="1")%>%
        rbind(data_test%>%
                      select(klas, klas_smote)%>%
                      mutate(type="2"))
smbinning.psi(data_psi, y="type",x="klas")        #PSI=0.07432464
smbinning.psi(data_psi, y="type",x="klas_smote")  #PSi=0.03056048
