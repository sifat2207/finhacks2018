#library-----
library(dplyr)
library(pROC)
library(caret)
library(xgboost)
library(smbinning)


# data------
data_train=read.csv("fraud_train.csv")%>%
  select(-X)
data_test=read.csv("fraud_test.csv")%>%
  select(-X)

#data preparation
#cek missing data
summary(data_train)
#isinya sama : flag_transaksi_finansial, status_transaksi, bank_pemilik_kartu
# terdapat 21 NA : rata_rata_nilai_transaksi, maksimum_nilai_transaksi, minimum_nilai_transaksi, rata_rata_jumlah transaksi
#tanggal_transaksi_awal tidak digunakan karna tidak dapat di convert ke dlm data tanggal

summary(data_test)



#features-------
vardep="flag_transaksi_fraud"
var_nominal=c("tipe_kartu","id_merchant", "tipe_mesin","tipe_transaksi",
              "nama_transaksi","id_negara","nama_kota","lokasi_mesin","pemilik_mesin",
              "kuartal_transaksi","kepemilikan_kartu","id_channel")
var_numerik=c("nilai_transaksi","rata_rata_nilai_transaksi","maksimum_nilai_transaksi",
              "minimum_nilai_transaksi","rata_rata_jumlah_transaksi")
data_train=data_train%>%
  select_(.dots=c(vardep,var_nominal,var_numerik))%>%
  mutate_at(vars(one_of(var_nominal)),funs(as.character))%>%
  mutate_at(vars(one_of(var_nominal)),funs(as.factor))
data_test=data_test%>%
  select_(.dots=c(var_nominal,var_numerik))%>%
  mutate_at(vars(one_of(var_nominal)),funs(as.character))%>%
  mutate_at(vars(one_of(var_nominal)),funs(as.factor))
for (nama in var_nominal){
  levels(data_test[,nama])=levels(data_train[,nama])
}




#information value
iv_table=smbinning::smbinning.sumiv(data_train, y=vardep)


formu=as.formula(paste(vardep,"~."))

#model xgboost-----
threshold <- 0.5

# convert categorical factor into one-hot encoding
data_numerik=data_train%>%
  select_(.dots = var_numerik)

for (nama in var_nominal){
  temp <- model.matrix(as.formula(paste0("~",nama,"-1")),data_train)
  data_numerik=cbind(data_numerik,temp)
}

data_matrix <- data.matrix(data_numerik)

train_label=data.matrix(data_train%>%select_(.dots=c(vardep)))
dtrain <- xgb.DMatrix(data = data_matrix, label=train_label )
model_xgboost <- xgboost(data = dtrain, # the data   
                         nround = 10, # max number of boosting iterations
                         objective = "binary:logistic")  # the objective function
rm(temp,data_matrix)

#validitas
data_train$pred <- predict(model_xgboost, dtrain)

# Calculate the area under the ROC curve
roc.curve <- roc(data_train[,vardep], data_train$pred, ci=T)


# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con=confusionMatrix(factor(data_train$pred>threshold), factor(data_train[,vardep]==1), positive="TRUE")



#####################################################################
#Xgboost dengan SMOTE-------
source('smote function.R')

set.seed(1234)
data_smote=data_train %>% select(-pred)
data_smote[,vardep]=as.character(data_smote[,vardep])
data_smote[,vardep]=as.factor(data_smote[,vardep])
data_smote <- SMOTE(formu, 
                    data=data_smote,
                    perc.over = 200, perc.under=300)
data_smote[,vardep]=as.character(data_smote[,vardep])
data_smote[,vardep]=as.numeric(data_smote[,vardep])

# convert categorical factor into one-hot encoding
data_numerik_smote=data_smote %>%
  select_(.dots=var_numerik)
for (nama in var_nominal){
  temp <- model.matrix(as.formula(paste0("~",nama,"-1")),data_smote)
  data_numerik_smote=cbind(data_numerik_smote,temp)
}

data_matrix_smote <- data.matrix(data_numerik_smote)
train_label_smote=data.matrix(data_smote%>%select_(.dots=c(vardep)))
dtrain_smote <- xgb.DMatrix(data = data_matrix_smote, label=train_label_smote )
model_xgboost_smote <- xgboost(data = dtrain_smote, # the data   
                               nround = 10, # max number of boosting iterations
                               objective = "binary:logistic")  # the objective function
rm(temp,data_matrix_smote)

#validitas
data_smote$pred <- predict(model_xgboost_smote, dtrain_smote)

# Calculate the area under the ROC curve
roc.curve_smote <- roc(data_smote[,vardep], data_smote$pred, ci=T)


# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con_smote=confusionMatrix(factor(data_smote$pred>threshold), factor(data_smote[,vardep]==1), positive="TRUE")

#####
#
data_train$pred_smote <- predict(model_xgboost_smote, dtrain)
roc.curve_train_smote <- roc(data_train[,vardep], data_train$pred_smote, ci=T)


# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con_train_smote=confusionMatrix(factor(data_train$pred_smote>threshold), factor(data_train[,vardep]==1), positive="TRUE")


#importance variable
all_=xgb.importance(colnames(dtrain),model=model_xgboost)
smote=xgb.importance(colnames(dtrain),model=model_xgboost_smote)
xgb.plot.importance(all_)
xgb.plot.importance(smote)

#prediksi data test------
# convert categorical factor into one-hot encoding
data_numerik_test=data_test%>%
  select_(.dots=var_numerik)
for (nama in var_nominal){
  temp <- model.matrix(as.formula(paste0("~",nama,"-1")),data_test)
  data_numerik_test=cbind(data_numerik_test,temp)
}

data_matrix_test <- data.matrix(data_numerik_test)

#train_label=data.matrix(data_train%>%select_(.dots=c(vardep)))
dtest <- xgb.DMatrix(data = data_matrix_test )
data_test$pred <- predict(model_xgboost, dtest)
data_test$pred_smote <- predict(model_xgboost_smote, dtest)

#perbandingan akurasi
table_akurasi=data.frame(data=c("train", "smote", "train"),
                         model=c("xgboost","smote-xgboost","smote-xgboost"),
                         auc=c(as.numeric(roc.curve$auc), as.numeric(roc.curve_smote$auc),as.numeric(roc.curve_train_smote$auc)),
                         accuracy=c(con$overall[1],con_smote$overall[1],con_train_smote$overall[1]),
                         recall=c(con$byClass[6],con_smote$byClass[6],con_train_smote$byClass[6]))


#stabilitas hasil prediksi-----
#all data training
batas_percentile=round(quantile(data_train$pred, probs = seq(0.05,0.95,0.5)),8)
data_train$klas=cut(data_train$pred,breaks =unique( c(0,batas_percentile,1)))

#data_train%>%group_by(klas)%>% summarise(min=min(pred), max=max(pred),bad=sum(flag_kredit_macet),n= n())

#smote data training
batas_percentile_smote=round(quantile(data_train$pred_smote, probs = seq(0.05,0.95,0.05)),8)
data_train$klas_smote=cut(data_train$pred_smote,breaks = unique(c(0,batas_percentile_smote,1)))

#PSI----
library(smbinning)
data_test$klas=cut(data_test$pred,breaks = c(0,batas_percentile,1))
data_test$klas_smote=cut(data_test$pred_smote,breaks = unique(c(0,batas_percentile_smote,1)))

data_psi=data_train%>%
  select(klas, klas_smote)%>%
  mutate(type="1")%>%
  rbind(data_test%>%
          select(klas, klas_smote)%>%
          mutate(type="2"))
psi_all=smbinning.psi(data_psi, y="type",x="klas")        
psi_smote=smbinning.psi(data_psi, y="type",x="klas_smote")  

psi_table=data.frame(model=c("xgboost","smote-xgboost"),
                     PSI=c(psi_all$psimg[nrow(psi_all$psimg),2],psi_smote$psimg[nrow(psi_smote$psimg),2]))
