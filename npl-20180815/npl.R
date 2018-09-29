#library-----
library(dplyr)
library(pROC)
library(caret)
library(xgboost)
library(smbinning)


# data------
data_train=read.csv("npl_train.csv")%>%
        select(-X)
data_test=read.csv("npl_test.csv")%>%
        select(-X)

#data preparation
#cek missing data
summary(data_train)
summary(data_test)

#features-------
vardep="flag_kredit_macet"
var_nominal=c("kode_cabang","skor_delikuensi")
var_numerik=c("jumlah_kartu","outstanding","limit_kredit","tagihan","total_pemakaian_tunai",              
              "total_pemakaian_retail","sisa_tagihan_tidak_terbayar","rasio_pembayaran",                   
              "persentasi_overlimit" ,"rasio_pembayaran_3bulan","rasio_pembayaran_6bulan",
              "jumlah_tahun_sejak_pembukaan_kredit","total_pemakaian",                    
              "sisa_tagihan_per_jumlah_kartu","sisa_tagihan_per_limit","total_pemakaian_per_limit" ,         
              "pemakaian_3bln_per_limit","pemakaian_6bln_per_limit","utilisasi_3bulan","utilisasi_6bulan")

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

# Partition data
set.seed(1234)
ind <- sample(2, nrow(data_train), replace = T, prob = c(0.8, 0.2))
# train <- data[ind==1,]
# test <- data[ind==2,]


#information value
iv_table=smbinning::smbinning.sumiv(data_train[ind==1,], y=vardep)

# variable yang tidak dapat di split :
# total_pemakaian_tunai, rasio_pembayaran_3bulan,rasio_pembayaran_6bulan,
# jumlah_tahun_sejak_pembukaan_kredit, pemakaian_6bln_per_limit

var_numerik=c("jumlah_kartu","outstanding","limit_kredit","tagihan",              
              "total_pemakaian_retail","sisa_tagihan_tidak_terbayar","rasio_pembayaran",                   
              "persentasi_overlimit" ,"total_pemakaian",                    
              "sisa_tagihan_per_jumlah_kartu","sisa_tagihan_per_limit","total_pemakaian_per_limit" ,         
              "pemakaian_3bln_per_limit","utilisasi_3bulan","utilisasi_6bulan")


formu=as.formula(paste(vardep,"~."))

#model xgboost-----
#threshold to calculate accuracy and recall 
threshold <- 0.5

# convert categorical factor into one-hot encoding
data_numerik=data_train%>%
        select_(.dots = var_numerik)

for (nama in var_nominal){
        temp <- model.matrix(as.formula(paste0("~",nama,"-1")),data_train)
        data_numerik=cbind(data_numerik,temp)
}

data_matrix_train <- data.matrix(data_numerik[ind==1,])
data_matrix_test <- data.matrix(data_numerik[ind==2,])

label_train=data.matrix(data_train[ind==1,]%>%select_(.dots=c(vardep)))
label_test=data.matrix(data_train[ind==2,]%>%select_(.dots=c(vardep)))

dtrain <- xgb.DMatrix(data = data_matrix_train, label=label_train)
dtest <- xgb.DMatrix(data = data_matrix_test, label=label_test)

watchlist <- list(train = dtrain, test = dtest)

xgb_params <- list(max_depth = 6, eta = 0.1, silent = 1, nthread = 2, 
                   objective = "binary:logistic", eval_metric = "auc")

# eXtreme Gradient Boosting Model
model_xgboost <- xgb.train(params = xgb_params,
                           data = dtrain,
                           nrounds = 99, 
                           watchlist)
e <- data.frame(model_xgboost$evaluation_log)
plot(e$iter, e$train_auc, col = 'blue')
lines(e$iter, e$test_auc, col = 'red')

rm(data_matrix_train,data_matrix_test,label_train,label_test, data_numerik, temp)

#validitas
data_train$pred =0
data_train[ind==1,"pred"] <- predict(model_xgboost, dtrain)
data_train[ind==2,"pred"] <- predict(model_xgboost, dtest)

# Calculate the area under the ROC curve
roc.curve_train <- roc(data_train[ind==1,vardep], data_train[ind==1,"pred"], ci=T)
roc.curve_test <- roc(data_train[ind==2,vardep], data_train[ind==2,"pred"], ci=T)

# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con_train=confusionMatrix(factor(data_train[ind==1,"pred"]>threshold), factor(data_train[ind==1,vardep]==1), positive="TRUE")
con_test=confusionMatrix(factor(data_train[ind==2,"pred"]>threshold), factor(data_train[ind==2,vardep]==1), positive="TRUE")



#####################################################################
#Xgboost dengan SMOTE-------
source('smote function.R')

set.seed(432)
data_smote=data_train[ind==1,] %>% select(-pred)
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

watchlist_smote <- list(train = dtrain_smote, test = dtest)

xgb_params_smote <- list(max_depth = 4, eta = 0.1, silent = 1, nthread = 2, 
                         objective = "binary:logistic", eval_metric = "auc")

# eXtreme Gradient Boosting Model
model_xgboost_smote <- xgb.train(params = xgb_params_smote,
                                 data = dtrain_smote,
                                 nrounds = 148, 
                                 watchlist=watchlist_smote)

rm(temp,data_matrix_smote)

#validitas
data_smote$pred <- predict(model_xgboost_smote, dtrain_smote)

data_train$pred_smote =0
data_train[ind==1,"pred_smote"] <- predict(model_xgboost_smote, dtrain)
data_train[ind==2,"pred_smote"] <- predict(model_xgboost_smote, dtest)

# Calculate the area under the ROC curve
roc.curve_smote <- roc(data_smote[,vardep], data_smote$pred, ci=T)
roc.curve_smote_train <- roc(data_train[ind==1,vardep], data_train[ind==1,"pred_smote"], ci=T)
roc.curve_smote_test <- roc(data_train[ind==2,vardep], data_train[ind==2,"pred_smote"], ci=T)


# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con_smote=confusionMatrix(factor(data_smote$pred>threshold), factor(data_smote[,vardep]==1), positive="TRUE")
con_smote_train=confusionMatrix(factor(data_train[ind==1,"pred_smote"]>threshold), factor(data_train[ind==1,vardep]==1), positive="TRUE")
con_smote_test=confusionMatrix(factor(data_train[ind==2,"pred_smote"]>threshold), factor(data_train[ind==2,vardep]==1), positive="TRUE")


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

dvalidasi <- xgb.DMatrix(data = data_matrix_test )
data_test$pred <- predict(model_xgboost, dvalidasi)
data_test$pred_smote <- predict(model_xgboost_smote, dvalidasi)

rm(data_matrix_test)
#perbandingan akurasi
table_akurasi=data.frame(model=c("xgboost","xgboost","smote-xgboost","smote-xgboost","smote-xgboost"),
                         data=c("train (model)","test","smote_train (model)","train","test"),
                         auc=c(as.numeric(roc.curve_train$auc),as.numeric(roc.curve_test$auc),as.numeric(roc.curve_smote$auc),
                               as.numeric(roc.curve_smote_train$auc),as.numeric(roc.curve_smote_test$auc)),
                         accuracy=c(con_train$overall[1],con_test$overall[1],con_smote$overall[1],con_smote_train$overall[1],con_smote_test$overall[1]),
                         recall=c(con_train$byClass[6],con_test$byClass[6],con_smote$byClass[6],con_smote_train$byClass[6],con_smote_test$byClass[6]))

#stabilitas hasil prediksi-----
#all data training
batas_percentile=round(quantile(data_train[ind==1,"pred"], probs = seq(0.05,0.95,0.5)),8)
data_train$klas=cut(data_train$pred,breaks =unique( c(0,batas_percentile,1)))

#data_train%>%group_by(klas)%>% summarise(min=min(pred), max=max(pred),bad=sum(flag_kredit_macet),n= n())

#smote data training
batas_percentile_smote=round(quantile(data_train[ind==1,"pred_smote"], probs = seq(0.05,0.95,0.05)),8)
data_train$klas_smote=cut(data_train$pred_smote,breaks = unique(c(0,batas_percentile_smote,1)))

#PSI----
library(smbinning)
data_test$klas=cut(data_test$pred,breaks = c(0,batas_percentile,1))
data_test$klas_smote=cut(data_test$pred_smote,breaks = unique(c(0,batas_percentile_smote,1)))

data_psi=data_train[ind==1,]%>%
        select(klas, klas_smote)%>%
        mutate(type="1")%>%
        rbind(data_train[ind==2,]%>%
                      select(klas, klas_smote)%>%
                      mutate(type="2"))%>%
        rbind(data_test%>%
                      select(klas, klas_smote)%>%
                      mutate(type="3"))
psi_all=smbinning.psi(data_psi, y="type",x="klas")        
psi_smote=smbinning.psi(data_psi, y="type",x="klas_smote")  

psi_table=data.frame(model=c("xgboost","smote-xgboost"),
                     PSI_train_test=c(psi_all$psimg[nrow(psi_all$psimg),2],psi_smote$psimg[nrow(psi_smote$psimg),2]),
                     PSI_test=c(psi_all$psimg[nrow(psi_all$psimg),3],psi_smote$psimg[nrow(psi_smote$psimg),3]),
                     num_klas=c(psi_all$psicnt%>%nrow()-1,psi_smote$psicnt%>%nrow()-1))%>%
        mutate(Inference_train=ifelse(PSI_train_test<0.1,"Insignificant change",
                                      ifelse(PSI_train_test<0.25, "Some minor change","Major shift in population")))%>%
        mutate(Inference_test=ifelse(PSI_test<0.1,"Insignificant change",
                                     ifelse(PSI_test<0.25, "Some minor change","Major shift in population")))

data_sent=read.csv("npl_test.csv")%>%
        select(X)%>%
        cbind(data_test%>%
                      select(probability=pred_smote)%>%
                      mutate(prediction=ifelse(probability>threshold,1,0)))%>%
        select(X,prediction, probability)
