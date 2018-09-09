
#library-----
library(dplyr)
library(pROC)
library(caret)
library(earth)
library(smbinning)
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


#features selection----
vardep="flag_kredit_macet"
var_nominal=c("kode_cabang","skor_delikuensi")
data_train=data_train%>%
        mutate_at(vars(one_of(var_nominal)),funs(as.character))%>%
        mutate_at(vars(one_of(var_nominal)),funs(as.factor))
data_test=data_test%>%
        mutate_at(vars(one_of(var_nominal)),funs(as.character))%>%
        mutate_at(vars(one_of(var_nominal)),funs(as.factor))

list_features=list()

#full features
list_features[["feature_1"]]=c("jumlah_kartu","outstanding","limit_kredit","tagihan",                            
                               "total_pemakaian_tunai","total_pemakaian_retail","sisa_tagihan_tidak_terbayar",
                               "kode_cabang","rasio_pembayaran" ,"persentasi_overlimit" ,"rasio_pembayaran_3bulan",
                               "rasio_pembayaran_6bulan","skor_delikuensi","jumlah_tahun_sejak_pembukaan_kredit",
                               "total_pemakaian","sisa_tagihan_per_jumlah_kartu","sisa_tagihan_per_limit",
                               "total_pemakaian_per_limit","pemakaian_3bln_per_limit","pemakaian_6bln_per_limit",
                               "utilisasi_3bulan","utilisasi_6bulan")


#feature selection dengan mars
formu=as.formula(paste(vardep,"~."))
earth1=earth(formu, data = data_train%>%select_(.dots=c(vardep,list_features[["feature_1"]])))
feature1=rownames(evimp(earth1))
list_features[["feature_2"]]=feature1

#feature selection dengan information value
iv_table=smbinning.sumiv(data_train%>%select_(.dots=c(vardep,list_features[["feature_1"]])),vardep)
list_features[["feature_3"]]=as.character((iv_table%>% filter(IV>=0.02))[,"Char"])




#model klasifikasi-----
list_model=list()
threshold <- 0.5
for (i in c(1:3)){
        #regresi logistik
        model_glm=glm(formu, 
                      data=data_train%>%select_(.dots=c(vardep,list_features[[paste0("feature_",i)]])),
                      family = binomial(link="logit"))
        #validitas
        data_train$pred <- predict(model_glm, type="response", newdata=data_train)
        
        # Calculate the area under the ROC curve
        roc.curve <- roc(data_train$flag_kredit_macet, data_train$pred, ci=T)
        
        
        # Calculates a cross-tabulation of observed and predicted classes 
        # with associated statistics 
        con=confusionMatrix(factor(data_train$pred>threshold), factor(data_train$flag_kredit_macet==1), positive="TRUE")
        
        list_temp=list()
        list_temp[["model"]]=model_glm
        list_temp[["ROC"]]=roc.curve
        list_temp[["confusionmatrix"]]=con
        list_model[[paste0("Model_",i)]]=list_temp
        
        
        #xgboost
        var_numerik=list_features[[paste0("feature_",i)]][!list_features[[paste0("feature_",i)]]%in%var_nominal]
        data_numerik=data_train%>%select_(.dots=c(var_numerik))
        if("kode_cabang" %in% list_features[[paste0("feature_",i)]]){
                # convert categorical factor into one-hot encoding
                region <- model.matrix(~kode_cabang-1,data_train)
                data_numerik=cbind(data_numerik,region)
        }
        if("skor_delikuensi" %in% list_features[[paste0("feature_",i)]]){
                # convert categorical factor into one-hot encoding
                delikuensi <- model.matrix(~skor_delikuensi-1,data_train)
                data_numerik=cbind(data_numerik,delikuensi)
        }
        data_matrix <- data.matrix(data_numerik)
        train_label=data.matrix(data_train%>%select_(.dots=c(vardep)))
        dtrain <- xgb.DMatrix(data = data_matrix, label=train_label )
        model_xgboost <- xgboost(data = dtrain, # the data   
                         nround = 2, # max number of boosting iterations
                         objective = "binary:logistic")  # the objective function
        
        #validitas
        data_train$pred <- predict(model_xgboost, dtrain)
        
        # Calculate the area under the ROC curve
        roc.curve <- roc(data_train$flag_kredit_macet, data_train$pred, ci=T)
        
        
        # Calculates a cross-tabulation of observed and predicted classes 
        # with associated statistics 
        con=confusionMatrix(factor(data_train$pred>threshold), factor(data_train$flag_kredit_macet==1), positive="TRUE")
        
        list_temp=list()
        list_temp[["model"]]=model_xgboost
        list_temp[["ROC"]]=roc.curve
        list_temp[["confusionmatrix"]]=con
        list_model[[paste0("Model_",i+3)]]=list_temp
}
rm(list_temp,model_glm,model_xgboost,roc.curve,con, earth1)




###--------------------------------------
#SMOTE
set.seed(1234)
data_smote=data_train%>%select_(.dots=c(vardep,list_features[["feature_1"]]))
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
data_numerik_smote=data_smote%>%select_(.dots=c(var_numerik))
data_numerik_smote=cbind(data_numerik_smote,region_smote,delikuensi_smote)

data_matrix_smote <- data.matrix(data_numerik_smote)
train_label_smote=data.matrix(data_smote%>%select_(.dots=c(vardep)))
dtrain_smote <- xgb.DMatrix(data = data_matrix_smote, label=train_label_smote )
model_xgboost_smote <- xgboost(data = dtrain_smote, # the data   
                         nround = 2, # max number of boosting iterations
                         objective = "binary:logistic")  # the objective function

#validitas
data_smote$pred <- predict(model_xgboost_smote, dtrain_smote)

# Calculate the area under the ROC curve
roc.curve <- roc(data_smote$flag_kredit_macet, data_smote$pred, ci=T)


# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con=confusionMatrix(factor(data_smote$pred>threshold), factor(data_smote$flag_kredit_macet==1), positive="TRUE")


#####
#xgboost
var_numerik=list_features[[paste0("feature_",1)]][!list_features[[paste0("feature_",1)]]%in%var_nominal]
data_numerik=data_train%>%select_(.dots=c(var_numerik))
if("kode_cabang" %in% list_features[[paste0("feature_",i)]]){
        # convert categorical factor into one-hot encoding
        region <- model.matrix(~kode_cabang-1,data_train)
        data_numerik=cbind(data_numerik,region)
}
if("skor_delikuensi" %in% list_features[[paste0("feature_",i)]]){
        # convert categorical factor into one-hot encoding
        delikuensi <- model.matrix(~skor_delikuensi-1,data_train)
        data_numerik=cbind(data_numerik,delikuensi)
}
data_matrix <- data.matrix(data_numerik)
train_label=data.matrix(data_train%>%select_(.dots=c(vardep)))
dtrain <- xgb.DMatrix(data = data_matrix, label=train_label )

data_train$pred <- predict(model_xgboost_smote, dtrain)
roc.curve_test <- roc(data_train$flag_kredit_macet, data_train$pred, ci=T)


# Calculates a cross-tabulation of observed and predicted classes 
# with associated statistics 
con_test=confusionMatrix(factor(data_train$pred>threshold), factor(data_train$flag_kredit_macet==1), positive="TRUE")
