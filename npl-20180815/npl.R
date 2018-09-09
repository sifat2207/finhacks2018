SMOTE <- function(form,data,
                  perc.over=200,k=5,
                  perc.under=200,
                  learner=NULL,...
)
        
        # INPUTS:
        # form a model formula
        # data the original training set (with the unbalanced distribution)
        # minCl  the minority class label
        # per.over/100 is the number of new cases (smoted cases) generated
        #              for each rare case. If perc.over < 100 a single case
        #              is generated uniquely for a randomly selected perc.over
        #              of the rare cases
        # k is the number of neighbours to consider as the pool from where
        #   the new examples are generated
# perc.under/100 is the number of "normal" cases that are randomly
#                selected for each smoted case
# learner the learning system to use.
# ...  any learning parameters to pass to learner
{
        
        # the column where the target variable is
        tgt <- which(names(data) == as.character(form[[2]]))
        minCl <- levels(data[,tgt])[which.min(table(data[,tgt]))]
        
        # get the cases of the minority class
        minExs <- which(data[,tgt] == minCl)
        
        # generate synthetic cases from these minExs
        if (tgt < ncol(data)) {
                cols <- 1:ncol(data)
                cols[c(tgt,ncol(data))] <- cols[c(ncol(data),tgt)]
                data <-  data[,cols]
        }
        newExs <- smote.exs(data[minExs,],ncol(data),perc.over,k)
        if (tgt < ncol(data)) {
                newExs <- newExs[,cols]
                data <- data[,cols]
        }
        
        # get the undersample of the "majority class" examples
        selMaj <- sample((1:NROW(data))[-minExs],
                         as.integer((perc.under/100)*nrow(newExs)),
                         replace=T)
        
        # the final data set (the undersample+the rare cases+the smoted exs)
        newdataset <- rbind(data[selMaj,],data[minExs,],newExs)
        
        # learn a model if required
        if (is.null(learner)) return(newdataset)
        else do.call(learner,list(form,newdataset,...))
}



# ===================================================
# Obtain a set of smoted examples for a set of rare cases.
# L. Torgo, Feb 2010
# ---------------------------------------------------
smote.exs <- function(data,tgt,N,k)
        # INPUTS:
        # data are the rare cases (the minority "class" cases)
        # tgt is the name of the target variable
        # N is the percentage of over-sampling to carry out;
        # and k is the number of nearest neighours to use for the generation
        # OUTPUTS:
        # The result of the function is a (N/100)*T set of generated
        # examples with rare values on the target
{
        nomatr <- c()
        T <- matrix(nrow=dim(data)[1],ncol=dim(data)[2]-1)
        for(col in seq.int(dim(T)[2]))
                if (class(data[,col]) %in% c('factor','character')) {
                        T[,col] <- as.integer(data[,col])
                        nomatr <- c(nomatr,col)
                } else T[,col] <- data[,col]
        
        if (N < 100) { # only a percentage of the T cases will be SMOTEd
                nT <- NROW(T)
                idx <- sample(1:nT,as.integer((N/100)*nT))
                T <- T[idx,]
                N <- 100
        }
        
        p <- dim(T)[2]
        nT <- dim(T)[1]
        
        ranges <- apply(T,2,max)-apply(T,2,min) # nilai max-min dari setiap variabel
        
        nexs <-  as.integer(N/100) # this is the number of artificial exs generated
        # for each member of T
        new <- matrix(nrow=nexs*nT,ncol=p)    # the new cases
        
        for(i in 1:nT) {
                
                # the k NNs of case T[i,]
                xd <- scale(T,T[i,],ranges)
                for(a in nomatr) xd[,a] <- xd[,a]==0
                dd <- drop(xd^2 %*% rep(1, ncol(xd)))
                kNNs <- order(dd)[2:(k+1)]
                
                for(n in 1:nexs) {
                        # select randomly one of the k NNs
                        neig <- sample(1:k,1)
                        
                        ex <- vector(length=ncol(T))
                        
                        # the attribute values of the generated case
                        difs <- T[kNNs[neig],]-T[i,]
                        new[(i-1)*nexs+n,] <- T[i,]+runif(1)*difs
                        for(a in nomatr)
                                new[(i-1)*nexs+n,a] <- c(T[kNNs[neig],a],T[i,a])[1+round(runif(1),0)]
                        
                }
        }
        newCases <- data.frame(new)
        for(a in nomatr)
                newCases[,a] <- factor(newCases[,a],levels=1:nlevels(data[,a]),labels=levels(data[,a]))
        
        newCases[,tgt] <- factor(rep(data[1,tgt],nrow(newCases)),levels=levels(data[,tgt]))
        colnames(newCases) <- colnames(data)
        newCases
}
############################################################################

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
