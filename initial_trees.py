import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import xgboost as xgb
import re

"""
Build stump trees
input:
K number of classes
X_train, X_test, X_valid training, testing, validation samples
output:
f features that are used in each tree
features name of the feature used in each tree
thresholds thresholds of the feature used in each tree
y_hat,y_hat_test,y_hat_valid predictions of each tree for each individual
"""
def stumptrees(K,X_train,X_test,X_valid):
    I = X_train.shape[0]
    I_test = X_test.shape[0]
    I_valid = X_valid.shape[0]
    feature_name=X_train.columns
    p=len(feature_name)
    #define rules in each stump tree
    f = np.array([]).reshape(0,p)
    thresholds = np.array([])
    points = np.linspace(0,100,101)[1:]
    features = np.array([], dtype=np.int32)
    for ind,feat in enumerate(feature_name):
        unique_val = np.unique(X_train.loc[:,feat])
        if len(unique_val) <= len(points)-1:
            threshold_feat = unique_val[:-1]
        else:
            threshold_feat = np.unique([np.percentile(X_train.loc[:,feat],i) for i in points])
        thresholds = np.concatenate((thresholds,threshold_feat))
        features = np.concatenate((features,np.array([feat]*len(threshold_feat))))
        f_feat = np.zeros((len(threshold_feat),p))
        f_feat[:,ind] = 1
        f = np.concatenate((f,f_feat))
    T_half = f.shape[0]
    #double trees with the opposite pred
    f = np.concatenate((f,f))
    features = np.concatenate((features,features))
    thresholds = np.concatenate((thresholds,thresholds))
    T = len(thresholds)
    pred_R = [1 for i in range(T_half)]+[0 for i in range(T_half)]
    pred_L = [0 for i in range(T_half)]+[1 for i in range(T_half)]
    #define predictions
    y_hat = np.zeros((T,I,K))
    for i in range(I):
        row_i = X_train.iloc[i,:]
        less = [row_i[features[i]]<=thresholds[i] for i in range(T_half)]
        less = less+less
        y_hat[:,i,1] = [pred_R[i] if less[i] else pred_L[i] for i in range(T)]
        y_hat[:,i,0] = np.ones(T)-y_hat[:,i,1]

    y_hat_test = np.zeros((T,I_test,K))
    for i in range(I_test):
        row_i = X_test.iloc[i,:]
        less = [row_i[features[i]]<=thresholds[i] for i in range(T_half)]
        less = less+less
        y_hat_test[:,i,1] = [pred_R[i] if less[i] else pred_L[i] for i in range(T)]
        y_hat_test[:,i,0] = np.ones(T)-y_hat_test[:,i,1]
        
    y_hat_valid = np.zeros((T,I_valid,K))
    for i in range(I_valid):
        row_i = X_valid.iloc[i,:]
        less = [row_i[features[i]]<=thresholds[i] for i in range(T_half)]
        less = less+less
        y_hat_valid[:,i,1] = [pred_R[i] if less[i] else pred_L[i] for i in range(T)]
        y_hat_valid[:,i,0] = np.ones(T)-y_hat_valid[:,i,1]

    return f,features,y_hat,y_hat_test,y_hat_valid,thresholds
"""
Build stump trees
input:
T_xgb number of XGBoost trees
y_train class variable for the training sample
X_train, X_test, X_valid training, testing, validation samples
output:
f features that are used in each tree
features name of the feature used in each tree
thresholds thresholds of the feature used in each tree
y_hat,y_hat_test,y_hat_valid predictions of each tree for each individual
"""
def xgbtrees(T_xgb,X_train,y_train,X_test,X_valid):
    I = X_train.shape[0]
    I_test = X_test.shape[0]
    I_valid = X_valid.shape[0]
    K = y_train.unique()
    feature_name=X_train.columns
    p=len(feature_name)
    
    XG=xgb.XGBClassifier(n_estimators=T_xgb, max_depth=1, learning_rate=1, objective='binary:logistic')
    XG.fit(X_train,y_train)
    #extract rules
    t_list_xg = XG.get_booster().get_dump()
    features = [re.search('\[(.*)\<',t_list_xg[i])[0][1:-1] for i in range(T_xgb)]
    thresholds =[float(re.search('\<(.*)\]',t_list_xg[i])[0][1:-1]) for i in range(T_xgb)]

    #double trees with the opposite pred
    features_num = [np.where(feature_name==features[i])[0][0] for i in range(T_xgb)]
    features = np.concatenate((features,features))
    features_num = np.concatenate((features_num,features_num))
    thresholds = np.concatenate((thresholds,thresholds))
    T = len(thresholds)
    f = np.zeros((T,p))
    for i in range(T):
        f[i][features_num[i]]=1
    
    pred_R = [1 for i in range(T//2)]+[0 for i in range(T//2)]
    pred_L = [0 for i in range(T//2)]+[1 for i in range(T//2)]
    #define predictions
    y_hat = np.zeros((T,I,K))
    for i in range(I):
        row_i = X_train.iloc[i,:]
        less = [row_i[features[i]]<=thresholds[i] for i in range(T//2)]
        less = less+less
        y_hat[:,i,1] = [pred_R[i] if less[i] else pred_L[i] for i in range(T)]
        y_hat[:,i,0] = np.ones(T)-y_hat[:,i,1]

    y_hat_test = np.zeros((T,I_test,K))
    for i in range(I_test):
        row_i = X_test.iloc[i,:]
        less = [row_i[features[i]]<=thresholds[i] for i in range(T//2)]
        less = less+less
        y_hat_test[:,i,1] = [pred_R[i] if less[i] else pred_L[i] for i in range(T)]
        y_hat_test[:,i,0] = np.ones(T)-y_hat_test[:,i,1]
        
    y_hat_valid = np.zeros((T,I_valid,K))
    for i in range(I_valid):
        row_i = X_valid.iloc[i,:]
        less = [row_i[features[i]]<=thresholds[i] for i in range(T//2)]
        less = less+less
        y_hat_valid[:,i,1] = [pred_R[i] if less[i] else pred_L[i] for i in range(T)]
        y_hat_valid[:,i,0] = np.ones(T)-y_hat_valid[:,i,1]

    return f,features,y_hat,y_hat_test,y_hat_valid,thresholds