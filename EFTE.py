import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gurobipy as gu
import xgboost as xgb
import pickle
from datetime import datetime
import os
from matplotlib import colors as mcolors

from initial_trees import *
from EFTE_model import *

def experiment_EFTE_oneseed(name,rand_stat):
    np.random.seed(rand_stat)
    if name == 'diabetes':
        data = pd.read_csv(os.path.join(__location__,'diabetes.csv'), sep = ',')
        data.Class = data.Class.astype('int')
        K = len(data.Class.unique())
        num_features = data.columns[:-1]
        feature_name = data.columns[:-1]
        data.loc[:,num_features] = (data.loc[:,num_features]-data.loc[:,num_features].min())/(data.loc[:,num_features].max()-data.loc[:,num_features].min())
        X_train, X_test, y_train, y_test = train_test_split(data.drop('Class',axis=1), data.Class, test_size=0.33, random_state=rand_stat)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=rand_stat)
        #sensitive individuals
        I_hat = list((X_train.reset_index())[(y_train.values).astype(bool)].index)
        I_hat_test = list((X_test.reset_index())[(y_test.values).astype(bool)].index)
     
    if name == 'compas_old':     
        data = pd.read_csv('compas.csv', sep = ',')
        feature_name = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count','days_b_screening_arrest', 'decile_score', 'two_year_recid']#,'is_recid' 'c_jail_in', 'c_jail_out']
        data = data.loc[:,feature_name]
        data = data[(data.days_b_screening_arrest <= 30) & (data.days_b_screening_arrest >= -30) & (data.c_charge_degree != "O") & (data.score_text != 'N/A')]# & (data.is_recid != -1)
        num_names = ['age', 'priors_count','days_b_screening_arrest', 'decile_score', 'two_year_recid']
        categ_names = ['c_charge_degree', 'race', 'age_cat', 'score_text', 'sex']
        data = pd.concat([pd.get_dummies(data.loc[:,categ_names]),data.loc[:,num_names]], axis = 1)
        data['Class'] =  data.two_year_recid           
        data = data.drop(['two_year_recid'], axis=1)
        K = len(data.Class.unique())
        num_features = data.columns[:-1]
        feature_name = data.columns[:-1]
        data.loc[:,num_features] = (data.loc[:,num_features]-data.loc[:,num_features].min())/(data.loc[:,num_features].max()-data.loc[:,num_features].min())
        X_train, X_test, y_train, y_test = train_test_split(data.drop('Class',axis=1), data.Class, test_size=0.33, random_state=rand_stat)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=rand_stat)
        I_hat = list((X_train.reset_index())[np.logical_and((X_train.loc[:,'race_African-American'].values).astype(bool),y_train.values==0)].index)
        I_hat_test = list((X_test.reset_index())[np.logical_and((X_test.loc[:,'race_African-American'].values).astype(bool),y_test.values==0)].index)
    
    #parameters for the EFTE    
    cs_list = [0, 0.125, 0.25, 0.5, 1]
    max_w_list = [2**(-i) for i in range(6)[::-1]]
    eps_list = [2**(-i) for i in range(1,4)[::-1]]
    Phi_list=list(range(1,len(X_train.column)+1))

    print('Run RF')
    RF_unconstr = RandomForestClassifier(n_estimators=500,max_features='sqrt',random_state=rand_stat,n_jobs=1)#class_weight='balanced')
    RF_unconstr.fit(X_train, y_train)
    features = RF_unconstr.feature_importances_
    feature_imp=features.argsort()[::-1]

    Models_cs = []
    Res_train_cs_acc = []
    Res_test_cs_acc = []
    Res_test_cs_acc_sens = []
    Res_test_cs_acc_nonsens = []
    print('Build initial trees')
    f,features,y_hat,y_hat_test,y_hat_valid,thresholds=stumptrees(K,X_train,X_test,X_valid)

    T = y_hat.shape[0]
    Res_train_RF_h_acc = []
    Res_test_RF_h_acc = []
    Res_test_RF_h_acc_sens = []
    Res_test_RF_h_acc_nonsens = []
    print('Run EFTE')
    for ind,Phi in enumerate(Phi_list): 
        #run benchmark
        idx = feature_imp[range(Phi)]
        RF_h = RandomForestClassifier(max_features='sqrt',n_estimators=500,random_state=rand_stat,n_jobs=1)
        RF_h.fit(X_train.iloc[:,idx], y_train)
        Res_train_RF_h_acc.append(accuracy_score(RF_h.predict(X_train.iloc[:,idx]),y_train))
        Res_test_RF_h_acc.append(accuracy_score(RF_h.predict(X_test.iloc[:,idx]),y_test))
        Res_test_RF_h_acc_sens.append(accuracy_score(RF_h.predict(X_test.values[I_hat_test,:][:,idx]),y_test.values[I_hat_test]))
        Res_test_RF_h_acc_nonsens.append(accuracy_score(RF_h.predict(X_test.values[list(set(range(len(y_test)))-set(I_hat_test)),:][:,idx]),y_test.values[list(set(range(len(y_test)))-set(I_hat_test))]))    
        
        #run EFTE
        models_cs = []
        res_train_cs_acc = []
        res_test_cs_acc = []
        res_test_cs_acc_sens = []
        res_test_cs_acc_nonsens = []
        for alpha in cs_list:
            r_best_cs = EFTE_tune(eps_list,max_w_list, Phi,f,y_hat,y_train,I_hat,y_hat_valid,y_valid,rand_stat, alpha=alpha)
            models_cs.append(r_best_cs)
            if r_best_cs.get('stat') !=3 and r_best_cs.get('w') != None:
                y_EFTE_cs_test = EFTE_predict(r_best_cs.get('w'),y_hat_test,rand_stat)
                res_train_cs_acc.append(accuracy_score(EFTE_predict(r_best_cs.get('w'),y_hat,rand_stat),y_train))
                res_test_cs_acc.append(accuracy_score(y_EFTE_cs_test,y_test))  
                res_test_cs_acc_sens.append(accuracy_score(y_EFTE_cs_test[I_hat_test],y_test.values[I_hat_test]))
                res_test_cs_acc_nonsens.append(accuracy_score(y_EFTE_cs_test[list(set(range(len(y_test)))-set(I_hat_test))],y_test.values[list(set(range(len(y_test)))-set(I_hat_test))]))
            else:
                res_train_cs_acc.append(np.nan)
                res_test_cs_acc.append(np.nan)
                res_test_cs_acc_sens.append(np.nan)
                res_test_cs_acc_nonsens.append(np.nan)
                
        Res_train_cs_acc.append(res_train_cs_acc)
        Res_test_cs_acc.append(res_test_cs_acc)
        Res_test_cs_acc_sens.append(res_test_cs_acc_sens)  
        Res_test_cs_acc_nonsens.append(res_test_cs_acc_nonsens)  
        Models_cs.append(models_cs)            
    
    return K, T, name, Phi_list, cs_list, Res_train_cs_acc,Res_test_cs_acc,Res_test_cs_acc_sens,Res_test_cs_acc_nonsens,Res_train_RF_h_acc,Res_test_RF_h_acc,Res_test_RF_h_acc_sens,Res_test_RF_h_acc_nonsens,Models_cs

"""
Run 5 MC simulations
"""
def experiment_EFTE_MC(name):

    Res_train_cs_acc=[]
    Res_test_cs_acc=[]
    Res_test_cs_acc_sens=[]
    Res_test_cs_acc_nonsens=[]
    Res_train_RF_h_acc = []
    Res_test_RF_h_acc = []
    Res_test_RF_h_acc_sens = []
    Res_test_RF_h_acc_nonsens = []
    Models_cs=[]  
    
    rand_stat = 12
    for i in range(5):
        print('seed',rand_stat+i)
        K, T, name, Phi_list, cs_list, Res_train_cs_acc_it,Res_test_cs_acc_it,Res_test_cs_acc_sens_it,Res_test_cs_acc_nonsens_it,Res_train_RF_h_acc_it,Res_test_RF_h_acc_it,Res_test_RF_h_acc_sens_it,Res_test_RF_h_acc_nonsens_it,Models_cs_it = experiment_EFTE_oneseed(name,rand_stat+i)
        Res_train_cs_acc.append(Res_train_cs_acc_it)
        Res_test_cs_acc.append(Res_test_cs_acc_it)
        Res_test_cs_acc_sens.append(Res_test_cs_acc_sens_it)
        Res_test_cs_acc_nonsens.append(Res_test_cs_acc_nonsens_it)
        Res_train_RF_h_acc.append(Res_train_RF_h_acc_it)
        Res_test_RF_h_acc.append(Res_test_RF_h_acc_it)
        Res_test_RF_h_acc_sens.append(Res_test_RF_h_acc_sens_it)
        Res_test_RF_h_acc_nonsens.append(Res_test_RF_h_acc_nonsens_it)
        Models_cs.append(Models_cs_it)  
    return K, T, name, Phi_list, cs_list, Res_train_cs_acc,Res_test_cs_acc,Res_test_cs_acc_sens,Res_test_cs_acc_nonsens,Res_train_RF_h_acc,Res_test_RF_h_acc,Res_test_RF_h_acc_sens,Res_test_RF_h_acc_nonsens,Models_cs

def main():
    name = 'diabetes'
    print('Start the experiment for',name, 'dataset')
    K, T, name, Phi_list, cs_list, Res_train_cs_acc,Res_test_cs_acc,Res_test_cs_acc_sens,Res_test_cs_acc_nonsens,Res_train_RF_h_acc,Res_test_RF_h_acc,Res_test_RF_h_acc_sens,Res_test_RF_h_acc_nonsens,Models_cs = experiment_EFTE_MC(name)
    
    with open(os.path.join(__location__,'%s_%s.pickle'%(name+'_tuning_powers2_newtrees_test',str(datetime.now())[:10])), 'wb') as handle:
        pickle.dump(Models_cs, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    Res_train_cs_acc_av=np.nanmean(np.array(Res_train_cs_acc), axis=0)
    Res_test_cs_acc_av=np.nanmean(np.array(Res_test_cs_acc), axis=0)
    Res_test_cs_acc_sens_av=np.nanmean(np.array(Res_test_cs_acc_sens), axis=0)
    Res_test_cs_acc_nonsens_av=np.nanmean(np.array(Res_test_cs_acc_nonsens), axis=0)

    Res_train_RF_h_acc_av=np.nanmean(np.array(Res_train_RF_h_acc), axis=0)
    Res_test_RF_h_acc_av=np.nanmean(np.array(Res_test_RF_h_acc), axis=0)
    Res_test_RF_h_acc_sens_av=np.nanmean(np.array(Res_test_RF_h_acc_sens), axis=0)
    Res_test_RF_h_acc_nonsens_av=np.nanmean(np.array(Res_test_RF_h_acc_nonsens), axis=0)
    
    #plot the obtained results for the EFTE and the benchmark
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    max_y=0.61
    min_y=0
    fig = plt.figure(figsize=(8, 8)) 
    plt.xlim(np.min(Phi_list),np.max(Phi_list))
    plt.ylim(min_y,max_y)
    if Phi_list[-1]>=10:
        plt.xticks(range(5,Phi_list[-1]+1,5))
    plt.plot(Phi_list,1-Res_test_RF_h_acc_av,'-k',linewidth=2,marker = '^')    
    for i in range(len(cs_list)):
        plt.plot(Phi_list,[1-Res_test_cs_acc_av[j][i] for j in range(len(Phi_list))],color = colors[list(colors)[28+i]],linestyle='--', marker = '^')
    plt.legend(['RF']+[r'EFTE, $\alpha$='+str(cs_list[i]) for i in range(len(cs_list))], fontsize=12)#,r'RF top $\Phi$'
    plt.grid(True)     
    plt.xlabel('the maximum number of features in EFTE and RF', fontsize=12)
    plt.ylabel('average misclassification error (%)', fontsize=12)
  

    fig = plt.figure(figsize=(8, 8)) 
    plt.xlim(np.min(Phi_list),np.max(Phi_list))
    plt.ylim(min_y,max_y)
    if Phi_list[-1]>=10:
        plt.xticks(range(5,Phi_list[-1]+1,5))
    plt.plot(Phi_list,1-Res_test_RF_h_acc_sens_av,'-k',linewidth=2, marker = '^') 
    for i in range(len(cs_list)):
        plt.plot(Phi_list,[1-Res_test_cs_acc_sens_av[j][i] for j in range(len(Phi_list))],color = colors[list(colors)[28+i]],linestyle='--', marker = '^')   
    plt.legend(['RF']+[r'EFTE, $\alpha$='+str(cs_list[i]) for i in range(len(cs_list))], fontsize=12)
    plt.grid(True)     
    plt.xlabel('the maximum number of features in EFTE and RF', fontsize=12)
    plt.ylabel('average misclassification error in $\\mathcal{I}_1$ (%)', fontsize=12) 


if __name__ == "__main__":
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    main()

