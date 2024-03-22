import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import gurobipy as gu
import time
from sklearn.metrics import accuracy_score

"""
Mixed Integer Linear Programming problem of the EFTE
input:
Phi maximum number of selected features
f features that are used in each tree
y_hat prediction of each tree for each individual
y_val true value of the class
I_hat list of sensetive individuals
alpha weight of the fairness term
eps, max_w parameters of the model
phi_start, w_start, xi_start warm start
TimeLimit timelimit in seconds
output:
w weights related to trees in the EFTE
phi selected features
xi proxy of the misclassification error
"""
def EFTE(Phi,f,y_hat,y_val,I_hat,alpha=0,eps=0.1,max_w=0.02, phi_start = None, w_start = None, xi_start = None, TimeLimit = None):
    T=y_hat.shape[0] #number of trees
    I=y_hat.shape[1] #number of individuals
    K=y_hat.shape[2] #number of classes
    p=f.shape[1]
    t_possible = np.where(f.sum(axis=1)<=Phi)[0]
    T_possible = len(t_possible)
    if T_possible==0:
        return {'stat':3}
    
    model = gu.Model("EFTE")
    w = model.addVars(T, name = 'w', vtype = gu.GRB.CONTINUOUS, lb = 0, ub = 1)
    phi = model.addVars(p, name = 'phi', vtype = gu.GRB.BINARY)
    xi = model.addVars(I, name = 'xi', vtype = gu.GRB.CONTINUOUS, lb = 0)

    if phi_start is not None:
       for pp in range(p):
           phi[pp].start = phi_start[pp]
    if w_start is not None:
       for t in range(T):
           w[t].start = w_start[t]
    if xi_start is not None:
       for i in range(I):
           xi[i].start = xi_start[i]    
           
    if T_possible<T:
        for t in (set(range(T))-set(t_possible)):
            w[t].ub = 0
            w[t].lb = 0
            
    if alpha != 0:
        model.setObjective(sum(xi[i] for i in range(I))/I+alpha*sum(xi[i] for i in I_hat)/len(I_hat), gu.GRB.MINIMIZE)
    else:
        model.setObjective(sum(xi[i] for i in range(I))/I, gu.GRB.MINIMIZE)

    model.addConstr(sum(w[t] for t in range(T)) == 1)
    model.addConstrs(max_w*phi[j] >= w[t] for j in range(p) for t in (set(np.where(f[:,j]==1)[0])& set(t_possible)))# 
    model.addConstr(sum(phi[j] for j in range(p))<= Phi)
    model.addConstrs(sum(w[t]*y_hat[t,i,int(np.array(y_val)[i])] for t in  range(T))>=sum(w[t]*y_hat[t,i,k] for t in range(T))-xi[i]+eps for i in range(I) for k in set(range(K))-{int(np.array(y_val)[i])})
 
    model.Params.OutputFlag = 0
    if TimeLimit is not None:
        model.Params.TimeLimit = TimeLimit
    model.Params.Seed = 12
    model.optimize()
    
    try:
        var = [v.x for v in model.getVars()]
        w = var[:T]
        phi = var[T:T+p]
        xi = var[T+p:T+p+I]
        obj = model.objVal
        stat = model.status
        runtime = model.runtime
        return {'w':w,'phi':phi,'xi':xi, 'stat':stat, 'runtime':runtime, 'obj':obj, 'MIPGap':model.MIPGap,'eps':eps,'max_w':max_w}
    except:
        stat = model.status
        return {'stat':stat,'eps':eps,'max_w':max_w}
"""
Prediction of the EFTE
input:
w weights of the EFTE
y_hat prediction of each tree for each individual
output:
y_new predictions of the EFTE
"""
def EFTE_predict(w,y_hat,rand_stat):
    I=y_hat.shape[1] #number of individuals
    K=y_hat.shape[2] #number of classes
    scores = np.zeros((I,K))
    y_new = np.zeros(I)
    for i in range(I):
        for k in range(K):
            tmp = np.sum(np.multiply(y_hat[:,i,k],w))
            scores[i,k]=tmp
        #if a tie then random allocation
        if len(np.where(scores[i,:] == np.max(scores[i,:]))[0])>1:
            np.random.seed(rand_stat)
            y_new[i]=int(np.random.choice(np.where(scores[i,:] == np.max(scores[i,:]))[0],1))
        else:
            y_new[i]= int(np.where(scores[i,:] == np.max(scores[i,:]))[0])
    return y_new

"""
Tuning of the EFTE
input:
eps_list list of parameter eps
max_w_list list of maximum weights
alpha weight of the fairness term
Phi maximum number of features
f features that are used in each tree
y_hat prediction of each tree for each individual on training
y_train true value of the class on training
I_hat list of sensetive individuals on training
y_hat_valid prediction of each tree for each individual on validation
y_valid true value of the class on validation
output:
r_best the best model in terms of accuracy on validation
"""
def EFTE_tune(eps_list,max_w_list, Phi,f,y_hat,y_train,I_hat,y_hat_valid,y_valid,rand_stat, alpha=0):
    acc_best = 0
    for eps in eps_list:
        for max_w in max_w_list:  
            start = time.time()
            r=EFTE(Phi,f,y_hat,y_train,I_hat,alpha=alpha,eps=eps,max_w=max_w, TimeLimit = 300)
            end = time.time()
            print('Phi', Phi,'alpha',alpha,'eps',eps,'max_w',max_w,str(np.round(end-start,2))+'s')
            if r.get('stat') !=3 and r.get('w') != None:
                y_new_valid = EFTE_predict(r.get('w'),y_hat_valid,rand_stat)
    
                if acc_best < accuracy_score(y_new_valid,y_valid):
                    r_best = r
                    acc_best=accuracy_score(y_new_valid,y_valid)     
    return r_best