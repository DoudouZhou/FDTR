#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################
#### SEPSIS ####
################
# import the required functions and packages:
from utils import *
import os, sys
import numpy.ma as ma
import random
import matplotlib.pyplot as plt
import numpy as np
import copy

H = 5

# Import data:

sepsis_df = pd.read_csv('sepsis3_data.tsv', sep='\t')

sepsis_df['spo2_MISSIND'] = pd.isna(sepsis_df.spo2)
sepsis_df['resp_rate_MISSIND'] = pd.isna(sepsis_df.resp_rate)
###
sys.stderr.write('\nFilter complete cases only\n')
# Filter complete cases only
completeCase_IDs =  []
for id_ in tqdm(set(sepsis_df.stay_id)):
    intersec = len(set(sepsis_df.loc[sepsis_df['stay_id'] == id_,'bloc']).intersection(set(list(range(H)))))
    if intersec == H:
        completeCase_IDs.append(id_)
len(completeCase_IDs)       
#25568 
sepsis_df = sepsis_df.loc[sepsis_df['stay_id'].isin(completeCase_IDs),:]
sepsis_df = sepsis_df.loc[sepsis_df['bloc']<H,:]
# Remove values where reward is nan
sepsis_df.loc[pd.isna(sepsis_df.lactate),'lactate'] = 0


max_lactate = np.max(sepsis_df.loc[:,'lactate'])
sepsis_df.loc[:,'lactate'] = sepsis_df.loc[:,'lactate']/max_lactate


# assign number IDs to ICU units
ICUs,ICU_dict,k = list(set(sepsis_df.first_careunit)),{},0
for c in ICUs:
    ICU_dict[c] = k
    k += 1
sepsis_df['K'] = [ICU_dict[c] for c in list(sepsis_df.first_careunit)]

# patient level covariates
Pcovs = [ 'hemoglobin','potassium', 'weight', 'temperature','sbp']

# Generate ICU wide variables
def ICU_level_cov(df,patient_val,new_var):
    # Create indicatos of patient_val per ICU unit        
    patient_val_icu = []


    # Generate ICU specific vars
    # Check whic ICU unit has positive level for patient_val
    for k in set(df['K']):
        if np.mean(list(df.loc[df['K'] ==  k,[patient_val]][patient_val]))>0:
            patient_val_icu.append(k)
    df[new_var] = df.apply(lambda row: 1*(row.K in patient_val_icu), axis=1)
    return df

# Dialisis use in the ICU
sepsis_df = ICU_level_cov(df=sepsis_df,patient_val='dialysis_active',new_var='use_dialisis')
# Oxygen saturation measurements (amount of oxygen-carrying hemoglobin in the blood relative to the amount of hemoglobin not carrying oxyge)
sepsis_df = ICU_level_cov(df=sepsis_df,patient_val='spo2_MISSIND',new_var='use_spo2')

Hcovs = ['use_dialisis','use_spo2']

##################
##################
##################

#  Code actions based on IV fluids: 0, and less than the quantiles 0.25, 0.5, 0.75

q25 = np.quantile(sepsis_df['input_4hourly'][sepsis_df['input_4hourly'] != 0],.25)
q5 = np.quantile(sepsis_df['input_4hourly'][sepsis_df['input_4hourly'] != 0],.5)
q75 = np.quantile(sepsis_df['input_4hourly'][sepsis_df['input_4hourly'] != 0],.75)


sepsis_df['A'] = 0 
sepsis_df.loc[(sepsis_df['input_4hourly'] != 0) & (sepsis_df['input_4hourly'] < q25),'A'] = 1
sepsis_df.loc[(sepsis_df['input_4hourly'] >= q25) & (sepsis_df['input_4hourly'] < q5),'A'] = 2
sepsis_df.loc[(sepsis_df['input_4hourly'] >= q5) & (sepsis_df['input_4hourly'] < q75),'A'] = 3
sepsis_df.loc[(sepsis_df['input_4hourly'] >= q75),'A'] = 4
a_No = len(set(sepsis_df['A']))

K = len(ICUs)
hospitals,hospitals_test = {},{}
sys.stderr.write('\nGenerating datasets\n')
for k in tqdm(range(K)):
    loc_df = sepsis_df.loc[ sepsis_df['K'] ==  k,:]
    loc_IDs = list(set(loc_df['stay_id']))
    print(len(loc_IDs))
    random.seed(116687)
    random.shuffle(loc_IDs)
    train_ids = loc_IDs[int(len(loc_IDs)*.2):]
    test_ids = loc_IDs[:int(len(loc_IDs)*.2)]
    sys.stderr.write('\nICU ' +str(k)+' has '+str(len(train_ids))+', '+str(len(test_ids))+' train and test samples respectively'+'\n')
    train = sepsis_df.loc[(sepsis_df['stay_id'].isin(train_ids)) & (sepsis_df['K'] ==  k),:]
    test = sepsis_df.loc[(sepsis_df['stay_id'].isin(test_ids)) & (sepsis_df['K'] ==  k),:]
    #train, test = train_test_split(sepsis_df.loc[ sepsis_df['K'] ==  k,:], test_size=0.2)
    states,actions,rewards = {},{},{}
    states_test,actions_test,rewards_test = {},{},{}
    for h in range(H):
        # Remove NAs
        # train
        Hvec = np.array(train.loc[train['bloc'] ==h,Hcovs])
        Hvec = np.where(np.isnan(Hvec), ma.array(Hvec, mask=np.isnan(Hvec)).mean(axis=0), Hvec)
        Hvec_test = np.array(test.loc[test['bloc'] ==h,Hcovs])
        Hvec_test = np.where(np.isnan(Hvec_test), ma.array(Hvec_test, mask=np.isnan(Hvec_test)).mean(axis=0), Hvec_test)
        
        Pvec = np.array(train.loc[train['bloc'] ==h,Pcovs])
        Pvec = np.where(np.isnan(Pvec), ma.array(Pvec, mask=np.isnan(Pvec)).mean(axis=0), Pvec)
        Pvec_test = np.array(test.loc[test['bloc'] ==h,Pcovs])
        Pvec_test = np.where(np.isnan(Pvec_test), ma.array(Pvec_test, mask=np.isnan(Pvec_test)).mean(axis=0), Pvec_test)

        Rvec = train.loc[train['bloc'] == h,'lactate'].to_list()
        Rvec = np.where(np.isnan(Rvec), ma.array(Rvec, mask=np.isnan(Rvec)).mean(axis=0), Rvec)
        #if max(Rvec)!=0:
        #    Rvec /= max(Rvec)
        Rvec_test = test.loc[test['bloc'] == h,'lactate'].to_list()
        Rvec_test = np.where(np.isnan(Rvec_test), ma.array(Rvec_test, mask=np.isnan(Rvec_test)).mean(axis=0), Rvec_test)
        #if max(Rvec_test)!=0:
        #    Rvec_test /= max(Rvec_test)

        states[h] = {'H':Hvec,'P':Pvec}
        states_test[h] = {'H':Hvec_test,'P':Pvec_test}
        actions[h] = train.loc[train['bloc'] == h,'A'].to_list()
        actions_test[h] = test.loc[test['bloc'] == h,'A'].to_list()
        rewards[h] = Rvec
        rewards_test[h] = Rvec_test
        
    hospitals[k] = {'states':states,'actions':actions,'rewards':rewards}
    hospitals_test[k] = {'states':states_test,'actions':actions_test,'rewards':rewards_test}

from utils_sepsis import *
# local value function estimates
sys.stderr.write('\nlocal value function estimates\n')
hospitals = estimate_local_Vs(hospitals,a_No)

# Compute Z vectors for algorithm
sys.stderr.write('\nCompute Z vectors for algorithm\n')

hospitals = compute_Zs(hospitals)

# Dimensions of the phi vectors:
d0,d1 = phi0(s=states[0]['H'],a=actions[0]).shape[1],phi1(s=states[0]['P'],a=actions[0]).shape[1]

sys.stderr.write('\nTraining policy with Algorithm 2\n')
# Train policy with Algorithm 2
beta_tilde,Lamda_tilde,Lamda_tilde_inv,alphaK,V_tilde,pi_tilde = LDTR(hospitals,a_No,d0,d1)


sys.stderr.write('\nTraining policy with Algorithm 1\n')
# Train policy with Algorithm 1
beta_hat,Lamda_hat,Lamda_hat_inv,alpha,V_hat,pi_hat = FDTR(hospitals,a_No,V_tilde,d0,d1)


Beta_hat = {}
np.set_printoptions(precision=3,suppress=True)
for h in range(H): 
    for k in range(K):
        if k == 0:
            Beta_hat[h] = beta_hat[k][h]
        else:
            Beta_hat[h] = np.vstack((Beta_hat[h],beta_hat[k][h]))
    Beta_hat[h] = np.around(Beta_hat[h],3)

m1 = np.mean(Beta_hat[0],axis=0)
m2 = np.mean(Beta_hat[1],axis=0)
m3 = np.mean(Beta_hat[2],axis=0)
m4 = np.mean(Beta_hat[3],axis=0)
m5 = np.mean(Beta_hat[4],axis=0)

theta0all = np.vstack((m1,m2,m3,m4,m5))
np.savetxt('betahat_all.csv',theta0all,delimiter=',')

Qlearn0 = {k:train_Qlearn0(hospitals,H,k=k) for k in range(K)}

# This uses the value functions and regressions already estimated and a Q-function for each time-step
def local_Qlearn(h,states,k):
    #k = 1#choices(range(len(hospitals)), k=1)[0]    
    Qfun = hospitals[k]['Qregs']
    n_k = len(states[h]['H'])
    # Compute Q function values 
    Q_sa = []
    for a in range(a_No):
        #X_a = np.hstack((phi0(s=states[h]['H'],a=np.tile(a,n_k)),phi1(s=states[h]['P'],a=np.tile(a,n_k))))
        X_a = phi1(s=states[h]['P'],a=np.tile(a,n_k))
        Q_sa.append(list(Qfun[h].predict(X_a)))
        
    return np.argmax(np.array(Q_sa),0)

def single_Qlearn(h,states,Qlearn0=Qlearn0,k=1):
    #k = 1#choices(range(len(hospitals)), k=1)[0]    
    n_k = len(states[h]['H'])
    # Compute Q function values 
    Q_sa = []
    for a in range(a_No):
        #X_a = np.hstack((phi0(s=states[h]['H'],a=np.tile(a,n_k)),phi1(s=states[h]['P'],a=np.tile(a,n_k))))
        X_a = phi1(s=states[h]['P'],a=np.tile(a,n_k))
        Q_sa.append(list(Qlearn0[k].predict(X_a)))
    return np.argmax(np.array(Q_sa),0)

def single_Qlearn_votes(h,states,k,Qlearn0=Qlearn0):
    n_k = len(states[h]['H'])
    a_max  = []
    for k in range(K):
        # Compute Q function values 
        Q_sa = []
        for a in range(a_No):
            #X_a = np.hstack((phi0(s=states[h]['H'],a=np.tile(a,n_k)),phi1(s=states[h]['P'],a=np.tile(a,n_k))))
            X_a = phi1(s=states[h]['P'],a=np.tile(a,n_k))
            Q_sa.append(list(Qlearn0[k].predict(X_a)))
        a_max.append(list(np.argmax(np.array(Q_sa),0)))            
    return mode(np.array(a_max))[0][0]

# Compute Q function values 
def Qlearn_model(h,k,states,actions,Qlearn0=Qlearn0):
    a = hospitals[k]['actions'][h]
    #X_a = np.hstack((phi0(s=states[h]['H'],a=a),phi1(s=states[h]['P'],a=a)))
    X_a = phi1(s=states[h]['P'],a=a)
    return list(Qlearn0[k].predict(X_a))

# Step importance sampling evaluation
def Vstep_WIS(hospitals,k,a_No,H,log_regs,policy=None,beta_hat_h=None,Lamda=None,alpha=None):
    states,actions,rewards = hospitals[k]['states'],hospitals[k]['actions'],hospitals[k]['rewards']
    # Select actions based on estimated policy
    mu_hat,prop_scores = {},{}
    rho_prod = [[1]*states[0]['H'].shape[0]]
    for h in range(H):
        if policy == 'FDTR' or policy == 'LDTR':
            mu_hat[h],_ = LDTR_V(states,H,h,a_No,Lamda[h],beta_hat_h[h],alpha)
        elif policy == 'LDTRMV':
            mu_hat[h] = LDTR_MV(states,H,h,a_No,Lamda,beta_hat_h,alpha)[0]   #dPEVI_MV(states,H,K,h,a_No,Lamda,beta_hat_h,beta,d1)[0]
        else:
            mu_hat[h] = policy(h,states,k=k)
        # 
        X = np.hstack((states[h]['H'],states[h]['P']))
        prop_scores[h] = log_regs[k].predict_proba(X)
        # Compute the importance sampling product weigths
        rho_prod.append([rho_prod[h][i]*(actions[h][i]==mu_hat[h][i])/prop_scores[h][i][actions[h][i]] for i in range(states[0]['H'].shape[0])])

    # Compute the value function using importance sampling
    V_mu = []
    for i in range(states[0]['H'].shape[0]):
        curr_V= 0
        # patient history
        for h in range(H):
            r = rewards[h][i]
            if np.mean(rho_prod[h+1]) != 0:     
                curr_V += rho_prod[h+1][i]*r/np.mean(rho_prod[h+1])
            else:
                curr_V += 0
        
        V_mu.append(curr_V)
    return V_mu,mu_hat

########################################################################################################################
# Train IS weights

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_IPWs(hospitals):
    log_regs = {}
    H = len(hospitals[0]['states'])
    for k in range(len(hospitals)):
        states,actions = hospitals[k]['states'],hospitals[k]['actions']
        logreg = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2',C=.01,multi_class='multinomial'))
        
        X,y = np.hstack((states[0]['H'],states[0]['P'])),np.ravel(actions[0])
        for h in range(1,H):
            X = np.vstack((X,np.hstack((states[h]['H'],states[h]['P']))))                
            y = np.hstack((y,np.ravel(actions[h])))
        pi = logreg.fit(X,y)
        log_regs[k] = pi
    return log_regs

log_regs = train_IPWs(hospitals)

sys.stderr.write('\nTesting policies\n')
wisVs_single_Q,wisVs_single_Q_vts,wisVs_Qlearn,wisVs_FDTR,wisVs_LDTR_single,wisVs_LDTR_MV = [],[],[],[],[],[]
pi_single_Q,pi_single_Q_vts,pi_Qlearn,pi_FDTR,pi_LDTR_single,pi_LDTR_MV = np.zeros(a_No),np.zeros(a_No),np.zeros(a_No),np.zeros(a_No),np.zeros(a_No),np.zeros(a_No)
pi_observe = np.zeros(a_No)
for k in tqdm(range(K)):
    V,pi = Vstep_WIS(hospitals_test,k,a_No,H,log_regs,policy=single_Qlearn)    
    wisVs_single_Q = wisVs_single_Q + V
    pi_single_Q = pi_single_Q + np.array([np.sum([np.sum(pi[key] == a) for key in range(H)]) for a in range(a_No)])
       
    V,pi = Vstep_WIS(hospitals_test,k,a_No,H,log_regs,policy=single_Qlearn_votes)
    wisVs_single_Q_vts = wisVs_single_Q_vts + V
    pi_single_Q_vts = pi_single_Q_vts + np.array([np.sum([np.sum(pi[key] == a) for key in range(H)]) for a in range(a_No)])
 
    V,pi = Vstep_WIS(hospitals_test,k,a_No,H,log_regs,policy=local_Qlearn)
    wisVs_Qlearn = wisVs_Qlearn + V
    pi_Qlearn = pi_Qlearn + np.array([np.sum([np.sum(pi[key] == a) for key in range(H)]) for a in range(a_No)])

    V,pi = Vstep_WIS(hospitals_test,k,a_No,H,log_regs,policy='LDTR',beta_hat_h=beta_tilde[k],Lamda=Lamda_tilde_inv[k],alpha=alphaK[k])
    wisVs_LDTR_single = wisVs_LDTR_single + V
    pi_LDTR_single = pi_LDTR_single + np.array([np.sum([np.sum(pi[key] == a) for key in range(H)]) for a in range(a_No)])

    V,pi = Vstep_WIS(hospitals_test,k,a_No,H,log_regs,policy='LDTRMV',beta_hat_h=beta_tilde,Lamda=Lamda_tilde_inv,alpha=alphaK)
    wisVs_LDTR_MV = wisVs_LDTR_MV + V
    pi_LDTR_MV = pi_LDTR_MV + np.array([np.sum([np.sum(pi[key] == a) for key in range(H)]) for a in range(a_No)])

    V,pi = Vstep_WIS(hospitals_test,k,a_No,H,log_regs,policy='FDTR',beta_hat_h=beta_hat[k],Lamda=Lamda_hat_inv[k],alpha=alpha)
    wisVs_FDTR = wisVs_FDTR + V
    pi_FDTR = pi_FDTR + np.array([np.sum([np.sum(pi[key] == a) for key in range(H)]) for a in range(a_No)])
    
    pi = hospitals_test[k]['actions']
    pi_observe = pi_observe + np.array([np.sum([np.sum( np.array(pi[key]) == a) for key in range(H)]) for a in range(a_No)])
    
    
import seaborn as sns

methods_dict = {'Vs_single_Q':'Q-learn (H)','Vs_single_Q_vts':'Q-learn (1-MV)','Vs_Qlearn':'Q-learn (1)','Vs_dPEVI':'FDTR','Vs_dPEVI_single':'LDTR','Vs_dPEVI_MV':'LDTR (MV)'}
df = pd.DataFrame(data=np.array([wisVs_single_Q,wisVs_single_Q_vts,wisVs_Qlearn,wisVs_LDTR_single,wisVs_LDTR_MV,wisVs_FDTR]).T, columns=['Vs_single_Q','Vs_single_Q_vts','Vs_Qlearn','Vs_dPEVI_single','Vs_dPEVI_MV','Vs_dPEVI'])
df = pd.melt(df)


df['value'][df['value']<=0] = np.NaN
df['value'][df['value']>H] = np.NaN
df['Method'] = [methods_dict[m] for m in list(df.variable)]
plt.figure()
plt.ylim((0.3,1))
sns.barplot(x="Method", y='value', data=df)
plt.ylabel('Value Estimate')
plt.xticks(rotation=15)
plt.savefig('sepsis_results.png')

