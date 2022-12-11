#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from random import choices
from scipy.stats import multivariate_normal,uniform,mode
from tqdm import tqdm
import pandas as pd
import pickle

import os


# Hospital level feature map
def phi0(s,a,map=None):
    a = np.array(a).reshape((s.shape[0], 1))
    if map is None:# linear feature map
        return np.hstack((s,a*s,(a**2)*s))
        #return np.hstack((s,a*s))
    
def phi1(s,a,map=None):
    a = np.array(a).reshape((s.shape[0], 1))
    if map is None:# linear feature map
        return np.abs(np.hstack((s,a*s,(a**2)*s)))
        #return np.hstack((s,a*s))
# Compute Zs as defined in paper
def compute_Zs(hospitals):
    K =  len(hospitals)
    H = len(hospitals[0]['actions'])
    for k in tqdm(range(K)):
        Zsh,Phis, Phis0, Phis1 = {},{},{},{}
        for h in range(H):
            ph0 = phi0(s=hospitals[k]['states'][h]['H'],a=hospitals[k]['actions'][h])
            ph1 = phi1(s=hospitals[k]['states'][h]['P'],a=hospitals[k]['actions'][h])
            n,d1 = ph1.shape        
            Zsh[h] = np.hstack((ph0,np.zeros((n,k*d1)),ph1,np.zeros((n,(K-k-1)*d1))))
            Phis[h] = np.hstack((ph0,ph1))
            Phis0[h] = ph0
            Phis1[h] = ph1
        hospitals[k]['Zs'],hospitals[k]['Phis'],hospitals[k]['Phis0'],hospitals[k]['Phis1'] = Zsh,Phis,Phis0,Phis1   
    return hospitals


#########################
###### Local Dynamic Treatment Regime (LDTR) ######  
#########################
#Input: hospitals = {0:{actions:{0:, 1:,..., H-1:}, Phis: , rewards:, states:,} }
def LDTR(hospitals,a_No,d0,d1):
    K = len(hospitals)
    n_ks = {k:len(hospitals[k]['actions'][0]) for k in range(K)}
    H = len(hospitals[0]['actions'])
    # xi =.99 or .95
    # c= 5,10,20,50
    lamda,c,xi,d = 1,.002,.95,d0+d1
    zeta = {k:np.log(2*d*H*n_ks[k]/xi) for k in range(K)}
    alphaK = {k:c*d*H*np.sqrt(zeta[k]) for k in range(K)}

    V_tilde = {k:{H:np.zeros(n_ks[k])} for k in range(K)}
    beta_tilde = {k:{H:np.zeros(d)} for k in range(K)}
    pi_tilde = {k:{H:np.zeros(n_ks[k])} for k in range(K)}
    Lamda_tilde = {k:{H:np.eye(d)} for k in range(K)}
    Lamda_tilde_inv = {k:{H:np.eye(d)} for k in range(K)}
    
    #########
    for k in tqdm(range(K)):
        Y = {}
        for h in range(H-1,-1,-1):
            Phi_sa = hospitals[k]['Phis'][h]
            Lamda_tilde[k][h] = Phi_sa.T.dot(Phi_sa)+lamda*np.eye(d)
            Lamda_tilde_inv[k][h] = np.linalg.inv(Lamda_tilde[k][h])
            
            # Compute pseudo-outcomes with local value function estimates
            Y[h] = hospitals[k]['rewards'][h] + V_tilde[k][h+1]
            # Estimate  linear regression parameters with Ridge reg.    
            beta_tilde[k][h] = Lamda_tilde_inv[k][h].dot(Phi_sa.T.dot(Y[h]))
            pi_tilde[k][h],V_tilde[k][h] = LDTR_V(hospitals[k]['states'],H,h,a_No,Lamda_tilde_inv[k][h],beta_tilde[k][h],alphaK[k])
            print(k,h,np.quantile(pi_tilde[k][h],[0,0.25,0.5,1]))
            print(k,h, [np.mean(pi_tilde[k][h]==a) for a in range(a_No)] )
    return beta_tilde,Lamda_tilde,Lamda_tilde_inv,alphaK,V_tilde,pi_tilde
    ##
    

def LDTR_V(states,H,h,a_No,Lamdakh_inv,betahk,alpha):
    n = states[h]['H'].shape[0]
    Qhat_sa  = []
    for a in range(a_No):
        ph0 = phi0(s=states[h]['H'],a=np.tile(a,n))
        ph1 = phi1(s=states[h]['P'],a=np.tile(a,n))
        Z_a = np.hstack((ph0,ph1))
        
        GammValues = [Gamma(Z=Z_a[i],Lambda_inv=Lamdakh_inv,alpha=alpha) for i in range(n)]
        Qbar_sa = [Qbar(Z=Z_a[i],betahat=betahk,GammVal=GammValues[i]) for i in range(n)]
        Qhat_sa.append([Qhat(QbarVal=Qbar_sa[i],H=H,h=h) for i in range(n)])
    pi_hat  = np.argmax(np.array(Qhat_sa),0)
    V_hat  = np.max(np.array(Qhat_sa),0)
    return pi_hat,V_hat

def Gamma(Z,Lambda_inv,alpha):
    return alpha*np.sqrt(Z.dot(Lambda_inv).dot(Z))

def Qbar(Z,betahat,GammVal):    
    return Z.dot(betahat)-GammVal

def Qhat(QbarVal,H,h):    
    val = min(QbarVal,H-h)# do not use +1 because h starts aat 0 not 1
    # apply ReLU function
    return max(val,0)


#########################
###### Algorithm Federated Dynamic Treatment Regime (FDTR)######  
#########################
def FDTR(hospitals,a_No,V_tilde,d0,d1):
    K = len(hospitals)
    #n = len(hospitals[0]['actions'][0])
    n_ks = {k:len(hospitals[k]['actions'][0]) for k in range(K)}
    N = np.sum([len(hospitals[k]['actions'][0]) for k in range(K)])
    H = len(hospitals[0]['actions'])
    # xi =.99 or .95
    # c= 5,10,20,50
    lamda,c,xi,d = 1,.008,.99,d0+d1
    zeta = np.log(2*d*H*N/xi)
    alpha = c*d*H*np.sqrt(zeta)
    #######
    Proj0 = {k:{H:np.eye(d0)} for k in range(K)}
    Proj1 = {k:{H:np.eye(d0)} for k in range(K)}
    
    #Lamda = {h:lamda*np.eye((d0+K*d1)) for h in range(H)}
    #
    #beta_hat = {}
    #V_k = {H:np.zeros(n)}
    
    V_hat = {k:{H:np.zeros(n_ks[k])} for k in range(K)}
    beta_hat = {k:{H:np.zeros(d0+d1)} for k in range(K)}
    pi_hat = {k:{H:np.zeros(n_ks[k])} for k in range(K)}
    Lamda_hat = {k:{H:np.eye(d)} for k in range(K)}
    Lamda_hat_inv = {k:{H:np.eye(d)} for k in range(K)}

    for h in tqdm(range(H-1,-1,-1)):
        Y = {}
        Lamda = lamda*np.eye(d)
        ZY = np.zeros(d0)
        #ZY =  {h:np.zeros((d0+K*d1)) for h in range(H)}
        
        for j in range(K):
            #Compute the projection matrix (Eq 12)
            Phis0_jh =  hospitals[j]['Phis0'][h]
            Phis1_jh =  hospitals[j]['Phis1'][h]
            mat_tmp = np.linalg.pinv(Phis1_jh.T.dot(Phis1_jh) ) #+ 100*lamda*np.eye(d1) 
            mat_tmp = Phis1_jh.dot(mat_tmp)
            mat_tmp = mat_tmp.dot(Phis1_jh.T)
            mat_tmp = np.eye(n_ks[j]) - mat_tmp
            Proj0[j][h] = Phis0_jh.T.dot(mat_tmp)
            Proj1[j][h] = Proj0[j][h].dot(Phis0_jh)
            
            
            Lamda[0:d0,0:d0] = Lamda[0:d0,0:d0] + Proj1[j][h]
            
            Y[j] = hospitals[j]['rewards'][h] + V_tilde[j][h+1]
            ZY += Proj0[j][h].dot(Y[j])
                   
        # Estimate  linear regression parameters with Ridge reg.    
        for k in tqdm(range(K)):
            Phis_kh =  hospitals[k]['Phis'][h]
            Lamda_hat[k][h] = Phis_kh.T.dot(Phis_kh) + Lamda
            Lamda_hat[k][h][0:d0,0:d0] = Lamda_hat[k][h][0:d0,0:d0] -  Proj1[k][h]
            Yk = hospitals[k]['rewards'][h] + V_hat[k][h+1]
            
            Z_hk = Phis_kh.T.dot(Yk)
            Z_hk[0:d0] = Z_hk[0:d0] + ZY - Proj0[k][h].dot(Y[k])
            
            Lamda_hat_inv[k][h] = np.linalg.inv(Lamda_hat[k][h]) 
            beta_hat[k][h] = Lamda_hat_inv[k][h].dot(Z_hk)
            
            pi_hat[k][h],V_hat[k][h] = LDTR_V(hospitals[k]['states'],H,h,a_No,Lamda_hat_inv[k][h],beta_hat[k][h],alpha)
            print(k,h,np.quantile(pi_hat[k][h],[0,0.25,0.5,1]))
            print(k,h, [np.mean(pi_hat[k][h]==a) for a in range(a_No)] )
    return beta_hat,Lamda_hat,Lamda_hat_inv,alpha,V_hat,pi_hat


def LDTR_MV(states,H,h,a_No,Lamda,beta_hat_h,alpha):
    a_star = []
    K= len(Lamda)
    for k in range(K):
        a_local,_ = LDTR_V(states,H,h,a_No,Lamda[k][h],beta_hat_h[k][h],alpha[k])
        a_star.append(list(a_local))
    
    return mode(np.array(a_star))[0]





### Q-learning Algorithms

########
########
# This trains a single Q-function for all times 
from sklearn import linear_model

def train_Qlearn0(hospitals,H,k):
    # Compute the value function within hospital k:
    Ys = np.hstack([hospitals[k]['V_hats'][h] for h in range(H)])
    Xs = np.vstack([phi1(s=hospitals[k]['states'][h]['P'],a=hospitals[k]['actions'][h]).tolist() for h in range(H)])
    # linear Q-function fitting
    reg = linear_model.LinearRegression()
    reg.fit(Xs, Ys)
    return reg



def estimate_local_Vs(hospitals,a_No):
    # Compute the value function within each hospital:
    for k in tqdm(range(len(hospitals))):
        n_k = len(hospitals[k]['actions'][0])
        H = len(hospitals[k]['actions'])
        V_hat,Qreg = {H:np.zeros(n_k)},{}
        for h in range(H-1,-1,-1):            
            # state
            #X = np.hstack((phi0(s=hospitals[k]['states'][h]['H'],a=hospitals[k]['actions'][h]),phi1(s=hospitals[k]['states'][h]['P'],a=hospitals[k]['actions'][h])))
            X = phi1(s=hospitals[k]['states'][h]['P'],a=hospitals[k]['actions'][h])
            #  pseudo-reward function
            y = hospitals[k]['rewards'][h] + V_hat[h+1]
            # linear Q-function fitting
            reg = linear_model.LinearRegression()
            reg.fit(X, y)
            # Store regression object for local Q-learning policy
            Qreg[h] = reg
            # Compute Q function values
            Q_sa = []
            for a in range(a_No):
                #X_a = np.hstack((phi0(s=hospitals[k]['states'][h]['H'],a=np.tile(a,n_k)),phi1(s=hospitals[k]['states'][h]['P'],a=np.tile(a,n_k))))
                X_a = phi1(s=hospitals[k]['states'][h]['P'],a=np.tile(a,n_k))
                Q_sa.append(list(reg.predict(X_a)))
            
            V_hat[h] =np.max(np.array(Q_sa).T,1)
            ##
            tmp = np.argmax(np.array(Q_sa).T,1)
            print(k,h,set(tmp)) 
            ##
        hospitals[k]['V_hats'] = V_hat
        hospitals[k]['Qregs'] = Qreg
    return hospitals

