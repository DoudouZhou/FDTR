import numpy as np
from random import choices
from scipy.stats import multivariate_normal,uniform,mode
from tqdm import tqdm
import pandas as pd
import pickle

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def MDP(Hs_dim,Ps_dim,a_No,H,episodes_No,Hseed=1,policy=None,beta_hat_h=None,Lamda=None,beta=None,S_bound = 3,K=None):
    d0,d1 = phi0(s=np.zeros((1,Ps_dim)),a=[0]).shape[1],phi1(s=np.zeros((1,Ps_dim)),a=[0]).shape[1]
    #### Sample policy parameters
    ## Hopsital level parameters (shared by all hospitals)
    np.random.seed(116687)
    theta0 = np.random.rand(d0,H)#5*(np.random.rand(d0,H)-.5)

    ## Patient level parameters (specific to each hospital)
    np.random.seed(Hseed)
    theta_hk = np.random.rand(d1,H)
    # vectors for transition dirstibution
    mu_hk = np.random.rand(d1,H)

    # Initial state covs from hospital:
    Hs0 = np.random.normal(loc=0.0, scale=1.0, size=Hs_dim)
    Hs0 = np.tile(Hs0,(episodes_No,1))

    # Draw first states from MVN distribution and store in state dictionary
    Ps0 = np.random.rand(episodes_No,Ps_dim)
    # Concatenate hospital &  patient level covs. for first state
    states = {0:{'H':Hs0,'P':Ps0}}
    rewards,actions,phis_d =  {},{},{}
    for h in range(H):
        # choose an action given the state according to the input policy or at random
        if policy is None:
            actions[h] = choices(range(a_No), k=episodes_No)
        elif policy == 'alg1':
            actions[h],_ = fPEVI_hat(states,H,h,a_No,Lamda,beta_hat_h,beta,K,k=1,d1=d1)
        elif policy == 'alg2':
            actions[h],_ = dPEVI_hat(states,H,h,a_No,Lamda,beta_hat_h,beta,k=1,d1=d1)
        elif policy == 'alg2MV':
            actions[h] = dPEVI_MV(states,H,K,h,a_No,Lamda,beta_hat_h,beta,d1)
        else:
            actions[h] = policy(h,states)
        # Compute feature transformations phi1(s,a), phi2(s,a)
        phis_d[h] = {0:phi0(s=states[h]['H'],a=actions[h]),1:phi1(s=states[h]['P'],a=actions[h])}
        #Draw from reward function 
        rewards[h] = get_reward(phi=phis_d[h],th0=theta0[:,h],thK=theta_hk[:,h])
        rewards[h] /= max(rewards[h])
        # Draw next states
        states[h+1] = next_state(ph1=phis_d[h][1],s=states[h],mu_of_s=mu_hk[:,h],Hs_dim=Hs_dim,Ps_dim=Ps_dim,S_bound=S_bound)
    del states[H]
    V = np.sum([np.mean(rewards[h]) for h in range(H)])
    return(V,states,actions,rewards)

# 
# Function to compute next state given current state and action for episodes_No
def next_state(ph1,s,mu_of_s,Hs_dim,Ps_dim,S_bound):
    
    ### Will use importance sampling to draw from transition density kernel: s'~P(|s,a)=phi(s,a)^t mu(s')
    # Draw from multivariate normal proposal s'~q(s,a)
  
    next_s = np.random.rand(ph1.shape[0],Hs_dim+Ps_dim)
    
    # Compute proposal density q(s,a) for each s,a pair:
    q_proposal = np.mean(uniform.pdf(next_s),1)
    #q_proposal = multivariate_normal.pdf(next_s,cov=np.eye(ph1.shape[1]))
    
    # Compute target density P(s'|s,a) up to a constant (unnormalized)
    p_target = ph1.dot(mu_of_s)
#    p_target /= sum(p_target)
    # compute importance sampling weights (and  normalize):
    ISws = p_target/q_proposal
    #ISws /= sum(ISws)
    # Multiply samples by IS weights
    next_s *= ISws.reshape((len(ISws),1))

    outliers_of_S = abs(next_s)>S_bound
    next_s[outliers_of_S] = S_bound*np.sign(next_s[outliers_of_S])
    # return dict with hospital dimensions and patient dimensions
    # set hospital level covs to the mean of all patients
    next_s[:,:Hs_dim] = np.tile(np.mean(next_s[:,:Hs_dim],0),(next_s.shape[0],1))
    return {'H':next_s[:,:Hs_dim],'P':next_s[:,-Ps_dim:]}

# Function to draw rewards
def get_reward(phi,th0,thK):
    # Draw a reward for state and action pair according to s%*%beta_ah
    r_means = phi[0].dot(th0)+phi[1].dot(thK)
    r_raw = r_means + np.random.normal(loc=0.0, scale=1.0, size=len(r_means))
    r_clipped = r_raw#np.array([0 if r<0 else 1 if r>1 else r for r in r_raw])
    return r_clipped
    

# Hospital level feature map
def phi0(s,a,map=None):
    a = np.array(a).reshape((s.shape[0], 1))
    if map is None:# linear feature map
        return np.hstack((s,a*s))
def phi1(s,a,map=None):
    a = np.array(a).reshape((s.shape[0], 1))
    if map is None:# linear feature map
        return np.abs(np.hstack((s,a*s)))
##################################################################
################## Functions for Distributed RL ##################
##################################################################
from sklearn import linear_model

def estimate_local_Vs(hospitals,a_No):
    # Compute the value function within each hospital:
    for k in tqdm(range(len(hospitals))):
        n_k = len(hospitals[k]['actions'][0])
        H = len(hospitals[k]['actions'])
        V_hat,Qreg = {H:np.zeros(n_k)},{}
        for h in range(H-1,-1,-1):            
            # state
            X = np.hstack((phi0(s=hospitals[k]['states'][h]['H'],a=hospitals[k]['actions'][h]),phi1(s=hospitals[k]['states'][h]['P'],a=hospitals[k]['actions'][h])))
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
                X_a = np.hstack((phi0(s=hospitals[k]['states'][h]['H'],a=np.tile(a,n_k)),phi1(s=hospitals[k]['states'][h]['P'],a=np.tile(a,n_k))))
                Q_sa.append(list(reg.predict(X_a)))

            V_hat[h] =np.max(np.array(Q_sa).T,1)
        hospitals[k]['V_hats'] = V_hat
        hospitals[k]['Qregs'] = Qreg
    return hospitals


# Compute Zs as defined in paper
def compute_Zs(hospitals):
    K =  len(hospitals)
    H = len(hospitals[0]['actions'])
    for k in tqdm(range(K)):
        Zsh,Phis = {},{}
        for h in range(H):
            ph0 = phi0(s=hospitals[k]['states'][h]['H'],a=hospitals[k]['actions'][h])
            ph1 = phi1(s=hospitals[k]['states'][h]['P'],a=hospitals[k]['actions'][h])
            n,d1 = ph1.shape        
            Zsh[h] = np.hstack((ph0,np.zeros((n,k*d1)),ph1,np.zeros((n,(K-k-1)*d1))))
            Phis[h] = np.hstack((ph0,ph1))
        hospitals[k]['Zs'],hospitals[k]['Phis'] = Zsh,Phis        
    return hospitals



#########################
###### Algorithm 1 ######  
#########################
def dPEVI(hospitals,a_No,fed_Vhat,k,d0,d1):
    n = len(hospitals[0]['actions'][0])
    K = len(hospitals)
    H = len(hospitals[0]['actions'])
    N = n*K
    # xi =.99 or .95
    # c= 5,10,20,50
    lamda,c,xi,d = 1,.005,.99,d0+d1
    zeta= np.log(2*d*H*N/xi)
    beta = c*d*H*np.sqrt(K*zeta)
    # 
    Lamda = {h:lamda*np.eye((d0+K*d1)) for h in range(H)}
    ZY =  {h:np.zeros((d0+K*d1)) for h in range(H)}
    beta_hat = {}
    V_k = {H:np.zeros(n)}
    for h in range(H-1,-1,-1):
        Y = {}
        for j in range(K):
            # Compute ridge feature matrix: (line 4)
            Z_h = hospitals[j]['Zs'][h]
            Lamda[h] += Z_h.T.dot(Z_h)
            # Compute pseudo-outcomes with local value function estimates (line 7)
            Y[j] = hospitals[j]['rewards'][h] + fed_Vhat[j][h+1]#V_local[h+1]#hospitals[j]['V_hats'][h+1]
            ZY[h] += Z_h.T.dot(Y[j])
            
        # Estimate  linear regression parameters with Ridge reg.    
        beta_hat[h] = np.linalg.inv(Lamda[h]).T.dot(ZY[h])
        _,V_k[h] =  fPEVI_hat(hospitals[k]['states'],H,h,a_No,Lamda,beta_hat,beta,K,k,d1)
    return beta_hat,Lamda,beta

def fPEVI_hat(states,H,h,a_No,Lamda,beta_hat_h,beta,K,k,d1):
    n = states[h]['H'].shape[0]
    
    Qhat_sa  = []
    for a in range(a_No):
        ph0 = phi0(s=states[h]['H'],a=np.tile(a,n))
        ph1 = phi1(s=states[h]['P'],a=np.tile(a,n))
        Z_a = np.hstack((ph0,np.zeros((n,k*d1)),ph1,np.zeros((n,(K-k-1)*d1))))
        
        GammValues = [Gamma(Z=Z_a[i],L=Lamda[h],beta=beta) for i in range(n)]    
        Qbar_sa = [Qbar(Z=Z_a[i],betahat=beta_hat_h[h],GammVal=GammValues[i]) for i in range(n)]
        Qhat_sa.append([Qhat(QbarVal=Qbar_sa[i],H=H,h=h) for i in range(n)])
    pi_hat  = np.argmax(np.array(Qhat_sa),0)
    V_hat  = np.max(np.array(Qhat_sa),0)
    return pi_hat,V_hat

def Gamma(Z,L,beta):
    return beta*np.sqrt(Z.dot(np.linalg.inv(L)).dot(Z))

def Qbar(Z,betahat,GammVal):    
    return Z.dot(betahat)-GammVal

def Qhat(QbarVal,H,h):    
    val = min(QbarVal,H-h)# do not use +1 because h starts aat 0 not 1
    # apply ReLU function
    return max(val,0)



#########################
###### Algorithm 2 ######  
#########################
def dPEVI2(hospitals,a_No,k,d0,d1):
    K = len(hospitals)
    n_ks = {k:len(hospitals[k]['actions'][0]) for k in range(K)}
    H = len(hospitals[0]['actions'])
    # xi =.99 or .95
    # c= 5,10,20,50
    lamda,c,xi,d = 1,.005,.95,d0+d1
    zeta = {k:np.log(2*d*H*n_ks[k]/xi) for k in range(K)}
    beta = {k:c*d*H*np.sqrt(zeta[k]) for k in range(K)}

    fed_Vhat = {k:{H:np.zeros(n_ks[k])} for k in range(K)}

    #########
    for k in range(K):
        Lamda,Y,beta_hat_h = {},{},{}
        for h in range(H-1,-1,-1):
            Phi_sa = hospitals[k]['Phis'][h]
            Lamda[h] = Phi_sa.T.dot(Phi_sa)+lamda*np.eye(d0+d1)
            # Compute pseudo-outcomes with local value function estimates
            Y[h] = hospitals[k]['rewards'][h] + fed_Vhat[k][h+1]
            # Estimate  linear regression parameters with Ridge reg.    
            beta_hat_h[h] = np.linalg.inv(Lamda[h]).dot(Phi_sa.T.dot(Y[h]))
            _,fed_Vhat[k][h] = dPEVI_hat(hospitals[k]['states'],H,h,a_No,Lamda,beta_hat_h,beta,k,d1)
    return beta_hat_h,Lamda,beta,fed_Vhat
    ##
    

def dPEVI_hat(states,H,h,a_No,Lamda,beta_hat_h,beta,k,d1):
    n = states[h]['H'].shape[0]
    Qhat_sa  = []
    for a in range(a_No):
        ph0 = phi0(s=states[h]['H'],a=np.tile(a,n))
        ph1 = phi1(s=states[h]['P'],a=np.tile(a,n))
        Z_a = np.hstack((ph0,ph1))
        
        GammValues = [Gamma(Z=Z_a[i],L=Lamda[h],beta=beta[k]) for i in range(n)]
        Qbar_sa = [Qbar(Z=Z_a[i],betahat=beta_hat_h[h],GammVal=GammValues[i]) for i in range(n)]
        Qhat_sa.append([Qhat(QbarVal=Qbar_sa[i],H=H,h=h) for i in range(n)])
    pi_hat  = np.argmax(np.array(Qhat_sa),0)
    V_hat  = np.max(np.array(Qhat_sa),0)
    return pi_hat,V_hat



def dPEVI_MV(states,H,K,h,a_No,Lamda,beta_hat_h,beta,d1):
    a_star = []
    for k in range(K):
        a_local,_ = dPEVI_hat(states,H,h,a_No,Lamda,beta_hat_h,beta,k,d1)
        a_star.append(list(a_local))
    
    return mode(np.array(a_star))[0]

### Q-learning Algorithms

########
########
# This trains a single Q-function for all times 
def train_Qlearn0(hospitals,H,k):
    # Compute the value function within hospital k:
    Ys = np.hstack([hospitals[k]['V_hats'][h] for h in range(H)])
    Xs = np.vstack([np.hstack((phi0(s=hospitals[k]['states'][h]['H'],a=hospitals[k]['actions'][h]),phi1(s=hospitals[k]['states'][h]['P'],a=hospitals[k]['actions'][h]))).tolist() for h in range(H)])
    # linear Q-function fitting
    reg = linear_model.LinearRegression()
    reg.fit(Xs, Ys)
    return reg



####################################
####################################
####################################


def simulations(Hs_dim,Ps_dim,a_No,H,episodes_No,K,seed):
    np.random.seed(seed)
    d0,d1 = phi0(s=np.zeros((1,Ps_dim)),a=[0]).shape[1],phi1(s=np.zeros((1,Ps_dim)),a=[0]).shape[1]
    # Generate K datasets with random policy and store them in hospitals dictionary
    hospitals = {}
    print('\nGenerating datasets\n')
    for k in tqdm(range(K)):
        _,states,actions,rewards = MDP(Hs_dim,Ps_dim,a_No,H,episodes_No,Hseed=k,policy=None)
        
        hospitals[k] = {'states':states,'actions':actions,'rewards':rewards}
    # local value function estimates
    print('\nlocal value function estimates\n')
    hospitals = estimate_local_Vs(hospitals,a_No)
    # Compute Z vectors for algorithm
    print('\nCompute Z vectors for algorithm\n')
    hospitals = compute_Zs(hospitals)
    ###

    print('\nTraining policy with Algorithm 2\n')
    # Train policy with Algorithm 2
    beta_hat_h2,Lamda2,beta2,fed_Vhat = dPEVI2(hospitals,a_No,k,d0,d1)

    print('\nTraining policy with Algorithm 1\n')
    # Train policy with Algorithm 1
    beta_hat_h,Lamda,beta = dPEVI(hospitals,a_No,fed_Vhat,k,d0,d1)

    
    Qlearn0 = {k:train_Qlearn0(hospitals,H,k=k) for k in range(K)}

    # This uses the value functions and regressions already estimated and a Q-function for each time-step
    def local_Qlearn(h,states):
        k = 1#choices(range(len(hospitals)), k=1)[0]    
        Qfun = hospitals[k]['Qregs']
        n_k = len(states[h]['H'])
        # Compute Q function values 
        Q_sa = []
        for a in range(a_No):
            X_a = np.hstack((phi0(s=states[h]['H'],a=np.tile(a,n_k)),phi1(s=states[h]['P'],a=np.tile(a,n_k))))
            Q_sa.append(list(Qfun[h].predict(X_a)))
            
        return np.argmax(np.array(Q_sa),0)

    def single_Qlearn(h,states,Qlearn0=Qlearn0):
        k = 1#choices(range(len(hospitals)), k=1)[0]    
        n_k = len(states[h]['H'])
        # Compute Q function values 
        Q_sa = []
        for a in range(a_No):
            X_a = np.hstack((phi0(s=states[h]['H'],a=np.tile(a,n_k)),phi1(s=states[h]['P'],a=np.tile(a,n_k))))
            Q_sa.append(list(Qlearn0[k].predict(X_a)))
        return np.argmax(np.array(Q_sa),0)
    
    def single_Qlearn_votes(h,states,Qlearn0=Qlearn0):
        n_k = len(states[h]['H'])
        a_max  = []
        for k in range(K):
            # Compute Q function values 
            Q_sa = []
            for a in range(a_No):
                X_a = np.hstack((phi0(s=states[h]['H'],a=np.tile(a,n_k)),phi1(s=states[h]['P'],a=np.tile(a,n_k))))
                Q_sa.append(list(Qlearn0[k].predict(X_a)))
            a_max.append(list(np.argmax(np.array(Q_sa),0)))            
        return mode(np.array(a_max))[0]
    

    print('\nTesting policies\n')
    Vs_single_Q,Vs_single_Q_vts,Vs_Qlearn,Vs_dPEVI,Vs_dPEVI_single,Vs_dPEVI_MV = [],[],[],[],[],[]
    for k in tqdm(range(K)):
        V,_,_,_ = MDP(Hs_dim,Ps_dim,a_No,H,episodes_No,Hseed=k,policy=single_Qlearn)
        Vs_single_Q.append(V)
        V,_,_,_ = MDP(Hs_dim,Ps_dim,a_No,H,episodes_No,Hseed=k,policy=single_Qlearn_votes)
        Vs_single_Q_vts.append(V)
        V,_,_,_ = MDP(Hs_dim,Ps_dim,a_No,H,episodes_No,Hseed=k,policy=local_Qlearn)
        Vs_Qlearn.append(V)    
        V,_,_,_ = MDP(Hs_dim,Ps_dim,a_No,H,episodes_No,Hseed=k,policy='alg2',beta_hat_h=beta_hat_h2,Lamda=Lamda2,beta=beta2,K=K)
        Vs_dPEVI_single.append(V)    
        V,_,_,_ = MDP(Hs_dim,Ps_dim,a_No,H,episodes_No,Hseed=k,policy='alg2MV',beta_hat_h=beta_hat_h2,Lamda=Lamda2,beta=beta2,K=K)
        Vs_dPEVI_MV.append(V)    
        V,_,_,_ = MDP(Hs_dim,Ps_dim,a_No,H,episodes_No,Hseed=k,policy='alg1',beta_hat_h=beta_hat_h,Lamda=Lamda,beta=beta,K=K)
        Vs_dPEVI.append(V)    
        
    # save simulation results
    df = pd.DataFrame(data=np.array([Vs_single_Q,Vs_single_Q_vts,Vs_Qlearn,Vs_dPEVI_single,Vs_dPEVI_MV,Vs_dPEVI]).T, columns=[ 'Vs_single_Q','Vs_single_Q_vts','Vs_Qlearn','Vs_dPEVI_single','Vs_dPEVI_MV','Vs_dPEVI'])
    df.to_csv('../Results/sim_Results_Hs_dim_'+str(Hs_dim)+'_Ps_dim_'+str(Ps_dim)+'_a_No_'+str(a_No)+'_H_'+str(H)+'_episodes_No_'+str(episodes_No)+'_K_'+str(K)+'.csv', index=False)
    # save beta hat
    f = open('../Results/sim_betaHats_Hs_dim_'+str(Hs_dim)+'_Ps_dim_'+str(Ps_dim)+'_a_No_'+str(a_No)+'_H_'+str(H)+'_episodes_No_'+str(episodes_No)+'_K_'+str(K)+'.pkl',"wb")
    pickle.dump({'beta_hat_h':beta_hat_h},f)
    f.close()

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#########################
### Propensity Scores ###
#########################
# Numpy format data
###############
def train_IPWs(hospitals):
    log_regs = {}
    H = len(hospitals[0]['states'])
    for k in range(len(hospitals)):
        states,actions = hospitals[k]['states'],hospitals[k]['actions']
        logreg = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2',C=.01,multi_class='multinomial'))
        
        X,y = np.hstack((states[0]['H'],states[0]['P'])),np.ravel(actions[h])
        for h in range(1,H):
            X = np.vstack((X,np.hstack((states[h]['H'],states[h]['P']))))                
            y = np.hstack((y,np.ravel(actions[h])))
        pi = logreg.fit(X,)
        log_regs[k] = pi
    return log_regs

# Step importance sampling evaluation
def Vstep_IS(hospitals,k,a_No,H,log_regs,policy=None,beta_hat_h=None,Lamda=None,beta=None):
    states,actions,rewards = hospitals[k]['states'],hospitals[k]['actions'],hospitals[k]['rewards']
    # Select actions based on estimated policy
    mu_hat,prop_scores = {},{}
    for h in range(H):
        if policy == 'alg1':
            mu_hat[h],_ = fPEVI_hat(states,H,h,a_No,Lamda,beta_hat_h,beta,K,k,d1)
        elif policy == 'alg2':
            mu_hat[h],_ = dPEVI_hat(states,H,h,a_No,Lamda,beta_hat_h,beta,k,d1)
        elif policy == 'alg2MV':
            mu_hat[h] = dPEVI_MV(states,H,K,h,a_No,Lamda,beta_hat_h,beta,d1)[0]
        else:
            mu_hat[h] = policy(h,states)
        # 
        X = np.hstack((states[h]['H'],states[h]['P']))
        prop_scores[h] = log_regs[k].predict_proba(X)
    # Compute the value function using importance sampling
    V_mu = []
    for i in range(states[0]['H'].shape[0]):
        curr_V,rho = 0,1
        for h in range(H):
            # patient history
            a,r = actions[h][i],rewards[h][i]
            rho *= (a==mu_hat[h][i])/prop_scores[h][i][a]
            curr_V += rho*r
        V_mu.append(curr_V)
    return np.mean(V_mu), np.std(V_mu)

# Step importance sampling evaluation
def Vstep_WIS(hospitals,k,a_No,H,log_regs,policy=None,beta_hat_h=None,Lamda=None,beta=None):
    states,actions,rewards = hospitals[k]['states'],hospitals[k]['actions'],hospitals[k]['rewards']
    # Select actions based on estimated policy
    mu_hat,prop_scores = {},{}
    rho_prod = [[1]*states[0]['H'].shape[0]]
    for h in range(H):
        if policy == 'alg1':
            mu_hat[h],_ = fPEVI_hat(states,H,h,a_No,Lamda,beta_hat_h,beta,K,k,d1)
        elif policy == 'alg2':
            mu_hat[h],_ = dPEVI_hat(states,H,h,a_No,Lamda,beta_hat_h,beta,k,d1)
        elif policy == 'alg2MV':
            mu_hat[h] = dPEVI_MV(states,H,K,h,a_No,Lamda,beta_hat_h,beta,d1)[0]
        else:
            mu_hat[h] = policy(h,states)
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
            curr_V += rho_prod[h+1][i]*r/np.mean(rho_prod[h+1])
        V_mu.append(curr_V)
    return np.mean(V_mu), np.std(V_mu)

'''
with     theta_hk = 50*np.random.rand(d1,H) on line 19
# MDP parameters
Hs_dim,Ps_dim = 5,5 # state dimension
a_No = 2 # Number of actions
H = 5 # Episode length
episodes_No = int(1e5) 3,4,5 #  sample size
K=10

Hs_dim , Ps_dim , a_No , H , episodes_No , K 
'''


'''

# MDP parameters
Hs_dim,Ps_dim = 2,2 # state dimension
a_No = 6 # Number of actions
H = 15 # Episode length
episodes_No = int(1e3) #  sample size
K=10
'''
'''
df = pd.DataFrame(data=np.array([Vs_single_Q,Vs_Qlearn,Vs_dPEVI1,Vs_dPEVI2]).T, columns=[ 'Vs_single_Qlearn','Vs_Qlearn','Vs_dPEVI1','Vs_dPEVI2'])
df = pd.melt(df)
sns.barplot(x="variable", y="value", data=df)
'''



'''
# Generate K datasets with random policy and store them in hospitals dictionary
oracle_hospitals = {}
print('\nComputing the oracle theta0: generating datasets\n')
for k in tqdm(range(K)):
    _,oracle_s,oracle_a,oracle_r = MDP(Hs_dim,Ps_dim,a_No,H,episodes_No=int(1e6),Hseed=k,policy=None)    
    oracle_hospitals[k] = {'states':oracle_s,'actions':oracle_a,'rewards':oracle_r}


# Compute Z vectors for algorithm
print('\nComputing the oracle theta0: compute Z vectors for algorithm\n')
oracle_hospitals = compute_Zs(oracle_hospitals)
###

print('\nnComputing the oracle theta0: training policy with Algorithm 2\n')
# Train policy with Algorithm 2
beta_hat_h2,Lamda2,beta2,fed_Vhat = dPEVI2(oracle_hospitals,a_No,k,d0,d1)

print('\nTraining policy with Algorithm 1\n')
# Train policy with Algorithm 1
beta_hat_h,Lamda,beta = dPEVI(oracle_hospitals,fed_Vhat,k,d0,d1)
'''

'''
a=pd.read_csv('../Results/sim_Results_Hs_dim_'+str(Hs_dim)+'_Ps_dim_'+str(Ps_dim)+'_a_No_'+str(a_No)+'_H_'+str(H)+'_episodes_No_'+str(episodes_No)+'_K_'+str(K)+'.csv')

file_to_read = open('../Results/sim_betaHats_Hs_dim_'+str(Hs_dim)+'_Ps_dim_'+str(Ps_dim)+'_a_No_'+str(a_No)+'_H_'+str(H)+'_episodes_No_'+str(episodes_No)+'_K_'+str(K)+'.pkl', "rb")

loaded_dictionary = pickle.load(file_to_read)
'''
