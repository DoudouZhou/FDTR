from utils import *
import os, sys

Hs_dim , Ps_dim , a_No , H , episodes_No , K =  int(sys.argv[2]),  int(sys.argv[4]), int(sys.argv[6]), int(sys.argv[8]), int(sys.argv[10]), int(sys.argv[12])
print('Hs_dim_'+str(Hs_dim)+'_Ps_dim_'+str(Ps_dim)+'_a_No_'+str(a_No)+'_H_'+str(H)+'_episodes_No_'+str(episodes_No)+'_K_'+str(K))
simulations(Hs_dim,Ps_dim,a_No,H,episodes_No,K,1)