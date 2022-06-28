# JD+NMF avec budget sur #TL et #NMF

import argparse
import sys
import scipy
from scipy import fftpack
import numpy as np
from qndiag import qndiag2
from tlnmf.utils import unitary_projection
from utils import compute_jdloss, compute_tlnmfloss, compute_LS,\
    compute_A, compute_B,save_obj, get_values#, solve_NMF
from utils import solve_NMF_me, solve_NMF_mm

from tlnmf.functions import is_div
from tlnmf.transform_learning_gcm_newton import compute_loss
from tlnmf.nmf import update_nmf_sparse, update_nmf_smooth, update_nmf_sparse_nono

import matplotlib.pyplot as plt

sys.path.insert(1, './jointdiag')

parser = argparse.ArgumentParser()
parser.add_argument('-K', '--nmf_rank', type = int, default = 5)
parser.add_argument('-nbrun', '--num_runs', type = int, default = 10)
parser.add_argument('-S','--num_samples', type = int, default = 10)
parser.add_argument('-nbnmf','--num_nmf_runs', type = int, default = 10)
parser.add_argument('-rseed','--rand_seed', type = int, default = 100)
parser.add_argument('-barseed','--rand_barseed', type = int, default = 2020)
parser.add_argument('-initseed','--rand_initseed', type = int, default = 2021)
parser.add_argument('-runid','--run_id', type = int, default = -1)
parser.add_argument('-epsc', '--eps_c', type = float, default = 1e-16)
parser.add_argument('-nmfme','--nmf_me', type = int, default = 0) # NMF MM default

args = parser.parse_args()

K = args.nmf_rank # NMF rank
nbrun = args.num_runs # nb of random init
s = args.num_samples
rseed = args.rand_seed
barseed = args.rand_barseed
ratio = 10 # T_TL=1,T_NMF=ratio
this_runid = args.run_id

barK = 5
N, M = 50, 10
maxiter = 1000 # 200 # 100
nmfiter = maxiter
eps_nmf = args.eps_c # 1e-16
eps_S = eps_nmf # assume the same in the code

lamin = 1e-16
verb = False
nls = 30
gtol = 1e-8

#### GENERATE bar W , bar H with barseed
np.random.seed(barseed)

barPhi = fftpack.dct(np.eye(M), 3, norm='ortho') # = type 2 = transpose of type 3
barW = np.random.gamma(size=(M, barK),shape=1,scale=2) # + 1e-4
barH = np.random.gamma(size=(barK, N),shape=1,scale=2) # + 1e-4

print('min bar W bar H is', np.min(barW @ barH))

#### GENERATE Y with rseed
np.random.seed(rseed)
# sample s samples of Y from GCM
Ys = np.zeros((s,M,N))
barV = barW @ barH
for ids in range(s):
    for n in range(N):
        xn = np.zeros((M))
        for m in range(M):
            xn[m] = np.random.randn()*np.sqrt(barV[m,n]) # Normal(0,[bar W bar H]_mn)
            #print('xn[m]',xn[m])
        Ys[ids,:,n] = barPhi.T @ xn

# esitmate Cov matrices C : (N,M,M) from Ys
estC = np.zeros((N,M,M))
for n in range(N):
    for ids in range(s):
        estC[n,:,:] = estC[n,:,:] + Ys[ids,:,n].reshape((M,1)) @ Ys[ids,:,n].reshape((1,M))
    estC[n,:,:] = estC[n,:,:] / s
    estC[n,:,:] += np.eye(M)*eps_S

C = np.zeros((N,M,M)) # Sigma_n matrix
for n in range(N):
    C[n,:,:] = barPhi.transpose() @ np.diag(barV[:,n]) @ barPhi

#### GENERATE nbrun Phi0,W0,H0 with rseed
rng = np.random.RandomState(seed=args.rand_initseed) # =rseed)
l_Phi0 = []
l_W0 = []
l_H0 = []
for runid in range(nbrun):
    Phi0 = unitary_projection(rng.randn(M, M))
    W0 = np.abs(rng.randn(M, K)) + 1.
    W0 = W0 / np.sum(W0, axis=0)
    H0 = np.abs(rng.randn(K, N)) + 1.
    l_Phi0.append(Phi0)
    l_W0.append(W0)
    l_H0.append(H0)

np.seterr(under='warn')

alossL = np.zeros((nbrun))  # L_S''
alossISNMF = np.zeros((nbrun)) # I_S''
alossC = np.zeros((nbrun)) # C_S''
alossD = np.zeros((nbrun)) # D_S''
regul = 0

explossL = np.zeros((nbrun)) # L_inf''
alossA = np.zeros((nbrun)) # A_S''
alossB = np.zeros((nbrun)) # B_S''
blossA = np.zeros((nbrun)) # bar A_S
blossB = np.zeros((nbrun)) # bar B_S
bexplossL = np.zeros((nbrun)) # bar L_inf

# eval groundtruth value
barL = compute_LS(barPhi,C)
W0_eval = None
H0_eval = None
def cb_eval_Gs(Phi,W=None,H=None,ite=None):
    lower_Gs = compute_LS(Phi,estC)
    V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0) # + eps_S
    V_bar = (Phi@barPhi.T)**2 @ (barW @ barH)
    higher_Gs = lower_Gs + is_div(V + eps_S,V_bar,0)
    L = compute_LS(Phi,C)
    Cs = None
    if ite is None:
        if W is not None and H is not None:
            V_hat = np.dot(W,H)
            Cs = compute_tlnmfloss(V,V_hat,eps_nmf)
    else:
        # run NMF for R * ite ierations startring from W0_eval,H0_eval
        niter = ite*ratio
        #W,H = solve_NMF(Phi,Ys,niter,eps_nmf,W0_eval,H0_eval)
        if args.nmf_me == 1:
            W,H = solve_NMF_me(Phi,Ys,niter,eps_nmf,W0_eval,H0_eval)
        else:
            W,H = solve_NMF_mm(Phi,Ys,niter,eps_nmf,W0_eval,H0_eval)
        
        V_hat = np.dot(W,H)
        Cs = compute_tlnmfloss(V,V_hat,eps_nmf)
        #print('Cs',Cs,'ite',ite)
    #print('cb_eval_Gs gap',higher_Gs-lower_Gs)
    return dict(lower_Gs=lower_Gs, higher_Gs=higher_Gs, Cs=Cs, L=L, barL=barL)

print('s=',s)
outfol = './jdresults/ex17e_jdnmfb_finite/'
# './jdresults/ex17e' + '_Ttl' + str(args.iter_pertl) + '_Tnmf' + str(args.iter_pernmf)

nbnmf = args.num_nmf_runs # nb. nmf init / NOT nmf iterations, which is nmfiter
for runid in range(nbrun):
    if (this_runid == -1) or (this_runid == runid):
        print('*****BEGIN******')
        Phi_init = l_Phi0[runid] # unitary_projection(rng.randn(M, M))
        W0_eval = l_W0[runid]
        H0_eval = l_H0[runid]
        
        # JD-othrogonal-QN
        Phi, infos = qndiag2(estC,Phi_init,max_iter=maxiter,verbose=verb,lambda_min=lamin,\
                             max_ls_tries=nls,gtol=gtol,cb_eval=cb_eval_Gs)
        save_obj(infos,outfol + 'ratio' + str(ratio) + '_S' + str(s) +\
                 '_rseed' + str(rseed) + '_runid' + str(runid))
        
        print('****END*******')
