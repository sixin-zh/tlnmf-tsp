# TL-NMF with cross INIT from JD+NMF

import argparse
import numpy as np
from os import path, mkdir
from time import time

from scipy import fftpack
from tlnmf import tl_nmf_batch
import matplotlib.pyplot as plt
from tlnmf.utils import unitary_projection
from utils import compute_jdloss, compute_tlnmfloss, compute_LS, compute_A, compute_B, save_obj, get_values # , solve_NMF
from utils import solve_NMF_me, solve_NMF_mm
from tlnmf.functions import is_div
#from tlnmf.transform_learning_gcm_newton import compute_loss, compute_V
#from tlnmf.nmf import update_nmf_sparse, update_nmf_smooth, update_nmf_sparse_nono
from qndiag import qndiag2

parser = argparse.ArgumentParser()
parser.add_argument('-K', '--nmf_rank', type = int, default = 5)
parser.add_argument('-nbrun', '--num_runs', type = int, default = 10)
parser.add_argument('-S','--num_samples', type = int, default = 10)
parser.add_argument('-rseed','--rand_seed', type = int, default = 100)
parser.add_argument('-barseed','--rand_barseed', type = int, default = 2020)
parser.add_argument('-initseed','--rand_initseed', type = int, default = 2021)
parser.add_argument('-pertl', '--iter_pertl', type = int, default = 1) # T_tl
parser.add_argument('-pernmf', '--iter_pernmf', type = int, default = 1) # T_nmf
parser.add_argument('-nbnmf','--num_nmf_runs', type = int, default = 0) # last stage NMF
parser.add_argument('-track','--to_track', type = int, default = 0) # to track
parser.add_argument('-epsc', '--eps_c', type = float, default = 1e-16)
parser.add_argument('-nmfme','--nmf_me', type = int, default = 0) # NMF MM default

args = parser.parse_args()

K = args.nmf_rank # NMF rank
nbrun = args.num_runs # nb of random init
s = args.num_samples
rseed = args.rand_seed
barseed = args.rand_barseed
to_track = args.to_track

#nmf_pertl = 1 # 100 # 1, with 100 seems faster

barK = 5
N = 50
M = 10
eps_nmf = args.eps_c #  1e-16
eps_S = eps_nmf # assume the same in the code

#maxiter = 1000
#nmfiter = maxiter
iter_tl = 1000
maxiter = iter_tl
nmfiter = maxiter # 1:1
#iter_pertl = 1
tol_tl = 1e-12
#assert(iter_pertl == 1)

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
rng = np.random.RandomState(seed=args.rand_initseed) # rseed)
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

# run JD+NMF to init TL-NMF
for runid in range(nbrun):
    print('*****JD init******',runid)
    Phi_init = l_Phi0[runid] # unitary_projection(rng.randn(M, M))
    Phi, infos = qndiag2(estC,Phi_init,max_iter=maxiter,verbose=verb,lambda_min=lamin,\
                         max_ls_tries=nls,gtol=gtol)
    W0 = l_W0[runid].copy()
    H0 = l_H0[runid].copy()
    if args.nmf_me == 1:
        W,H = solve_NMF_me(Phi,Ys,nmfiter,eps_nmf,W0,H0)
    else:
        W,H = solve_NMF_mm(Phi,Ys,nmfiter,eps_nmf,W0,H0)
    #W,H = solve_NMF(Phi,Ys,nmfiter,eps_nmf,W0,H0)
    l_Phi0.append(Phi)
    l_W0.append(W)
    l_H0.append(H)

alossL = np.zeros((2*nbrun)) # L_S'
#alossLideal = np.zeros((nbrun))
alossC = np.zeros((2*nbrun)) # C_S'
#alossCideal = np.zeros((nbrun))
alossISNMF = np.zeros((2*nbrun)) # I_S'
#alossISNMFideal = np.zeros((nbrun))
explossL = np.zeros((2*nbrun)) # L_inf'
alossD = np.zeros((2*nbrun)) # D_S'
bexplossL = np.zeros((2*nbrun)) # bar L_inf

regul = 0

print('s=',s)
nbnmf = args.num_nmf_runs

if K == barK:
    outfol = './jdresults/ex17e' + '_Ttl' + str(args.iter_pertl) + '_Tnmf' + str(args.iter_pernmf)
else:
    outfol = './jdresults/ex17e' + '_Ttl' + str(args.iter_pertl) + '_Tnmf' + str(args.iter_pernmf) + '_K' + str(K)
    
if not path.exists(outfol):
    mkdir(outfol)

print('out to',outfol)

#if to_track==1:
#    def cb_eval_Gs(Phi,W,H):
#        lower_Gs = compute_LS(Phi,estC)
#        V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)
#        V_bar = (Phi@barPhi.T)**2 @ (barW @ barH)
#        higher_Gs = lower_Gs + is_div(V,V_bar,eps_S)
#        L = compute_LS(Phi,C)
#        #print('cb_eval_Gs gap',higher_Gs-lower_Gs)
#        V_hat = np.dot(W, H)
#        Cs = compute_tlnmfloss(V,V_hat,eps_nmf)
#        return dict(lower_Gs=lower_Gs, higher_Gs=higher_Gs, Cs = Cs, L=L)
#else:
cb_eval_Gs = None
    
for runid in range(2*nbrun):
    t0 = time()
    Phi, W, H, Phi_init, infos = tl_nmf_batch(Ys, K, verbose=False, rng=rng,\
                                              Phi = l_Phi0[runid], W = l_W0[runid], H = l_H0[runid], \
                                              max_iter=iter_tl, n_iter_tl=args.iter_pertl, \
                                              n_iter_nmf = args.iter_pernmf, \
                                              tol=tol_tl, regul=0, eps_nmf=eps_nmf, cb_eval=cb_eval_Gs, nmfme=args.nmf_me)
    fit_time = time() - t0
    #if to_track == 1:
    #    outname = '/ex17e_tlnmf2_best2_finite_S' + str(s) + '_rseed' + str(rseed) + '_runid' + str(runid)
    #    save_obj(infos, outfol + outname)

    V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)  # data spectrogram
    V_hat = np.dot(W, H)
    alossISNMF[runid] = is_div(V,V_hat,eps_nmf)
    
    #V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)  # final spectrogram
    #V_hat = W @ H
    alossL[runid] = compute_LS(Phi,estC)
    alossC[runid] = compute_tlnmfloss(V,V_hat,eps_nmf)
    V_bar = (Phi@barPhi.T)**2 @ (barW @ barH)
    alossD[runid] = is_div(V+eps_S,V_bar,0)

    print('tlnmf runid',runid,'L_S\'',alossL[runid])
    print('tlnmf runid',runid,'C_S\'',alossC[runid])

    explossL[runid] = compute_LS(Phi,C) # / N
    bexplossL[runid] = compute_LS(barPhi,C) # / N
    print('runid',runid,'L_inf^DOT-bar L_inf',explossL[runid]-bexplossL[runid])

outdata = {
    'nbrun': nbrun,
    'rseed': rseed,
    'alossL': alossL,
    'alossISNMF': alossISNMF,
    'alossC': alossC,
    'alossD': alossD,            
    'explossL': explossL,
    'bexplossL':bexplossL,
}

if to_track == 1:
    outname = '/ex17e_tlnmf2_best2_finite_S' + str(s) + '_rseed' + str(rseed) + '_epsc' + str(eps_nmf)
    save_obj(outdata, outfol + outname)
