import argparse
import numpy as np
from os import path, mkdir
from time import time

from scipy import fftpack
from tlnmf import tl_nmf_batch
import matplotlib.pyplot as plt
from tlnmf.utils import unitary_projection
from utils import compute_jdloss, compute_tlnmfloss, compute_LS, compute_A, compute_B, save_obj, get_values
from tlnmf.functions import is_div
from tlnmf.transform_learning_gcm_newton import compute_loss, compute_V
from tlnmf.nmf import update_nmf_sparse, update_nmf_smooth, update_nmf_sparse_nono

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
#iter_pertl = 1
tol_tl = 1e-12
#assert(iter_pertl == 1)

#### GENERATE bar W , bar H with barseed
np.random.seed(barseed)

barPhi = fftpack.dct(np.eye(M), 3, norm='ortho') # = type 2 = transpose of type 3
barW = np.random.gamma(size=(M, barK),shape=1,scale=2) # + 1e-4
barH = np.random.gamma(size=(barK, N),shape=1,scale=2) # + 1e-4

#plt.imshow(barW)
#plt.title('barW')
#plt.show()

#plt.imshow(barH)
#plt.title('barH')
#plt.show()

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

alossL = np.zeros((nbrun)) # L_S'
#alossLideal = np.zeros((nbrun))
alossC = np.zeros((nbrun)) # C_S'
#alossCideal = np.zeros((nbrun))
alossISNMF = np.zeros((nbrun)) # I_S'
#alossISNMFideal = np.zeros((nbrun))
explossL = np.zeros((nbrun)) # L_inf'
alossD = np.zeros((nbrun)) # D_S'

alossA = np.zeros((nbrun)) # A_S'
alossB = np.zeros((nbrun)) # B_S'
blossA = np.zeros((nbrun)) # bar A_S
blossB = np.zeros((nbrun)) # bar B_S
bexplossL = np.zeros((nbrun)) # bar L_inf

regul = 0

print('s=',s)
nbnmf = args.num_nmf_runs

outfol = './jdresults/ex17e' + '_Ttl' + str(args.iter_pertl) + '_Tnmf' + str(args.iter_pernmf)

# sn + '_run' + str(runid) + '_Ttl' + str(args.iter_pertl) + '_Tnmf' + str(args.iter_pernmf)
if not path.exists(outfol):
    mkdir(outfol)

if to_track==1:
    def cb_eval_Gs(Phi,W,H):
        lower_Gs = compute_LS(Phi,estC)
        V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)
        V_bar = (Phi@barPhi.T)**2 @ (barW @ barH)
        higher_Gs = lower_Gs + is_div(V,V_bar,eps_S)
        L = compute_LS(Phi,C)
        #print('cb_eval_Gs gap',higher_Gs-lower_Gs)
        V_hat = np.dot(W, H)
        Cs = compute_tlnmfloss(V,V_hat,eps_nmf)
        return dict(lower_Gs=lower_Gs, higher_Gs=higher_Gs, Cs = Cs, L=L)
else:
    cb_eval_Gs = None
    
for runid in range(nbrun):
    t0 = time()
    Phi, W, H, Phi_init, infos = tl_nmf_batch(Ys, K, verbose=False, rng=rng,\
                                              Phi = l_Phi0[runid], W = l_W0[runid], H = l_H0[runid], \
                                              max_iter=iter_tl, n_iter_tl=args.iter_pertl, \
                                              n_iter_nmf = args.iter_pernmf, \
                                              tol=tol_tl, regul=0, eps_nmf=eps_nmf, cb_eval=cb_eval_Gs, nmfme=args.nmf_me)
    fit_time = time() - t0
    if to_track == 1:
        outname = '/ex17e_tlnmf2_finite_S' + str(s) + '_rseed' + str(rseed) + '_runid' + str(runid)
        save_obj(infos, outfol + outname)

    if 0:
        # plot time objectives
        plt.figure()
        obj_list = infos['obj_list']
        t = np.linspace(0, fit_time, len(obj_list))
        plt.plot(t, obj_list)
        plt.xlabel('Time (sec.)')
        plt.ylabel('Objective function')
        plt.savefig(outfol + outname + '_timeobj.png')
        plt.show()
             #  + '_nmf' + str(nmf_pertl))
             
    if 0:       
        plt.plot(infos['gs_obj_iter'],get_values(infos['gs_obj_dict'],'lower_Gs'),'-')
        plt.plot(infos['gs_obj_iter'],get_values(infos['gs_obj_dict'],'higher_Gs'),'--')
#        plt.plot(infos['gs_obj_iter'],get_values(infos['gs_obj_dict'],'L'),':')
        plt.legend(['$L_S$','$L_S+D_S$']) # ,'$L_{\infty}$'])
    
        plt.title('TL-NMF (S=%d)' % (s))
        plt.xlabel('nb. updates of $\Phi$')
        #plt.ylim([1800,1900])
        #plt.xlim([0,100])
        plt.grid('on')
        plt.savefig(outfol + outname)
        plt.show()

    # re-compute NMF
    if 0:
        V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)  # data spectrogram
        alossISNMF[runid] = -1
        V_hat = None
        for idn in range(nbnmf):
            W = np.abs(rng.randn(M, K)) + 1.
            W = W / np.sum(W, axis=0)
            H = np.abs(rng.randn(K, N)) + 1.
            V_hat_ = np.dot(W, H) # + eps  # Initial factorization
            for ite in range(nmfiter):
                W, H = update_nmf_sparse(V, W, H, V_hat_, regul, eps=eps_nmf) # sparse
                V_hat_ = np.dot(W, H)  # Initial factorization
            #V_hat = np.matmul(W, H) # final factorization            
            nmf_loss = is_div(V,V_hat_,eps_nmf)
            if idn == 0 or nmf_loss < alossISNMF[runid]:
                alossISNMF[runid] = nmf_loss
                V_hat = V_hat_
    else:
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

#    alossC[runid] = compute_tlnmfloss(V,V_hat,eps_nmf)
#    alossISNMF[runid] = is_div(V,V_hat,eps_nmf)

    if 0:
        V_bar = (Phi@barPhi.T)**2 @ (barW @ barH)
        alossA[runid] = compute_A(V,V_bar,eps_nmf)
        alossB[runid] = compute_B(V,V_bar,eps_nmf)
        bV = np.mean(np.matmul(barPhi, Ys) ** 2, axis=0) 
        bV_bar = (barW @ barH)
        blossA[runid] = compute_A(bV,bV_bar,eps_nmf)
        blossB[runid] = compute_B(bV,bV_bar,eps_nmf)
    
        #print('runid',runid,'loss L=',alossL[runid])
        print('runid',runid,'A_S\'-bar A_S',alossA[runid]-blossA[runid])
        print('runid',runid,'L_inf\'-bar L_inf',explossL[runid]-bexplossL[runid])
        #    print('runid',runid,'B_S\'-bar B_S',alossB[runid]-blossB[runid])

#print('loss C = %.3e (%.3e)' % (np.min(alossC), np.max(alossC)))
#print('loss L = %.3e (%.3e)' % (np.min(alossL), np.max(alossL)))

outdata = {
    'nbrun': nbrun,
    'rseed': rseed,
    'alossL': alossL,
    'alossISNMF': alossISNMF,
    'alossC': alossC,
    'alossD': alossD,            
    'explossL': explossL,
#    'aPhi': Phi,
#    'alossA':alossA,
#    'alossB':alossB,
#    'blossA':blossA,
#    'blossB':blossB,
    'bexplossL':bexplossL,
}

if to_track == 1:
    outname = '/ex17e_tlnmf2_finite_S' + str(s) + '_rseed' + str(rseed) + '_epsc' + str(eps_nmf)
    save_obj(outdata, outfol + outname)
