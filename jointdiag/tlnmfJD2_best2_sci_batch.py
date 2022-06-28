# JD + NMF, no tracking of each step's performance, perform nbruns runs with random_seed
# + nbruns with cross init from TL-NMF
# Use JD-orthogonal-QN to learn a Phi, which will joint-diag all E(y_n y_n')
# then perform NMF

from time import time
from os import path, mkdir

import matplotlib as mpl
mpl.use('Agg')
FONT_SIZE = 20

import matplotlib.pyplot as plt
import numpy as np
import pickle

import soundfile as sf
from scipy import fftpack
import scipy.io as sio
import scipy.signal as sig
import scipy

from tlnmf.nmf import update_nmf_sparse
from tlnmf.utils import unitary_projection
from tlnmf.functions import is_div, new_is_div, is_div_eps
from tlnmf import tl_nmf_batch
from qndiag import qndiag2
from utils import getwin,  seperate_signal_from_WH_sci, hash_str2int2, eval_bss2 # , solve_NMF_eps
from utils import solve_NMF_me, solve_NMF_mm
from utils import signal_to_frames_win0
from utils import compute_jdloss, compute_tlnmfloss, compute_tlnmfloss_eps,\
                  compute_LS, convert_Ys_by_blocks, plot_atoms16, compute_losses, compute_LS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sn', '--signal_name', type = str, default = "nonstationary440_sim1")
parser.add_argument('-tn', '--test_signal_name', type = str, default = "test_nonstationary440_sim1")
parser.add_argument('-K', '--rank', type = int, default = 2)
parser.add_argument('-s', '--number_samples', type = int, default = 10)
parser.add_argument('-hla','--Hlambda', type = float, default = 0)
parser.add_argument('-ws', '--window_size', type = float, default = 40e-3)
parser.add_argument('-epsc', '--eps_c', type = float, default = 5e-7) # 1e-7
parser.add_argument('-epsnmf', '--eps_nmf', type = float, default = 1e-16)
parser.add_argument('-itl', '--iter_tl', type = int, default = 500) # max nb iter of JD
parser.add_argument('-pertl', '--iter_pertl', type = int, default = 1) # T_tl
parser.add_argument('-pernmf', '--iter_pernmf', type = int, default = 1) # T_nmf
parser.add_argument('-toltl', '--tol_tl', type = float, default = 1e-12)
parser.add_argument('-nbruns', '--num_runs', type = int, default = 10)
parser.add_argument('-nbtest', '--num_test', type = int, default = 10)
parser.add_argument('-win', '--window', type = int, default = 4) # win id for getwin
parser.add_argument('-inmf', '--iter_nmf', type = int, default = 50) # 500 # NMF for seperation
parser.add_argument('-cache', '--use_cache', type = int, default = 1) # to load saved results if 1
parser.add_argument('-loadphi','--load_phi', type = str, default = "")
parser.add_argument('-rseed', '--random_seed', type = int, default = 100) # seed for init of Phi,W,H
parser.add_argument('-nmfme','--nmf_me', type = int, default = 0) # NMF MM default

args = parser.parse_args()

np.seterr(under='warn')
    
# Read the song
sn = args.signal_name
tn = args.test_signal_name
mat_adr = 'datasets/' + sn + '.mat'
print('load training data at:',mat_adr)

mat_dic = sio.loadmat(mat_adr)
ys = mat_dic['ys']
fs = mat_dic['fs'][0]

test_adr = 'datasets/' + tn + '.mat'
test_dic = sio.loadmat(test_adr)
test_ys = test_dic['ys']
assert(fs == test_dic['fs'])

print('signals',ys.shape)
print('fs',fs)
L = ys.shape[0] # number of samples
s = args.number_samples
assert(L>=s)
L = s
print('number of samples s=',L)

ws = args.window_size # 40e-3
segsize = int(ws*fs)

K = args.rank
eps_c = args.eps_c
eps_nmf = args.eps_nmf # eps_c
itl = args.iter_tl # number of jd iters

assert(abs(eps_c - eps_nmf) < 1e-16)

assert args.Hlambda == 0

name = 'tlnmfJD2_sci_batch' + '_K' + str(K)  + '_S' + str(s) +\
       '_win' + str(args.window) + '_ws' + str(int(ws*1000)) + 'ms' +\
       '_epsnmf' + str(eps_nmf) + '_epsc' + str(eps_c)

nbrun = args.num_runs
rseed = args.random_seed
print('name=', name)
print('nbrun=', nbrun)
outfol = './results_jd_best2/' + sn + '_rseed' + str(rseed) + '_itl' +\
         str(args.iter_tl) + '_inmf' + str(args.iter_nmf) + '_nbrun' + str(nbrun)
if args.nmf_me:
    outfol += '_me'
else:
    outfol += '_mm'
    
loadphi = args.load_phi
if loadphi is not "":
    outfol = outfol + '_loadphi' + str(hash_str2int2(loadphi.encode('UTF-8')))
if not path.exists(outfol):
    mkdir(outfol)
print('outfol',outfol)

# prepare training data to short-time frames Ys
for l in range(L):
    signal = ys[l,:]

    win = getwin(args.window)
    if win is not None:
        iF,iT,X = sig.stft(signal,fs,window=win,nperseg=segsize,return_onesided=False) # ,boundary='even')
        Y = np.real(scipy.ifft(X,axis=0))
    else:
        Y = signal_to_frames_win0(signal,fs,ws)

    if l==0:
        M = Y.shape[0]
        N = Y.shape[1]
        F = M
        Ys = np.zeros((L,F,N))

    Ys[l,:,:] = Y # (M,N)

print('Ys',Ys.shape)
bYs,bL,bF,bN = Ys,L,F,N    

# prepare test data to short-time frames test_Ys
if 0:
    tL = args.num_test # test_ys.shape[0]
    test_Ys = np.zeros((tL,F,N))
    for l in range(tL):
        signal = test_ys[l,:]
        win = getwin(args.window)
        iF,iT,X = sig.stft(signal,fs,window=win,nperseg=segsize,return_onesided=False) # ,boundary='even')
        Y = np.real(scipy.ifft(X,axis=0))
        test_Ys[l,:,:] = Y # (M,N)
    
# esitmate Cov matrices C : (N,M,M) from Ys
estC = np.zeros((bN,M,M))
for n in range(bN):
    for l in range(bL):
        estC[n,:,:] = estC[n,:,:] + bYs[l,:,n].reshape((M,1)) @ bYs[l,:,n].reshape((1,M))
    estC[n,:,:] = estC[n,:,:] / bL

# esitmate Cov matrices C : (N,M,M) from test_Ys
if 0:
    test_estC = np.zeros((bN,M,M))
    for n in range(N):
        for l in range(tL):
            test_estC[n,:,:] = test_estC[n,:,:] +\
                               test_Ys[l,:,n].reshape((M,1)) @ test_Ys[l,:,n].reshape((1,M))
        test_estC[n,:,:] = test_estC[n,:,:] / tL

# add eps_c to estC
estC = estC + eps_c*np.eye(M)
print('estC',estC.shape)

if 0:
    for n in range(bN):
        if n == 10:
            plt.figure(figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
            plt.imshow(estC[n,:,:])
            plt.title('C[n,:,:] (n=' + str(n) + ')',size=FONT_SIZE*1.5)
            plt.savefig(outfol + '/figs/' + name + '_estC' + '_n' + str(n) + '.png')
            plt.show()

# init
#### GENERATE nbrun Phi0,W0,H0 with rseed
rng = np.random.RandomState(args.random_seed)
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

# RUN TL-NMF to obtain another nbrun init for JD+NMF
# define NMF params
assert( args.iter_nmf > 0 )
niter = args.iter_nmf
regul = args.Hlambda * float(K) / M
regul_type = 'sparse'
assert(regul==0)
for runid in range(nbrun):
    print('*****TL-NMF init******',runid)
    Phi, W, H, Phi_init, infos = tl_nmf_batch(Ys, K, verbose=False, rng=rng,\
                                              Phi = l_Phi0[runid], W = l_W0[runid],\
                                              H = l_H0[runid],\
                                              max_iter=args.iter_tl, n_iter_tl=args.iter_pertl, \
                                              n_iter_nmf = args.iter_pernmf, \
                                              tol=args.tol_tl, regul=regul, \
                                              eps_nmf=eps_nmf, eps_c = eps_c, nmfme=args.nmf_me)
    l_Phi0.append(Phi)
    l_W0.append(W)
    l_H0.append(H)
   
# JD
cache = args.use_cache  
Phi_best = None
best_lossL = None
best_jd_id = 0
alossL = np.zeros((2*nbrun))
if not path.exists(outfol + '/' + name + '.pkl') or cache==0:
    for runid in range(2*nbrun):
        Phi0 = l_Phi0[runid]
        t0 = time()
        if '.pkl' in loadphi:
            print('use loadphi init of Phi')
            ckpt = pickle.load( open(args.load_phi, "rb" ) )
            Phi_init = ckpt['Phi']
            if runid >= 1:
                break
        elif loadphi=='dct':
            print('use dct-ii init of Phi')
            Phi_init = fftpack.dct(np.eye(M), 3, norm='ortho') # DCT-II
            if runid >= 1:
                break
        else:
            print('use random init of Phi')            
            Phi_init = Phi0
        Phi, infos = qndiag2(estC,B0=Phi_init,cb_eval=None,max_iter=itl)
        LS = compute_LS(Phi,estC)
        alossL[runid] = LS
        fit_time = time() - t0
        print('fit time',fit_time)
        # save Phi,W,H,Phi_init,infos into pickle
        if best_lossL is None or alossL[runid] < best_lossL:
            best_lossL = alossL[runid]
            Phi_best = Phi.copy()
            best_jd_id = runid
            infos_best = infos
            
    ckpt = {}
    ckpt['alossL'] = alossL
    ckpt['Phi_best'] = Phi_best
    ckpt['best_jd_id'] = best_jd_id
    ckpt['infos_best'] = infos
    ckpt['best_lossL'] = best_lossL
    pickle.dump(ckpt, open(outfol + '/' + name + '.pkl', "wb"))
else:
    # load saved results
    ckpt = pickle.load( open(outfol + '/' + name + '.pkl', "rb" ) )
    Phi_best = ckpt['Phi_best']
    best_lossL = ckpt['best_lossL']    

# SOLVE NMF FROM JD solution with nbrun init
W_best = None
H_best = None
best_nmf_loss = None
best_nmf_id = 0
alossISNMF = np.zeros((2*nbrun))
alossC = np.zeros((2*nbrun))
V = np.mean(np.matmul(Phi_best, Ys) ** 2, axis=0)  # data spectrogram
#V_bar =  np.mean(np.matmul(Phi_best, test_Ys) ** 2, axis=0)  # data spectrogram
#D_best = is_div(V+eps_nmf,V_bar,0)

outfile = outfol + '/' + name + '_best.pkl'
if not path.exists(outfile) or cache==0:
    for runid in range(2*nbrun):
        print('*****NMF******',runid)
        # NMF
        W0 = l_W0[runid].copy()
        H0 = l_H0[runid].copy()
        if args.nmf_me == 1:
            W,H = solve_NMF_me(Phi_best,Ys,niter,eps_nmf,W0,H0)
        else:
            W,H = solve_NMF_mm(Phi_best,Ys,niter,eps_nmf,W0,H0) 
        #W,H = solve_NMF_eps(Phi_best,Ys,niter,eps_nmf,eps_c,W0,H0)
#        W,H = solve_NMF(Phi_best,Ys,niter,eps_nmf,W0,H0)
        V_hat = np.matmul(W, H) # final factorization            
        nmf_loss = is_div_eps(V,V_hat,eps_nmf,eps_c)
        # is_div(V,V_hat,eps_nmf)
        alossISNMF[runid] = nmf_loss
        alossC[runid] = compute_tlnmfloss_eps(V,V_hat,eps_nmf,eps_c)
        # compute_tlnmfloss(V,V_hat,eps_nmf)
        print('Cs',alossC[runid])
    
        if best_nmf_loss is None or alossISNMF[runid] < best_nmf_loss:
            best_nmf_loss = alossISNMF[runid]
            W_best = W.copy()
            H_best = H.copy()
            best_nmf_id = runid

    # save results
    outdata = {
        'alossC': alossC,
        'alossISNMF': alossISNMF,
#        'D_best': D_best, 
        'nbrun': nbrun,
        'rseed': rseed,
        'Phi_best': Phi_best,
        'W_best': W_best,
        'H_best': H_best,
        'Phi': Phi_best,
        'W': W_best,
        'H': H_best,        
        'best_nmf_loss': best_nmf_loss,
        'best_lossL': best_lossL,
        'best_nmf_id':best_nmf_id,
        'best_jd_id':best_jd_id,
    }
    
    pickle.dump(outdata, open(outfile, "wb"))
    print('write outdata to', outfile)
else:
    outdata = pickle.load( open(outfile, "rb" ) )
    Phi_best = outdata['Phi_best']
    W_best = outdata['W_best']
    H_best = outdata['H_best']        
