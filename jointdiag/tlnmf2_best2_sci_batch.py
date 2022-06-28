# Use TL-NMF to learn Phi,W,H from S samples of Y
# + nbruns with cross init from JD+NMF
# perform nbruns runs with random_seed, NOT runid=random_seed + 0..nbruns-1

from time import time
from os import path, mkdir

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
FONT_SIZE = 20

from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import pickle

import soundfile as sf
from scipy import fftpack
import scipy.io as sio
import scipy.signal as sig
import scipy

#from tlnmf.nmf import update_nmf_sparse, update_nmf_eps
from tlnmf.functions import new_is_div, penalty, is_div
from tlnmf import tl_nmf_batch
from qndiag import qndiag2
from utils import getwin, seperate_signal_from_WH_sci, hash_str2int2, eval_bss2 # , solve_NMF_eps
from utils import solve_NMF_me, solve_NMF_mm
from utils import signal_to_frames_win0
from tlnmf.utils import unitary_projection
from utils import compute_jdloss, compute_tlnmfloss, compute_LS,\
    plot_atoms16, compute_losses,\
    compute_losses_eps, compute_tlnmfloss_eps

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sn', '--signal_name', type = str, default = "nonstationary440_sim1")
parser.add_argument('-tn', '--test_name', type = str, default = "nonstationary440_sim1") # same for train, or use test_nonstationary440_sim1
parser.add_argument('-s', '--number_samples', type = int, default = 10)
parser.add_argument('-t', '--test_samples', type = int, default = 10)
parser.add_argument('-K', '--rank', type = int, default = 2)
parser.add_argument('-hla','--Hlambda', type = float, default = 0)
parser.add_argument('-ws', '--window_size', type = float, default = 40e-3)
parser.add_argument('-epsc', '--eps_c', type = float, default = 1e-11) # eps_S
parser.add_argument('-epsnmf', '--eps_nmf', type = float, default = 5e-7) # eps_0
parser.add_argument('-itl', '--iter_tl', type = int, default = 500) # T
parser.add_argument('-pertl', '--iter_pertl', type = int, default = 1) # T_tl , 5
parser.add_argument('-pernmf', '--iter_pernmf', type = int, default = 1) # T_nmf
parser.add_argument('-toltl', '--tol_tl', type = float, default = 1e-12)
#parser.add_argument('-runid', '--run_id', type = int, default = 1)
parser.add_argument('-nbruns', '--num_runs', type = int, default = 10)
parser.add_argument('-win', '--window', type = int, default = 4) # win id for getwin
#parser.add_argument('-inmf', '--iter_nmf', type = int, default = 50) # 500 , last-stage NMF for seperation
parser.add_argument('-cache', '--use_cache', type = int, default = 1) # to load saved results if 1
parser.add_argument('-loadphi','--load_phi', type = str, default = "")
parser.add_argument('-loadWH','--load_WH', type = str, default = "")
parser.add_argument('-rseed', '--random_seed', type = int, default = 100) # seed for init of Phi,W,H
parser.add_argument('-nmfme','--nmf_me', type = int, default = 0) # NMF MM default

args = parser.parse_args()

np.seterr(under='warn')

# Read training data
sn = args.signal_name
mat_adr = 'datasets/' + sn + '.mat'
print('load mat data at:',mat_adr)
mat_dic = sio.loadmat(mat_adr)
ys = mat_dic['ys']
fs = mat_dic['fs'][0][0]
print('signals',ys.shape)
print('fs',fs)
L = ys.shape[0] # number of traning samples
s = args.number_samples
assert(L>=s)
L = s # reset number of traning samples
print('number of samples s=',L)

# read test data
tn = args.test_name # number of test samples
mat_adr = 'datasets/' + tn + '.mat'
mat_dic = sio.loadmat(mat_adr)
test_ys = mat_dic['ys']
test_fs = mat_dic['fs'][0]
assert(test_fs == fs)
nb_test = args.test_samples

ws = args.window_size # 40e-3
segsize = int(ws*fs)

K = args.rank
eps_c = args.eps_c # c
eps_nmf = args.eps_nmf # _nmf

assert(abs(eps_c - eps_nmf) < 1e-16)

if args.Hlambda == 0:
    name = 'tlnmf2_sci_batch' + '_K' + str(K) + '_S' + str(L) + '_win' + str(args.window)
#    name = 'tlnmf2_sci_batch_' + sn + '_K' + str(K) + '_S' + str(L) + '_win' +\
#           str(args.window) + '_ws' + str(int(ws*1000))  + 'ms_epsc' + str(eps_c) + 'epsnmf' + str(eps_nmf)
else:
    assert(0)

#runid = args.run_id
nbrun = args.num_runs

FOL = './results_' + sn + '/'
print('name=', name)
#print('runid=', runid)
print('nbrun=', nbrun)
outfol = 'tlnmf2_best2' + '_itl' + str(args.iter_tl) + '_Ttl' + str(args.iter_pertl) + '_Tnmf' + str(args.iter_pernmf) +\
         '_epsnmf' + str(eps_nmf) + '_epsc' + str(eps_c) + '_ws' + str(int(ws*1000))  + 'ms'
if args.nmf_me:
    outfol += '_me'
else:
    outfol += '_mm'

loadphi = args.load_phi
if loadphi is not "":
    outfol = outfol + '_loadphi' + str(hash_str2int2(loadphi.encode('UTF-8')))
loadWH = args.load_WH
if loadWH is not "":
    outfol = outfol + '_loadWH' + str(hash_str2int2(loadWH.encode('UTF-8')))
outfol = outfol + '_nbrun' + str(nbrun) # '_run' + str(runid)

if not path.exists(FOL + outfol):
    mkdir(FOL + outfol)
print('output to', FOL + outfol)

# test data
'''
for l in range(nb_test):
    signal = test_ys[l,:]
    win = getwin(args.window)
    if win is not None:
        iF,iT,X = sig.stft(signal,fs,window=win,nperseg=segsize,return_onesided=False) # ,boundary='even')
        Y = np.real(scipy.ifft(X,axis=0))
    else:
        # use non-overlap, rect win (no border effect)
        Y = signal_to_frames_win0(signal,fs,ws)

    if l == 0:
        M = Y.shape[0]
        N = Y.shape[1]
        F = M        
        test_Ys = np.zeros((nb_test,F,N))
    test_Ys[l,:,:] = Y
'''

# train data
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

        if 1:
            plt.imshow(Y)
            plt.show()
            
        if 0:
            # plot y(t)
            plt.figure(figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(signal)
            plt.xlabel('t',size=FONT_SIZE)
            ax = plt.gca()
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(FONT_SIZE*1.2)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(FONT_SIZE*1.2)
            plt.savefig(FOL + outfol + '/' + name + '_signal.eps')
            plt.show()
        
    Ys[l,:,:] = Y

# esitmate Cov matrices C : (N,M,M) from Ys
estC = np.zeros((N,M,M))
for n in range(N):
    for ids in range(L): # L is s
        estC[n,:,:] = estC[n,:,:] + Ys[ids,:,n].reshape((M,1)) @ Ys[ids,:,n].reshape((1,M))
    estC[n,:,:] = estC[n,:,:] / L
estC += eps_c*np.eye(M)

print('Ys',Ys.shape)
regul = args.Hlambda * float(K) / M
print('regul',regul)

# init
l_Phi0 = []
l_W0 = []
l_H0 = []
rng = np.random.RandomState(args.random_seed)
for runid in range(nbrun):
    if loadphi=='dct':
        print('use dct-ii init of Phi')
        Phi0 = fftpack.dct(np.eye(M), 3, norm='ortho') # DCT-II
    else:
        print('use random init of Phi')
        Phi0 = unitary_projection(rng.randn(M, M))
    W0 = np.abs(rng.randn(M, K)) + 1.
    W0 = W0 / np.sum(W0, axis=0)
    H0 = np.abs(rng.randn(K, N)) + 1.
    l_Phi0.append(Phi0)
    l_W0.append(W0)
    l_H0.append(H0)

# run JD+NMF to init TL-NMF
# define NMF params
assert( args.iter_pertl == 1)
nmfiter = args.iter_tl * args.iter_pernmf
regul = args.Hlambda * float(K) / M
regul_type = 'sparse'
assert(regul==0)
for runid in range(nbrun):
    print('*****JD init******',runid)
    Phi_init = l_Phi0[runid] # unitary_projection(rng.randn(M, M))
    Phi, infos = qndiag2(estC,B0=Phi_init,max_iter=args.iter_tl)
    W0 = l_W0[runid].copy()
    H0 = l_H0[runid].copy()
    #W,H = solve_NMF(Phi,Ys,nmfiter,eps_nmf,W0,H0)
    if args.nmf_me == 1:
        W,H = solve_NMF_me(Phi,Ys,nmfiter,eps_nmf,W0,H0)
    else:
        W,H = solve_NMF_mm(Phi,Ys,nmfiter,eps_nmf,W0,H0)   
    #W,H = solve_NMF_eps(Phi,Ys,nmfiter,eps_nmf,eps_c,W0,H0)
    l_Phi0.append(Phi)
    l_W0.append(W)
    l_H0.append(H)
    
cache = args.use_cache

Phi_best = None
W_best = None
H_best = None
lossC_best = None
lossL_best = None
lossI_best = None
infos_best = None
fit_time_best = None
alossL = np.zeros((2*nbrun))
alossISNMF = np.zeros((2*nbrun))
alossC = np.zeros((2*nbrun))
if not path.exists(FOL + outfol + '/' + name + '.pkl') or cache==0:
    cb_eval_Gs = None
    for runid in range(2*nbrun):
        t0 = time()
        Phi0 = l_Phi0[runid]
        W0 = l_W0[runid]
        H0 = l_H0[runid]
        Phi, W, H, Phi_init, infos = tl_nmf_batch(Ys, K, verbose=True, rng=rng, Phi=Phi0, W=W0, H=H0,\
                                                  max_iter=args.iter_tl, n_iter_tl=args.iter_pertl,\
                                                  n_iter_nmf=args.iter_pernmf,\
                                                  tol=args.tol_tl, regul=regul,\
                                                  eps_nmf=eps_nmf, eps_c = eps_c,\
                                                  cb_eval=cb_eval_Gs, nmfme=args.nmf_me)
        fit_time = time() - t0
        print('fit time',fit_time)
        
        #Cs0,Ls0,Is0 = compute_losses(Phi,W,H,estC,Ys,eps_nmf)
        Cs0,Ls0,Is0 = compute_losses_eps(Phi,W,H,estC,Ys,eps_nmf,eps_c)
        alossL[runid] = Ls0
        alossC[runid] = Cs0
        alossISNMF[runid] = Is0
        if lossC_best is None or lossC_best > Cs0:
            lossC_best = Cs0
            lossL_best = Ls0
            lossI_best = Is0
            Phi_best = Phi
            W_best = W
            H_best = H
            infos_best = infos
            fit_time_best = fit_time

    # keep only the best
    Cs0 = lossC_best
    Ls0 = lossL_best
    Is0 = lossI_best
    Phi = Phi_best
    W = W_best
    H = H_best
    fit_time = fit_time_best
    # save Phi,W,H,Phi_init,infos into pickle
    ckpt = {}
    ckpt['Phi'] = Phi
    ckpt['W'] = W
    ckpt['H'] = H
    #ckpt['Phi_init'] = Phi_init
    ckpt['infos'] = infos_best
    ckpt['fit_time'] = fit_time
    ckpt['Cs'] = Cs0
    ckpt['Ls'] = Ls0
    ckpt['Is'] = Is0
    ckpt['alossL'] = alossL
    ckpt['alossC'] = alossC
    ckpt['alossISNMF'] = alossISNMF
#    print("TL-NMF best Cs,Ls,Is:",lossC_best,lossL_best,lossI_best)
    pickle.dump(ckpt, open(FOL + outfol + '/' + name + '.pkl', "wb"))
else:
    # load saved results
    ckpt = pickle.load( open(FOL + outfol + '/' + name + '.pkl', "rb" ) )
    Phi = ckpt['Phi']
    W = ckpt['W']
    H = ckpt['H']
    #Phi_init = ckpt['Phi_init']
    infos = ckpt['infos']
    fit_time = ckpt['fit_time']
    Cs0 = ckpt['Cs']
    Ls0 = ckpt['Ls']
    Is0 = ckpt['Is']

print("TL-NMF best Cs,Ls,Is:",Cs0,Ls0,Is0)
