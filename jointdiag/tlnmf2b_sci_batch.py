# Use TL-NMF to learn Phi,W,H from S samples of Y, with budget eval. 

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

from tlnmf.nmf import update_nmf_sparse
from tlnmf.functions import new_is_div, penalty, is_div
from tlnmf import tl_nmf_batch, signal_to_frames, synthesis_windowing
from utils import getwin, seperate_signal_from_WH_sci, eval_bss2 # , solve_NMF
#from utils import solve_NMF_me, solve_NMF_mm
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
#parser.add_argument('-epsc', '--eps_c', type = float, default = 1e-11)
parser.add_argument('-epsnmf', '--eps_nmf', type = float, default = 5e-7) # 1e-7
parser.add_argument('-itl', '--iter_tl', type = int, default = 500) # T
parser.add_argument('-pertl', '--iter_pertl', type = int, default = 1) # T_tl , 5
parser.add_argument('-pernmf', '--iter_pernmf', type = int, default = 1) # T_nmf
parser.add_argument('-toltl', '--tol_tl', type = float, default = 1e-12)
parser.add_argument('-runid', '--run_id', type = int, default = 1)
#parser.add_argument('-nbruns', '--num_runs', type = int, default = 10)
parser.add_argument('-win', '--window', type = int, default = 4) # win id for getwin
parser.add_argument('-inmf', '--iter_nmf', type = int, default = 50) # 500 , last-stage NMF for seperation
parser.add_argument('-cache', '--use_cache', type = int, default = 1) # to load saved results if 1
parser.add_argument('-loadphi','--load_phi', type = str, default = "")
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
fs = mat_dic['fs'][0]
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
eps_c = args.eps_nmf # c
eps_nmf = args.eps_nmf

if args.Hlambda == 0:
    name = 'tlnmf2b_sci_batch_' + sn + '_K' + str(K) + '_S' + str(L) + '_win' +\
           str(args.window) + '_ws' + str(int(ws*1000))  + 'ms_epsnmf' + str(eps_nmf)
else:
    assert(0)

runid = args.run_id

rng = np.random.RandomState(args.random_seed) #  + args.run_id-1)

print('name=', name)
print('runid=', runid)
outfol = sn + '_itl' + str(args.iter_tl) + '_run' + str(runid) + '_Ttl' + str(args.iter_pertl) + '_Tnmf' + str(args.iter_pernmf)

if args.nmf_me:
    outfol += '_me'
else:
    outfol += '_mm'
    
print('output to', './results/' + outfol)
if not path.exists('./results/' + outfol):
    mkdir('./results/' + outfol)

# test data
for l in range(nb_test):
    signal = test_ys[l,:]
    win = getwin(args.window)
    iF,iT,X = sig.stft(signal,fs,window=win,nperseg=segsize,return_onesided=False)
    Y = np.real(scipy.ifft(X,axis=0))
    if l == 0:
        M = Y.shape[0]
        N = Y.shape[1]
        F = M        
        test_Ys = np.zeros((nb_test,F,N))
    test_Ys[l,:,:] = Y

# train data
for l in range(L):
    signal = ys[l,:]
    
    win = getwin(args.window)
    iF,iT,X = sig.stft(signal,fs,window=win,nperseg=segsize,return_onesided=False)
    Y = np.real(scipy.ifft(X,axis=0))

    if l==0:
        M = Y.shape[0]
        N = Y.shape[1]
        F = M
        Ys = np.zeros((L,F,N))

        # plot y(t)
        plt.figure(figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(signal)
        plt.xlabel('t',size=FONT_SIZE)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(FONT_SIZE*1.2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(FONT_SIZE*1.2)
        plt.savefig('./results/' + outfol + '/' + name + '_signal.eps')
        plt.show()
        
    Ys[l,:,:] = Y

# esitmate Cov matrices C : (N,M,M) from Ys
estC = np.zeros((N,M,M))
for n in range(N):
    for ids in range(L): # L is s
        estC[n,:,:] = estC[n,:,:] + Ys[ids,:,n].reshape((M,1)) @ Ys[ids,:,n].reshape((1,M))
    estC[n,:,:] = estC[n,:,:] / L
estC += eps_nmf*np.eye(M)

print('Ys',Ys.shape)
regul = args.Hlambda * float(K) / M
print('regul',regul)

# init to the right random seed
for rid in range(runid):
    Phi0 = unitary_projection(rng.randn(M, M))
    W0 = np.abs(rng.randn(M, K)) + 1.
    W0 = W0 / np.sum(W0, axis=0)
    H0 = np.abs(rng.randn(K, N)) + 1.

Cs0,Ls0,Is0 = compute_losses(Phi0,W0,H0,estC,Ys,eps_nmf)
print("Init Cs,Ls,Is:",Cs0,Ls0,Is0)

# define NMF hook
assert( args.iter_nmf > 0 )
niter = args.iter_nmf
regul = args.Hlambda * float(K) / M
regul_type = 'sparse'
assert(regul==0)

cache = args.use_cache

if not path.exists('./results/' + outfol + '/' + name + '.pkl') or cache==0:
    if 1:
        def cb_eval_Gs(Phi,W,H):
            #global estC,Ys,eps_nmf
            lower_Gs = compute_LS(Phi,estC)
            V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)
            V_hat = np.dot(W, H)
            Cs = compute_tlnmfloss(V,V_hat,eps_nmf)
            #Cs = compute_tlnmfloss_eps(V,V_hat,eps_nmf,eps_c)
            return dict(lower_Gs=lower_Gs, Cs = Cs)
    
    t0 = time()
    Phi, W, H, Phi_init, infos = tl_nmf_batch(Ys, K, verbose=True, rng=rng, Phi=Phi0, W=W0, H=H0,\
                                              max_iter=args.iter_tl, n_iter_tl=args.iter_pertl,\
                                              n_iter_nmf=args.iter_pernmf,\
                                              tol=args.tol_tl, regul=regul, eps_nmf=eps_nmf,\
                                              cb_eval=cb_eval_Gs, nmfme=args.nmf_me) # , eps_c = eps_c)
    fit_time = time() - t0
    print('fit time',fit_time)
    # save Phi,W,H,Phi_init,infos into pickle
    ckpt = {}
    ckpt['Phi'] = Phi
    ckpt['W'] = W
    ckpt['H'] = H
    ckpt['Phi_init']=Phi_init
    ckpt['infos'] = infos
    ckpt['fit_time'] = fit_time
    
    Cs0,Ls0,Is0 = compute_losses(Phi,W,H,estC,Ys,eps_nmf)
    ckpt['Cs'] = Cs0
    ckpt['Ls'] = Ls0
    ckpt['Is'] = Is0
    print("TL-NMF Cs,Ls,Is:",Cs0,Ls0,Is0)

    pickle.dump(ckpt, open('./results/' + outfol + '/' + name + '.pkl', "wb"))
else:
    # load saved results
    ckpt = pickle.load( open('./results/' + outfol + '/' + name + '.pkl', "rb" ) )
    Phi = ckpt['Phi']
    W = ckpt['W']
    H = ckpt['H']
    Phi_init = ckpt['Phi_init']
    infos = ckpt['infos']
    fit_time = ckpt['fit_time']
    Cs0 = ckpt['Cs']
    Ls0 = ckpt['Ls']
    Is0 = ckpt['Is']
    
    
# Plot the most important atoms:
Phis, idx_sorted = plot_atoms16(Phi,Ys,'./results/' + outfol + '/' + name + '_atoms.png')
#plt.show()

# plot time objectives
plt.figure()
obj_list = infos['obj_list']
t = np.linspace(0, fit_time, len(obj_list))
plt.plot(t, obj_list)
plt.xlabel('Time (sec.)')
plt.ylabel('Objective function')
plt.savefig('./results/' + outfol + '/' + name + '_timeobj.png')

# plot W,H
V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)  # final spectrogram
# reorder W from energy
V_hat = np.matmul(W, H) # final factorization
#print('jdloss L is',compute_jdloss(Phi,estC))
#print('loss C is', new_is_div(V,V_hat,eps_nmf))
#print('NMF is_div is',is_div(V,V_hat,eps_nmf)) # BUG !
jdloss = compute_LS(Phi,estC) #  compute_jdloss(Phi,estC)
print('TLNMF loss C is',compute_tlnmfloss(V,V_hat,eps_nmf))
print('jdloss L_S',jdloss)
print('NMF loss I is',is_div(V,V_hat,eps_nmf))

Ws = W[idx_sorted[::-1],:]

# compute NMF decomposition of V = |Phi*Y|^2 = W*H

plt.figure()
plt.subplot(121)
cmax = np.max(np.log10(V+eps_nmf))
plt.imshow(np.log10(V+eps_nmf),vmin=cmax-4,vmax=cmax)
plt.title('log V')
plt.xlabel('n')
plt.ylabel('f')
plt.colorbar()
plt.gca().invert_yaxis()

plt.subplot(122)
plt.imshow(np.log10(V_hat+eps_nmf),vmin=cmax-4,vmax=cmax)
plt.title('log WH')
plt.colorbar()
plt.xlabel('n')
plt.ylabel('f')
plt.gca().invert_yaxis()

plt.savefig('./results/' + outfol + '/' + name +  '_Vh.png')   

#plt.figure()
plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot(121)
if K>=2:
    plt.plot(range(1,F+1),Ws[:,0],'o',markersize=12)
    plt.plot(range(1,F+1),Ws[:,1],'.',markersize=12)
    if K==3:
        plt.plot(range(1,F+1),Ws[:,2],'x',markersize=12)

plt.title('$[W]_{mk}$',size=FONT_SIZE*1.5)
plt.xlabel('m',size=FONT_SIZE)
plt.grid('on')
plt.xlim(0,24)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
    
if K==2:
    plt.legend(['k=1','k=2'],prop={"size":FONT_SIZE})
elif K==3:
    plt.legend(['k=1','k=2','k=3'],prop={"size":FONT_SIZE})

ax = plt.subplot(122)
if K>=2:
    plt.plot(range(1,N+1),H.T[:,0],'-')
    plt.plot(range(1,N+1),H.T[:,1],'--')
    if K==3:
        plt.plot(range(1,N+1),H.T[:,2],':')

plt.title('$[H]_{kn}$',size=FONT_SIZE*1.5)
plt.xlabel('n',size=FONT_SIZE)
plt.xlim(0,N)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(0)

print('savefit to','./results/' + outfol + '/' + name +  '_WH.png')
plt.savefig('./results/' + outfol + '/' + name +  '_WH.png')
sio.savemat('./results/' + outfol + '/' + name + '_Wsorted.mat', {'Ws':Ws,'H':H,'Phis':Phis,'y':ys[0,:]})
              
# seperate one example into K comopnents
if 1 and args.window > 0:
    outdir = './results/' + outfol
    for l in range(L):
        Y = Ys[l,:,:] # seperate training data
        seperate_signal_from_WH_sci(Y,Phi,W,H,K,eps_nmf,win,fs,segsize,outdir,name + '_l' + str(l))

if 0 and args.window > 0:
    outdir = './results/' + outfol
    for l in range(nb_test):
        Y = test_Ys[l,:,:] # seperate test data
        seperate_signal_from_WH_sci(Y,Phi,W,H,K,eps_nmf,win,fs,segsize,outdir,name + '_l' + str(l))

plt.show()
