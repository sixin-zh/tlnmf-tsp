# JD + NMF with evaluation of budget
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
from tlnmf.functions import is_div, new_is_div
from qndiag import qndiag2
from utils import getwin,  seperate_signal_from_WH_sci, hash_str2int2, eval_bss2#, solve_NMF
from utils import solve_NMF_me, solve_NMF_mm
from utils import compute_jdloss, compute_tlnmfloss, compute_LS, convert_Ys_by_blocks, plot_atoms8, compute_losses
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sn', '--signal_name', type = str, default = "nonstationary440_sim1") # nonstationary12c
parser.add_argument('-K', '--rank', type = int, default = 2)
parser.add_argument('-s', '--number_samples', type = int, default = 10)
parser.add_argument('-bs', '--block_size', type = int, default = 1)
parser.add_argument('-bso', '--block_size_overlap', type = int, default = 0)
parser.add_argument('-hla','--Hlambda', type = float, default = 0)
parser.add_argument('-ws', '--window_size', type = float, default = 40e-3)
parser.add_argument('-epsc', '--eps_c', type = float, default = 5e-7) # 1e-7
#parser.add_argument('-epsnmf', '--eps_nmf', type = float, default = 1e-16)
parser.add_argument('-itl', '--iter_tl', type = int, default = 500) # max nb iter of JD
parser.add_argument('-ratio', '--iter_ratio', type = int, default = 1) # JD+NMF, 1:ratio for TL-NMF
parser.add_argument('-runid', '--run_id', type = int, default = 1)
parser.add_argument('-win', '--window', type = int, default = 4) # win id for getwin
parser.add_argument('-inmf', '--iter_nmf', type = int, default = 500) # 50 for BSS_EVAL before # NMF iterations for seperation
parser.add_argument('-cache', '--use_cache', type = int, default = 1) # to load saved results if 1
parser.add_argument('-loadphi','--load_phi', type = str, default = "")
parser.add_argument('-dctinit','--dctinit', type = int, default = 0)
parser.add_argument('-rseed', '--random_seed', type = int, default = 100) # seed for init of Phi,W,H
parser.add_argument('-nmfme','--nmf_me', type = int, default = 0) # NMF MM default

args = parser.parse_args()

np.seterr(under='warn')

EVAL_BSS = 0
if EVAL_BSS:
    # https://fr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    # python setup.py install  --user build --build-base=/tmp/
    # matlab.engine.shareEngine
    import matlab.engine
    import matlab
    eng = matlab.engine.start_matlab() # matlab.engine.connect_matlab()
    
#rng = np.random # .RandomState(0)
rng = np.random.RandomState(args.random_seed) #  + args.run_id-1)

# Read the song
sn = args.signal_name
mat_adr = 'datasets/' + sn + '.mat'
print('load mat data at:',mat_adr)

mat_dic = sio.loadmat(mat_adr)
ys = mat_dic['ys']
fs = mat_dic['fs'][0]

print('signals',ys.shape)
print('fs',fs)
L = ys.shape[0] # number of samples
s = args.number_samples
assert(L>=s)
L = s
print('number of samples s=',L)
bs = args.block_size
if bs>1:
    print('block size S=',bs)
    bso = args.block_size_overlap

ws = args.window_size # 40e-3
segsize = int(ws*fs)

K = args.rank
eps_c = args.eps_c
eps_nmf = eps_c # eps_nmf # eps_c 
itl = args.iter_tl # number of jd iters
ratio = args.iter_ratio # number of nmf iters per jd iter
dctinit = args.dctinit # if 1, init Phi with DCT

assert args.Hlambda == 0

#if itl == 0:
if bs > 1:
    assert(False)
    name = 'tlnmfJD2b_sci_batch_' + sn + '_K' + str(K)  + '_S' + str(s) +\
           '_bs' + str(bs) + '_bso' + str(bso) +\
           '_win' + str(args.window) + '_ws' + str(int(ws*1000)) + 'ms_epsc' + str(eps_c)
elif dctinit > 0:
    name = 'tlnmfJD2b_sci_batch_' + sn + '_K' + str(K)  + '_S' + str(s) +\
           '_win' + str(args.window) + '_ws' + str(int(ws*1000)) + 'ms_epsc' + str(eps_c) + '_dctinit'
else:
    name = 'tlnmfJD2b_sci_batch_' + sn + '_K' + str(K)  + '_S' + str(s) +\
           '_win' + str(args.window) + '_ws' + str(int(ws*1000)) + 'ms_epsc' + str(eps_c)
#else:
 #   assert(false)
  #  name = 'tlnmfJD2_sci_batch_' + sn + '_K' + str(K) + '_S' + str(s) + '_bs' + str(bs) + '_win' +\
   #        str(args.window) + '_ws' + str(int(ws*1000)) + 'ms_epsc' + str(eps_c) + '_itl' +str(itl)

runid = args.run_id
print('name=', name)
print('runid=', runid)
#outfol = './results_jd_epsc/' + sn + '_run' + str(runid)
outfol = './results_jd/' + sn + '_rseed' + str(args.random_seed) +\
         '_itl' + str(itl) + '_ratio' + str(ratio) + '_run' + str(runid)

if args.nmf_me:
    outfol += '_me'
else:
    outfol += '_mm'
    
loadphi = args.load_phi
if loadphi is not "":
    assert(0)
    outfol = outfol + 'loadphi' + str(hash_str2int2(loadphi.encode('UTF-8')))
if not path.exists(outfol):
    mkdir(outfol)
print('outfol',outfol)

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

    Ys[l,:,:] = Y # (M,N)

# convert Ys into blocks, with a new L=L*bs,F=F,N=N/bs
print('Ys',Ys.shape)
if bs>1:
    bYs,bL,bF,bN = convert_Ys_by_blocks(Ys,L,F,N,bs,bso=bso)
    print('bYs',bYs.shape)
else:
    bYs,bL,bF,bN = Ys,L,F,N    

# esitmate Cov matrices C : (N,M,M) from Ys
estC = np.zeros((bN,M,M))
for n in range(bN):
    for l in range(bL):
        estC[n,:,:] = estC[n,:,:] + bYs[l,:,n].reshape((M,1)) @ bYs[l,:,n].reshape((1,M))
    estC[n,:,:] = estC[n,:,:] / bL

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

# init to the right random seed
for rid in range(runid): # assume runid starts from 0
    if dctinit == 0:
        Phi0 = unitary_projection(rng.randn(M, M))
    else:
        Phi0 = fftpack.dct(np.eye(M), 3, norm='ortho')    
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

# JD
cache = args.use_cache
if EVAL_BSS:
    ys1 = mat_dic['ys1'][0:L,:]
    ys2 = mat_dic['ys2'][0:L,:]
    T = ys1.shape[1]
    ys1_ = matlab.double(ys1.tolist())
    ys2_ = matlab.double(ys2.tolist())

W0_eval = W0
H0_eval = H0
if not path.exists(outfol + '/' + name + '.pkl') or cache==0:
    def cb_eval_Gs(Phi,W=None,H=None,ite=None):
        lower_Gs = compute_LS(Phi,estC)
        if EVAL_BSS and K==2 and bs==1:
            W1,H1 = solve_NMF(Phi,Ys,niter,eps_c,W0,H0)
            bss_sdr,bss_sir,bss_sar = eval_bss2(eng,Phi,W1,H1,L,T,Ys,ys1_,ys2_,eps_nmf,win,fs,segsize,outfol)
            print('bss sdr',bss_sdr)
            return dict(lower_Gs=lower_Gs, sdr = bss_sdr, sir=bss_sir, sar=bss_sar)
        else:
            if ite is None:
                return dict(lower_Gs=lower_Gs)
            else:
                niter = ite*ratio
                V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0) # + eps_S
                #W,H = solve_NMF(Phi,Ys,niter,eps_c,W0_eval,H0_eval)
                if args.nmf_me == 1:
                    W,H = solve_NMF_me(Phi,Ys,niter,eps_c,W0_eval,H0_eval)
                else:
                    W,H = solve_NMF_mm(Phi,Ys,niter,eps_c,W0_eval,H0_eval)
                V_hat = np.dot(W,H)
                Cs = compute_tlnmfloss(V,V_hat,eps_c)
                print('iter Cs',ite,Cs)                    
                return dict(lower_Gs=lower_Gs, Cs=Cs)

    t0 = time()
    if '.pkl' in loadphi:
        ckpt = pickle.load( open(args.load_phi, "rb" ) )
        Phi_init = ckpt['Phi']
    else:
        Phi_init = Phi0 #  unitary_projection(rng.randn(M, M))
    Phi, infos = qndiag2(estC,B0=Phi_init,cb_eval=cb_eval_Gs,max_iter=itl)
    fit_time = time() - t0
    print('fit time',fit_time)
    # save Phi,W,H,Phi_init,infos into pickle
    ckpt = {}
    ckpt['Phi'] = Phi
    ckpt['Phi_init']=Phi_init
    ckpt['infos'] = infos
    pickle.dump(ckpt, open(outfol + '/' + name + '.pkl', "wb"))
else:
    # load saved results
    ckpt = pickle.load( open(outfol + '/' + name + '.pkl', "rb" ) )
    Phi = ckpt['Phi']
    infos = ckpt['infos']
    if EVAL_BSS and K==2 and bs==1:
        W1,H1 = solve_NMF(Phi,Ys,niter,eps_c,W0,H0) 
        bss_sdr,bss_sir,bss_sar = eval_bss2(eng,Phi,W1,H1,L,T,Ys,ys1_,ys2_,eps_nmf,win,fs,segsize,outfol)
        print('bss sdr',bss_sdr)

Cs0,Ls0,Is0 = compute_losses(Phi,W0,H0,estC,Ys,eps_nmf)
print("After JD Cs,Ls,Is:",Cs0,Ls0,Is0)

#jdloss = compute_LS(Phi,estC) #  compute_jdloss(Phi,estC)
#print('jdloss L_S',jdloss)

# Peroform NMF after JD finished
if args.nmf_me == 1:
    W,H = solve_NMF_me(Phi,Ys,niter,eps_c,W0,H0)
else:
    W,H = solve_NMF_mm(Phi,Ys,niter,eps_c,W0,H0)
#W,H = solve_NMF(Phi,Ys,niter,eps_c,W0,H0) 
Cs0,Ls0,Is0 = compute_losses(Phi,W,H,estC,Ys,eps_nmf)
print("JD+NMF Cs,Ls,Is:",Cs0,Ls0,Is0)

# Plot the most important atoms:
Phis, idx_sorted = plot_atoms8(Phi,Ys,outfol + '/' + name + '_atoms.png')

# reorder W from energy
V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)
V_hat = np.matmul(W, H) # final factorization
print('TLNMF loss C is',compute_tlnmfloss(V,V_hat,eps_c)) # ,eps_nmf))
print('NMF loss I is',is_div(V,V_hat,eps_c))
#print('loss C is', new_is_div(V,V_hat,eps_nmf))
#print('NMF is_div is',is_div(V,V_hat,eps_nmf)) # FIX bug!
Ws = W[idx_sorted[::-1],:]

# PLOT
#plt.figure()
plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot(121)
if K>=2:
    plt.plot(range(1,F+1),Ws[:,0],'o',markersize=12)
    plt.plot(range(1,F+1),Ws[:,1],'.',markersize=12)
    if K==3:
        plt.plot(range(1,F+1),Ws[:,2],'x',markersize=12)

plt.title('w_{km}',size=FONT_SIZE*1.5)
plt.xlabel('m',size=FONT_SIZE)
plt.grid('on')
plt.xlim(0,12)
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

plt.title('h_{kn}',size=FONT_SIZE*1.5)
plt.xlabel('n',size=FONT_SIZE)
plt.xlim(0,N)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(0)

plt.savefig(outfol + '/' + name +  '_WH.png')
sio.savemat(outfol + '/' + name + '_Wsorted.mat', {'Ws':Ws,'H':H,'Phis':Phis,'y':ys[0,:]})

# compute NMF decomposition of V = |Phi*Y|^2 = W*H
plt.figure()
plt.subplot(121)
cmax = np.max(np.log10(V+eps_c))
plt.imshow(np.log10(V+eps_c),vmin=cmax-4,vmax=cmax)
plt.title('log V')
plt.xlabel('n')
plt.ylabel('f')
plt.colorbar()
plt.gca().invert_yaxis()

plt.subplot(122)
plt.imshow(np.log10(V_hat+eps_c),vmin=cmax-4,vmax=cmax)
plt.title('log WH')
plt.colorbar()
plt.xlabel('n')
plt.ylabel('f')
plt.gca().invert_yaxis()

plt.savefig(outfol + '/' + name +  '_Vh.png')    

# seperate one example into K comopnents
if args.window > 0 and False:
    for l in range(L):
        Y = Ys[l,:,:]
        seperate_signal_from_WH_sci(Y,Phi,W,H,K,eps_c,win,fs,segsize,outfol,name + '_l' + str(l))

plt.show()

if EVAL_BSS:
    eng.quit()
