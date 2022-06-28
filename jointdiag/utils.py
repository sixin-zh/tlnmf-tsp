try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

import scipy.signal as sig
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt

import librosa, librosa.display
import soundfile as sf

from tlnmf import synthesis_windowing
from tlnmf.functions import is_div, is_div_eps
from tlnmf.nmf import update_nmf_eps, update_nmf_mm, update_nmf_me # update_nmf_sparse

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj0(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

from scipy import fftpack
import numpy as np

def getwin(winid):
    win = None
    if winid == 1:
        win = ('tukey',0.5)
    elif winid == 2:
        win = ('tukey',0.8)
    elif winid == 3:
        win = ('tukey',0.2)
    elif winid == 4:
        win = ('tukey',0.1)
    elif winid == 5:
        win = ('hamming')
    return win

def compute_dct_spectrum(Y):
    F = Y.shape[0]
    Phi = fftpack.dct(np.eye(F), 3, norm='ortho')
    X = np.dot(Phi, Y)
    V = X ** 2
    return V

def frames_to_signal(Y,signal,fs,ws):
    n_samples = len(signal)
    n_window_samples = 2 * np.floor(ws * fs / 2)
    N_box = int(np.ceil(n_samples / n_window_samples))
    n_window_samples = int(n_window_samples)
    y = synthesis_windowing(Y, n_window_samples, N_box)
    return y.reshape(n_samples)

def signal_to_frames_win0(signal, fs, window_size):
    n_samples = len(signal)
    n_window_samples = int(np.floor(window_size * fs))
    N_box = int(np.ceil(n_samples / n_window_samples))
    M = n_window_samples
    scaling = 1.0/M
    x = signal
    #print(M,N_box)
    Y = np.zeros((M,N_box))
    for kk in range(1,N_box+1):
        Y[:,kk-1] = x[np.arange(0,M)+(kk-1)*M] * scaling
    return Y

def seperate_signal_from_WH_sci(Y,Phi,W,H,K,eps_nmf,win,fs,segsize,outfol,name=None):
    X1 = np.dot(Phi, Y)
    V_hat = np.dot(W, H)
    iPhi = np.linalg.inv(Phi)
    if 1:
        X = scipy.fft(Y, axis=0)
        iT_,y_0 = sig.istft(X,fs,window=win,nperseg=segsize,input_onesided=False)
        y0 = np.real(y_0)
        sf.write(outfol + '/' + name + '_original.wav', y0, fs)
    ys = []
    #plt.figure()
    for k in range(K):
        mask = (W[:,k:k+1] @H[k:k+1,:]+eps_nmf/K) / (V_hat+eps_nmf) # avoid underflow in @
        Zxx_k = X1 * mask
        Y_k = iPhi @ Zxx_k
        X_k = scipy.fft(Y_k, axis=0)
        iT_,y_k = sig.istft(X_k,fs,window=win,nperseg=segsize,input_onesided=False)
        y_k = np.real(y_k)
        y_k = y_k.reshape((np.size(y_k),1))
        ys.append(y_k)
        if name is not None:
          #  sio.savemat(outfol + '/' + name + '_piece' + str(k) + '.mat', {'y_k':y_k, 'fs':fs, 'mask':mask})
            sf.write(outfol + '/' + name + '_piece_k' + str(k+1) + '.wav', y_k, fs)
        if 1:
            plt.figure(figsize=[6,9])
            logVv = np.log10( (V_hat+eps_nmf) * mask )
            logV = np.log10(mask * (np.abs(X1)**2+eps_nmf))                           
            cmax = max(np.max(logVv),np.max(logV))
            if 1:
                plt.subplot(2, 1, 1)                
                #plt.imshow(logVv,cmap=plt.get_cmap('gray'),vmin=cmax-6,vmax=cmax)
                plt.imshow(logVv,cmap=plt.get_cmap('jet'),vmin=cmax-6,vmax=cmax)
                plt.gca().invert_yaxis()
                #plt.ylabel('m')
                plt.colorbar()
                plt.title('WH times mask k=' + str(k+1) + ' (log)')
            if 1:
                plt.subplot(2, 1, 2)
                #cmax = np.max(logV)
                plt.imshow(logV,cmap=plt.get_cmap('jet'),vmin=cmax-6,vmax=cmax)
                plt.colorbar()
                plt.gca().invert_yaxis()
                plt.title('|Phi Y|^2 times mask k=' + str(k+1) + ' (log)')
            if 0:
                plt.subplot(2, 1, 2)
                plt.plot(np.arange(y_k.shape[0])/fs, y_k ) # ,cmap=plt.get_cmap('gray'))
                plt.ylabel('Amp.')
                plt.xlabel('Time [s]')
                plt.title('signal')
                #plt.colorbar()
            plt.savefig(outfol + '/' + name +  '_mask_k' + str(k+1) +'.png')
            
    return ys

# from qndiag
def transform_set(M, D, diag_only=False):
    n, p, _ = D.shape
    if not diag_only:
        op = np.zeros((n, p, p))
        for i, d in enumerate(D):
            op[i] = M.dot(d.dot(M.T))
    else:
        assert(0)
        op = np.zeros((n, p))
        for i, d in enumerate(D):
            op[i] = np.sum(M * d.dot(M.T), axis=0)
    return op

def loss(D, is_diag=False, weights=None):
    n, p = D.shape[:2]
    if not is_diag:
        diagonals = np.diagonal(D, axis1=1, axis2=2)
    else:
        diagonals = D
    sign, logdets = np.linalg.slogdet(D) # [1]
    if weights is None:
        return ( np.sum(np.log(diagonals)) - np.sum(logdets) )
    else:
        return -1

def loss2(D, is_diag=False, weights=None):
    n, p = D.shape[:2]
    if not is_diag:
        diagonals = np.diagonal(D, axis1=1, axis2=2)
    else:
        diagonals = D
    if weights is None:
        return np.sum(np.log(diagonals)+1)
    else:
        return -1
    
def compute_tlnmfloss(A,B,eps):
    #    M, N = A.shape
    f = (A + eps) / (B + eps)
    return np.sum(f + np.log(B+eps))

# Cs with eps_S = eps_C, eps_0 = eps_nmf
def compute_tlnmfloss_eps(A,B,eps_nmf,eps_c):
    f = (A + eps_c) / (B + eps_nmf)
    return np.sum(f + np.log(B+eps_nmf))
    
def compute_A(A,B,eps):
#    M, N = A.shape
    f = (A + eps) / (B + eps)
    return np.sum(f -1)

def compute_B(A,B,eps):
#    M, N = A.shape
    f = (A + eps) / (B + eps)
    return np.sum(np.log(f)) # f -1)
# + np.log(B+eps))

def compute_LS(B,C):
    D = transform_set(B,C)
    return loss2(D)

def compute_jdloss(B,C):
    D = transform_set(B,C) # D = B C B^t
    return loss(D)

def compute_jdloss3(Phi,Ys,epsc=0):
    # Sigma_n = 1/S \sum_s y_{s,n} y_{s,n}^t
    # return 1/2N sum_n det Diag ( Phi Sigma_n Phi')
    S, M, N = Ys.shape
    Xs = np.matmul(Phi, Ys)
    Vs = Xs ** 2
    op = np.mean(Vs,axis=0) + epsc
    estC = np.zeros((N,M,M))
    for n in range(N):
        estC[n,:,:] = (Ys[:,:,n].T @ Ys[:,:,n]) /S
    estC = estC + epsc*np.eye(M)
    sign, logdets = np.linalg.slogdet(estC)
    return 0.5 * ( np.sum(np.log(op)) - np.sum(logdets) )  / N

def compute_jdgrad(B,C):
    D = transform_set(B,C)
    N = D.shape[0]
    M = D.shape[1]
    diagonals = np.diagonal(D, axis1=1, axis2=2)
    # Gradient
    G = np.average(D / diagonals[:, :, None], axis=0) - np.eye(M)
    #G = np.zeros((M,M))
    #for n in range(N):        
    #    G += np.diag(1/np.diag(D[n,:,:])) @ D[n,:,:]
    #G = G / N  - np.eye(M)
    #print(G)
    return G

import hashlib
def hash_str2int2(s):
    return int(hashlib.sha1(s).hexdigest(), 16) % (100)


def get_values(dic,key):
    values = []
    for iv in dic:
        values.append(iv[key])
    return values

def convert_Ys_by_blocks(Ys,L,F,N,bs,bso=0,eps1=1e-16):
    assert(Ys.shape[0]==L)
    assert(Ys.shape[1]==F)
    assert(Ys.shape[2]==N)
    if bso ==0:
        bN = N // bs # skip the end of the blocks
        ys = np.zeros((L*bs,F,bN))
        for bid in range(bN):
            for l in range(L):
                for b in range(bs):
                    ys[bs*l+b,:,bid] = Ys[l,:,b+bid*bs]
        return ys,L*bs,F,bN # int(N/bs)
    elif bso ==1: # half-overlap
        assert(bs % 2 == 0)
        bN = (N // bs) * 2 - 1
        ys = np.zeros((L*bs,F,bN))
        for bid in range(bN): # number of blocks
            for l in range(L): # nb samples
                for b in range(bs): # block size S
                    ys[bs*l+b,:,bid] = Ys[l,:,b+bid*(bs//2)]
        return ys,L*bs,F,bN # int(N/bs)
    elif bso ==2: # quad-overlap
        assert(bs % 4 == 0)
        bN = (N // bs) * 4 - 3
        ys = np.zeros((L*bs,F,bN))
        for bid in range(bN): # number of blocks
            for l in range(L): # nb samples
                for b in range(bs): # block size S
                    ys[bs*l+b,:,bid] = Ys[l,:,b+bid*(bs//4)]
        return ys,L*bs,F,bN # int(N/bs)
    elif bso == 3: # use only odd columns of Y
        Ys = Ys[:,:,np.arange(0,N,2)]
        #print('Ys size',Ys.shape)
        N = N//2
        bN = N // bs # skip the end of the blocks
        ys = np.zeros((L*bs,F,bN))
        for bid in range(bN):
            for l in range(L):
                for b in range(bs):
                    ys[bs*l+b,:,bid] = Ys[l,:,b+bid*bs]
        return ys,L*bs,F,bN
    elif bso == 4: # normalized block
        # use eps1 to avoid numerical issue??
        bN = N // bs # skip the end of the blocks
        ys = np.zeros((L*bs,F,bN))
        for bid in range(bN):
            for l in range(L):
                for b in range(bs):
                    ys[bs*l+b,:,bid] = Ys[l,:,b+bid*bs] / (np.sqrt(np.sum(Ys[l,:,b+bid*bs]**2)) + eps1)
        return ys,L*bs,F,bN
    elif bso == 5:
        # filter out segments with large variation
        assert(L==1)

                    
    return None

def plot_atoms8(Phi,Ys,outname=None):
    Xs = np.matmul(Phi, Ys)
    powers = np.sum(Xs**2,axis=2) # np.linalg.norm(Xs, axis=1)
    power = np.mean(powers,axis=0)
    shape_to_plot = (4, 2)
    n_atoms = np.prod(shape_to_plot)
    idx_sorted = np.argsort(power)
    idx_to_plot = idx_sorted[-n_atoms:][::-1]
    f, ax = plt.subplots(*shape_to_plot)
    f.set_size_inches(18, 6)
    Phis = Phi[idx_to_plot,:]
    print('Phis',Phis.shape)

    for axe, idx in zip(ax.ravel(), range(0,8)):
        axe.plot(Phis[idx])
        axe.axis('off')
        
    if outname is not None:
        plt.savefig(outname,  dpi=80)

    return Phis, idx_sorted

def plot_atoms16(Phi,Ys,fs,outname=None):
    Xs = np.matmul(Phi, Ys)
    powers = np.sum(Xs**2,axis=2) # np.linalg.norm(Xs, axis=1)
    power = np.mean(powers,axis=0)
    shape_to_plot = (4, 4)
    n_atoms = np.prod(shape_to_plot)
    idx_sorted = np.argsort(power)
    idx_to_plot = idx_sorted[-n_atoms:][::-1]
    f, ax = plt.subplots(*shape_to_plot)
    f.set_size_inches(18, 12)
    Phis = Phi[idx_to_plot,:]
    print('Phi sorted shape',Phis.shape)

    for axe, idx in zip(ax.ravel(), range(0,16)):
        axe.plot(Phis[idx])
        axe.title.set_text('m='+str(idx+1))
        axe.title.set_fontsize(20)
        if 0:
            M = Phi.shape[0]
            alpha,psi,amp = regress_atom(Phis[idx],repeat=1000)
            fit = np.zeros(M)
            for m in range(M):
                fit[m] = amp*np.cos(alpha*m+psi)
            axe.plot(fit)
            opta = (alpha/np.pi) % 2 # alpha%(2*np.pi)/np.pi
            opta = min(opta, 2-opta) # use neg freq to make sure opta in 0 nad 1
            axe.title.set_text('%.1f Hz' % (opta/2*fs))
            print('%.1f Hz\t atomidx =%d' % (opta/2*fs,idx+1))
        axe.axis('off')
        
    if outname is not None:
        plt.savefig(outname,  dpi=80)

    return Phis, idx_sorted

def compute_losses(Phi,W,H,estC,Ys,eps_nmf):
    Ls0 = compute_LS(Phi,estC) # alrady with epsC
    V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)
    V_hat = np.dot(W, H)
    Is0 = is_div(V,V_hat,eps_nmf)
    Cs0 = compute_tlnmfloss(V,V_hat,eps_nmf)
    return Cs0,Ls0,Is0

def compute_losses_eps(Phi,W,H,estC,Ys,eps_nmf,eps_c):
    Ls0 = compute_LS(Phi,estC) # alrady with eps_c
    V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0) # no eps_c
    V_hat = np.dot(W, H)
    Is0 = is_div_eps(V,V_hat,eps_nmf, eps_c)
    Cs0 = compute_tlnmfloss_eps(V,V_hat,eps_nmf,eps_c) # add eps_c inside
    return Cs0,Ls0,Is0

def eval_bss2(eng,Phi,W,H,L,T,Ys,ys1_,ys2_,eps_nmf,win,fs,segsize,outfol):
#    import matlab.engine
    # assume ys1_ and ys2_ are alreay in matlab double format
    import matlab
    K = 2
#    assert(ys2.shape[1]==ys1.shape[1])
    y1rs = np.zeros((L,T))
    y2rs = np.zeros((L,T))
    for l in range(L):
        Y = Ys[l,:,:] # seperate signal Y using W and H into K components
        ys12 = seperate_signal_from_WH_sci(Y,Phi,W,H,K,eps_nmf,win,fs,segsize,outfol)
        y1rs[l,:] = ys12[0][:,0]
        y2rs[l,:] = ys12[1][:,0]
    #print(np.linalg.norm(ys1), np.linalg.norm(ys2), np.linalg.norm(y1rs),np.linalg.norm(y2rs))
    # convert to matlab format
    # https://fr.mathworks.com/help/matlab/matlab_external/pass-data-to-matlab-from-python.html    
    y1rs_ = matlab.double(y1rs.tolist())
    y2rs_ = matlab.double(y2rs.tolist())
    bss_sdr,bss_sir,bss_sar = eng.call_bsseval(ys1_,ys2_,y1rs_,y2rs_,nargout=3)
    del y1rs_, y2rs_
    return bss_sdr,bss_sir,bss_sar

def solve_NMF_mm(Phi,Ys,niter,eps_c,W0,H0):
    V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)
    #print('run NMF with niter=',niter)
    W = W0.copy() # np.abs(rng.randn(F, K)) + 1.
    #W = W / np.sum(W, axis=0)
    H = H0.copy() # np.abs(rng.randn(K, N)) + 1.
    V_hat = np.dot(W, H) # + eps  # Initial factorization
    for ite in range(niter):
        W, H = update_nmf_mm(V, W, H, V_hat, eps=eps_c) # in-place of W,H
        V_hat = np.dot(W, H)  # Initial factorization
    #print(np.linalg.norm(W0))
    return W,H

def solve_NMF_me(Phi,Ys,niter,eps_c,W0,H0):
    V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)
    #print('run NMF with niter=',niter)
    W = W0.copy() # np.abs(rng.randn(F, K)) + 1.
    #W = W / np.sum(W, axis=0)
    H = H0.copy() # np.abs(rng.randn(K, N)) + 1.
    V_hat = np.dot(W, H) # + eps  # Initial factorization
    for ite in range(niter):
        W, H = update_nmf_me(V, W, H, V_hat, eps=eps_c) # in-place of W,H
        V_hat = np.dot(W, H)  # Initial factorization
    #print(np.linalg.norm(W0))
    return W,H

def solve_NMF_sparse(Phi,Ys,niter,eps_c,W0,H0):
    regul = 0
    # NMF: estimate W,H for |Phi Y|^2 over training set
    V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)
    #print('run NMF with niter=',niter)
    W = W0.copy() # np.abs(rng.randn(F, K)) + 1.
    #W = W / np.sum(W, axis=0)
    H = H0.copy() # np.abs(rng.randn(K, N)) + 1.
    V_hat = np.dot(W, H) # + eps  # Initial factorization
    for ite in range(niter):
        W, H = update_nmf_sparse(V, W, H, V_hat, regul, eps=eps_c) # in-place of W,H
        V_hat = np.dot(W, H)  # Initial factorization
    #print(np.linalg.norm(W0))
    return W,H

def solve_NMF_eps(Phi,Ys,niter,eps_nmf,eps_c,W0,H0):
    #regul = 0
    # NMF: estimate W,H for |Phi Y|^2 over training set
    V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)
    #print('run NMF with niter=',niter)
    W = W0.copy() # np.abs(rng.randn(F, K)) + 1.
    #W = W / np.sum(W, axis=0)
    H = H0.copy() # np.abs(rng.randn(K, N)) + 1.
    V_hat = np.dot(W, H) # + eps  # Initial factorization
    for ite in range(niter):
        W, H = update_nmf_eps(V, W, H, V_hat, eps_nmf = eps_nmf, eps_c = eps_c) # in-place of W,H
        V_hat = np.dot(W, H)  # Initial factorization
    #print(np.linalg.norm(W0))
    return W,H

def regress_atom(phi, alpha0=0, psi0=0, repeat=300):
    # solve MSE fit of obj = ( phi(m) - amp*cos(alpha * m + psi) )**2
#    print('regress phi', phi)
    M = phi.shape[0]
    amp0 = np.max(phi)
    arrm = np.arange(M)
#    print(arrm)
    def loss(x):
        alpha = x[0]
        psi = x[1]
        amp = x[2]
        diff = phi - amp*np.cos(alpha*arrm+psi)
        obj = np.sum(diff**2)
#        grad = np.zeros(2)
        
#        for m in range(M):
#            diff = phi[m] - amp*np.cos(alpha*m+psi)
#            obj += diff**2
#            grad[0] += 2*diff*(np.sin(alpha*m+psi))*m
#            grad[1] += 2*diff*(np.sin(alpha*m+psi))

        return obj # , grad
    
    # call a optimizer
    x0 = np.zeros(3)
    x0[0] = alpha0
    x0[1] = psi0
    x0[2] = amp0
    res = scipy.optimize.minimize(loss,x0)
    rep = 0
    bestloss = loss(x0)
    while rep < repeat:
        x0[0] = alpha0 + np.random.rand()*2*np.pi
        x0[1] = psi0 + np.random.rand()*2*np.pi
        x0[2] = amp0 * (1 + 0.1*np.random.rand())
        res = scipy.optimize.minimize(loss,x0,method='BFGS') # SLSQP')
        if res.success is True:
            rep += 1
            if loss(res.x) < bestloss:
                bestloss = loss(res.x)
                bestx = res.x
    x = bestx
    #print('best alpha',x[0])
    return x[0], x[1], x[2]

def get_Ys_from_sn(sn,s,window=4,ws=40e-3):
    #sn = 'timit_test_sa1'
    mat_adr = sn + '.mat'
    print('load mat data at:',mat_adr)
    mat_dic = sio.loadmat(mat_adr)
    ys = mat_dic['ys']
    fs = mat_dic['fs'][0][0]
    win = getwin(window)
    segsize = int(ws*fs)
    print('signals',ys.shape)
    print('fs',fs)
    L = ys.shape[0] # number of traning samples
    #s = args.number_samples
    L = s # reset number of traning samples
    print('number of samples s=',L)
    for l in range(L):
        signal = ys[l,:]    
        iF,iT,X = sig.stft(signal,fs,window=win,nperseg=segsize,return_onesided=False)
        Y = np.real(scipy.ifft(X,axis=0))
        if l==0:
            M = Y.shape[0]
            N = Y.shape[1]
            F = M
            Ys = np.zeros((L,F,N))
        Ys[l,:,:] = Y
    return Ys, fs

def plot_wavaudio(file,title):
    x, sr = librosa.load(file)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, alpha=0.6)
    plt.title(title)
    plt.ylim(-1, 1)
    plt.show()  


def get_dft_phi(N):
    assert(N % 2 == 0)
    Phi = np.zeros((N,N)) # each row is an atom
    for k in range(N//2+1):
        if k == 0:
            Phi[0,:] = 1/np.sqrt(N)
        elif k == N//2:
            for n in range(N):
                Phi[N-1,n] = (-1)**n/np.sqrt(N)
        else:
            for n in range(N):
                Phi[2*k-1,n] = np.cos(2*np.pi/N*k*n)
            Phi[2*k-1,:] = Phi[2*k-1,:] / np.linalg.norm(Phi[2*k-1,:])
            for n in range(N):
                Phi[2*k,n] = np.sin(2*np.pi/N*k*n)
            Phi[2*k,:] = Phi[2*k,:] / np.linalg.norm(Phi[2*k,:])
    return Phi
