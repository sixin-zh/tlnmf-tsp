from time import time
from os import path, mkdir

import matplotlib.pyplot as plt
plt.style.use(['science','ieee'])

import numpy as np
import pickle

import soundfile as sf
from scipy import fftpack
import scipy.io as sio
import scipy.signal as sig
import scipy

from utils import plot_atoms8, load_obj, getwin

def show_Phi(Phis,idx_sorted):    
    n_atoms = 8
    shape_to_plot = (4, 2)
    plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
    f, ax = plt.subplots(*shape_to_plot)
    f.set_size_inches(18, 6)    
    idx_to_plot = idx_sorted[-n_atoms:][::-1]
    for axe, idx in zip(ax.ravel(), range(0,n_atoms)):
        axe.plot(Phis[idx])
        axe.axis('off')
    plt.savefig('./figs/notes_solutions' + str(plotid) + '_' + mode + '_Phi.png',dpi=80)

def show_WH(K,N,F,Ws,H,FONT_SIZE = 20):
    plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot(121)
    if K>=2:
        plt.plot(range(1,F+1),Ws[:,0],'o',markersize=12)
        plt.plot(range(1,F+1),Ws[:,1],'.',markersize=12)
        if K==3:
            plt.plot(range(1,F+1),Ws[:,2],'x',markersize=12)

    plt.title('$[\mathbf{W}]_{mk}$',size=FONT_SIZE)
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

    plt.title('$[\mathbf{H}]_{kn}$',size=FONT_SIZE)
    plt.xlabel('n',size=FONT_SIZE)
    plt.xlim(0,N)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE)

    plt.savefig('./figs/notes_solutions' + str(plotid) + '_' + mode + '_WH.png')

for plotid in [0,2]:
    for mode in ['tlnmf','jdnmf']:
        dic = load_obj('./notes_solutions' + str(plotid) + '_' + mode)
        
        Phis = dic['Phis'] 
        idx_sorted = dic['idx_sorted']
        Ws = dic['Ws']
        K = dic['K']
        N = dic['N']
        F = dic['F']
        H = dic['H']


        #show_Phi(Phis,idx_sorted)
        show_WH(K,N,F,Ws,H)
