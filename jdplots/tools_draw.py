import matplotlib.pyplot as plt
import numpy as np

def draw_WH(W,H,k=0):
    # Plot the W and H of NMF
    f, ax = plt.subplots(*(1,2))
    #print(W.transpose()[0])
    ax[0].plot((W.transpose()[k]))
    #ax.hold()
    #print(wf)
    ax[0].set_title('w_fk (k=%d)' % (k))
    ax[0].set_xlabel('f')

    ax[1].plot(H[k])
    ax[1].set_title('h_kn (k=%d)' % (k))
    ax[1].set_xlabel('n')
    #ax[1].set_ylim(0,1.5*np.max(H))

def draw_atoms(Phi0,title,F=8):
    F0 = Phi0.shape[0]
    shape_to_plot = (int(F/2), 2)
    n_atoms = np.prod(shape_to_plot)
    idx_to_plot = np.arange(F0)
    #print(idx_to_plot)
    f, ax = plt.subplots(*shape_to_plot)
    f.suptitle(title)
    cmin = min(np.min(Phi0), -np.max(Phi0))
    cmax = -cmin
    for axe, idx in zip(ax.ravel(), idx_to_plot):
        axe.plot(Phi0[idx])
        axe.set_title('atom id='+str(idx))
        axe.set_ylim(ymin=cmin,ymax=cmax)
#    plt.show()

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax
