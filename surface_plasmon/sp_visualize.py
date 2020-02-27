'''
@Author: Yingshi Chen

@Date: 2019-11-26 16:10:19
@
# Description: 
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import pickle
import seaborn as sns;      sns.set()
import cv2
data_root="D:/MetaLab"
pkl_acti_visual=f"{data_root}/dump/all_activations.pkl"

def HeatMap(data,title="",noAxis=True,cbar=True):
    dpi = 100
    dump_dir = f"{data_root}/dump/"
    sns.set(font_scale=2)
    assert (data is not None)
    #xitas = args.xitas
    ticks = np.linspace(0, 1, 10)
    #xlabels = [int(i) for i in np.linspace(300, 2000, 10)]
    #ylabels = ["{:.3g}".format(i) for i in np.linspace(x0, x1, 10)]

    s = max(data.shape[1] / dpi, data.shape[0] / dpi)

    #fig.set_size_inches(18.5, 10.5)
    cmap = 'coolwarm'  # "plasma"  #https://matplotlib.org/examples/color/colormaps_reference.html
    #cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    figsize = (s , s )
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax = sns.heatmap(data, ax=ax, cmap=cmap,cbar=False, xticklabels=False, yticklabels=False)
    path = '{}/all/{}_.jpg'.format(dump_dir,title)
    fig.savefig(path,bbox_inches='tight', pad_inches = 0)
    image = cv2.imread(path)
    #image = fig2data(ax.get_figure())      #会放大尺寸，难以理解
    cv2.imshow("",image);       cv2.waitKey(0)
    plt.close("all")
    return image,path

#{'name':name,'shape':acti.shape,'activation':acti}
def plot_activations(acti_infos,img_path=None):
    layer_activation = acti_infos['activation']
    n_col = 2
    n_row = layer_activation.shape[0] // n_col
    n_row = min(n_row,2)
#https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
    #f, ax = plt.subplots(n_row, n_column)
    f = plt.figure(figsize=(n_col+1, n_row+1))
    gs = gridspec.GridSpec(n_row, n_col,wspace=0.05, hspace=0.05,
                           bottom=0.0,top=1,  left=0, right=1)
    for i in range(n_row):
        for j in range(n_col):
            ax = plt.subplot(gs[i, j])
            channel_image = layer_activation[i * n_col + j,0,:,:]
            # image post-processing for better visualization
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            #if i==0 and j==0:              HeatMap(channel_image)

            ax.imshow(channel_image, cmap='coolwarm')#'viridis'
            #ax.axis('off')
            #ax.set_xticklabels([])
            #ax.set_yticklabels([])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    #f.set_size_inches(n_column, n_row)
    plt.show()
    if img_path is not None:
        plt.savefig(img_path,bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    with open(pkl_acti_visual, "rb") as fp:
        activations = pickle.load(fp)
    for acti in activations:
        plot_activations(acti)