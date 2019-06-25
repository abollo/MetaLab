# Pytorch image segmentation    https://github.com/CSAILVision/semantic-segmentation-pytorch
# Pytorch edge detection        Learning to Predict Crisp Boundaries
#           https://github.com/sniklaus/pytorch-hed https://github.com/meteorshowers/RCF-pytorch
#           https://github.com/DCurro/CannyEdgePytorch

from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    plt.show()
    return fig, ax

def segment_0(path):
    #image = data.binary_blobs()
    image = data.load(path)
    gray=image[:,:,0]
    val = filters.threshold_local(gray,block_size=131)
    val = gray > val
    #mask = gray < val
    #image_show(gray)
    image_show(val)
    return

    seg.active_contour(gray)
    #segmented = image > (value concluded from histogram i.e 50, 70, 120)
    text_threshold = filters.threshold_local(gray,block_size=51, offset=10)  # Hit tab with the cursor after the underscore to get all the methods.
    image_show(gray < text_threshold);
    #image_show(segmented);
    #image_show(image)

if __name__ == '__main__':
    segment_0("E:\MetaLab\hyper_9.jpg")
