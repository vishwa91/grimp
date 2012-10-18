#!/bin/python

from scipy import *
from scipy.linalg import *
from scipy.ndimage import *
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d as conv
import matplotlib.pyplot as plt
import networkx as nx
import Image

'''
We need to decide what comprises our feature vector. We can use:
1. RGB intensities
2. Gradient of the patch. Hence we need to decide patch size also
3. Position of the patch in the image. Because we can say that patches
   which are spacially close tend to be more similar.
4. I am not sure, but we could use fourier transform also, since frequency
   content tend to be same.

There is not hard and fast rule, but we could use 8x8 pixels as a patch
Once we are done with these parameters, we need to create edges. I suggest
creating weighted network, weight being proportional to how similar the patches
are. Once again, we need to quantify what we mean by similar patches.

Once we have a weighted network, we could go ahead and calculate centralities.
Not that this is just for the sake of doing it, but because I feel that every
object has a single node which has high centrality.

Then, the final part is to do a community detection. However, if we know for
sure that we will have X nodes which have high centrality, we could as well
do graph partitioning.

'''
PATCH_SIZE = 8

im = imread('template.jpg')
imR = im[:,:,0]
imG = im[:,:,1]
imB = im[:,:,2]

x, y = imR.shape
residX = x % PATCH_SIZE
residY = y % PATCH_SIZE

xdim = x - residX
ydim = y - residY

# As of now, we will just adjust the image brutely so that dimensions are
# multiples of patch size
imR = imR[:xdim, :ydim]
imG = imG[:xdim, :ydim]
imB = imB[:xdim, :ydim]

# We now need to create our feature vector. We need to iterate through the
# whole image and extract patches. Tiresome job!

iterx = xdim / PATCH_SIZE
itery = ydim / PATCH_SIZE
fvector = []
for i in range(iterx):
    for j in range(itery):
        x1 = i * PATCH_SIZE
        y1 = j * PATCH_SIZE
        x2 = x1 + PATCH_SIZE
        y2 = y1 + PATCH_SIZE

        imchunkR = imR[x1:x2, y1:y2]
        imchunkG = imG[x1:x2, y1:y2]
        imchunkB = imB[x1:x2, y1:y2]

        # Our first features will be average R, G, B values
        Rval = sum(imchunkR) / (PATCH_SIZE*PATCH_SIZE)
        Gval = sum(imchunkG) / (PATCH_SIZE*PATCH_SIZE)
        Bval = sum(imchunkB) / (PATCH_SIZE*PATCH_SIZE)

        # Next feature should be the position of the patch
        xpos = (x1 + x2)/2
        ypos = (y1 + y2)/2

        # We can take a fourier transform and find the maximum frequency.
        # This would tell us how non uniform is our image.

        IMR = abs(fft2(imchunkR))
        IMG = abs(fft2(imchunkG))
        IMB = abs(fft2(imchunkB))

        FXR, FYR = where(IMR == IMR.max())
        FXG, FYG = where(IMG == IMG.max())
        FXB, FYB = where(IMB == IMB.max())

        # Spacial comparision will connect nearby neighbours. However,
        # frequency content comparision may connect two plane areas. Not sure
        # how to avoid that thing. Anyways, I am not sure that would matter,
        # since our weights should take care of that.

        # As of now, create the feature vector.
        
        fvector.append([[Rval, Gval, Bval], [xpos, ypos],
                        [[FXR, FYR], [FXG, FYG], [FXB, FYB]]])

node_count = len(fvector)
imgraph = nx.Graph()
for i in range(node_count):
    node_name = 'node_'+str(i)
    Cval, pos, FFT = fvector[i]
    imgraph.add_node(i, Cval=Cval, pos=pos, FFT=FFT)
    
