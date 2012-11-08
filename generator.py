#!/bin/python

from scipy import *
from scipy.linalg import *
from scipy.ndimage import *
from scipy.signal import convolve2d as conv
from scipy.special import *

import networkx as nx

def _preprocess_image(im, patch_size=8):
    '''
        This routine processes the image to return the Y,Cb,Cr components and
        also truncates the image, so that the dimensions are integral
        multiples of the patch size.
    '''

    x, y, z = im.shape
    x_residue = x % patch_size
    y_residue = y % patch_size

    xdim = x - x_residue
    ydim = y - y_residue

    # Create the Y, Cb, Cr channels. More info about YCbCr domain can be
    # found in: http://en.wikipedia.org/wiki/YCbCr

    Y = (0.229*im[:xdim,:ydim,0] + 0.587*im[:xdim,:ydim,1] +
         0.114*im[:xdim,:ydim,2])
    Cb = (128 - 0.168763*im[:xdim,:ydim,0] - 0.331264*im[:xdim,:ydim,1] - 
          0.5*im[:xdim,:ydim,2])
    Cr = (128 + 0.5*im[:xdim,:ydim,0] - 0.418688*im[:xdim,:ydim,1] - 
          0.081312*im[:xdim,:ydim,2])

    return array([Y, Cb, Cr])

def _create_feature_vector(im, patch_size = 8):
    '''
        Creates the feature vector for the images, which will be passed
        to the create_graph function.
        The features are:
        1. Cb of the patch
        2. Cr of the patch
        3. Ep at 0.5 scaling
        4. Ep at 1.0 scaling
        5. Ep at 1.5 scaling
        6. Ep at 2.0 scaling
        7. Ep at 2.5 scaling
        8. Distance between patch centers.

        The first 7 features have exponential weightage but distance is
        logarithimic.
    '''
    imY, imCb, imCr = im
    xdim, ydim = imY.shape
    iterx = xdim / patch_size
    itery = ydim / patch_size
    fvector = []

    # Create the differentiation kernels for orientation histogram
    kernx = array([-1, 0, 1,
                   -1, 0, 1,
                   -1, 0, 1]).reshape(3,3)
    kerny = array([-1, -1, -1,
                    0,  0,  0,
                    1,  1,  1]).reshape(3,3)

    for i in range(iterx):
        for j in range(itery):
            x1 = i * patch_size
            y1 = j * patch_size
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            imchunkY = imY[x1:x2, y1:y2]
            imchunkCb = imCb[x1:x2, y1:y2]
            imchunkCr = imCr[x1:x2, y1:y2]

            # Find the average Cb and Cr values.
            Cb = sum(imchunkCb) / (patch_size * patch_size)
            Cr = sum(imchunkCr) / (patch_size * patch_size)

            # Find the position of the center of the patch
            xpos = (x1 + x2)/2
            ypos = (y1 + y2)/2
            pos = [xpos, ypos]
            fvector.append([Cb, Cr, pos])
            # We now need to find Entropy at 5 scales.
            scale_vector = [0.5, 1.0, 1.5, 2.0, 2.5]
            
            for var in scale_vector:                
                # Calculate the orientation histogram
                
                nbins = 18
                imchunk = gaussian_filter(imchunkY, sqrt(var))
                imx = conv(imchunk, kernx)
                imy = conv(imchunk, kerny)
                grad = 180 + 180*arctan2(imy, imx)/pi

                # Segregate the orientations into bins
                bin_len_l = 0
                bin_len_h = 0
                hist = []
                for k in range(nbins):
                    bin_len_l = bin_len_h
                    bin_len_h += 360.0 / nbins
                    x, y = where((imchunk <= bin_len_h)*(imchunk > bin_len_l))
                    hist.append(len(x))
                H = array(hist)
                x = where(H == 0)       # This is to avoid log(0)
                H[x] = 1

                # Calculate the entropy.
                E = sum(H * log(H))
                fvector[-1].append(E)
    return fvector

def _create_graph(fvector, xdim, ydim):
    '''
        Create a graph for the given feature vector.
    '''

    node_count = len(fvector)
    G = nx.Graph()
    for i in range(node_count):
        Cb, Cr, pos, e1, e2, e3, e4, e5 = fvector[i]
        G.add_node(i, Cb=Cb, Cr=Cr, pos=pos, e1=e1,
                   e2=e2,e3=e3,e4=e4,e5=e5)
        # To save time, create edges simultaneously.

        if i == 0:
            pass
        else:
            count = 0
            for node in [d for n,d in G.nodes_iter(data=True)]:
                Cb0 = node['Cb']
                Cr0 = node['Cr']
                pos0 = node['pos']
                e10 = node['e1']
                e20 = node['e2']
                e30 = node['e3']
                e40 = node['e4']
                e50 = node['e5']

                # Create the differences vector
                V = array([Cb0-Cb, Cr0-Cr, e1-e10,
                           e2-e20, e3-e30, e4-e40, e5-e50])
                dist_max = hypot(xdim, ydim) * 1.0
                dist = hypot(pos[0]-pos0[0], pos[1]-pos0[1])

                weight = exp(-dot(V, V.T)) * log(dist_max/(1+dist))
                if weight > 10e-10:
                    G.add_edge(i, count, weight=weight)
                count += 1
    return G

def create_graph(im, patch_size=8):
    '''
        The main routine which should be called for creating the image graph
    '''
    im = _preprocess_image(im, patch_size)
    fvector = _create_feature_vector(im, patch_size)
    x, y, z = im.shape

    return _create_graph(fvector, x, y)
if __name__ == '__main__':
    import time
    im = imread('ball1.jpg')
    print time.clock()
    im = _preprocess_image(im)
    fvector = _create_feature_vector(im)
    x, y, z = im.shape
    G = _create_graph(fvector, x, y)
    print time.clock()
