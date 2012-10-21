#!/bin/python

from scipy import *
from scipy.linalg import *
from scipy.ndimage import *
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d as conv
from scipy.special import *
import matplotlib.pyplot as plt
from time import clock
import networkx as nx
import Image

'''
As of now, we have frozen the following features:
1. Y
2. Cb
3. Cr
4. Entropy at 1:1 scale.
5. Position

For the entropy, we need the orientation histogram. We will use simple sobel
operator to find the gradient at each point and construct the histogram with
9 channels of angles.

'''
PATCH_SIZE = 8

im = imread('images.jpg')
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

kernx = array([-1, 0, 1,
               -1, 0, 1,
               -1, 0, 1]).reshape(3,3)
kerny = array([-1, -1, -1,
                0,  0,  0,
                1,  1,  1]).reshape(3,3)
for i in range(iterx):
    for j in range(itery):
        x1 = i * PATCH_SIZE
        y1 = j * PATCH_SIZE
        x2 = x1 + PATCH_SIZE
        y2 = y1 + PATCH_SIZE

        imchunkR = imR[x1:x2, y1:y2]
        imchunkG = imG[x1:x2, y1:y2]
        imchunkB = imB[x1:x2, y1:y2]

        # Our first features will be Y, Cb, Cr values
        Rval = sum(imchunkR) / (PATCH_SIZE*PATCH_SIZE)
        Gval = sum(imchunkG) / (PATCH_SIZE*PATCH_SIZE)
        Bval = sum(imchunkB) / (PATCH_SIZE*PATCH_SIZE)

        # Calculation of Y Cb Cr from R G B is from wikipedia:
        # http://en.wikipedia.org/wiki/YCbCr
        Y = 0.229*Rval + 0.587*Gval + 0.114*Bval
        Cb = 128 - 0.168763*Rval - 0.331264*Gval - 0.5*Bval
        Cr = 128 + 0.5*Rval - 0.418688*Gval - 0.081312*Bval

        # Next feature should be the position of the patch
        xpos = (x1 + x2)/2
        ypos = (y1 + y2)/2
        pos = [xpos, ypos]

        # We need to now calculate the orientation histogram. We will use
        # differentiation operator for the same
        imchunk = (imchunkR + imchunkG + imchunkB)/3
        imx = conv(imchunk, kernx)
        imy = conv(imchunk, kerny)
        grad = 180 + 180*arctan2(imy, imx)/3.1415926

        # We will now segregate our gradient map into bins
        bin_len_l = 0
        bin_len_h = 0
        hist = []
        for i in range(9):
            bin_len_l = bin_len_h
            bin_len_h += 40
            x, y = where((imchunk <= bin_len_h)*(imchunk > bin_len_l))
            hist.append(len(x))
        H = array(hist)
        x = where(H == 0)
        H[x] = 1

        # Entropy formula is emperical(For us!)
        entropy = sum(H * log(H))

        fvector.append([Y, Cb, Cr, pos, entropy])        

t1 = clock()
node_count = len(fvector)
imgraph = nx.Graph()
for i in range(node_count):
    node_name = 'node_'+str(i)
    Y, Cb, Cr, pos, entropy = fvector[i]
    imgraph.add_node(i, Y=Y, Cb = Cb, Cr = Cr,
                     pos=pos, entropy=entropy)

# We have created our nodes. Now we need to create edges. We can start off by
# saying that weight is proportional to exp(-distance). This will ensure that
# farther nodes have as less weight as possible.

# As of now, we will create a clique out of the whole graph

for i in range(node_count):
    for j in range(i, node_count):
        
        Y1 = imgraph.node[i]['Y']
        Y2 = imgraph.node[j]['Y']
        Cb1 = imgraph.node[i]['Cb']
        Cb2 = imgraph.node[j]['Cb']
        Cr1 = imgraph.node[i]['Cr']
        Cr2 = imgraph.node[j]['Cr']
        x1,y1 = imgraph.node[i]['pos']
        x1,y1 = imgraph.node[j]['pos']
        E1 = imgraph.node[i]['entropy']
        E2 = imgraph.node[j]['entropy']

        v1 = array([Y2-Y1, Cb2-Cb1, Cr2-Cr1, E2-E1])
        weight = exp(-1 * dot(v1, v1.T)) * 100 / hypot(x2-x1, y2-y1)
        if weight > 10e-80:
            #print weight
            imgraph.add_edge(i, j, weight=weight)
    
t2 = clock()
print t2-t1

import community
partition = community.best_partition(imgraph)

#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(imgraph)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(imgraph, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))

nx.draw_networkx_edges(imgraph,pos, alpha=0.5)
plt.show()

components = nx.connected_components(imgraph)
bounding_box = []
im1 = copy(im)
for c in components:
    x1, y1 = imgraph.node[c[0]]['pos']
    x2, y2 = imgraph.node[c[0]]['pos']
    for node_index in c:
        x, y = imgraph.node[node_index]['pos']
        if (x < x1) or (y < y1):
            x1 = x
            y1 = y
        elif (x > x2) or (y > y2):
            x2 = x
            y2 = y
        im1[x,y,:] = 255
    bounding_box.append([x1, y1, x2, y2])

for bound in bounding_box:
    x1, y1, x2, y2 = bound
    im[x1:x2, y1, :] = 0
    im[x1:x2, y2, :] = 0
    im[x1, y1:y2, :] = 0
    im[x2, y1:y2, :] = 0
Image.fromarray(im).convert('RGB').save('attempt1.jpg')

