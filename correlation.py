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
import community

def corr2d(mat1, mat2):
    '''
        Normalised cross correlation of two 2D matrices. The theory can be
        found at http://en.wikipedia.org/wiki/Cross-correlation
    '''

    sigma1 = standard_deviation(mat1)
    sigma2 = standard_deviation(mat2)

    product = (mat1 - mean(mat1)) * (mat2 - mean(mat2))/(sigma1 * sigma2)
    x, y = mat1.shape

    corr_coef = sum(product)/(1.0*x*y)
    return corr_coef

class correlate_graph:
    """
    Create instances of this class to implement object detection using cross
    correlation of image patches.
    """
    def __init__(self, im, patch_size = 8):
        
        self.imR = im[:,:,0]
        self.imG = im[:,:,1]
        self.imB = im[:,:,2]
        self.patch_size = patch_size

        x, y = self.imR.shape
        resid_x = x % self.patch_size
        resid_y = y % self.patch_size

        self.xdim = x - resid_x
        self.ydim = y - resid_y
        self.fvector = self.create_features()
        self.graph = self.create_graph(self.fvector)
        
    def create_features(self):
        '''
            Not much to do here but to return the patches and position of the
            pacth
        '''

        fvector = []
        iterx = self.xdim // self.patch_size
        itery = self.ydim // self.patch_size
        for i in range(iterx):
            for j in range(itery):
                x1 = i * self.patch_size
                x2 = x1 + self.patch_size
                y1 = j * self.patch_size
                y2 = y1 + self.patch_size
                
                imchunkR = self.imR[x1:x2, y1:y2]
                imchunkG = self.imG[x1:x2, y1:y2]
                imchunkB = self.imB[x1:x2, y1:y2]
                pos = [(x2+x1)/2, (y2+y1)/2]
                fvector.append([imchunkR, imchunkG, imchunkB, pos])
        return fvector

    def create_graph(self, fvector):
        n_nodes = len(fvector)
        # Create nodes first
        G = nx.Graph()
        for i in range(n_nodes):
            G.add_node(i, pos=fvector[i][3])

        # Create edges now. The weight will be proportional to the correlation
        # of the 3 channels
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                imR1 = fvector[i][0]
                imR2 = fvector[j][0]
                #corR = corrcoef(imR1, imR2)
                #x = where(isnan(corR))
                #corR[x] = 1.

                imG1 = fvector[i][1]
                imG2 = fvector[j][1]
                #corG = corrcoef(imG1, imG2)
                #x = where(isnan(corG))
                #corG[x] = 1.

                imB1 = fvector[i][2]
                imB2 = fvector[j][2]
                #corB = corrcoef(imB1, imB2)
                #x = where(isnan(corB))
                #corB[x] = 1.
                cor_coef = corr2d((imR1+imG1+imB1)/3, (imR2+imG2+imB2)/3)
                x1, y1 = fvector[i][3]
                x2, y2 = fvector[j][3]
                dist = hypot(x2-x1, y2-y1)
                weight = cor_coef.prod() / (2 + dist)
                #print weight
                #weight = corR.prod() * corG.prod() * corB.prod()
                if (weight > 10e-80):
                    G.add_edge(i, j, weight=weight)
        return G

def process_graph(imgraph, partition):
    """
    This routine will process the graph and attempt to divide it into
    communities.
    """

    # The values of the partition will represent the community to which they
    # belong. The keys are the node numbers.
    
    n_communities = max(partition.values()) + 1
    n_nodes = len(partition)
    communities = range(n_communities)

    # The next one is a sad hack. Need to find an elegant way.
    for i in range(n_communities):
        communities[i] = []
        
    for i in range(n_nodes):
        communities[partition[i]].append(i)
        
    return communities

def save_partition_snapshot(imgraph, partition):
    """
    Save the partition shapshot. This is just for reference.
    """
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
    plt.savefig('images/__partition_snapshot.png')

def draw_line(im, point1, point2):
    """
        Routine to draw a line between two images in a ndimage array.
        This can also be done in PIL image, but it may take time to convert
        from ndarray to PIL and back again
    """
    x1, y1 = point1
    x2, y2 = point2

    dist = hypot(x2-x1, y2-y1)
    theta = arctan2(y2-y1, x2-x1)

    r = array(range(int(dist +1)))
    x = x1 + r * cos(theta)
    y = y1 + r * sin(theta)
    x = x.astype(int)
    y = y.astype(int)
    im[x, y] = 0
    im[x1-1:x1+1, y1-1:y1+1] = [255,0,0]
    im[x2-1:x2+1, y2-1:y2+1] = [255,0,0]

    return im

im = imread('mlogo.jpg')
imgraph = correlate_graph(im, 8).graph
partition = community.best_partition(imgraph)
comm = process_graph(imgraph, partition)

positions = []
im_temp = im.copy()
count = 0
for group in comm:
    pos = []
    old_pos = None
    im1 = im_temp.copy()
    for index in group:
        x, y = imgraph.node[index]['pos']
        pos.append([x, y])
        if old_pos == None:
            old_pos = [x, y]
            im[x-2:x+2, y-2:y+2] = [0,0,255]
        else:
            im = draw_line(im, old_pos, [x, y])
            old_pos = [x, y]
        X = 4
        im1[x-X:x+X, y-X:y+X] += 50
    im[x-2:x+2, y-2:y+2] = [255,255,0]    
    
    Image.fromarray(im1).convert('RGB').save('images/im'+str(count)+'.jpg')
    count += 1
    positions.append(pos)

count = 0
for t in positions:
    t = array(t)
    x1 = min(t[:,0])
    x2 = max(t[:,0])
    y1 = min(t[:,1])
    y2 = max(t[:,1])
    im1 = copy(im)
    im[x1:x2, y1] = 0
    im[x1:x2, y2] = 0
    im[x1, y1:y2] = 0
    im[x2, y1:y2] = 0
    #Image.fromarray(im1).convert('RGB').save('images/im'+str(count)+'.jpg')
    count += 1
Image.fromarray(im).show()
save_partition_snapshot(imgraph, partition)
