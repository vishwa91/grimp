#!/bin/python

import os
from scipy import *
from scipy.ndimage import *
import matplotlib.pyplot as plt
from time import clock
import networkx as nx
import Image
import community

def community_detect(G):
    '''
        Try implementing community detection using our own algorithms.
    '''
    
def process_graph(imgraph):
    """
    This routine will process the graph and attempt to divide it into
    communities.
    """

    # The values of the partition will represent the community to which they
    # belong. The keys are the node numbers.

    partition = community.best_partition(imgraph)
    n_communities = max(partition.values()) + 1
    n_nodes = len(partition)
    communities = range(n_communities)

    # The next one is a sad hack. Need to find an elegant way.
    for i in range(n_communities):
        communities[i] = []
        
    for i in range(n_nodes):
        communities[partition[i]].append(i)
        
    return communities, partition

def _draw_line(im, point1, point2):
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

def save_community_snapshot(im, imgraph, community, patch_size=8):
    '''
        This routine is for getting a visual insight into the community
        partition. The routine highlights the communities and saves the images
        in the ./images folder.
    '''
    im_temp = im.copy()     # Save a copy, which we will meddle with.

    positions = []
    count = 0
    try:
        os.mkdir('images')
    except OSError:
        pass
    
    for group in community:
        pos = []
        old_pos = None
        im1 = im_temp.copy()    # This image will be saved
        if len(group) == 1:
            print 'Seems like one of the high entropy regions. Will not be saved.'
            continue
        for index in group:
            x, y = imgraph.node[index]['pos']
            pos.append([x, y])
            if old_pos == None:
                old_pos = [x, y]
                im[x-2:x+2, y-2:y+2] = [0,0,255]
            else:
                im = _draw_line(im, old_pos, [x, y])
                old_pos = [x, y]
            X = patch_size // 2
            im1[x-X:x+X, y-X:y+X] += 50
        im[x-2:x+2, y-2:y+2] = [255,255,0]    
        
        Image.fromarray(im1).convert('RGB').save('images/im'+str(count)+'.jpg')
        count += 1
        positions.append(pos)
    Image.fromarray(im).convert('RGB').save('images/group_snapshot.jpg')
    return positions
        
def save_partition_snapshot(imgraph, partition):
    """
    Save the partition shapshot. This is just for reference. This is a slow
    routine and does not give much insight into the community. Use
    save_community_snapshot instead.
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
