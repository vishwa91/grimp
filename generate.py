# -*- coding: utf-8 -*-

import sys

import Image
import scipy as sp
import numpy as np
import scipy.ndimage as ni
import networkx as nx
import itertools as it

FEATURE_VECTOR_SIZE = 7
NUM_HISTOGRAM_BUCKETS = 18 # This value actually has to be chosen carefully.
                           # If there are too many buckets in comparison to the
                           # pixel group size, then some buckets may end up
                           # having 0. Then, when we take log(histogram), that
                           # instance will go to negative infinity!

def _create_feature_vector(pixel_group):
    """
    Generates the feature vector, given a square bunch of pixels.
    
    ``pixel_group`` by itself is actually a list of pixel groups (sub-images).
    Each sub-image will become a different part of the vector
    """
    
    # Initialise some values that we'll use later
    feature_vector = sp.empty(FEATURE_VECTOR_SIZE)
    num_pixels = pixel_group[0].shape[0]
    # Find the angles of each point in degrees
    x = sp.arange(-num_pixels/2, num_pixels/2)
    grid = sp.meshgrid(x, x)
    angle = sp.angle(grid[0] + 1j*grid[1], deg=True)
    
    # Create histrogram buckets and find indices of the spectrum array that
    # fall into a particular bucket
    diff = -360/NUM_HISTOGRAM_BUCKETS
    buckets = sp.arange(180, -180+diff, diff)
    indices = {}
    for i in range(0, NUM_HISTOGRAM_BUCKETS):
        indices[i] = sp.where((angle <= buckets[i]) * 
                              (angle > buckets[i+1]) )
    buckets = buckets[:-1]
    
    # Average out the Cb and Cr components and add it to the feature vector
    feature_vector[0] = sp.dot(sp.ones((1, num_pixels)), 
                               pixel_group[0].dot(sp.ones((num_pixels, 1))))
    feature_vector[0] /= num_pixels * num_pixels
    feature_vector[1] = sp.dot(sp.ones((1, num_pixels)), 
                               pixel_group[1].dot(sp.ones((num_pixels, 1))))
    feature_vector[1] /= num_pixels * num_pixels
    
    # The other five elements are the orientation entropies at different scales
    for i in range(2, FEATURE_VECTOR_SIZE):
        # First calculate the centre-shifted fourier transform of the pixel
        # group, and then apply a log transformation to get the magnitude
        # spectrum
        transformed_pixel_group = np.fft.fft2(pixel_group[i])
        centre_shifted_pixel_group = np.fft.fftshift(transformed_pixel_group)
        fourier_spectrum = sp.log(abs(centre_shifted_pixel_group) + 1)
        
        # Calculate the orientation histogram of the log magnitude spectrum
        # by summing over groups of angles. The histogram value at a given
        # angle should give the power in the log magnitude spectrum around that
        # angle (approximately)
        histogram = sp.empty(buckets.shape)
        for j in range(NUM_HISTOGRAM_BUCKETS):
            histogram[j] = fourier_spectrum[indices[j]].sum()
        
        # Finally, calculate the orientation entropy based on the standard
        # statistical formula:
        #       E = H(θ) * log(H(θ))
        if not histogram.all():
            entropy = 0
        else:
            entropy = - (histogram * sp.log(histogram)).sum()
        if sp.isnan(entropy):
            print histogram
            print fourier_spectrum
            sys.exit(1)
        feature_vector[i] = entropy
    
    return feature_vector

def _create_nodes(source, graph, patches):
    """
    Creates nodes from a source vector and a patch set and adds them to a graph
    
    Parameters
    ----------
    source : list of numpy.ndarray's
        Source list comprising the Y, Cb and Cr components, along with the
        greyscale image at different scalings.
    graph : networkx.Graph
        Graph to which to add the nodes
    patches : list of tuples 
        Each patch (tuple) specifies an array slice that depicts a pixel group
    """
    
    # Iterate over the patches and apply them to the source vector
    for patch in patches:
        graph.add_node(
            (patch.x, patch.y),
            feature_vector=_create_feature_vector(source[patch.slice])
        )
        #print 'Feature vector of node (',patch.x, patch.y,'):',np.around(graph.node[(patch.x, patch.y)]['feature_vector'])

def _create_edges(graph):
    """
    Creates the edges of the graph by setting the weights of connected nodes
    """
    nodes = graph.nodes(data = True)
    
    for (node1, node2) in it.combinations(nodes, 2):
        diff_vector = (  node1[1]['feature_vector']
                       - node2[1]['feature_vector'] )
        weight = sp.exp(-sp.dot(diff_vector, diff_vector))
        #print 'Nodes', node1[0], node2[0], ':', np.around(diff_vector), ',', weight
        graph.add_edge(node1[0], node2[0], weight=weight)

def _scale_image(image, scaling):
    """
    Creates a scaled version of the image by applying a gaussian filter whose
    variance is equal to ``scaling``.
    """
    # ``scaling`` is the variance. We need to pass standard deviation as a
    # parameter.
    sigma = sp.sqrt(scaling)
    
    # Find the scaled version of the image
    scaled_image = sp.empty(image.shape)
    ni.gaussian_filter(image, sigma, output=scaled_image)
    return scaled_image

def create_graph(pil_image, pixel_group_size=8):
    """
    Creates a graph from an image.
    
    Parameters
    ----------
    image : Image.Image
        The image to be converted into a graph
    pixel_group_size : int
        The number of pixels on the side of the square of each patch that is
        converted into a node
        
    Returns
    -------
    graph : networkx.Graph
        The graph created by grouping pixels. Blocks are determined by
        ``pixel_group_size``. Each node of the graph is a feature vector.
        Each edge of the graph is an integer weight whose value is determined
        by the 'similarity' of the feature vectors of the nodes it joins.
    """
    
    # Each node comprises of a feature vector. We need to find each element of
    # the feature vector. These are:
    # Y, Cb, Cr, Eρ1, Eρ2, Eρ3, Eρ4 and Eρ5.
    
    # First, for Y, Cb and Cr:
    Y, Cb, Cr = pil_image.convert('YCbCr').split()
    Y, Cb, Cr = [ sp.asarray(Y),
                  sp.asarray(Cb),
                  sp.asarray(Cr),
                ]
    source = [Cb, Cr]
    print 'Y: '
    print Y
    print 'Cb: '
    print Cb
    print 'Cr: '
    print Cr
    
    # For each of the orientation entropy values, we need to create a scaled
    # version of the image. For this, first convert the PIL image into a
    # scipy ndimage. Then, ``_scale_image`` will take care of the rest.
    image = sp.asarray(pil_image)
    source.extend([ _scale_image(Y, 0.5),
                    _scale_image(Y, 1),
                    _scale_image(Y, 1.5),
                    _scale_image(Y, 2),
                    _scale_image(Y, 2.5),
                 ])
    maxx = image.shape[0]
    maxy = image.shape[1]
    print 'maxx:',maxx
    print 'maxy:',maxy
    source = sp.array(source)
    
    #np.set_printoptions(threshold='nan', linewidth=130)
    print np.around(source)

    # We now have a ``source`` vector which contains the basic elements from
    # which the feature vector of each node will be generated
    print 'Done creating the source vector'
    
    # We now need to split up the image into patches of size pixel_group_size
    # First, we create a list of patches. Each 'patch' in this list will be a 
    # tuple describing an array slice of the image that produces this patch.
    class Patch(object):
        pass
    patches = []
    i = pixel_group_size
    while(i <= maxx):
        j = pixel_group_size
        while(j <= maxy):
            # We want to extract source[:, x:x+delta_x, y:y+delta_y]
            patch = Patch()
            patch.slice = (  slice(0, FEATURE_VECTOR_SIZE + 1),
                             slice(i-pixel_group_size, i), 
                             slice(j-pixel_group_size, j),
                          )
            patch.x = i / pixel_group_size - 1
            patch.y = j / pixel_group_size - 1
            patches.append(patch)
            j += pixel_group_size
        i += pixel_group_size
    print 'Number of patches:', len(patches)
    print 'Done making patches'
    
    # Now we proceed to actually create the graph
    graph = nx.Graph()
    _create_nodes(source, graph, patches)
    print 'Done creating nodes'
    _create_edges(graph)
    print 'Done creating edges'
    
    maxx -= (maxx % pixel_group_size)
    maxy -= (maxy % pixel_group_size)
    return graph, maxx, maxy

def _distance(node1, node2):
    """
    Calculates the "distance" between two nodes.
    ``node1`` and ``node2`` should be tuples with x and y coordinates.
    """
    delta_x = node1[0] - node2[0]
    delta_y = node1[1] - node2[1]
    return sp.sqrt(delta_x*delta_x + delta_y*delta_y)
    
def localize(graph, size):
    """
    Localizes an image's graph: that is, it removes edges that connect patches
    far apart from each other. The way the graph is constructed,
    """
    local_graph = graph.copy()
    # There has got to be a more efficient way of doing this...
    print 'Localizing...'
    for (node, neighbours) in graph.adjacency_iter():
        #print node
        for neighbour in neighbours:
            if(_distance(node, neighbour) >= size):
                #print '\t', neighbour, 'chucked'
                try:
                    local_graph.remove_edge(node, neighbour)
                except nx.exception.NetworkXError:
                    pass
            else:
                pass
                #print '\t', neighbour
    return local_graph
