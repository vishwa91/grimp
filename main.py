#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import Image
import scipy as sp
import numpy as np
import scipy.ndimage as ni
import networkx as nx
import itertools

FEATURE_VECTOR_SIZE = 7
NUM_HISTOGRAM_BUCKETS = 18 # This value actually has to be chosen carefully.
                           # If there are too many buckets in comparison to the
                           # pixel group size, then some buckets may end up
                           # having 0. Then, when we take log(histogram), that
                           # instance will go to negative infinity!

#TODO: this can be removed as it is not being used.
class HashableNdarray(object):
    """
    Wrapper class around ``feature_vector`` in order to make it hashable.
    numpy.ndarrays are not hashable, but they must be in order to make them
    into a node on a networkx.Graph.
    """
    # User defined classes are hashable by default. They compare unequal with
    # any object but themselves and their hash value is their id.
    # http://docs.python.org/reference/datamodel.html#object.__hash__
    
    def __init__(self, feature_vector):
        self.feature_vector = feature_vector

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
    i = 0
    for patch in patches:
        graph.add_node(i, feature_vector=_create_feature_vector(source[patch]))
        i += 1

def _create_edges(graph):
    """
    Creates the edges of the graph by setting the weights of connected nodes
    """
    n = graph.number_of_nodes()
    for i in range(n):
        for j in range(i+1, n):
            weight = sp.exp( - (graph.node[i]['feature_vector']  
                                - graph.node[j]['feature_vector']) ** 2 )
            graph.add_edge(i, j, weight=weight)

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
    source = sp.array(source)
    
    # We now have a ``source`` vector which contains the basic elements from
    # which the feature vector of each node will be generated
    print 'Done creating the source vector'
    
    # We now need to split up the image into patches of size pixel_group_size
    # First, we create a list of patches. Each 'patch' in this list will be a 
    # tuple describing an array slice of the image that produces this patch.
    patches = []
    i = pixel_group_size
    while(i < maxx):
        j = pixel_group_size
        while(j < maxy):
            # We want to extract source[:, x:x+delta_x, y:y+delta_y]
            patches.append(( slice(0, FEATURE_VECTOR_SIZE + 1),
                             slice(i-pixel_group_size, i), 
                             slice(j-pixel_group_size, j),
                          ))
            j += pixel_group_size
        i += pixel_group_size
    print 'Done making patches'
    
    # Now we proceed to actually create the graph
    graph = nx.Graph()
    _create_nodes(source, graph, patches)
    print 'Done creating nodes'
    _create_edges(graph)
    print 'Done creating edges'
    
create_graph(Image.open('template.jpg'))
