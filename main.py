#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Image

import scipy as sp

from generate import create_graph, localize
import random_walk as rw

from saliency_framework import salient_node

IMAGE_PATH = '16x16.png'
PIXEL_GROUP_SIZE = 4

image = Image.open(IMAGE_PATH)
graph, maxx, maxy = create_graph(image, PIXEL_GROUP_SIZE)

Ei_Ti, Ei_Tj, Epi_Ti = rw.do_global(graph)

# Use a size of 1.5, because sqrt(2) < 1.5 < sqrt(3)
# In other words, it will restrict the localization to the eight patches just
# surrounding a given patch
local_graph = localize(graph, 1.5)

local_Ei_Ti, local_Ei_Tj, local_Epi_Ti = rw.do_local(local_graph)

print 'Global Epi_Ti', Epi_Ti
print 'Local Epi_Ti', local_Epi_Ti

node_number = salient_node(Epi_Ti, local_Epi_Ti)
print 'Salient node number:', node_number
print 'Salient node coordinates: (', (node_number / (maxx/PIXEL_GROUP_SIZE) + 1), (node_number % (maxy/PIXEL_GROUP_SIZE) + 1), ')'
#print multiple_nodes
#print 'All most-salient-nodes:'
#for node in multiple_nodes:
#    print (node / (maxx/PIXEL_GROUP_SIZE) + 1), (node % (maxy/PIXEL_GROUP_SIZE) + 1)

