#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Image

import scipy as sp
from networkx import read_yaml, write_yaml

from generate import create_graph, localize
import random_walk as rw

from saliency_framework import salient_node

GRAPH_FILE_PATH = 'graph_of_template.yaml'
IMAGE_PATH = '16x16.png'

image = Image.open(IMAGE_PATH)
graph = create_graph(image, 4)

n = graph.number_of_nodes()

print n

P = rw.generate_transition_matrix(graph, n)
print 'Probability transition marix:'
print sp.where(P < 0.001, sp.zeros((n, n)), sp.around(P, 3))
print 'Done getting P'

eq_pi = rw.equilibrium_distribution(P)
print 'Eigenvector: '
print eq_pi
print 'Done getting pi'

W = rw.equilibrium_transition_matrix(eq_pi, n)
print 'Done getting W'

Z = rw.fundamental_matrix(P, W, n)
print 'Done getting Z'

Ei_Ti, Ei_Tj, Epi_Ti = rw.hitting_times(eq_pi, Z, n)
print 'Done getting hitting times'

# Use a size of 1.5, because sqrt(2) < 1.5 < sqrt(3)
# In other words, it will restrict the localization to the eight patches just
# surrounding a given patch
local_graph = localize(graph, 1.5)

local_n = local_graph.number_of_nodes()

local_P = rw.generate_transition_matrix(local_graph, local_n)
print 'Done getting local P'

print
print local_P
print

local_eq_pi = rw.equilibrium_distribution(local_P)
print 'Done getting local pi'

print
print local_eq_pi
print

local_W = rw.equilibrium_transition_matrix(local_eq_pi, local_n)
print 'Done getting local W'

local_Z = rw.fundamental_matrix(local_P, local_W, local_n)
print 'Done getting local Z'

local_Ei_Ti, local_Ei_Tj, local_Epi_Ti = rw.hitting_times(local_eq_pi, local_Z,
                                                          local_n)
print 'Done getting local hitting times'

print salient_node(Ei_Ti, local_Ei_Ti)
