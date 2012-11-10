#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Image

from networkx import read_yaml, write_yaml

from generate import create_graph
import random_walk as rw

GRAPH_FILE_PATH = 'graph_of_template.yaml'
IMAGE_PATH = 'block.jpg'

#try:
#    graph = read_yaml(GRAPH_FILE_PATH)
#except IOError:
graph = create_graph(Image.open(IMAGE_PATH))

#write_yaml(graph, GRAPH_FILE_PATH)

n = graph.number_of_nodes()

P = rw.generate_transition_matrix(graph, n)
print 'Done getting P'

eq_pi = rw.equilibrium_distribution(P)
print 'Done getting pi'

W = rw.equilibrium_transition_matrix(eq_pi, n)
print 'Done getting W'

Z = rw.fundamental_matrix(P, W, n)
print 'Done getting Z'

Ei_Ti, Ei_Tj, Epi_Ti = rw.hitting_times(eq_pi, Z, n)
print 'Done getting hitting times'
