#bin/python

from generator import *
from processor import *

patch_size = 10
im = imread('new_image.jpg')
G = create_graph(im,patch_size)
community = process_graph(G)
save_community_snapshot(im, G, community, patch_size)
