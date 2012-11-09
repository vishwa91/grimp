#bin/python

from generator import *
from processor import *
from canny import Canny
patch_size = 8
imname = 'flower.jpg'
im = imread(imname)
im_edge = Canny(imname, 1).grad[1:-1, 1:-1]
im_high = empty_like(im)
im_high[:,:,0] = im_edge[:,:]
im_high[:,:,1] = im_edge[:,:]
im_high[:,:,2] = im_edge[:,:]
G = create_graph(gaussian_filter(im, 2)+im_high,patch_size)
community = process_graph(G)
save_community_snapshot(im, G, community, patch_size)
