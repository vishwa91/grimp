#bin/python

from generator import *
from processor import *
import metis

patch_size = 8
imname = 'src/new_image.jpg'
im = imread(imname)
imtemp = im.copy()
print 'Starting graph processing'
G = create_graph(im,patch_size)
comm, partition = process_graph(G)
save_community_snapshot(im, G, comm, patch_size)
node_list = [d for n,d in G.nodes_iter(data=True)]
print 'Level Zero done.'
(edgecuts, parts) = metis.part_graph(G, 2, recursive = True)

for i in range(len(parts)):
    if parts[i]==1:
        x, y = G.node[i]['pos']
        t = patch_size // 2
        imtemp[x-t:x+t, y-t:y+t] += 50
Image.fromarray(imtemp).convert('RGB').save('partition_check.jpg')
