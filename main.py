#bin/python

from generator import *
from processor import *
import metis

patch_size = 6
imname = 'src/microsoft.jpg'
im = imread(imname)
imtemp = im.copy()
print 'Starting graph processing'
G, pos_vector = create_graph(im,patch_size)
G.graph['edge_weight_attr'] = 'weight'
#G.graph['node_weight_attr'] = 'e1'
comm, partition = process_graph(G)
save_community_snapshot(im, G, comm, patch_size)
node_list = [d for n,d in G.nodes_iter(data=True)]
print 'Level Zero done.'
(edgecuts, parts) = metis.part_graph(G, nparts=2, contig=False)
nparts = max(parts)
for p in range(nparts+1):
    imout = imtemp.copy()
    for i in range(len(parts)):
        if parts[i]==p:
            x, y = G.node[i]['pos']
            t = patch_size // 2
            imout[x-t:x+t, y-t:y+t] += 50
    Image.fromarray(imout).convert('RGB').save('partition_check'+str(p)+'.jpg')

imgen = im.copy()
for i in G.nodes(True):
    if(i[1]['e1']):
        x, y = i[1]['pos']
        imgen[x-t:x+t,y-t:y+t] += 50
Image.fromarray(imgen).show()
L = nx.laplacian(G)
