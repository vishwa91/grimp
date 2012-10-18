#!/bin/python

from scipy import *
from scipy.linalg import *
from scipy.ndimage import *
import Image

'''
We need to decide what comprises our feature vector. We can use:
1. RGB intensities
2. Gradient of the patch. Hence we need to decide patch size also
3. Position of the patch in the image. Because we can say that patches
   which are spacially close tend to be more similar.
4. I am not sure, but we could use fourier transform also, since frequency
   content tend to be same.

There is not hard and fast rule, but we could use 8x8 pixels as a patch
Once we are done with these parameters, we need to create edges. I suggest
creating weighted network, weight being proportional to how similar the patches
are. Once again, we need to quantify what we mean by similar patches.

Once we have a weighted network, we could go ahead and calculate centralities.
Not that this is just for the sake of doing it, but because I feel that every
object has a single node which has high centrality.

Then, the final part is to do a community detection. However, if we know for
sure that we will have X nodes which have high centrality, we could as well
do graph partitioning.

'''
