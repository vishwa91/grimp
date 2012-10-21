Grimp
=====

An image processing program for salient node detection based on network-theoretical algorithms.

This program was primarily written as part of the course 'Networks: Models, Theory and Algorithms' for our course project.

Fundamentals of its functioning
-------------------------------

Most of what is done here is amply described in a bunch of papers:
1. http://www3.ntu.edu.sg/home/ASDRajan/cvpr09.pdf
2. http://ieeexplore.ieee.org/xpls/abs\_all.jsp?arnumber=5497152
3. http://ieeexplore.ieee.org/xpls/abs\_all.jsp?arnumber=4907085
 
For the most part, we are trying to reproduce their results, but we won't hesitate to try out a few experiments of our own, here and there.

The functioning of this program can be broadly broken down into a few different parts that are described below.

### Create a graph from the image

This process is fairly straightforward. We just bunch together every so many pixels and make it a node. We choose an 8x8 patch for this purpose. The node has certain properties that are contained in its so-called "feature vector". This feature vector contains a few elements that are understood to be crucial in determining the most important characteristics of the pixel group it represents. Thus two nodes with similar values of corresponding elements of the feature vector are said to be similar. Quite possibly therefore, the two of them are part of the same object and so on. More about this later.

### Create the feature vector

Each node has a feature vector comprising of the elements:
- The average Y-component of the pixel group
- The average C<sub>b</sub>-component
- The average C<sub>r</sub>-component
- The orientation entropy E<sub>ρ1</sub> at a scaling of 0.5
- The orientation entropy E<sub>ρ2</sub> at a scaling of 1
- The orientation entropy E<sub>ρ3</sub> at a scaling of 1.5
- The orientation entropy E<sub>ρ4</sub> at a scaling of 2
- The orientation entropy E<sub>ρ5</sub> at a scaling of 2.5

The orientation entropy itself is defined in terms of the orientation histogram and the scalings refer to the variances of the gaussian kernels convoluted with the image in order to capture the image complexity at different scales and reduce the dependency of the feature vector on the size of the pixel group. Refer to the aforementioned papers, and to http://en.wikipedia.org/wiki/Scale\_space for more information.

### Create the edges of the graph by defining a weight measure

We then proceed to define a weight measure in order to determine what the edges of the graph are. These will later be used when we perform the random walk on the graph in order to find salient objects.

The weight measure we use is defined as the exponential of the negative of the square of the norm of the difference of the feature vectors of the two nodes in consideration. (Strictly, there is also a variance term in the denominator, inside the exponential, but we fix it to unity).

Thus, we now have a fully connected graph, with the weight of an edge being closer to 1 if the two nodes are more 'similar', in that the difference of their feature vectors has a smaller norm.

### The random walk and the probability distribution of states

We now consider a random walk on this graph, starting at an arbitrary point. The walk proceeds in a manner such that it is more likely to choose an edge that haas a larger weight. The idea is that a random walk is likely to remain in the same object for a longer time, since the nodes of that object are likely to have very similar values for the elements of their feature vectors. In other words, given a starting node within a certain object, the mean first passage time of other nodes within this object should be small, while that of nodes outside this object should be large.

If we model the graph as an adjacency matrix, then the matrix also becomes similar to the probability transition matrix of the Markov chain, subject to normalisation along each column. The eigenvector with eigenvalue unity will provide the equilibrium condition of the probability distribution of states. Using this as a starting point, we can calculate mean first passage times of all nodes.
