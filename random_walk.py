# -*- coding: utf-8 -*-

import scipy as sp
import scipy.linalg as linalg
import networkx as nx

def generate_transition_matrix(graph, n=None):
    """
    Generates the probability transition matrix for a given graph. It is
    assumed that all edges have weights and that the probability of transition
    from one node to a neighbouring node is proportional to the weight of the
    edge connecting the two nodes, normalised over all current outgoing edges.
    
    Parameters
    ----------
    graph : networkx.Graph
        Graph whose transition matrix is to be found. It is assumed that the
        weight of each edge can be found under the attribute name ``weight``.
    n : int (optional)
        Number of nodes in the graph.
    
    Returns
    -------
    P : numpy.ndarray
        Markov probability transition matrix for the given graph
    
    Notes
    -----
    This method works only for an undirected graph. If the adjacency matrix is
    represented as A[i,j], then it is assumed that an edge "goes" from node j
    to node i. The transition matrix is appropriately defined. The equilibrium
    distribution of states would therefore be the right eigenvectors of the
    resulting matrix.
    """
    
    # First, get the adjacency matrix of the graph
    P = sp.array(nx.adjacency_matrix(graph))
    
    # We'll also need the number of nodes
    if n is None:
        n = graph.number_of_nodes()
    
    # The probability transition matrix is simply the adjacency matrix itself,
    # but with values along each column scaled so that every column sums up to
    # unity.
    k = sp.array(graph.degree(weight='weight').values())    # Vector of degrees
    # Stack degrees vertically:
    K = sp.multiply(k.T, sp.ones(n).reshape((1, n)))
    P /= K
    
    # If any elements of K were zero, the corresponding elements of P would go
    # infinite. But logically, if the degree of a vertex is zero, then there is
    # no way to get to it, or out of it. Therefore, it should just appear as a
    # zero in the probability transition matrix. We explicitly set this.
    z = sp.where(sp.isnan(P))
    P[z] = 0
    
    return sp.asarray_chkfinite(P)

def equilibrium_distribution(P):
    """
    Finds the equilibrium distribution of states for a given Markov probability
    transition matrix ``P``. It is assumed that a transition occurs from node j
    to node i, for any element P[i,j] of the matrix.
    """
    
    # The equilibrium distribution of states is simply the right eigenvector of
    # the transition matrix with eigenvalue 1, since at equilibrium we must 
    # have
    #                               P.π = π
    w, vr = linalg.eig(P)
    
    # Find the index of the eigenvalue 1
    
    # TODO: It's not very clear as to why I need to do the extra '[1]'. For the
    # 'block.jpg' example, index contained two values - 63 and 303. On using 63
    # I was not able to invert (I-P+W) to get the fundamental matrix because
    # this matrix was singular for some reason. However, 303 appeared to work.
    
    index = sp.where(w == 1)[0]
    return vr[:, index]

def equilibrium_transition_matrix(eq_pi, n=None):
    """
    Creates the equilibrium probability transition matrix, that is, P^(n) in
    the limit of large n, where P is the probability transition matrix.
    
    Parameters
    ----------
    eq_pi : numpy.ndarray
        The equilibrium distribution of states of the probability transition
        matrix P
    """
    
    # The required matrix is simply the equilibrium distribution vector itself,
    # stacked horizontally to create a square matrix
    if n is None:
        n = eq_pi.size()
    return sp.multiply(eq_pi.reshape((n, 1)), sp.ones((1, n)))

def fundamental_matrix(P, W, n=None):
    """
    Computes the fundamental matrix ``Z`` of the Markov chain, given the
    probability transition matrix, and the transition matrix at equilibrium.
    """
    
    # The required formula is simply:
    #                         Z = (I - P + W)^(-1)
    # where I is the identity matrix
    if n is None:
        n = sp.sqrt(P.size())
    return linalg.inv(sp.identity(n) - P + W)

def hitting_times(eq_pi, Z, n=None):
    """
    Calculates the hitting times (mean first passage times) of various each
    node from the equilibrium distribution of states.
    
    Three quantities are determined and returned:
    Ei(Ti) : Expected number of steps to return to the state 'i' if the Markov
        chain is started in the state 'i'.
    Ei(Tj) : Expected number of steps to reach state 'j' if the Markov chain is
        started in state 'i'.
    Eπ(Ti) : The hitting time of state 'i' if we start with the equilibrium
        state distribution π.
    
    These three quantities are given in terms of the equilibrium distribution
    of states and the fundamental matrix as follows:
                            Ei(Ti) = 1 / (πi)
                            Ei(Tj) = Ej(Tj).(Zjj - Zij)
                            Eπ(Ti) = Ei(Ti).Zii
    """

    if n is None:
        n = eq_pi.size()
    
    Ei_Ti = 1 / eq_pi
    
    # First, we calculate Zj, which is a matrix each of whose columns is
    # replaced by the diagonal element of that column
    Zj = sp.multiply(sp.ones((n, 1)), Z.diagonal().reshape((1, n)))
    # Next, we calculate Ej, which is a matrix, whose diagonal elements are
    # Ei_Ti
    Ej = sp.diag(Ei_Ti)
    # Finally, Ei_Tj is simply the product of the two matrices given by:
    Ei_Tj = sp.multiply(Ej, Zj - Z)
    
    Ei_Tj_test = sp.empty((n, n))
    for i in range(n):
        for j in range(n):
            Ei_Tj_test[i][j] = Ei_Ti[j, 0] * (Z[j][j] - Z[i][j])
    
    # FIXME
    if (Ei_Tj == Ei_Tj_test).all():
        print 'It worked!'
    else:
        print 'It didn\'t work :-/'
        print (Ei_Tj - Ei_Tj_test)
    
    Epi_Ti = sp.multiply(Ei_Ti.reshape((1, n)), Z.diagonal().reshape((1, n)))
    
    return Ei_Ti, Ei_Tj, Epi_Ti

