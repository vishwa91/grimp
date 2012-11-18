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
    print 'Adjacency matrix:'
    print sp.around(P)
    # We'll also need the number of nodes
    if n is None:
        n = graph.number_of_nodes()
    
    # The probability transition matrix is simply the adjacency matrix itself,
    # but with values along each column scaled so that every column sums up to
    # unity.
    k = sp.dot(sp.ones((1, n)), P)    # Vector of degrees
    print 'Vector of degrees:', k
    # Stack degrees vertically:
    K = sp.multiply(sp.ones((n, 1)), k)
    P /= K
    print 'Left stochastic check:', sp.dot(sp.ones((1, n)), P)
    
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
    print 'Eigenvalues:'
    print list(sp.real(w))
    
    return w, vr

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
        n = eq_pi.size
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
        n = sp.sqrt(P.size)
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
        n = eq_pi.size
    
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
            Ei_Tj_test[i][j] = Ei_Ti[j] * (Z[j][j] - Z[i][j])
    
    # FIXME
    if (Ei_Tj == Ei_Tj_test).all():
        print 'It worked!'
    else:
        print 'It didn\'t work :-/'
        #print (Ei_Tj - Ei_Tj_test)
    
    Epi_Ti = sp.multiply(Ei_Ti.reshape((1, n)), Z.diagonal().reshape((1, n)))
    
    return Ei_Ti, Ei_Tj, Epi_Ti

def do_global(graph):
    n = graph.number_of_nodes()
    print n

    P = generate_transition_matrix(graph, n)
    print 'Probability transition marix:'
    print sp.where(P < 0.001, sp.zeros((n, n)), sp.around(P, 3))
    print 'Done getting P'

    w, vr = equilibrium_distribution(P)
    # Find the index of the eigenvalue 1

    index = sp.where((w > 0.999999999999999) * (w < 1.000000000000001))[0]
    print index
    #if index.size == 0:
    #    if roundoff == 0:
    #        raise ValueError
    #    P = sp.around(P, roundoff-1)
    #    return equilibrium_distribution(P, roundoff-1)

    # Instead of randomly choosing the first eigenvalue, choose the first
    # eigen value which yields a non-singular fundamental matrix
    eigen_index = 0
    while(1):
        try:
            chosen_index = index[eigen_index]
            print 'Trying with index', chosen_index
        except IndexError:
            print 'No eigenvalue of 1 exists, or no eigenvector of 1 exists',
            print 'which allows the fundamental matrix to be non-singular'
            raise

        eq_pi = vr[:, chosen_index]
        print 'Eigenvector: '
        print eq_pi
        print 'Done getting pi'
        print 'Finiteness check:',
        try:
            eq_pi = sp.asarray_chkfinite(eq_pi)
            print 'OK'
        except ValueError:
            eigen_index += 1
            continue

        W = equilibrium_transition_matrix(eq_pi, n)
        print 'Done getting W'

        try:
            Z = fundamental_matrix(P, W, n)
            print 'Done getting Z'
            break
        except linalg.LinAlgError:
            eigen_index += 1

    Ei_Ti, Ei_Tj, Epi_Ti = hitting_times(eq_pi, Z, n)
    print 'Done getting hitting times'

    return Ei_Ti, Ei_Tj, Epi_Ti

def do_local(local_graph):
    local_n = local_graph.number_of_nodes()

    local_P = generate_transition_matrix(local_graph, local_n)
    print 'Local probability transition matrix:'
    print sp.around(local_P, 3)
    print 'Done getting local P'

    w, vr = equilibrium_distribution(local_P)
    # Find the index of the eigenvalue 1

    index = sp.where((w > 0.999999999999999) * (w < 1.000000000000001))[0]
    print index
    #if index.size == 0:
    #    if roundoff == 0:
    #        raise ValueError
    #    P = sp.around(P, roundoff-1)
    #    return equilibrium_distribution(P, roundoff-1)

    eigen_index = 0
    while(1):
        try:
            chosen_index = index[eigen_index]
        except IndexError:
            print 'No eigenvalue of 1 exists, or no eigenvector of 1 exists'
            print 'which allows the fundamental matrix to be non-singular'
            raise

        local_eq_pi = vr[:, chosen_index]
        print 'Local eigenvector:'
        print local_eq_pi
        print 'Done getting local pi'
        print 'Finiteness check:',
        try:
            local_eq_pi = sp.asarray_chkfinite(local_eq_pi)
            print 'OK'
        except ValueError:
            eigen_index += 1
            continue

        local_W = equilibrium_transition_matrix(local_eq_pi, local_n)
        print 'Done getting local W'

        try:
            local_Z = fundamental_matrix(local_P, local_W, local_n)
            print 'Done getting local Z'
            break
        except linalg.LinAlgError:
            eigen_index += 1

    local_Ei_Ti, local_Ei_Tj, local_Epi_Ti = hitting_times(local_eq_pi, local_Z,
                                                           local_n)
    print 'Done getting local hitting times'

    return local_Ei_Ti, local_Ei_Tj, local_Epi_Ti
