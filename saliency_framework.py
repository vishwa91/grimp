# -*- coding: utf-8 -*-

import scipy as sp

def salient_node(global_Epi_Ti, local_Epi_Ti):
    """
    Finds the index corresponding to the most salient node.
    """
    
    # We define a saliency measure as follows:
    NSali = global_Epi_Ti / local_Epi_Ti.reshape(global_Epi_Ti.shape)
    print 'NSali:', NSali
    Ns = sp.argmax(NSali)
    return Ns

