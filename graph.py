# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:32:24 2017

@author: Jannik
"""
import sys
sys.path.append("H:/Geography/MASTERARBEIT/src/github/_LBP")
import _graph

def _add_edge_filter(values, edgeSet):
    """Create edge in `graph` between central element of `values` and the rest.
    Add an edge between the middle element in `values` and
    all other elements of `values` into `graph`.  ``values[len(values) // 2]``
    is expected to be the central value of the footprint used.
    Parameters
    ----------
    values : array
        The array to process.
    graph : RAG
        The graph to add edges in.
    Returns
    -------
    0 : float
        Always returns 0. The return value is required so that `generic_filter`
        can put it in the output array, but it is ignored by this filter.
    """
    
    values = values.astype(int, copy = False)
    
    _graph._graphLabels(values, edgeSet)
    
    
#    for edge in edges:
#        if not edgeSet.has_edge(edge[0], edge[1]):
#            edgeSet.add_edge(edge[0], edge[1])
    
    return 0