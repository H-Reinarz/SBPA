#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:45:55 2017

@author: hre070
"""

#Weighting function: Cumulative difference over bins   
def cumu_diff(node1, node2, **kwargs):
    c_diff = 0
    for k,v in node1["BOW"].items():
        c_diff += abs(v-node2["BOW"][k])
    return c_diff
