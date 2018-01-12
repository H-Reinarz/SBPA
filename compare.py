# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:14:54 2017

@author: Jannik
"""

### Compare Histogram formulas 
### http://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist#comparehist

from math import sqrt


def intersection(h1, h2):
    '''Computes intersection of two histograms. High intersection = 1.0; 
    Zero intersection = 0.0'''
    if len(h1) != len(h2):
        raise ValueError("Computing histogram intersection failed: Histograms "
                         "do not have the same length")
    diff = 0.0
    diffNorm = 0.0
    for i in enumerate(h1):
        diff += min(h1[i[0]], h2[i[0]])  
    diffNorm = 1.0 - (diff / sum(h2))
    return diffNorm



#def ChiSqr(h1, h2):
#    if len(h1) != len(h2):
#        raise ValueError("Computing histogram Chi-Squared failed: Histograms "
#        "do not have the same length")
#    diff = 0.0
#    diffNorm = 0.0
#    for i in range(len(h1)):
#        diff = ((h1[i] - h2[i])**2) / h1[i] # NOT WORKING
#    return diff
    


def correlation(h1, h2):
    '''Computes correlation of two histograms. High Correlation = 1.0; 
    Zero Correlation = 0.0'''
    if len(h1) != len(h2):
        raise ValueError("Computing histogram correlation failed: Histograms "
                         "do not have the same length")
    diff = 0.0
    h1mean = sum(h1) / len(h1)
    h2mean = sum(h2) / len(h2)
    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    for i in enumerate(h1):
        numerator += (h1[i[0]] - h1mean) * (h2[i[0]] - h2mean)
        denominator1 += (h1[i[0]] - h1mean)**2
        denominator2 += (h2[i[0]] - h2mean)**2
    diff = numerator / sqrt(denominator1 * denominator2)
    diffNorm = 1.0 - diff
    return diffNorm



def hellinger(h1, h2):
    '''Computes Hellinger distance, which is related to Bhattacharyya 
    coefficient.'''
    if len(h1) != len(h2):
        raise ValueError("Computing histogram hellinger distance failed: "
                         "Histograms do not have the same length")
    diff = 0.0
    h1mean = sum(h1) / len(h1)
    h2mean = sum(h2) / len(h2)
    sigma = 0.0
    for i in enumerate(h1):
        sigma += sqrt(h1[i[0]] * h2[i[0]])
    diff = sqrt(1 - (1 / sqrt(h1mean*h2mean*(len(h1)**2))) * sigma)
    return diff