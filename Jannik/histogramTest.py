# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:14:54 2017

@author: Jannik
"""

### Compare Histogram formulas http://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist#comparehist

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from math import sqrt

img1 = io.imread("H:\\Geography\\MASTERARBEIT\\Data\\testData\\patch1.jpg")
img2 = io.imread("H:\\Geography\\MASTERARBEIT\\Data\\testData\\patch3.jpg")

hsv1 = color.rgb2hsv(img1)
hsv2 = color.rgb2hsv(img2)

hist1, _ = np.histogram(hsv1[:,:,0], bins = 100)
hist2, _ = np.histogram(hsv2[:,:,0], bins = 100)

def Intersection(h1, h2):
    if len(h1) != len(h2):
        raise ValueError("Computing histogram intersection failed: Histograms do not have the same length")
    diff = 0.0
    diffNorm = 0.0
    for i in range(len(h1)):
        diff += min(h1[i], h2[i])
        
    diffNorm = 1.0 - (diff / sum(h2))
    return diffNorm

print(Intersection(hist1, hist2))


def ChiSqr(h1, h2):
    if len(h1) != len(h2):
        raise ValueError("Computing histogram Chi-Squared failed: Histograms do not have the same length")
    diff = 0.0
    diffNorm = 0.0
    for i in range(len(h1)):
        diff = ((h1[i] - h2[i])**2) / h1[i]
        
    
    return diffNorm

#print(ChiSqr(hist1, hist2))


def Correlation(h1, h2):
    if len(h1) != len(h2):
        raise ValueError("Computing histogram correlation failed: Histograms do not have the same length")
    diff = 0.0
    h1mean = sum(h1) / len(h1)
    h2mean = sum(h2) / len(h2)
    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    for i in range(len(h1)):
        numerator += (h1[i] - h1mean) * (h2[i] - h2mean)
        denominator1 += (h1[i] - h1mean)**2
        denominator2 += (h2[i] - h2mean)**2
    diff = numerator / sqrt(denominator1 * denominator2)
    
    diffNorm = 1.0 - diff
    return diffNorm

print(Correlation(hist1, hist2))


def Hellinger(h1, h2):
    if len(h1) != len(h2):
        raise ValueError("Computing histogram hellinger distance failed: Histograms do not have the same length")
        
    diff = 0.0
    h1mean = sum(h1) / len(h1)
    h2mean = sum(h2) / len(h2)
    sigma = 0.0
    for i in range(len(h1)):
        sigma += sqrt(h1[i] * h2[i])
    diff = sqrt(1 - (1 / sqrt(h1mean*h2mean*(len(h1)**2))) * sigma )
    
    return diff

print(Hellinger(hist1, hist2))


"""
# https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

intersection0 = return_intersection(hist1, hist2)

# http://blog.datadive.net/histogram-intersection-for-change-detection/
def histogram_intersection(h1, h2, bins):
   bins = np.diff(bins)
   sm = 0
   for i in range(len(bins)):
       sm += min(bins[i]*h1[i], bins[i]*h2[i])
   return sm

intersection1 = histogram_intersection(hist1, hist2, _)

# http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
 
	# return the chi-squared distance
	return d

chi2 = chi2_distance(hist1, hist2)

def chi2(histA, histB, bins):
    
    d = 0
    for i in range(len(bins)-1):
        d += ((histA[i] - histB[i])**2)/histA[i]
    return d

chi22 = chi2(hist1, hist2, _)
"""