#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:22:51 2017

@author: hre070
"""

from math import pi, sqrt

from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
import numpy as np
from numpy import fft
from skimage import draw
from skimage.filters import gaussian
#from skimage.restoration import estimate_sigma
from skimage.measure import find_contours
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse

def create_radius_map(shape, array_dtype=np.float32):
    '''Create an array where the value of a cell is its distance to the centroid
    of the array. Main parameter is the shape of the array.'''

    radius_array = np.zeros(shape=shape, dtype=array_dtype)

    middle = (radius_array.shape[0]/2, radius_array.shape[1]/2)

    for row in range(radius_array.shape[0]):
        for col in range(radius_array.shape[1]):

            distance = euclidean((row+1, col+1), middle)
            #print(distance)
            radius_array[row, col] = int(distance)

    return radius_array



def dft_summary(freq, width=1):
    '''Return a dictionary with the mean value of the shifted 2D-DFT
    for each concentrical ring with the specified width.'''

    class_array = create_radius_map(freq.shape)//width

    #plt.imshow(class_array)

    value_dict = dict()

    for ring in np.unique(class_array):
        values = freq[class_array == ring]
        mean = np.mean(values)
        value_dict[ring] = mean

    return value_dict


def xcut_auc(value_dict, cut_perc, width):
    '''Return the position on the x-axis (the radius) where the specified
    percentage of the area under the curve (cut_perc) is reached.
    This function is designed to use the output of dft_summary() (value_dict).'''

    area = sum(value_dict.values())
    cut_value = area/100*cut_perc

    for key, value in value_dict.items():
        if cut_value < value:
            return key - int(round(cut_value/(value/width)))
        else:
            cut_value -= value


def lowpass_filter(shifted_dft, radius, stretch, upper, lower):
    '''Applies a lowpass filter to the shifted 2D-DFT.
    Therefore an array of values of a sigmoid function (specified by 'radius' and stretch')
    between 'upper' and 'lower' is created and concentrically aranged to constitue the filter'''

    span = upper - lower
    sigmoid = [upper-(3*(x**2)-2*(x**3))*span for x in list(np.linspace(0, 1, stretch))]

    radius_array = create_radius_map(shifted_dft.shape)

    #Filter creation
    filter_array = np.zeros(shape=radius_array.shape, dtype=np.float32)
    filter_array[radius_array <= radius] = upper
    filter_array[radius < radius_array] = lower

    for index, factor in enumerate(sigmoid):
        add = index+1
        filter_array[radius_array == radius+add] = factor


    return filter_array


def get_wavelength(dft_shape, radius):
    '''Return the ratio of a concentrical radius to the
    largest dimension of a shifted 2D-DFT array as the wavelangth
    of the frequency at that point.'''
    denom = max(dft_shape)

    wavelength = (2*pi)/(radius*2*pi/denom)

    return wavelength


class FrequencySetup:
    '''Class to provide the 2D-DFT (shifted) of an image
    alongside wavelength and filtered version for texture analysis.'''

    def __init__(self, image, ring_width, cut_percent, lp_stretch, lp_upper, lp_lower):
        self.dft = fft.fftshift(fft.fft2(image))

        self.summary = dft_summary(self.dft_log(), ring_width)

        self.cut_value = xcut_auc(self.summary, cut_percent, ring_width)

        self.low_pass = lowpass_filter(self.dft, self.cut_value, lp_stretch, lp_upper, lp_lower)

        self.filtered_dft = self.dft * self.low_pass

        self.lbp_radius = int(round(get_wavelength(self.dft.shape, self.cut_value)/2))

    def dft_log(self):
        '''Return log-transformed DFT for plotting.'''
        return np.log(np.abs(self.dft))

    def filtererd_dft_log(self):
        '''Return log-transformed filtered DFT for plotting.'''
        return np.log(np.abs(self.filtered_dft))

    def result(self):
        '''Return the input image after applying the lowpass filter
        in the frequency domain.'''
        return np.abs(fft.ifft2(fft.ifftshift(self.filtered_dft)))
    
    
    
    
    
    
    
    
    
    
    
class DFTanalyzer:
    '''Class to provide the 2D-DFT (shifted) of an image
    and a derived ellipse model representing the the frequencies
    of interest. Based on that model, several parameters
    for image analysis are provided.'''
    
    def __init__(self, image):
        self.dft = fft.fftshift(fft.fft2(image))
        
    @property    
    def abs_log_dft(self):
        return np.abs(np.log(self.dft))
    
    def fit_model(self, cut_percent, gauss_sigma):
        #Fit ellipse model
        al_dft = self.abs_log_dft
        
        gauss_dft = gaussian(al_dft, gauss_sigma)
        
        contour_value = gauss_dft.min()+((gauss_dft.max() - gauss_dft.min())*cut_percent/100)
        contours = find_contours(gauss_dft, contour_value)
    
        assert(len(contours) == 1)
        
        self.contour = contours[0]
        
        self.ellipse = EllipseModel()
        
        self.ellipse.estimate(self.contour[:, ::-1])
        
        center = tuple([x/2 for x in self.dft.shape])
        
        offset = euclidean(center, (self.ellipse.params[1], self.ellipse.params[0]))
        
        half_diagonal = sqrt(sum((x**2 for x in self.dft.shape)))/2
        
        assert((offset/half_diagonal) <= 0.03)
      
        #derive wavelength and texture parameters
        xy_points = self.ellipse.predict_xy(np.linspace(0, 2*np.pi, 4*self.ellipse.params[0]))
        
        max_x = round(xy_points[:,0].max())
        min_y = round(xy_points[:,1].min())
        
        wavelength_x = self.dft.shape[1]/(max_x - center[1])
        
        wavelength_y = self.dft.shape[0]/(center[0] - min_y)
        
        self.wavelength = round((wavelength_x + wavelength_y)/2)
        
        self.texture_radius = round(self.wavelength/2)
        
        self.min_patch_size = round(self.texture_radius**2 * pi)
        
        
        
        
        
    def apply_lowcut(self, upper, lower, gauss_sigma=1):
        cx, cy, a, b, theta = self.ellipse.params
        
        self.low_cut = np.zeros_like(self.dft, dtype=np.float64) + lower
        rr, cc = draw.ellipse(cy, cx, b, a, self.low_cut.shape, (theta*-1))
        self.low_cut[rr, cc] = upper
        
        self.low_cut = gaussian(self.low_cut, gauss_sigma)
        
        filtered_dft = self.dft * self.low_cut
        
        self.filtered_img =  np.abs(fft.ifft2(fft.ifftshift(filtered_dft)))
        
        
    
    

#======================================================================================

if __name__ == "__main__":


    from skimage import io
    from skimage.color import rgb2gray
    from skimage.filters import gaussian
#    from numpy.fft import *

    image = io.imread("/home/hre070/MA/DJI_0095_CLIP.jpg")
    im_gray = rgb2gray(image)

    start = 50
    width = 400


    end = start+width

    clip = im_gray[start:end, start:end]


    #Generate random noise within brightness spectrum of the image

#    factor = 10000
#    lower = clip.min()*factor
#    upper = clip.max()*factor
#
#    noise = np.random.randint(lower, upper, size=clip.shape)/factor
#
#
#    clip += noise



    #clip = im_gray

    #

    #Apply shifted furier transform
#    dft = fftshift(fft2(clip))

    #dft_log = np.log(np.abs(dft))




#    result = dft_summary(dft_log, 5)
#
#    x_cut = xcut_auc(result, 80, 5)
#
#    plot_values = np.array(list(result.values()))
#
#    deriv1 = np.diff(plot_values)
#    deriv2 = np.diff(deriv1)



    #CLASS TEST
    test = FrequencySetup(clip, 5, 80, 10, 1.0, 0.1)

    print(test.cut_value, test.lbp_radius)

    plt.imshow(test.result(), cmap="gray")


#    f, ax = plt.subplots(nrows=2, figsize=(20,20))
#
#    ax[0].imshow(clip, cmap="gray")
#    ax[1].imshow(dft_log)
#    ax[2].imshow(new_image, cmap="gray")

#    f, ax = plt.subplots(nrows=3, figsize=(10,10), sharex=True)
#
#    ax[0].plot(plot_values)
#    ax[0].axvline(x_cut, color='r')
#    ax[1].plot(deriv1, color="r")
#    ax[2].plot(deriv2, color="g")

    #print(result)
