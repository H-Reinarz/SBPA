from skimage.measure import compare_psnr
import sys
sys.path.append("H:/Geography/MASTERARBEIT/src/github/")
sys.path.append("H:/Geography/MASTERARBEIT/src/github/_LBP")
import skimage.io as io
import lbp
import lbp_bins
from skimage.color import rgb2gray
import numpy as np

img = io.imread("H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/orthoClipRA1_badRes.jpg")





def TrainLbp(img, methods, n_points, radius):
    outputDict = {}
    i = 0
    img_gray = rgb2gray(img)
    for method in methods:
        for n_point in n_points:
            for r in radius:
                outputDict[i] = [CorrelateLbp(img_gray, method, n_point, r), method, n_point, r]
                i += 1
    keyResult = min(outputDict, key=lambda y:abs(float(outputDict[y][0])-0))
    return outputDict[keyResult]
                
    
def CorrelateLbp(img, method, n_points, radius):
    lbp_img = lbp.local_binary_pattern(img, None, n_points, radius, method, True)
    BINS = lbp_bins.lbp_bins(n_points, method)
    lookup = dict(enumerate(BINS))
    lbp_random = np.vectorize(lookup.__getitem__)(np.random.randint(len(BINS), size=lbp_img.shape))
    return np.corrcoef(lbp_img.flat, lbp_random.flat)[0][1]
    
                

methods = ["default", "var"]
n_points = range(8,20)
radius = range(1,10)
bob = TrainLbp(img, methods, n_points, radius)
