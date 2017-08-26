import sys
sys.path.append("H:/Geography/MASTERARBEIT/src/github/Jannik/_LBP")

import lbp
from skimage import io

#http://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html

image = io.imread("D:/Bilder/bilbo-baggins.jpg", as_grey = True )
result = lbp.local_binary_pattern(image, 8, 1, 'var')
import matplotlib.pyplot as plt
plt.imshow(result)