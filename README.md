# Similarity Based Patch Agglomeration (SPBA)
Classes and algorithms for aerial image segmentation focussing on unsupervised texture based clustering. Development is done in Anaconda 3 on top of SciKit Image.

# Installation
On Windows you will need a compiler like Visual Studio. If you use Visual Studio you will need the C++ and Python extensions It turned out to be a good solution to first install Visual Studio and afterwards Anaconda. This ensures the Anaconda Cython installation gets the correct include and library paths it needs for compiling the scripts.

Install Visual Studio
https://visualstudio.microsoft.com/de/vs/community/

Install Anaconda for Python 3.6 
https://anaconda.org/anaconda/python

On Linux everything should work fine as long as you have the build-essentials (gcc, etc.)

## Compile Cython files
Navigate to the _LBP folder and run ```python setup.py build_ext --inplace``` in your Anaconda prompt (Windows) or your terminal (Linux).

