from skimage import io
import utils as u
import numpy as np
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb, rgb2lab
from matplotlib import pyplot as plt
import sys
sys.path.append("D:/janni/Documents/Geographie/Masterarbeit/src/github")
sys.path.append("D:/janni/Documents/Geographie/Masterarbeit/src/github/_LBP")
import bow_rag
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
##############
image = io.imread("D:/janni/Documents/Geographie/Masterarbeit/Data/ra_neu/ra2_small_gauss.jpg")
image = u.ZerosToOne(image, 1)
image = img_as_float(image)

segments_slic = slic(image, n_segments=50, compactness=10, sigma=1)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
#f, ax = plt.subplots(figsize=(10, 10))
#ax.imshow(mark_boundaries(image, segments_slic))
im_gray = rgb2gray(image)
imageLab = rgb2lab(image)

Graph = bow_rag.BOW_RAG(segments_slic)
print(nx.info(Graph))
Graph.add_attribute('color', imageLab, np.mean)
Graph.normalize_attribute('color', value=255)
Graph.add_attribute('var', im_gray, np.var)

n_clusters=2
fs_attrs = {'color':1, 'var':.5}
fs0 = Graph.basic_feature_space_array(fs_attrs)

################
#Cluster0
connectivity = nx.adjacency_matrix(Graph, weight=None)
#Graph.clustering("run0", 'AgglomerativeClustering', fs0, n_clusters=n_clusters, linkage="ward", connectivity=connectivity)
cluster_obj = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                               connectivity=connectivity).fit(fs0.array)       
for node, label in zip(fs0.order, cluster_obj.labels_):
    Graph.node[node]["run0"] = str(fs0.label) + str(label)

#Cluster1
fs1 = Graph.attribute_divided_fs_arrays(fs_attrs, 'run0')
for fs in fs1:
#    if len(fs[1]) < superpixel_min:
#        print("Cluster too small")
#        g.node[fs[1][0]][attr_name] = str(fs.label)# + str(3)
#        continue
    
    subset = Graph.subgraph(nbunch=list(fs.order))
    
    pixel_count = 0
    
    if 357913 >= 0:
        for n in subset:
            pixel_count += subset.node[n]['pixel_count']
        if pixel_count <= 357913:
            #print("In FS ", fs.label, " are ", pixel_count, " Pixel. It gets Name ", fs.label)
            for n in subset:
                Graph.node[n]["run1"] = str(fs.label)
            continue
#    
    connectivity = nx.adjacency_matrix(subset, weight=None)
                
    cluster_obj = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                       connectivity=connectivity).fit(fs.array)       
    for node, label in zip(fs.order, cluster_obj.labels_):
        Graph.node[node]["run1"] = str(fs.label) + str(label)
        
#Cluster2
fs2 = Graph.attribute_divided_fs_arrays(fs_attrs, 'run1')
for fs in fs2:
#    if len(fs[1]) < superpixel_min:
#        print("Cluster too small")
#        g.node[fs[1][0]][attr_name] = str(fs.label)# + str(3)
#        continue
    
    subset = Graph.subgraph(nbunch=list(fs.order))
    
    pixel_count = 0
    
    if 357913 >= 0:
        for n in subset:
            pixel_count += subset.node[n]['pixel_count']
        if pixel_count <= 357913:
            #print("In FS ", fs.label, " are ", pixel_count, " Pixel. It gets Name ", fs.label)
            for n in subset:
                Graph.node[n]["run2"] = str(fs.label)
            continue
#    
    connectivity = nx.adjacency_matrix(subset, weight=None)
                
    cluster_obj = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                       connectivity=connectivity).fit(fs.array)       
    for node, label in zip(fs.order, cluster_obj.labels_):
        Graph.node[node]["run2"] = str(fs.label) + str(label)
#################
    
from skimage import color
cluster_img = Graph.produce_cluster_image('run1')
out = color.label2rgb(cluster_img, image, kind='avg')
out = mark_boundaries(out, cluster_img, (0, 0, 0))

from visual import plot_node_attribute
f, ax = plt.subplots(figsize=(18, 18))

ax.imshow(image)
ax.imshow(out, alpha = .4)
ax.imshow(cluster_img, cmap=plt.cm.spectral, alpha=.2)
plot_node_attribute(ax, Graph, 'labels', 24)

