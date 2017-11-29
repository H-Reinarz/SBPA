#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Aug  7 14:25:44 2017

@author: hre070
'''

#Imports
import statistics as stats
from collections import namedtuple, Counter
import copy
import networkx as nx
#from itertools import repeat
import numpy as  np
from bow_container import hist
from skimage.future.graph import RAG
from skimage.measure import regionprops
import sklearn.cluster



def calc_attr_value(*, array, func, **kwargs):
    '''Helper function to apply a given function to
    a numpy array (i.e. an image) and return the result.
    If the array has multiple dimensions, a list of values is returned.'''
    #account for multi channels: one hist per channel
    if len(array.shape) == 2:
        result = [func(array[:, dim], **kwargs) for dim in range(array.shape[1])]
    else:
        result = func(array, **kwargs)

    return result




#Subclass of RAG specified for BOW classification
class BOW_RAG(RAG):
    '''Subclass of the 'region adjacency graph' (RAG) in skimage to accomodate for
    dynamic attribute assignment, neighbourhood weighting and node clustering.'''

    config = namedtuple('AttributeConfig', ['img', 'func', 'kwargs'])

    fs_spec = namedtuple('fs_spec', ['array', 'order', 'label'])



    def __init__(self, seg_img, **attr):
        '''BOW_RAG is initialized with the parents initializer along
        with additional attributes.'''

        #Call the RAG constructor
        super().__init__(label_image=seg_img, connectivity=1, data=None, **attr)

        #Store seg_img as attribute
        self.seg_img = seg_img


        #Node attribute reference information
        self.attr_func_configs = {}
        self.attr_norm_val = {}

        #Init edge weight statistics
        self.edge_weight_stats = {}


        #Set indipendent node attributes
        for node in self.__iter__():
            #get color values for super pixel
            label_mask = self.seg_img == node

            #Assign attributes to node
            self.node[node].update({'labels': [node],
                                    'pixel_count': seg_img[label_mask].size})


    def deepcopy_node(self, node):
        '''Return a deep copy of a node dictionary.'''
        #create mutable copy of the node for calculation
        return copy.deepcopy(self.node[node])


    def add_attribute(self, name, image, function, **func_kwargs):
        '''Adds an attribute ('name') to each node by calling 'calc_attr_value()'
        on the subset of the image that is represented by the node.'''

        self.attr_func_configs[name] = BOW_RAG.config(image, function, func_kwargs)

        #Set node attributes
        for node in self.__iter__():
            #get color values for super pixel
            label_mask = self.seg_img == node
            masked_image = image[label_mask]

            attr_value = calc_attr_value(array=masked_image, func=function, **func_kwargs)

            #Assign attributes to node
            self.node[node].update({name:attr_value})


    def subgraph(self, nbunch):
        """Return the subgraph induced on nodes in nbunch.

        The induced subgraph of the graph contains the nodes in nbunch
        and the edges between those nodes.
        """
        bunch = self.nbunch_iter(nbunch)
        # create new graph and copy subgraph into it
        H = nx.Graph()
        # copy node and attribute dictionaries
        for n in bunch:
            H.node[n] = self.node[n]
        # namespace shortcuts for speed
        H_adj = H.adj
        self_adj = self.adj
        # add nodes and edges (undirected method)
        for n in H.node:
            Hnbrs = H.adjlist_dict_factory()
            H_adj[n] = Hnbrs
            for nbr, d in self_adj[n].items():
                if nbr in H_adj:
                    # add both representations of edge: n-nbr and nbr-n
                    Hnbrs[nbr] = d
                    H_adj[nbr][n] = d
        H.graph = self.graph
        return H



    def add_attribute_from_lookup(self, new_attribute, attribute, lookup_dict):
        '''Assign a new node attribute with values from the provided
        look up dictionary corresponding to an existing attribute.'''

        for node in self.__iter__():
            key = self.node[node][attribute]
            self.node[node].update({new_attribute: lookup_dict[key]})




    def add_regionprops(self):
        '''Function to assign geometric properties of the represented region
        as node attributes. IN DEVELOPMENT!'''

        self.seg_img += 1

        for reg in regionprops(self.seg_img):
            self.node[reg.label-1]["Y"] = round(reg.centroid[0]/self.seg_img.shape[0], 3)
            self.node[reg.label-1]["X"] = round(reg.centroid[1]/self.seg_img.shape[1], 3)

        self.seg_img -= 1


    def normalize_attribute(self, attribute, value=None):
        '''Normalize a node attribute with a given denominator.'''

        self.attr_norm_val[attribute] = value

        for node in self.__iter__():
            if isinstance(self.node[node][attribute], list):
                for index, element in enumerate(self.node[node][attribute]):
                    if isinstance(element, hist):
                        element.normalize(self.node[node]['pixel_count'])
                    else:
                        self.node[node][attribute][index] = element/value

            else:
                if isinstance(self.node[node][attribute], hist):
                    self.node[node][attribute].normalize(self.node[node]['pixel_count'])
                else:
                    self.node[node][attribute] /= value




    def delete_attributes(self, attribute):
        '''Delete a given attribute.'''
        for node in self.__iter__():
            del self.node[node][attribute]


    def filter_by_attribute(self, attribute, values):
        '''Filter the nodes based on their value of a specified attribute.'''

        func = lambda node: self.node[node][attribute] in values

        return list(filter(func, self.__iter__()))



    def calc_edge_weights(self, weight_func):
        '''Apply a given weighting function to all edges.'''

        #Iterate over edges and calling weight_func on the nodes
        for node1, node2, data in self.edges_iter(data=True):
            data.update(weight_func(self, node1, node2))


    def get_edge_weight_list(self, attr_label='weight'):
        '''Return a sorted value list of a given edge attribute.'''
        return sorted(list(data[attr_label] for node1, node2, data in self.edges(data=True)))


    def calc_edge_weight_stats(self, attr_label='weight'):
        '''Perform descriptive stats on a given edge attribute.
        Result is stored as a graph attribute.'''
        weight_list = self.get_edge_weight_list(attr_label)

        self.edge_weight_stats['min'] = min(weight_list)
        self.edge_weight_stats['max'] = max(weight_list)
        self.edge_weight_stats['mean'] = stats.mean(weight_list)
        self.edge_weight_stats['median'] = stats.median(weight_list)
        self.edge_weight_stats['stdev'] = stats.stdev(weight_list)


    def get_edge_weight_percentile(self, perc, attr_label='weight', as_threshhold=False):
        '''Return the given percentile value for the value list af a specified attribute.
        When 'as_threshhold' is true, the mean of the percentile value
        and the next value is returned.'''
        weight_list = self.get_edge_weight_list(attr_label)

        index = round(len(weight_list)*(perc/100))

        if as_threshhold:
            result = (weight_list[index] +  weight_list[index+1]) /2
            return result
        else:
            return weight_list[index]



    def basic_feature_space_array(self, attr_config, label='', subset=None, exclude=()):
        '''Arange a specification of attributes into an array that contains
        one row per node. It serves as data points in feature space for clustering operations.
        Nodes are selectable via the 'subset' parameter
        and excludable via the 'exclude' parameter.'''


        if subset is None:
            subset = set(self.__iter__())
        else:
            subset = set(subset)

        exclude = set(exclude)

        weight_list = list()

        for attr, weight in attr_config.items():
            if isinstance(self.node[0][attr], list):
                for element in self.node[0][attr]:
                    weight_list.append(weight)
            else:
                weight_list.append(weight)

        mul_array = np.array(weight_list, dtype=np.float64)

        order_list = list()
        array_list = list()

        for node in self.__iter__():

            if node not in subset or node in exclude:
                continue

            a_row = list()

            for attr in attr_config.keys():
                if isinstance(self.node[node][attr], list):
                    for element in self.node[node][attr]:
                        a_row.append(element)
                else:
                    a_row.append(self.node[node][attr])

            array_list.append(a_row)
            order_list.append(node)

        fs_array = np.array(array_list, dtype=np.float64)
        fs_array *= mul_array

        return BOW_RAG.fs_spec(fs_array, tuple(order_list), label)





    def attribute_divided_fs_arrays(self, attr_config, div_attr, exclude=()):
        '''Return a feature space array for every value of a specified attribute.
        Nodes are excludable via the 'exclude' parameter.'''

        return_list = []

        div_attr_labels = {self.node[node][div_attr] for node in self.__iter__()}

        for label in div_attr_labels:
            nodes = self.filter_by_attribute(div_attr, label)
            fs_result = self.basic_feature_space_array(attr_config, label, nodes, exclude)

            return_list.append(fs_result)

        return return_list





    def hist_to_fs_array(self, attr_config, subset=(), label=''):
        '''Arange a attribute that is itself a histogram into an array that contains
        one row per node. It serves as data points in feature space for clustering operations.'''


        if subset is None:
            subset = set(self.__iter__())

        array_list = list()
        order_list = list()

        for node in self.__iter__():
            if node in subset:
                row_array = np.array([], dtype=np.float64)
                for attr, factor in attr_config.items():
                    if isinstance(self.node[node][attr], list):
                        for histogramm in self.node[node][attr]:
                            if not isinstance(histogramm, hist):
                                raise TypeError(f'Wrong type: {type(histogramm)} Must be "hist"!')
                            part = histogramm(mode='array', normalized=True)
                            part *= factor
                            row_array = np.append(row_array, part)
                    elif isinstance(attr, hist):
                        part = self.node[node][attr](mode='array', normalized=True)
                        part *= factor
                        row_array = np.append(row_array, part)
                    else:
                        raise TypeError(f'Wrong type: {type(histogramm)} Must be "hist"!')

                array_list.append(row_array)
                order_list.append(node)

        fs_array = np.array(array_list, dtype=np.float64)

        return BOW_RAG.fs_spec(fs_array, tuple(order_list), label)



    def clustering(self, attr_name, algorithm, fs_spec, **cluster_kwargs):
        '''Perform any clustering operation from sklearn.cluster on a given feature space array
        (as returnd by 'get_feature_space_array()' or 'hist_to_fs_array()').
        Return the cluster label of each node as an attribute.'''

        for node in self.__iter__():
            self.node[node][attr_name] = None

        cluster_class = getattr(sklearn.cluster, algorithm)

        if isinstance(fs_spec, BOW_RAG.fs_spec):
            cluster_obj = cluster_class(**cluster_kwargs).fit(fs_spec.array)       
            for node, label in zip(fs_spec.order, cluster_obj.labels_):
                #print(node, label)
                self.node[node][attr_name] = str(fs_spec.label) + str(label)
        
        elif isinstance(fs_spec, list):    
            for fs in fs_spec:
                if not isinstance(fs, BOW_RAG.fs_spec):
                    raise TypeError("Must be BOW_RAG.fs_spec!")

                cluster_obj = cluster_class(**cluster_kwargs).fit(fs.array)       
                for node, label in zip(fs.order, cluster_obj.labels_):
                    self.node[node][attr_name] = str(fs.label) + str(label)
    


#    def kmeans_clustering(self, attr_name, fs_array, k, **cluster_kwargs):
#        '''Perform the KMeans clustering from SKLearn on a geiven feature space array
#        (as returnd by 'get_feature_space_array()' or 'hist_to_fs_array()').
#        Return the cluster label of each node as an attribute.'''
#
#        cluster_obj = sklearn.cluster.KMeans(k, **cluster_kwargs).fit(fs_array)
#
#        for node_ix, label in enumerate(cluster_obj.labels_):
#            self.node[node_ix][attr_name] = label
#
#
#
#    def mean_shift_clustering(self, attr_name, fs_array, **ms_kwargs):
#        '''Perform the MeanShift clustering from SKLearn on a geiven feature space array
#        (as returnd by 'get_feature_space_array()' or 'hist_to_fs_array()').
#        Return the cluster label of each node as an attribute.'''
#
#        meanshift_obj = MeanShift(**ms_kwargs).fit(fs_array)
#
#        for node_ix, label in enumerate(meanshift_obj.labels_):
#            self.node[node_ix][attr_name] = label
#




    def produce_cluster_image(self, attribute, dtype=np.int64):
        '''Render an image (2D numpy array) of cluster labels based
        on a cluster label node attribute.'''

        attr_labels = {self.node[node][attribute] for node in self.__iter__()}
        label_dict = dict(zip(attr_labels, range(len(attr_labels))))
        
        print(label_dict)

        cluster_img = np.zeros_like(self.seg_img, dtype=dtype)

        for node in self.__iter__():
            for label in set(self.node[node]['labels']):
                mask = self.seg_img == label
                cluster_img[mask] = label_dict[self.node[node][attribute]]

        return cluster_img


    def neighbour_cross_tabulation(self, attribute):
        '''Tabulate the joint distribution of cluster labels
        for all adjacent nodes.'''

        count = Counter()
        for node1, node2, in self.edges():
            combo = tuple(sorted([self.node[node1][attribute], self.node[node2][attribute]]))
            count[combo] += 1
        return count




    @classmethod
    def old_init(cls, seg_img, tex_img, color_image, tex_bins, color_bins, **attr):
        '''Constructor of the first version of this class to ensure backwards compatibility.'''

        new_rag = cls(seg_img, **attr)

        new_rag.add_attribute('tex', tex_img, hist, vbins=tex_bins)
        new_rag.normalize_attribute('tex')

        new_rag.add_attribute('color', color_image, hist, bins=color_bins)
        new_rag.normalize_attribute('color')

        return new_rag








#Simple merging function
def _bow_merge_simple(graph, src, dst):
    '''Function to perform attribute transfer/recalculation
    of two nodes to be merged as part of a sequently merging
    algorithm.'''

    #pixel counter
    graph.node[dst]['pixel_count'] += graph.node[src]['pixel_count']

    #get color values for super pixel
    label_mask = (graph.seg_img == src) | (graph.seg_img == dst)

    for attr, fconfig in graph.attr_func_configs.items():

        masked_image = fconfig.img[label_mask]

        graph.node[dst][attr] = graph.calc_attr_value(data=masked_image,
                                                      func=fconfig.func, **fconfig.kwargs)

        #Normalize according to specs
        if attr in graph.attr_norm_val:
            graph.normalize_attribute(attr, graph.attr_norm_val[attr])
        #else: raise KeyError(f"Attribute '{attr}' has no stored normalization value")
