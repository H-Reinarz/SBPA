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

    fs_spec = namedtuple('fs_spec', ['array', 'order', 'label', 'connectivity'])



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

        #dict for storing cluster properties
        self.cluster_props = {}

        #Setup simple pixel index map
        index_map = np.arange(seg_img.size, dtype=np.int64)
        index_map.reshape(seg_img.shape)

        #Set indipendent node attributes
        for node in self.__iter__():
            #get color values for super pixel
            label_mask = self.seg_img == node

            #Assign attributes to node
            self.node[node].update({'labels': [node],
                                    'pixels': set(index_map[label_mask]),
                                    'pixel_count': seg_img[label_mask].size})


    def mask(self, nodes):
        '''asdf'''

        mask_set = set()
        for node in nodes:
            for pixel in self.node[node]['pixels']:
                mask_set.add(pixel)

        return np.array(list(mask_set), dtype=np.int64)


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
        Implemented in bow_rag to overwrite method of the base class
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


    def produce_connectivity_matrix(self, subset, weight=None):
        '''Return a connectivity matrix of a subset of nodes.'''
        sub_graph = self.subgraph(nbunch=list(set(subset)))
        connectivity = nx.adjacency_matrix(sub_graph, weight)

        return connectivity


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




    def delete_attribute(self, attribute):
        '''Delete a given attribute.'''
        for node in self.__iter__():
            del self.node[node][attribute]


    def filter_by_attribute(self, attribute, values, subset=None):
        '''Filter the nodes based on their value of a specified attribute.'''

        if subset is None:
            subset = set(self.__iter__())
        else:
            subset = set(subset)

        func = lambda node: self.node[node][attribute] in values

        return list(filter(func, subset))



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



    def basic_feature_space_array(self, attr_config, label=[], subset=None, exclude=()):
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

        con_matrix = self.produce_connectivity_matrix((subset - exclude))

        return BOW_RAG.fs_spec(fs_array, tuple(order_list), label, con_matrix)


#NOT UP TO DATE

    def attribute_divided_fs_arrays(self, attr_config, div_attr, max_layer=None, subset=None, exclude=()):
        '''Return a feature space array for every value of a specified attribute.
        Nodes are excludable via the 'exclude' parameter.'''
        
        temp_attr = 'temp_' + div_attr

        return_list = []

        for node in subset:
            if max_layer is not None:
                self.node[node][temp_attr] = '-'.join(self.node[node][div_attr][:max_layer])
            else:
                self.node[node][temp_attr] = '-'.join(self.node[node][div_attr])

        div_attr_labels = {self.node[node][temp_attr] for node in self.__iter__()}

        for label in div_attr_labels:
            nodes = self.filter_by_attribute(temp_attr, {label}, subset)
            fs_result = self.basic_feature_space_array(attr_config, label.split('-'), nodes, exclude)

            return_list.append(fs_result)

        self.delete_attribute(temp_attr)
        return return_list



    def hist_to_fs_array(self, attr_config, subset=None, label=[]):
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
                    elif isinstance(self.node[node][attr], hist):
                        part = self.node[node][attr](mode='array', normalized=True)
                        part *= factor
                        row_array = np.append(row_array, part)
                    else:
                        raise TypeError(f'Wrong type: {type(self.node[node][attr])} Must be "hist"!')

                array_list.append(row_array)
                order_list.append(node)

        fs_array = np.array(array_list, dtype=np.float64)

        con_matrix = self.produce_connectivity_matrix(subset)

        return BOW_RAG.fs_spec(fs_array, tuple(order_list), label, con_matrix)



    def clustering(self, attribute, algorithm, fs_spec, return_clust_obj=False, **cluster_kwargs):
        '''Perform any clustering operation from sklearn.cluster on a given feature space array
        (as returnd by 'get_feature_space_array()' or 'hist_to_fs_array()').
        Return the cluster label of each node as an attribute.'''
        
        #Assert all involved nodes have a list as the given attribute
        assertion_set = set()
        for node in fs_spec.order:
            assert(isinstance(self.node[node][attribute], list))
            assertion_set.add(len(self.node[node][attribute]))
        
        #Assert all nodes have the same number of cluster layers
        assert(len(assertion_set) == 1)

        cluster_class = getattr(sklearn.cluster, algorithm)

        if isinstance(fs_spec, BOW_RAG.fs_spec):
            cluster_obj = cluster_class(**cluster_kwargs)
            cluster_obj.fit(fs_spec.array)
            for node, label in zip(fs_spec.order, cluster_obj.labels_):
                #print(node, label)
                #Make sure the clustered feature space and the node have the same label
                if attribute in self.node[node]:
                    assert(fs_spec.label == self.node[node][attribute])
                    self.node[node][attribute].append(str(label))
                else:
                    self.node[node][attribute] = [str(label)]



            if return_clust_obj:
                return cluster_obj


        elif isinstance(fs_spec, list):
            return_clust_obj=False

            for fs in fs_spec:
                self.clustering(attribute, algorithm, fs_spec, return_clust_obj, **cluster_kwargs)


        else:
            raise TypeError("Must be BOW_RAG.fs_spec!")


    def cluster_affinity_attrs(self, attribute, algorithm, fs_spec, stretch=None, limit=None, centralize=1, **cluster_kwargs):
        '''Perform a clustering operation using the clustering() method and store the affinity of the node to each cluster
        (the distance to each )'''

        if stretch is None:
            stretch = (0, 1)
        elif len(stretch) != 2:
            raise ValueError("stretch must contain the two values min and max")
        elif not isinstance(stretch, tuple):
            raise TypeError("stretch must be tuple")

        cluster_obj = self.clustering(attribute, algorithm, fs_spec, return_clust_obj=True, **cluster_kwargs)

        n_clusters = cluster_obj.cluster_centers_.shape[0]


        distances = []
        for row, node in enumerate(fs_spec.order):

            self.node[node][attribute] = []

            node_vector = fs_spec.array[row]

            for cluster in range(n_clusters):
                center_vector = cluster_obj.cluster_centers_[cluster, :]

                distance = np.linalg.norm(center_vector - node_vector)
                distances.append(distance)

                self.node[node][attribute].append(distance)

        min_dist = min(distances)
        stretch_denom = max(distances) - min_dist



        for node in self.__iter__():
            for ix, affinity in enumerate(self.node[node][attribute]):
                affinity -= min_dist
                affinity /= stretch_denom

                affinity *= stretch[1] - stretch[0]
                affinity += stretch[0]
                self.node[node][attribute][ix] = affinity


    def single_feature(self, fs, layer):
        ''' Transforms multifeature clusters to single feature clusters on the graph,
        and adds them to the layer list'''
        
        if isinstance(fs, BOW_RAG.fs_spec):
            fs = [fs]
            
        processed = set() # Keep track which node has already been processed
        
        for _fs in fs:
            cluster = 0
            for node in _fs.order:
                if node in processed:
                    continue
                # Isolate current node
                self.isolate(self, node, cluster, processed, layer)
                cluster += 1
    
    
    def isolate(self, node, clusters, processed, layer):
        # Node is processed and gets new cluster ID
        processed.add(node)
            
        # Do the same for all neighbors with the same previous cluster
        for neighbour in self.neighbors_iter(node):
            if (self.node[node][layer] == self.node[neighbour][layer]) and neighbour not in processed:
                self.isolate(self, neighbour, clusters, processed, layer)   
        
        self.node[node][layer].append(str(clusters))
    
    
    def apply_group_metrics(self, fs, metric_config):
        '''IN DEVELOPMENT'''
        metric_dict = dict.fromkeys(metric_config)

        for metric, function in metric_config.items():
            metric_dict[metric] = function(self, fs)

        return metric_dict


    def produce_cluster_image(self, attribute, max_layer=None, dtype=np.int64):
        '''Render an image (2D numpy array) of cluster labels based
        on a cluster label node attribute.'''

        if max_layer is not None:
            attr_labels = {'-'.join(self.node[node][attribute][:max_layer]) for node in self.__iter__()}
        else:
            attr_labels = {'-'.join(self.node[node][attribute]) for node in self.__iter__()}

        label_dict = dict(zip(attr_labels, range(len(attr_labels))))

        print(label_dict)

        cluster_img = np.zeros_like(self.seg_img, dtype=dtype)

        for node in self.__iter__():
            for label in set(self.node[node]['labels']):
                mask = self.seg_img == label
                attr_string = '-'.join(self.node[node][attribute][:max_layer])
                cluster_img[mask] = label_dict[attr_string ]

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
