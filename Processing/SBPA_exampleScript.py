# Set up paths
####################################################################################################    
FULL_PATH_TO_IMAGE = "D:/janni/Documents/Geographie/Masterarbeit/Results/Bestof/DJI_0094_square.jpg"
FULL_PATH_TO_INI = "D:/janni/Documents/Geographie/Masterarbeit/Results/Bestof/Data/processing/SCENE2.ini"
OUTPUT_FOLDER = "D:/janni/Documents/Geographie/Masterarbeit/Results/TEST/"
####################################################################################################

#IMPORTS
##SYS PATH
import os
import sys
# This needs to be the folder that contains your repository folder
sys.path.append("../")

###General modules
import matplotlib
import numpy as np

###Skimage imports
from skimage import io, exposure
from skimage.color import rgb2gray, rgb2lab
from skimage.util import img_as_float, img_as_ubyte
from skimage.segmentation import slic
from skimage.filters import gaussian

###SBPA imports
import SBPA.config as conf
from SBPA.rgb_indices import make_rgb_indices, make_pca
from SBPA.frequency import DFTanalyzer
from SBPA.lbp import ni_lbp, radial_lbp, angular_lbp
from SBPA.histogram import Hist
from SBPA._LBP.lbp_bins import lbp_bins
from SBPA.ipag import IPAG, get_internal_distance
from SBPA.sbpa_utils import normalize_image, DoubleLogStream
from SBPA.processing import DynamicClustering, LogicStage, ClusterStage, SplittingStage, \
                            IsolateStage, threshhold, AbsorptionStage
from SBPA.metrics import *
from SBPA.visual import plot_sbpa


####################### SETUP #################################################

# Get name of the image file
IMAGE_FILENAME = os.path.basename(FULL_PATH_TO_IMAGE).split('.')[0]

# Read ini file
PARAMS = conf.get_params(FULL_PATH_TO_INI)

# If logfile should be created (specified in ini)
if PARAMS.script.log:
    LOGFILE = IMAGE_FILENAME+'.log'
    log = open(LOGFILE, 'w')
    log.close()
    log = open(LOGFILE, 'a')
    old_stdout = sys.stdout
    sys.stdout = DoubleLogStream(log, old_stdout)



####################### PRE-PROCESSING ########################################
print('Start image processing...')
image = img_as_float(io.imread(FULL_PATH_TO_IMAGE))
image_gray = rgb2gray(image)
IMAGE_SIZE = image.shape[0]*image.shape[1]


print('Start Frequency:')
# Inserting frequency wavelength and sigma by hand (specified in ini)
frequency = DFTanalyzer(image_gray)
manual_wavelength = PARAMS.freq.manual_wavelength
manual_sigma = PARAMS.freq.manual_sigma
frequency.wavelength = float(manual_wavelength)
frequency.filtered_img = gaussian(image_gray, sigma=float(manual_sigma))


print("Start RGBI Indices")
rgb_indices, rgb_indices_dict = make_rgb_indices(img_as_ubyte(image))
components, components_dict = make_pca(rgb_indices_dict, image)


print("Start Local Binary Pattern")
print("LBP Radius: ", frequency.texture_radius)
radius = frequency.texture_radius
n_points = 8

nilbp = ni_lbp(frequency.filtered_img, n_points, radius, method="ror")
radlbp = radial_lbp(frequency.filtered_img, n_points, radius*2, radius, 'ror')
anglbp = angular_lbp(frequency.filtered_img, 4, radius)

rad_lbp_BINS = lbp_bins(n_points, "ror")
ang_lbp_BINS = lbp_bins(4, "default")
ni_lbp_BINS = lbp_bins(n_points, "ror")


####################### SUPERPIXELS ###########################################
print("Start Superpixel")
# Check for absolute or relative number of superpixel (specified in the ini)
if PARAMS.sp.absolute:
    n_sp = int(PARAMS.sp.number)
else:
    n_sp = int(round(IMAGE_SIZE/PARAMS.sp.number))

print('Size:', IMAGE_SIZE)
print('Min patch size:', frequency.min_patch_size)
print("Desired number of Superpixels: ", n_sp, 'Ratio: ', IMAGE_SIZE/frequency.min_patch_size)

segments_slic = slic(image,
                     n_segments=n_sp,
                     compactness=PARAMS.sp.compactness,
                     sigma=PARAMS.sp.sigma)

segments_slic = segments_slic.astype(np.int32)

real_n_sp = len(np.unique(segments_slic))
print('Produced number of segments: {}'.format(real_n_sp))


####################### IMAGE PATCH ADJACENCY GRAPH (IPAG) ####################
print("Build IPAG")
image_lab = rgb2lab(image) # Image needs to be in LAB Color Space
pca_inverted = 1-components.dim1 # Principal components needs to be inverted
graph = IPAG(segments_slic)

print('Adding attributes to IPAG...')
graph.add_attribute('color', normalize_image(image_lab), np.mean) # Mean Color
graph.add_attribute('var', image_gray, np.var) # Variance
graph.add_attribute("pc1", pca_inverted, np.mean) # Mean pca results
# Adding texture information
graph.add_attribute('ni_lbp', nilbp, Hist, vbins=ni_lbp_BINS)
graph.normalize_attribute('ni_lbp')
graph.add_attribute('rad_lbp', radlbp, Hist, vbins=rad_lbp_BINS)
graph.normalize_attribute('rad_lbp')
graph.add_attribute('ang_lbp', anglbp, Hist, vbins=ang_lbp_BINS)
graph.normalize_attribute('ang_lbp')

# Calculating texture affinity clusters
lbp_config = {'ni_lbp':0.01, 'rad_lbp':0.01, 'ang_lbp':0.01}
lbp_fs = graph.hist_to_fs_array(lbp_config)
graph.cluster_affinity_attrs("texture", "KMeans", lbp_fs, n_clusters=5)




####################### FEATURE SPACE #########################################
print("Create Feature Space")
fs_attribute_weights = {'color':PARAMS.fs.color,
            'var':PARAMS.fs.var,
            'pc1': PARAMS.fs.pc1,
            "texture": PARAMS.fs.texture}

fs = graph.basic_feature_space_array(fs_attribute_weights)

cluster_field_name = 'cluster' # Attribute name for cluster results

internal_distance = get_internal_distance(fs)

# Setup pointers to the functions used for calculating the feature space metrics
metric_config = {"fs_var":metric(fs_variance,{}),
                 "pixel_size":metric(count_pixel,{}),
                 "multifeatures":metric(count_multi_features,{"attribute":cluster_field_name}),
                 "superpixel":metric(count_superpixel,{})}

fs_metrics = graph.apply_group_metrics(fs, metric_config)




####################### DYNAMIC CLUSTERING ####################################
print("Start dynamic clustering")
# Check if scale limit is absolute or relative (specified in ini file)
if PARAMS.proc.absolute:
    scale_limit = int(PARAMS.proc.scale)
else:
    scale_limit = int(round(IMAGE_SIZE*PARAMS.proc.scale))

# Set up dynamic clustering
stages = DynamicClustering()

# Define dynamic clustering stages
stages['var'] = LogicStage(False,
                             {"fs_var":threshhold('>=', fs_metrics["fs_var"] * PARAMS.proc.var_factor)},
                             'Variance Check')

stages['size'] = LogicStage(True,
                            {'pixel_size':threshhold('>=', scale_limit)},
                            'Entry Point')

stages['superpixel_check'] = LogicStage(False,
                                        {"superpixel":threshhold('>=', PARAMS.proc.min_sp)},
                                        'SP Check')

stages['MeanShift'] = ClusterStage(False,
                                   {'pixel_size':threshhold('>', scale_limit*PARAMS.proc.meanshift)},
                                   'Meanshift Stage',
                                   algorithm='MeanShift')

stages['KMeans'] = ClusterStage(False,
                                {'pixel_size':threshhold('>', scale_limit*PARAMS.proc.kmeans)},
                                'KMeans Stage',
                                algorithm='KMeans', n_clusters=PARAMS.proc.nc_kmeans)

stages['agglo'] = ClusterStage(False,
                                {'pixel_size':threshhold('>=', scale_limit)},
                                'Second agglo',
                                algorithm='AgglomerativeClustering', n_clusters=PARAMS.proc.nc_agglo)

stages['split'] = SplittingStage(True, descr='Splitter')

stages.post_processing_stage = AbsorptionStage(False,
                                               {'pixel_size':threshhold('<', scale_limit)},
                                               descr='Absorption', 
                                               norm_distance = internal_distance, 
                                               factors = PARAMS.proc.factors, 
                                               dist_threshhold = internal_distance * PARAMS.proc.dist_threshhold)



stages['isolate'] = IsolateStage(False, descr='Isolate')


# Link stages
stages.link_stages('size', 'var', stages.post_processing_stage)
stages.link_stages('var', 'superpixel_check')
stages.link_stages('superpixel_check', 'MeanShift')
stages.link_stages('MeanShift', 'isolate', 'KMeans') 
stages.link_stages('KMeans', 'isolate', 'agglo') 
stages.link_stages('isolate', 'split')
stages.link_stages('agglo', 'split')
stages.link_stages('split', 'size', 'KMeans')

stages.exc_recorder.raise_mode = 'all'
stages.set_exception_recorder()
stages.initiate_stages()

stages(graph, fs_attribute_weights, cluster_field_name, metric_config, 'size')


####################### OUTPUT ########################################################
segments_result = graph.produce_cluster_image('cluster', sort=True)
segments_result += 1 # Clusters in image need to start at 1
""" 
At this point we have
graph: contains the IPAG with dynamic clustering results, clustering history and initial state
segments_result: cluster image (pixel value = cluster id) derived from the graph
Ready for further processing
"""

print("FINISHED")


####################### PLOT ########################################################
# Define Boundary Color    
colormap_bounds = matplotlib.colors.ListedColormap([PARAMS.plot.boundary_color,[1,1,0]])
    
plot_sbpa(background = image, segments = segments_result,
          thickness = PARAMS.plot.thickness, boundary_color = colormap_bounds, 
          outputfile = os.path.join(OUTPUT_FOLDER, IMAGE_FILENAME+"_region_map_boundaries.png"))
