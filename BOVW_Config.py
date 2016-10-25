# Configuration file for BoVW
import os

# Initialize variables
GEN_MODEL = False
TEST_IMAGES = False
QUIET = False
K = 10
LIMIT = 5
FEATURES = 'rootsift'
CLUSTERING_ALGORITHM = 'kmeans'
K_MEANS_ITERATION = 3
DISTANCE_LIMIT = 100
FOV_LIMIT = 45

CORPUS_PATH = 'corpus'
if not os.path.exists(CORPUS_PATH):
    os.mkdir(CORPUS_PATH)

ROUTE_PATH = "route"
if not os.path.exists(ROUTE_PATH):
    os.mkdir(ROUTE_PATH)

DATASET_PATH = "dataset"
if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

QUERY_PATH = "queries"
if not os.path.exists(QUERY_PATH):
    os.mkdir(QUERY_PATH)

CODEBOOK_PATH = "codebooks"
if not os.path.exists(CODEBOOK_PATH):
    os.mkdir(CODEBOOK_PATH)
CODEBOOK_PATH += "/" + str(K) + ".p"

arguments = (DATASET_PATH, ROUTE_PATH, TEST_IMAGES, QUIET, K, CODEBOOK_PATH, QUERY_PATH, LIMIT)
# (DATASET_PATH, ROUTE_PATH, TEST_IMAGES, QUIET, K, CODEBOOK_PATH, QUERY_PATH, LIMIT)
# 0=DATASET_PATH 1=ROUTE_PATH 2=TEST_IMAGES 3=QUIET 4=K 5=CODEBOOK_PATH 6=QUERY_PATH 7=LIMIT
