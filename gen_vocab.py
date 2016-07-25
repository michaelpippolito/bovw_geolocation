import argparse
import numpy
import os
import cv2
import glob
import pickle
import base64
from scipy.cluster.vq import *
from cassandra.cluster import Cluster


# Extract and return the rootSIFT keypoints and descriptors from an image
def compute(imagePath, sift, eps=1e-7):
    image = cv2.imread(imagePath, 0)

    kp, des = sift.detectAndCompute(image, None)

    if des is not None:
        kp, des = sift.compute(image, kp)

        des /= (des.sum(axis=1, keepdims=True) + eps)
        des = numpy.sqrt(des)

        des = whiten(des)

        return kp, des
    else:
        return ([], None)


# Parse command line arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=False,
                help="Path to the dataset of images, default=dataset")
ap.add_argument("-k", "--kmeans", required=False,
                help="Value of k to cluster with, default=10")
ap.add_argument("-c", "--codebook", required=False,
                help="Path to store the codebooks, default=codebooks")
ap.add_argument("-q", "--quiet", required=False,
                help="Set true for less output, default=False")

args = vars(ap.parse_args())

# Initialize variables
DATASET_PATH = "dataset"
if args["dataset"] is not None:
    DATASET_PATH = args["dataset"]

K = 10
if args["kmeans"] is not None:
    K = int(args["kmeans"])

CODEBOOK_PATH = "codebooks"
if args["codebook"] is not None:
    CODEBOOK_PATH = args["codebook"]
if not os.path.exists(CODEBOOK_PATH):
    os.mkdir(CODEBOOK_PATH)
CODEBOOK_PATH += "/" + str(K) + ".p"

QUIET = False
if args["quiet"] is not None:
    if args["quiet"] in ["true", "True", "TRUE", "t", "T"]:
        QUIET = True

if not QUIET:
    print "Feature extraction..."
# Extract and store the descriptors of each image
sift = cv2.xfeatures2d.SIFT_create()  # SIFT extractor

des_list = 0  # To store descriptors
for imagePath in glob.glob(DATASET_PATH + "/*.jpg"):
    kp, des = compute(imagePath, sift)

    if type(des_list) is int:
        des_list = des
    else:
        des_list = numpy.append(des_list, des, axis=0)

if not QUIET:
    print "\t" + str(len(des_list)) + " features extracted!"
    print "Performing k-means clustering and storing codebook..."

codebook, variance = kmeans(des_list, K, 3)
pickle.dump(codebook, open(CODEBOOK_PATH, "wb"))

if not QUIET:
    print "\tCodebook stored at " + CODEBOOK_PATH
    print "Creating histograms"

# Initialize Cassandra database
cluster = Cluster()
session = cluster.connect("bovw")
session.execute("DROP TABLE IF EXISTS HISTOGRAMS")
session.execute("CREATE TABLE HISTOGRAMS(FILENAME TEXT PRIMARY KEY, HISTOGRAM TEXT)")

for imagePath in glob.glob(DATASET_PATH + "/*.jpg"):
    kp, des = compute(imagePath, sift)

    codes, distortion = vq(des, codebook)
    hist, bins = numpy.histogram(codes, K)

    histogram_pickle = base64.b64encode(pickle.dumps(hist, pickle.HIGHEST_PROTOCOL))

    session.execute("INSERT INTO HISTOGRAMS(FILENAME,HISTOGRAM) VALUES('" + imagePath[imagePath.rfind("/")+1:] + "', '" + histogram_pickle + "')")

if not QUIET:
    print "\tHistograms stored!"
