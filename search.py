import argparse
import numpy
import os
import cv2
import glob
import pickle
import base64
import scipy.spatial.distance as distCalc
from scipy.cluster.vq import *
from geopy.distance import vincenty
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
    else :
        return ([], None)

# Guess the latitude and longitude values of the query image
def guessLatLon(results, session) :
    avgLat = 0.0
    avgLon = 0.0
    count = 0.0
    for distance, path in results :
        if distance <= 10.0 :
            for image_data in session.execute("SELECT LAT,LON FROM LOCATIONS WHERE FILENAME='" + path + "'") :
                lat = image_data.lat
                lon = image_data.lon

                return (lat,lon)
        elif(distance <= 250.0) :
            for image_data in session.execute("SELECT LAT,LON FROM LOCATIONS WHERE FILENAME='" + path + "'") :
                avgLat += image_data.lat
                avgLon += image_data.lon
                count += 1.0

    avgLat /= count
    avgLon /= count
    return (avgLat,avgLon)


ap = argparse.ArgumentParser()

ap.add_argument("-q", "--query", required=False,
                help="Path to query image or directory of images, default=queries")
ap.add_argument("-c", "--codebook", required=False,
                help="Path to the codebook, default=codebooks/10.p")
ap.add_argument("-l", "--limit", required=False,
                help="Number of images to return, default=10")
ap.add_argument("--quiet", required=False,
                help="Set true for less output, default=False")

args = vars(ap.parse_args())

# Initialize Variables
QUERY_PATH = "queries"
if args["query"] is not None :
    QUERY_PATH = args["query"]

CODEBOOK_PATH = "codebooks/10.p"
if args["codebook"] is not None :
    CODEBOOK_PATH = args["codebook"]
CODEBOOK = pickle.load(open(CODEBOOK_PATH, "rb"))

QUIET = False
if args["quiet"] is not None:
    if args["quiet"] in ["true", "True", "TRUE", "t", "T"]:
        QUIET = True

K = len(CODEBOOK)

LIMIT = 10
if args["limit"] is not None :
    LIMIT = int(args["limit"])

sift = cv2.xfeatures2d.SIFT_create()

# Initialize Cassandra database
cluster = Cluster()
session = cluster.connect("bovw")

if os.path.isfile(QUERY_PATH) :
    q_kp, q_des = compute(QUERY_PATH, sift)
    codes, distortion = vq(q_des, CODEBOOK)
    q_hist, q_bins = numpy.histogram(codes, K)
    q_lat = 0.0
    q_lon = 0.0

    for query_data in session.execute("SELECT LAT,LON FROM QUERIES WHERE FILENAME='"+QUERY_PATH[QUERY_PATH.rfind("/")+1:] + "'") :
        q_lat = query_data.lat
        q_lon = query_data.lon

    histograms = session.execute("SELECT * FROM HISTOGRAMS")

    distances = []
    paths = []
    for h in histograms :
        path = h.filename
        hist = pickle.loads(base64.b64decode(h.histogram))

        distances.append(distCalc.euclidean(q_hist, hist))
        paths.append(path)

    results = zip(distances,paths)
    results = sorted(results)
    results = results[:LIMIT]

    guess_location = guessLatLon(results, session)
    print "For " + QUERY_PATH
    print "\tActual Location: " + str(q_lat) + ", " + str(q_lon)
    print "\tGuess Location: " + str(guess_location[0]) + ", " + str(guess_location[1])
    print "\tDistance apart: " + str(vincenty((q_lat,q_lon), guess_location).meters)
else :
    avgDistance = 0.0
    count = 0.0
    for imagePath in glob.glob(QUERY_PATH + "/*.jpg") :
        q_kp, q_des = compute(imagePath, sift)
        codes, distortion = vq(q_des, CODEBOOK)
        q_hist, q_bins = numpy.histogram(codes, K)
        q_lat = 0.0
        q_lon = 0.0

        for query_data in session.execute("SELECT LAT,LON FROM QUERIES WHERE FILENAME='" + imagePath[imagePath.rfind("/") + 1:] + "'"):
            q_lat = query_data.lat
            q_lon = query_data.lon

        histograms = session.execute("SELECT * FROM HISTOGRAMS")

        distances = []
        paths = []
        for h in histograms:
            path = h.filename
            hist = pickle.loads(base64.b64decode(h.histogram))

            distances.append(distCalc.euclidean(q_hist, hist))
            paths.append(path)

        results = zip(distances, paths)
        results = sorted(results)
        results = results[:LIMIT]

        guess_location = guessLatLon(results,session)
        avgDistance += vincenty((q_lat,q_lon),guess_location).meters
        count += 1.0

        if not QUIET :
            print "For " + imagePath
            print "\tActual Location: " + str(q_lat) + ", " + str(q_lon)
            print "\tGuess Location: " + str(guess_location[0]) + ", " + str(guess_location[1])
            print "\tDistance apart: " + str(vincenty((q_lat, q_lon), guess_location).meters)

    avgDistance /= count
    print "Average distance from target: " + str(avgDistance)