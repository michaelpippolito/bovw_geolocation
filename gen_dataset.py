import argparse
import urllib
import numpy
import os
from cassandra.cluster import Cluster

# Parse command line arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=False,
                help="Path to store images, default=./dataset/")
ap.add_argument("-r", "--route", required=False,
                help="GPX file to use for dataset creation, default=./route.gpx")
ap.add_argument("-t", "--test", required=False,
                help="Set true if using to generate test queries, default=False")
ap.add_argument("-q", "--quiet", required=False,
                help="Set true for less output, default=False")

args = vars(ap.parse_args())

# Initialize variables
DATASET_PATH = "dataset"
if args["dataset"] is not None:
    DATASET_PATH = args["dataset"]
if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

ROUTE_PATH = "route.gpx"
if args["route"] is not None:
    ROUTE_PATH = args["route"]

TEST_IMAGES = False
if args["test"] is not None:
    if args["test"] in ["true", "True", "TRUE", "t", "T"]:
        TEST_IMAGES = True

QUIET = False
if args["quiet"] is not None:
    if args["quiet"] in ["true", "True", "TRUE", "t", "T"]:
        QUIET = True

# Initialize Google Street View URL
STREET_VIEW_URL = "https://maps.googleapis.com/maps/api/streetview?size=500x500"
API_KEY = "AIzaSyAxnyL1XvyItXaU2MAq2u3Jzn9RYYSiSaA"

# Initialize Cassandra database
cluster = Cluster()
session = cluster.connect("bovw")
if not TEST_IMAGES:
    session.execute("DROP TABLE IF EXISTS LOCATIONS")
    session.execute("CREATE TABLE LOCATIONS(FILENAME TEXT PRIMARY KEY, LAT DOUBLE, LON DOUBLE)")
else:
    session.execute("DROP TABLE IF EXISTS QUERIES")
    session.execute("CREATE TABLE QUERIES(FILENAME TEXT PRIMARY KEY, LAT DOUBLE, LON DOUBLE)")

if not QUIET:
    print "Downloading images..."
# Parse the GPX file for the coordinates
# Download the image at those coordinates
count = 0
for line in open(ROUTE_PATH):
    if "gpxx:rpt" in line:
        # Parse the latitude and longitude
        lat = line[line.find('lat="') + 5:line.rfind('" lon=')]
        lon = line[line.find('lon="') + 5:line.rfind('"')]

        FOV = []
        # For dataset generation, download 4 images per coordinate
        # For test query generation, download 1 image per coordinate
        if not TEST_IMAGES:
            FOV.append(numpy.random.randint(0, 90))  # North
            FOV.append(numpy.random.randint(90, 180))  # East
            FOV.append(numpy.random.randint(180, 270))  # South
            FOV.append(numpy.random.randint(270, 360))  # West
        else:
            FOV.append(numpy.random.randint(0, 360))  # Random

        for fov in FOV:
            # Create the URL String
            URL = STREET_VIEW_URL + "&location=" + lat + "," + lon + "&heading=" + str(fov) + "&key=" + API_KEY
            urllib.urlretrieve(URL, DATASET_PATH + "/" + str(count) + ".jpg")

            # Insert image location and name into database
            if not TEST_IMAGES:
                session.execute(
                    "INSERT INTO LOCATIONS(FILENAME,LAT,LON) VALUES('" + str(count) + ".jpg'," + lat + "," + lon + ")")
            else:
                session.execute(
                    "INSERT INTO QUERIES(FILENAME,LAT,LON) VALUES('" + str(count) + ".jpg'," + lat + "," + lon + ")")
            count += 1

count += 1
if not QUIET:
    print "\tDownloaded " + str(count) + " images!"
