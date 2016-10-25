#  Capt Noah Lesch
#  BOVW Python File

import BOVW_Config as BC  # Configuration file
import BOVW_Modules as BOM
import glob
import numpy as np
import pickle
from geopy.distance import vincenty
from scipy.cluster.vq import vq
from sys import exit
import os

if __name__ == '__main__':
    print BC.arguments
    session = BOM.CONNECT_MSACS_DB()

    if BC.GEN_MODEL:

        session.execute("DROP TABLE IF EXISTS LOCATIONS")
        session.execute("CREATE TABLE LOCATIONS(FILENAME TEXT PRIMARY KEY, LAT DOUBLE, LON DOUBLE, FOV INT)")
        # The first iteration downloads the corpus so TEST_IMAGES should be false
        BOM.compiledataset(session, BC.TEST_IMAGES, BC.CORPUS_PATH, BC.QUIET, BC.DATASET_PATH)

        session.execute("DROP TABLE IF EXISTS QUERIES")
        session.execute("CREATE TABLE QUERIES(FILENAME TEXT PRIMARY KEY, LAT DOUBLE, LON DOUBLE, FOV INT)")
        # Now we can switch TEST_IMAGES to be true so our query set is populated and our route folder is parsed
        BOM.compiledataset(session, not BC.TEST_IMAGES, BC.ROUTE_PATH, BC.QUIET, BC.QUERY_PATH)

        # Now we will calculate the individual image features and store them in the FEATURES database
        # We will insert the downloaded images into the LOCATIONS and QUERIES database
        # We will also need to rename each file to not include its FOV in the filename
        session.execute("DROP TABLE IF EXISTS FEATURES")  # Drop old table if it exists
        session.execute("CREATE TABLE FEATURES(FILENAME TEXT PRIMARY KEY, NUM INT)")  # Prep new table for features

        ICFeatures = 0  # Initialize to hold features

        for image in glob.glob(BC.DATASET_PATH + '/*.jpg'):
            # LOCATIONS database
            count = image[image.find('/') + 1:image.find('_')]
            latlon = image[image.find('_') + 1:image.rfind('_')]
            lat = latlon[:latlon.find('_')]
            lon = latlon[latlon.find('_') + 1:]
            fov = image[image.rfind('_') + 1:image.find('.jpg')]
            session.execute(
                "INSERT INTO LOCATIONS(FILENAME,LAT,LON,FOV) VALUES('" + count + ".jpg',"
                    + lat + "," + lon + "," + fov + ")")
            os.rename(image, image[:image.find('_')]+'.jpg')
            image = image[:image.find('_')]+'.jpg'

            features = BOM.calcfeatures(image, BC.FEATURES)  # Calculate image features with desired feature type
            # features is a tuple  w/the form (kp, des)
            # features[0]=kp is a list of key points
            # features[1]=des is an ndarray of descriptors of size (kp-by-128)

            # print features # Number of key points = number of descriptors and each descriptor has 128 values

            # The first time this if-statement runs will be the only time the first path is TRUE
            # Each subsequent run the array will be an ndarray and the features from the image will
            # be appended.  This creates a running list of features which can be several hundred thousand
            # feature long.  After this list is completed then k-means and codebook calculation can be done.
            if type(ICFeatures) is int:
                ICFeatures = features[1]
            else:
                ICFeatures = np.append(ICFeatures, features[1], axis=0)

            # The database statement inserts the filename as the PK and also the number of features in the image
            session.execute("INSERT INTO FEATURES(FILENAME, NUM) VALUES(%s,%s)",
                            (str(image).split('/')[1], len(features[0])))

        # QUERIES database
        for image in glob.glob(BC.ROUTE_PATH + '/*.jpg'):
            count = image[image.find('/') + 1:image.find('_')]
            latlon = image[image.find('_') + 1:image.rfind('_')]
            lat = latlon[:latlon.find('_')]
            lon = latlon[latlon.find('_') + 1:]
            fov = image[image.rfind('_') + 1:image.find('.jpg')]
            session.execute(
                "INSERT INTO QUERIES(FILENAME,LAT,LON,FOV) VALUES('" + count + ".jpg',"
                    + lat + "," + lon + "," + fov + ")")

        codebook = BOM.calccodebook(ICFeatures, BC.CLUSTERING_ALGORITHM, BC.K, BC.K_MEANS_ITERATION)
        pickle.dump(codebook, open(BC.CODEBOOK_PATH, "wb"))

        if not BC.QUIET:
            print "\tCodebook stored at " + BC.CODEBOOK_PATH

        print codebook

        # The below block of code calculates and inserts the histograms for each dataset image into the database
        session.execute("DROP TABLE IF EXISTS HISTOGRAMS")
        session.execute("CREATE TABLE HISTOGRAMS(FILENAME TEXT PRIMARY KEY, HISTOGRAM TEXT)")

        for image in glob.glob(BC.DATASET_PATH + "/*.jpg"):
            features = BOM.calcfeatures(image, BC.FEATURES)

            encodedhist = BOM.createhistograms(codebook, BC.K, features[1])

            # The below code inserts the histogram data into the table
            # imagePath[imagePath.rfind("/")+1:] splits dataset/###.jpg at /, looks 1 char to the right and ends at the
            #   end of the string so we are left with ###.jpg
            session.execute("INSERT INTO HISTOGRAMS(FILENAME,HISTOGRAM) VALUES(%s,%s)",
                            (image[image.rfind("/") + 1:], encodedhist))
    # The below block of code calculates the ground truth matrix of our model

    groundTruth = [[0 for a in range(len(glob.glob(BC.DATASET_PATH + '/*.jpg')))] for b
                   in range(len(glob.glob(BC.QUERY_PATH + '/*.jpg')))]
    num_row, num_col = 0, 0
    for row in session.execute("SELECT * FROM QUERIES"):
        q_name = int(row.filename.split('.')[0])  # Query file numeric prefix
        q_lat = row.lat
        q_lon = row.lon
        q_fov = row.fov
        for data_query in session.execute("SELECT * FROM LOCATIONS"):
            d_name = int(data_query.filename.split('.')[0])
            d_lat = data_query.lat
            d_lon = data_query.lon
            d_fov = data_query.fov
            dist = vincenty((q_lat, q_lon), (d_lat, d_lon)).meters
            groundTruth[num_row][num_col] = BOM.computeGroundTruth(BC.DISTANCE_LIMIT, BC.FOV_LIMIT, dist, q_fov, d_fov)
            num_col += 1

        num_row += 1  # Increment row variable so we look at next query image
        num_col = 0  # Reset the column variable

    if not BC.QUIET:
        print "Ground truth table computed."
        BOM.relevantcalc(groundTruth, BC.QUIET)  # See how many relevant images are in the ground truth matrix

    # The below block of code executes the search function using the query images

    for query_image in glob.glob(BC.QUERY_PATH + '/*.jpg'):

        try:
            type(codebook)  # If codebook hasn't been prev. defined (i.e. GEN MODEL was false) we will need to load
        except NameError:
            codebook = pickle.load(open(BC.CODEBOOK_PATH, "rb"))

        query_kp, query_des = BOM.calcfeatures(query_image, BC.FEATURES)
        codes, distortion = vq(query_des, codebook)  # Calculate codebook mbrship of observed vectors vq(obs, code_book)
        q_hist, q_bins = np.histogram(codes, BC.K)  # Assigns codes from above to the K-bin histogram values

        for query_data in session.execute("SELECT LAT,LON,FOV FROM QUERIES WHERE FILENAME='" + query_image[query_image.rfind(
                "/") + 1:] + "'"):
            q_lat = query_data.lat
            q_lon = query_data.lon
            q_fov = query_data.fov

        histograms = session.execute("SELECT * FROM HISTOGRAMS")  # Grabs all histograms in database
        results = BOM.search(q_hist, histograms, BC.K, BC.LIMIT)  # Results returns a tuple of (dist, image name.jpg)

        # If our highest rated match is .99 or above we assume the query image is in the same location as corpus image
        try:
            if results[0][0] >= .99:
                for image_data in session.execute("SELECT LAT,LON FROM LOCATIONS WHERE FILENAME='" + results[0][1] + "'"):
                    distance = results[0][0]
                    lat = image_data.lat
                    lon = image_data.lon
                guesslocation = vincenty((q_lat, q_lon), (lat, lon)).meters

            elif results[0][0] < .99:
                avgLat = 0
                avgLon = 0
                count = 0
                for distance, path in results:
                    for image_data in session.execute("SELECT LAT,LON FROM LOCATIONS WHERE FILENAME='" + path + "'"):
                        avgLat += image_data.lat
                        avgLon += image_data.lon
                        count += 1.0
                        # guess_Location = BOM.guessLatLon(distance, path, image_data.lat, image_data.lon)
                lat = avgLat / count
                lon = avgLon / count
                guesslocation = vincenty((q_lat, q_lon), (avgLat, avgLon)).meters

            if not BC.QUIET:
                print "For " + query_image
                print "\tActual Location: " + str(q_lat) + ", " + str(q_lon)
                print "\tGuess Location: " + str(lat) + ", " + str(lon)
                print "\tDistance apart: " + str(vincenty((q_lat, q_lon), (lat, lon)).meters)
        # Otherwise if our results set is empty for some reason we handle except and gracefully exit
        except IndexError:
            print "Empty results set!"
            exit()  # sys.exit() function to terminate program since we can not proceed farther
