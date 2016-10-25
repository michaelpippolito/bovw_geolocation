# BoVW Modules for BoVW Program


def CONNECT_MSACS_DB():
    from cassandra.cluster import Cluster
    cluster = Cluster()
    session = cluster.connect()
    session.execute("CREATE KEYSPACE IF NOT EXISTS tst WITH replication = {'class':'SimpleStrategy', 'replication_factor' : 1}")
    session.set_keyspace("tst")
    return session


def compiledataset(session, test_images, model_path, verbose, img_destination):

    import urllib
    import numpy
    import glob
    STREET_VIEW_URL = "https://maps.googleapis.com/maps/api/streetview?size=500x500"
    API_KEY = "AIzaSyAxnyL1XvyItXaU2MAq2u3Jzn9RYYSiSaA"

    if not verbose:
        print "Downloading images..."
    # Parse the GPX file for the coordinates
    # Download the image at those coordinates
    count = 0
    gpx = glob.glob(model_path + '/*.gpx')
    for route in gpx:
        for line in open(route):
            if "lat=" in line:
                # Parse the latitude and longitude
                lat = line[line.find('lat="') + 5:line.rfind('" lon=')]
                lon = line[line.find('lon="') + 5:line.rfind('"')]

                FOV = []
                # For dataset generation, download 4 images per coordinate
                # For test query generation, download 1 image per coordinate
                if not test_images:
                    FOV.append(numpy.random.randint(0, 90))  # North
                    FOV.append(numpy.random.randint(90, 180))  # East
                    FOV.append(numpy.random.randint(180, 270))  # South
                    FOV.append(numpy.random.randint(270, 360))  # West
                else:
                    FOV.append(numpy.random.randint(0, 360))  # Random

                for fov in FOV:
                    # Create the URL String
                    URL = STREET_VIEW_URL + "&location=" + lat + "," + lon + "&heading=" + str(fov) + "&key=" + API_KEY
                    urllib.urlretrieve(URL, img_destination + "/" + str(count) + "_" + str(lat) + "_" + str(lon) + "_" + str(fov) + ".jpg")

                    # # Insert image location and name into database
                    # if not test_images:
                    #     session.execute(
                    #         "INSERT INTO LOCATIONS(FILENAME,LAT,LON,FOV) VALUES('" + str(
                    #             count) + ".jpg'," + lat + "," + lon + "," + str(fov) + ")")
                    # else:
                    #     session.execute(
                    #         "INSERT INTO QUERIES(FILENAME,LAT,LON,FOV) VALUES('" + str(
                    #             count) + ".jpg'," + lat + "," + lon + "," + str(fov) + ")")
                    count += 1

    if not verbose:
        print "\tDownloaded " + str(count) + " images!"
    return


def calcfeatures(img, features):

    if features == 'sift':
        return sift(img)
    elif features == 'rootsift':
        return rootsift(img)
    elif features == 'hog':
        pass  # Insert code to do hog features
    #  Insert additional elif/else statements as needed
    else:
        "Incorrect feature type specified in BOVW_Config!"
        return


def sift(img):
    from cv2 import xfeatures2d
    from cv2 import imread

    sift = xfeatures2d.SIFT_create()  # SIFT extractor
    image = imread(img, 0)
    kp, des = sift.detectAndCompute(image, None)

    return kp, des


def rootsift(img, eps=1e-7):
    from cv2 import xfeatures2d
    from cv2 import imread
    import numpy as np
    from scipy.cluster.vq import whiten

    image = imread(img, 0)
    sift = xfeatures2d.SIFT_create()  # SIFT extractor

    kp, des = sift.detectAndCompute(image, None)

    if des is not None:
        kp, des = sift.compute(image, kp)

        des /= (des.sum(axis=1, keepdims=True) + eps)
        des = np.sqrt(des)

        des = whiten(des)

        return kp, des
    else:
        return [], None


def calccodebook(descriptors, cluster_algorithm, clusters, iterations):
    # Arguments tuple = (DATASET_PATH,ROUTE_PATH,TEST_IMAGES,QUIET,K,CODEBOOK_PATH, QUERY_PATH, LIMIT)
    # 0=DATASET_PATH 1=ROUTE_PATH 2=TEST_IMAGES 3=QUIET 4=K 5=CODEBOOK_PATH 6=QUERY_PATH 7=LIMIT

    if cluster_algorithm == 'kmeans':
        from scipy.cluster.vq import kmeans
        codebook, variance = kmeans(descriptors, clusters, iterations)
        # descriptors = image feature descriptors
        # clusters = the K value used for k-means clustering
        # iterations = the number of times to perform kmeans clustering
        return codebook
    elif cluster_algorithm == "other_than_kmeans":
        pass  # Do other clustering algorithm related code
        return  # Needs to return some type of codebook related to another clustering algorithm


def createhistograms(codebook, clusters, descriptors):
    from numpy import histogram
    import base64
    from pickle import dumps
    from pickle import HIGHEST_PROTOCOL
    from scipy.cluster.vq import vq

    codes, distortion = vq(descriptors, codebook)
    hist, bins = histogram(codes, clusters)  # Imported function from numpy
    # print hist
    histogram_pickle = base64.b64encode(dumps(hist, HIGHEST_PROTOCOL))  # Imported function names from pickle

    return histogram_pickle


def computeGroundTruth(dist_limit, view_limit, actual_dist, query_fov, corpus_fov):

    # Bounds tuple (rows, cols, fovLim, distLim)

    # latlist = []
    # lonlist = []

    # The below if-nest determines the relevance of the query image for each location image.  This is done by
    # first determining the distance using the vincenty function above.  If the difference is > 10 we are not
    # relevant whatsoever.  If the difference is <= 10 we may be relevant if the field of view is within
    # +/- the fovLimit variable.  The below statements also account for wrap-around (i.e. 360 degrees is within
    # the fovLimit of say 10 degrees since it's only a 10 degree difference)

    if actual_dist > dist_limit:
        return 0  # If dist > 10 we are not relevant, no need to look at below fov code
    elif actual_dist <= dist_limit and (query_fov == corpus_fov):
        return 1  # The image is relevant
        # print "Q: " + str(q_fov) + " D: " + str(d_fov) + " are equal."  # Debug statement
    elif (query_fov > corpus_fov) and ((query_fov - corpus_fov) <= view_limit or ((360-view_limit) <= (query_fov - corpus_fov) <= 360)):  # Logic is: X && (Y || Z)
        return 1  # The image is relevant
        # print "Query Image: " + str(row) + " Data Image: " + str(data_query) + " Dist: " + str(dist) # Debug statement
        # print "Q: " + str(q_fov) + " D: " + str(d_fov) + " with Q > D"  # Debug statement
    elif (query_fov < corpus_fov) and (abs(query_fov - corpus_fov) <= view_limit or ((360 - view_limit) <= abs(query_fov - corpus_fov) <= 360)):  # Logic is X && ( Y || Z)
        # print "Query Image: " + str(row) + " Data Image: " + str(data_query) + " Dist: " + str(dist) # Debug statement
        return 1  # The image is relevant
        # print "Q: " + str(q_fov) + " D: " + str(d_fov) + " with Q < D"  # Debug statement
    else:
        return 0  # The image is not relevant


def relevantcalc(groundtruth_matrix, verbose):
    relevant = 0  # Initialize variable to hold relevant images
    rows = len(groundtruth_matrix)  # Determine number of rows in groundtruth matrix
    cols = len(groundtruth_matrix[0])  # Determine number of cols in groundtruth matrix

    for x in range(rows):
        for y in range(cols):
            if groundtruth_matrix[x][y] == 1:
                relevant += 1

    if not verbose:
        print "There are " + str(relevant) + " total relevant images in the dataset"
    return


def search(query_histogram, histograms, clusters, set_limit):
    import glob
    import numpy as np
    from scipy.cluster.vq import vq
    import pickle
    import base64
    from sklearn.preprocessing import normalize
    from geopy.distance import vincenty
    from cv2 import xfeatures2d

    avgDistance = 0.0
    count = 0.0
    # for imagePath in glob.glob(arguments[6] + "/*.jpg"): # This grabs all the .jpg images in the queries directory
    #     # q_kp, q_des = compute(imagePath, sift)  # Returns keypoints & descriptors of query; imagePath is queries/XX.jpg; This needs to be pulled out or replaced with inline code or abstracted
    #     # codes, distortion = vq(q_des, codebook)  # Calculate code book membership of observation vectors vq(obs, code_book)
    #     # q_hist, q_bins = np.histogram(codes, arguments[4])  # Assigns codes from above to the K-bin histogram values
    #     q_lat = 0.0
    #     q_lon = 0.0
    #     q_fov = 0


        # # The below for loop selects the query image from the database and assigns its lat/lon/fov to variables
        # for query_data in session.execute("SELECT LAT,LON,FOV FROM QUERIES WHERE FILENAME='" + imagePath[imagePath.rfind("/") + 1:] + "'"):
        #     q_lat = query_data.lat
        #     q_lon = query_data.lon
        #     q_fov = query_data.fov
        #
        # histograms = session.execute("SELECT * FROM HISTOGRAMS") # Grabs all histograms in database

    dataset = [row for row in histograms]  # Makes a 0-by-results size list (0-by-1080 for my case)

    corpus = np.array([]).reshape([0, clusters])  # Creates an empty array with shape 0-by-K
    corpus_fn = np.array([]).reshape([0, 1])  # Creates an empty array with shape 0-by-1
    corpus = np.array([np.array(pickle.loads(base64.b64decode(row.histogram))) for row in dataset])  # Creates an array of arrays (outer array is 0-by-1079 and each element has an array of size 0-by-189 (our vocab size))

    corpus = corpus / 1.0  # convert all array entries to float
    query_histogram = query_histogram / 1.0  # Converts histogram values to float values (must be floats to normalize)

    corpus_fn = [row.filename for row in dataset]  # Populates corpus_fn with the filename of each file from dataset
    normalize(corpus, 'l2', 1, False)  # Normalize corpus histogram values
    normalize(query_histogram, 'l2', 1, False)  # Normalize query histogram values

    dot = np.dot(corpus, query_histogram)  # Computes dot product of corpus and query histograms
    ind = dot.argsort()[-set_limit:][::-1]  # Returns the sorted indices of the high-to-low cosine values
    path = []

    for index in ind:
        path.append(dataset[index][0])
    # row = int(imagePath.split('.')[0].split('/')[1])  # Filename prefix of query image (i.e. 26) without extension

    # for image in range(len(recallmatrix[0])):
    #     recallmatrix[row][image] = dot[image]

    results = zip(dot[ind], path)  # Zips together the cosine distances and indices
    results = sorted(results, reverse=True)  # Sorts the results from high to low
    return results
