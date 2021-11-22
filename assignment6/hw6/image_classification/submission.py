import collections
import numpy as np
############################################################
# Problem 4.1

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    centroids = np.random.randn(patches.shape[0], k)

    numPatches = patches.shape[1]

    for i in range(maxIter):
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
        # Assignment step:
        z = np.zeros(numPatches)

        # Assignment
        for patch_i in range(numPatches):
            dist_to_centroids = np.zeros((k, ))
            for centroid_k in range(k):
                dist_to_centroids[centroid_k] = np.linalg.norm(patches.T[patch_i] - centroids.T[centroid_k])
            z[patch_i] = np.argmin(dist_to_centroids)


        # Update step:
        centroids_t = centroids.T       # Easier to index
        patches_t = patches.T
        for j in range(k):
            indices = np.where(z == j)[0]   # index to find points in patch with same centroid
            patch_same_centroid = patches_t[indices]
            centroids_t[j] = np.mean(patch_same_centroid, axis=0)
        centroids = centroids_t.T

        # raise Exception("Not yet implemented")
        # END_YOUR_CODE

    return centroids

############################################################
# Problem 4.2

def extractFeatures(patches, centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array of size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches, k))

    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    for patch_i in range(numPatches):   # loop through cols
        newFeature = np.zeros(k)
        curr_patch = patches.T[patch_i]
        for j in range(k):
            total_dist = 0
            centroid_j_pos = centroids.T[j]
            for centroid_K in range(k):
                centroid_k_pos = centroids.T[centroid_K]
                total_dist += np.linalg.norm(curr_patch - centroid_k_pos)
            avg_dist = total_dist / k
            a_ijk = avg_dist - np.linalg.norm(curr_patch - centroid_j_pos)
            newFeature[j] = max(a_ijk, 0)
        features[patch_i] = newFeature
    # raise Exception("Not yet implemented")
    # END_YOUR_CODE
    return features

############################################################
# Problem 4.3.1

import math
def logisticGradient(theta, featureVector, y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """

    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    yy = 2*y - 1
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        up = -featureVector[i] * yy * np.exp(-np.dot(theta, featureVector*yy))
        bottom = 1 + np.exp(-theta.dot(featureVector) * yy)
        grad[i] = up / bottom
    return grad
    # raise Exception("Not yet implemented.")
    # END_YOUR_CODE

############################################################
# Problem 4.3.2
    
def hingeLossGradient(theta, featureVector, y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    yy = 2*y - 1
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        if 1 - np.dot(theta, featureVector*yy) > 0:
            grad[i] = -featureVector[i] * yy
        else:
            grad[i] = 0.0
    # raise Exception("Not yet implemented.")
    return grad
    # END_YOUR_CODE

