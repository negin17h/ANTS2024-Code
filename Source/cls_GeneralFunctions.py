import numpy as np
import pandas as pd
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing

class GeneralFunctions():

    def generateRandom(seed = 720, elemNo=2, applySeed=False):
        if (applySeed):
            np.random.seed(seed)
        return np.random.random(elemNo)

    def generateRandomN(seed = 720, start = 0, end = 1, elemNo=2, applySeed=False):
        if (applySeed):
            np.random.seed(seed)
        rng = np.random.default_rng()
        return rng.normal(start, end,size=elemNo)

    def generateRandomUniform(lbound=0, ubound=1, elemNo=2, seed = 720, applySeed=False):
        if (applySeed):
            np.random.seed(seed)
        return np.random.uniform(size=elemNo, low=lbound, high=ubound)

    def getEuclideanDistanceBetweenTwoPoint(point_a,point_b):
        #return np.linalg.norm(a-b)
        return np.sqrt(np.sum((point_a - point_b) ** 2))

    def getManhattanDistance(point_a, point_b):
        """
        Calculate the L1 norm (Manhattan distance or Taxicab Norm) between two points.

        Parameters:
        - point_a: A list or tuple of coordinates for the first point.
        - point_b: A list or tuple of coordinates for the second point.

        Returns:
        - The Manhattan distance between point_a and point_b.
        """
        if len(point_a) != len(point_b):
            raise ValueError("Points must have the same dimension")

        distance = sum(abs(a - b) for a, b in zip(point_a, point_b))
        return distance

    def getReducedDimensionByPCA(vector, toDimension = 2):
        pca = PCA()
        pca.n_components = toDimension
        pcaData = pca.fit_transform(vector)
        return pcaData

    def getReducedDimensionByPCA_OriginalCoordinateSystem(vector, toDimension=2):
        # Fit PCA to your data
        pca = PCA(n_components=toDimension)
        pca.fit(vector)
        # Transform your data to the new space defined by the principal components
        pcaData = pca.transform(vector)
        # Inverse transform the transformed data back to the original space
        reconstructed_data = pca.inverse_transform(pcaData)
        # Now, perform PCA again to reduce the 10-dimensional reconstructed data to 2 dimensions
        pca_original_space = PCA(n_components=toDimension)
        pca_original_space.fit(reconstructed_data)
        final_transformed_data = pca_original_space.transform(reconstructed_data)

        return final_transformed_data