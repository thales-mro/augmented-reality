import cv2
import math
import numpy as np

def hamming_distance(a, b):
    """
    Calculates the hamming distance between two vectors

    Keyword arguments:
    a -- first vector
    b -- second vector
    """

    return np.sum(a != b)


def euclidean_distance(a, b):
    """
    Calculates the euclidian distance between two vectors

    Keyword arguments:
    a -- first vector
    b -- second vector
    """

    return np.linalg.norm(a-b)


class Matching:
    """
       A class used to execute the matching operations

       Methods
       -------
       match(inputPath, query, train, k)
            Execute the match operations
       _knn_match(query, train, k)
            Execute the match operations using knn
       _get_neighbors(query, train, k)
            Get the neighbors
       """

    def __init__(self):
        pass


    def match(self, query, train, k=2):
        """
        Match the query with the train using k-nearest neighbors

        Keyword arguments:
        query -- the query vector
        train -- the training vector
        k -- number of clusters
        """

        matches_knn = self._knn_match(query, train, k)

        # Apply ratio test
        matches = []
        for m, n in matches_knn:
            if m.distance < 0.75*n.distance:
                matches.append(m)

        return matches


    def _knn_match(self, query, train, k):
        """
        Perform the match using knn

        Keyword arguments:
        query -- the query vector
        train -- the training vector
        k -- number of clusters
        """

        matches = []

        # For each descriptor in the current
        for i, _ in enumerate(query):

            # Get the k-nearest neighbors
            neighbor = self._get_neighbors(train, query[i], k)

            matches_knn = set()

            # Create the k descriptor tuple
            for n in neighbor:
                matches_knn.add(cv2.DMatch(i, n[0], n[1]))

            matches.append(matches_knn)

        return matches

    def _get_neighbors(self, query, train, k):
        """
        Get the k-nearest neighbors

        Keyword arguments:
        query -- the query vector
        train -- the training vector
        k -- number of clusters
        """

        # Create the matrix of distances
        distances = np.zeros((np.shape(query)[0], 2), dtype=np.int)

        # For each pair, calculates the distance
        for x, _ in enumerate(query):
            distances[x] = (x, euclidean_distance(train, query[x]))

        # Sort based on distance using numpy notation
        distances = distances[distances[:, 1].argsort()]

        neighbors = []

        # Group in k clusteres
        for x in range(k):
            neighbors.append(distances[x])

        return neighbors
