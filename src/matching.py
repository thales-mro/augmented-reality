import cv2
import math
import numpy as np


class Matching:
    """
       A class used to execute the matching operations

       Methods
       -------
       match(inputPath, query, train, k)
            Execute the match operations
       _knnMatch(query, train, k)
            Execute the match operations using knn
       _hammingDistance(a, b)
            Calculate the hamming distance
       _euclideanDistance(a, b)
            Calculate the euclidian distance
       _getNeighbors(query, train, k)
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

        matches_knn = self._knnMatch(query, train, k)

        # Apply ratio test
        matches = []
        for m,n in matches_knn:
            if m.distance < 0.75*n.distance:
                matches.append(m)
                
        return matches


    def _knnMatch(self, query, train, k):
        """
        Perform the match using knn

        Keyword arguments:
        query -- the query vector
        train -- the training vector
        k -- number of clusters
        """
        
        matches = []
        
        # For each descriptor in the current
        for i in range(len(query)):
            
            # Get the k-nearest neighbors
            neighbor = self._getNeighbors(train, query[i], k)
        
            matches_knn = set()
            
            # Create the k descriptor tuple
            for n in neighbor:
                matches_knn.add(cv2.DMatch(i, n[0], n[1]))
            
            matches.append(matches_knn)
        
        return matches
    
    
    def _hammingDistance(self, a, b):
        """
        Calculates the hamming distance between two vectors

        Keyword arguments:
        a -- first vector
        b -- second vector
        """
        
        return np.sum(a != b)


    def _euclideanDistance(self, a, b):
        """
        Calculates the euclidian distance between two vectors

        Keyword arguments:
        a -- first vector
        b -- second vector
        """
        
        return np.linalg.norm(a-b)
        
        
    def _getNeighbors(self, query, train, k):
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
        for x in range(len(query)):
            distances[x] = (x, self._euclideanDistance(train, query[x]))
            
        # Sort based on distance using numpy notation
        distances = distances[distances[:,1].argsort()]
        
        neighbors = []
        
        # Group in k clusteres
        for x in range(k):
            neighbors.append(distances[x])
            
        return neighbors
    
    
