import math
import numpy as np
import random

def points_distance(x, x_next, y, y_next):
    """
    It calculates the euclidean distance between two points

    Keyword arguments:
    x -- x coordinate of first point
    x_next -- x coordinate of second point
    y -- y coordinate of first point
    y_next -- y coordinate of second point
    """
    return math.sqrt(((x_next - x)**2) + ((y_next - y)**2))

def least_squares(prev_points, curr_points):
    """
    It calculates least squares in order to find affine transform that transform a set of coordinates to another

    Keyword arguments:
    prev_points -- set of original coordinates
    curr_points -- set of transformed coordinates
    """
    n_points = prev_points.shape[0]
    x = np.zeros(shape=(2*n_points, 6), dtype=np.float32)
    for idx, _ in enumerate(prev_points):
        x[2*idx] = [prev_points[idx][0], prev_points[idx][1], 1, 0, 0, 0]
        x[2*idx + 1] = [0, 0, 0, prev_points[idx][0], prev_points[idx][1], 1]

    y = np.zeros(shape=(2*n_points, 1), dtype=np.float32)
    for idx, _ in enumerate(curr_points):
        y[2*idx][0] = curr_points[idx][0]
        y[2*idx + 1][0] = curr_points[idx][1]

    a = (np.linalg.inv((x.transpose()).dot(x))).dot(((x.transpose()).dot(y)))

    return a

class AffineTransform:
    """
    A class to calculate the Affine transform parameters

    """

    def __init__(self, success_rate, inliers_rate, threshold):
        self._success_rate = success_rate
        self._inliers_rate = inliers_rate
        self._threshold = threshold
        self._n_ransac_iterations = 0
        self._calculate_ransac_iterations()

    def _calculate_ransac_iterations(self):
        """
        It calculates the number of ransac iterations needed based
        on the probability of inliers in set and the desired success rate
        """
        n = math.log(1 - self._success_rate)/math.log(1 - (self._inliers_rate)**2)
        self._n_ransac_iterations = int(math.ceil(n))

    def set_success_rate(self, success_rate):
        """
        It updates success_rate variable value (used in RANSAC)

        Keyword arguments:
        success_rate -- new success rate (in percentage)
        """
        self._success_rate = success_rate
        self._calculate_ransac_iterations()

    def set_inliers_rate(self, inliers_rate):
        """
        It updates inliers_rate variable value (used in RANSAC)

        Keyword arguments:
        inliers_rate -- new inliers rate (in percentage)
        """
        self._inliers_rate = inliers_rate
        self._calculate_ransac_iterations()

    def set_threshold(self, threshold):
        """
        It updates threshold variable value (used in RANSAC)

        Keyword arguments:
        threshold -- new threshold
        """
        self._threshold = threshold

    def _matches_based_on_affine_matrix(self, previous_points, current_points, a):
        """
        It measures how good is the calculate affine transform matrix
        based on the threshold and candidate matches

        Keyword arguments:
        previous_points -- set of original coordinates
        current_points -- set of transformed coordinates
        """
        n_match_points = 0

        for idx, point in enumerate(previous_points):
            tf_x = a[0][0]*point[0] + a[1][0]*point[1] + a[2][0]
            tf_y = a[3][0]*point[0] + a[4][0]*point[1] + a[5][0]

            distance = points_distance(current_points[idx][0], tf_x, current_points[idx][1], tf_y)

            if distance <= self._threshold:
                n_match_points += 1

        return n_match_points

    def get_affine_transform_matrix(self, previous_f_points, current_f_points):
        """
        It runs RANSAC with least squares to find the affine transform
        matrix for the candidate matches

        Keyword arguments:
        previous_points -- set of original candidate coordinates
        current_points -- set of transformed candidate coordinates
        """
        n_points = len(current_f_points)
        point_matches = np.zeros((self._n_ransac_iterations, 3), dtype=int)
        best_match_points = 0
        best_a = []

        # starts RANSAC
        for i in range(self._n_ransac_iterations):
            # find 3-points combination which as not chosen before
            valid_eq_points = False
            while not valid_eq_points:

                idx = 0
                guess_array = np.full((3), -1)
                while idx < 3:

                    guess = random.randint(0, n_points - 1)
                    if not guess in guess_array:
                        guess_array[idx] = guess
                        idx += 1

                valid_eq_points = not (point_matches == guess_array).all(1).any()

            # save guess combination
            point_matches[i] = guess_array
            prev_frame_set = np.zeros(shape=(3, 2), dtype=np.float32)
            curr_frame_set = np.zeros(shape=(3, 2), dtype=np.float32)

            for idx, _ in enumerate(curr_frame_set):
                prev_frame_set[idx] = previous_f_points[guess_array[idx]]
                curr_frame_set[idx] = current_f_points[guess_array[idx]]

            # execute the least squares method with 3 match candidates
            a = least_squares(prev_frame_set, curr_frame_set)

            # analyze the affine transform based on the threshold and keeps the best
            n_matches = self._matches_based_on_affine_matrix(previous_f_points, current_f_points, a)
            if n_matches > best_match_points:
                best_match_points = n_matches
                best_a = a
        # print("Best number of matches:", best_match_points)
        # print("Ratio:", 100*best_match_points/n_points)
        # print(a)

        return best_a
