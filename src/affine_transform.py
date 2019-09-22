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
    It calculates least squares in order to find affine transform that transforms
    a set of coordinates to another

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

   # a = (np.linalg.inv((x.transpose()).dot(x))).dot(((x.transpose()).dot(y)))
    uu = x.copy()
    transp_times_x = (np.transpose(x)).dot(x)
    #print(uu == x)
    # #print(uu == x)
    det = np.linalg.det(transp_times_x)
    # # #print(x.shape[0])
    if det == 0:
        print("det")
        return np.zeros((6, 1))
    # singular matrix case
    # if np.linalg.matrix_rank(transp_times_x) != x.shape[0]:
    #     print("Singular matrixxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    #print("AAAA")
    #a = (np.linalg.inv((np.transpose(x)).dot(x))).dot(((np.transpose(x)).dot(y)))

    a = (np.linalg.inv(transp_times_x)).dot(((np.transpose(x)).dot(y)))


   # a = (np.linalg.inv(transp_times_x)).dot(((x.transpose()).dot(y)))
   # a = (np.linalg.inv(transp_times_x)).dot(((x.transpose()).dot(y)))
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
        unmatch_points_idx = []
        distances = []

        for idx, point in enumerate(previous_points):
            tf_x = a[0][0]*point[0] + a[1][0]*point[1] + a[2][0]
            tf_y = a[3][0]*point[0] + a[4][0]*point[1] + a[5][0]
            distance = points_distance(current_points[idx][0], tf_x, current_points[idx][1], tf_y)
            distances.append(distance)

        max_d = np.max(distances)

        normalized_distance = []
        for d in distances:
            normalized_distance.append(d/max_d)

        for idx, n_d in enumerate(normalized_distance):
            if n_d < self._threshold:
                n_match_points += 1
            else:
                unmatch_points_idx.append(idx)

        return n_match_points, unmatch_points_idx

    def get_affine_transform_matrix(self, previous_f_points, current_f_points):
        """
        It runs RANSAC with least squares to find the affine transform
        matrix for the candidate matches

        Keyword arguments:
        previous_points -- set of original candidate coordinates
        current_points -- set of transformed candidate coordinates
        """
        # n_points = len(current_f_points)
        point_matches = np.zeros((self._n_ransac_iterations, 3, 2), dtype=np.float32)
        best_number_matches = 0
        best_a = []
        previous_f_ransac_set = np.copy(previous_f_points)
        current_f_ransac_set = np.copy(current_f_points)

        # starts RANSAC
        for i in range(self._n_ransac_iterations):
            n_points = len(previous_f_ransac_set)
            # find 3-points combination which was not chosen before
            valid_eq_points = False
            while not valid_eq_points:

                idx = 0
                pf_guess_array = np.full((3, 2), -1)
                cf_guess_array = np.full((3, 2), -1)
                while idx < 3:

                    guess = random.randint(0, n_points - 1)
                    #if not guess in guess_array:
                    if not (pf_guess_array == previous_f_ransac_set[guess]).all(1).any():
                        pf_guess_array[idx][0] = previous_f_ransac_set[guess][0]
                        pf_guess_array[idx][1] = previous_f_ransac_set[guess][1]
                        cf_guess_array[idx][0] = current_f_ransac_set[guess][0]
                        cf_guess_array[idx][1] = current_f_ransac_set[guess][1]
                        idx += 1

                valid_eq_points = not np.any(np.all(pf_guess_array == point_matches, axis=(1, 2)))
            point_matches[i] = pf_guess_array

            # solve least squares for the guess combination
            a = least_squares(pf_guess_array, cf_guess_array)
            n_matches, unmatch_points_idx = self._matches_based_on_affine_matrix(
                previous_f_points, current_f_points, a)

            # if the number of matches obtained this time is greater than previous iterations, 
            # we have a new best affine transform matrix
            if n_matches > best_number_matches:
                best_number_matches = n_matches
                best_a = a
                match_rate = best_number_matches/len(previous_f_points)
                # if it achieves considerably good success rate, we can remove outlier candidates
                if match_rate > self._success_rate/2:
                    idx_to_be_deleted = []
                    for idx in unmatch_points_idx:
                        m = (previous_f_ransac_set == previous_f_points[idx]).all(1)
                        idx = np.where(m == True)
                        for el in idx[0]:
                            idx_to_be_deleted.append(el)
                    previous_f_ransac_set = np.delete(
                        previous_f_ransac_set, idx_to_be_deleted, axis=0)
                    current_f_ransac_set = np.delete(
                        current_f_ransac_set, idx_to_be_deleted, axis=0)

        # solve least squares one last time with all the survivor candidates
        a_final = least_squares(previous_f_ransac_set, current_f_ransac_set)
        n_matches, unmatch_points_idx = self._matches_based_on_affine_matrix(
            previous_f_points, current_f_points, a_final)
        if n_matches > best_number_matches:
            best_a = a_final
            best_number_matches = n_matches

        return np.reshape(best_a, (2, 3))
