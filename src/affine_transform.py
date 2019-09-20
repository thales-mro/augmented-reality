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
        unmatch_points_idx = []
        distances = []

        for idx, point in enumerate(previous_points):
            tf_x = a[0][0]*point[0] + a[1][0]*point[1] + a[2][0]
            tf_y = a[3][0]*point[0] + a[4][0]*point[1] + a[5][0]

            distance = points_distance(current_points[idx][0], tf_x, current_points[idx][1], tf_y)
            #print("Distance:", distance)
            distances.append(distance)

            '''if distance <= self._threshold:
                n_match_points += 1
            else:
                unmatch_points_idx.append(idx)'''

        max_d = np.max(distances)

        normalized_distance = []
        for d in distances:
            normalized_distance.append(d/max_d)

        for idx, nd in enumerate(normalized_distance):
            if nd < self._threshold:
                n_match_points += 1
            else:
                unmatch_points_idx.append(idx)
        print("number of matches:", n_match_points)
        #print("Normalized distances:", ])
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
        best_match_points = 0
        best_a = []
        best_guess_combination = []
        previous_f_ransac_set = np.copy(previous_f_points)
        current_f_ransac_set = np.copy(current_f_points)

        # starts RANSAC
        for i in range(self._n_ransac_iterations):
            n_points= len(previous_f_ransac_set)
            # find 3-points combination which as not chosen before
            valid_eq_points = False
            while not valid_eq_points:

                idx = 0
                pf_guess_array = np.full((3, 2), -1)
                cf_guess_array = np.full((3, 2), -1)
                while idx < 3:

                    guess = random.randint(0, n_points - 1)
                    if not (pf_guess_array == previous_f_ransac_set[guess]).all(1).any():
                    #if not guess in guess_array:
                        #print("Entrou aqui?", guess)
                        pf_guess_array[idx][0] = previous_f_ransac_set[guess][0]
                        pf_guess_array[idx][1] = previous_f_ransac_set[guess][1]
                        cf_guess_array[idx][0] = current_f_ransac_set[guess][0]
                        cf_guess_array[idx][1] = current_f_ransac_set[guess][1]
                        #guess_array[idx] = previous_f_ransac_set[idx]
                        idx += 1

                #print("Saiu do loop:", pf_guess_array, np.any(np.all(pf_guess_array == point_matches, axis=(1, 2))))
                valid_eq_points = not np.any(np.all(pf_guess_array == point_matches, axis=(1, 2)))
                #valid_eq_points = not (point_matches == guess_array).all(1).any()
            point_matches[i] = pf_guess_array

            a = least_squares(pf_guess_array, cf_guess_array)
            n_matches, unmatch_points_idx = self._matches_based_on_affine_matrix(previous_f_points, current_f_points, a)
            #print("Number of matches:", n_matches)
            if n_matches > best_match_points:
                best_match_points = n_matches
                best_guess_combination = pf_guess_array
                match_rate = best_match_points/len(previous_f_points)
                if match_rate > self._success_rate/2:
                    idx_to_be_deleted = []
                    for idx in unmatch_points_idx:
                        m = (previous_f_ransac_set == previous_f_points[idx]).all(1)
                        idx = np.where(m == True)
                        for el in idx[0]:
                            idx_to_be_deleted.append(el)
                    previous_f_ransac_set = np.delete(previous_f_ransac_set, idx_to_be_deleted, axis=0)
                    current_f_ransac_set = np.delete(current_f_ransac_set, idx_to_be_deleted, axis=0)
                best_a = a

        a_final = least_squares(previous_f_ransac_set, current_f_ransac_set)
        #print("a_final", a_final)
        n_matches, unmatch_points_idx = self._matches_based_on_affine_matrix(previous_f_points, current_f_points, a_final)
        print("Number of matches in final iteration:", n_matches)
        if n_matches > best_match_points:
            best_a = a_final

        print("At the end:", len(current_f_ransac_set))
        return best_a
