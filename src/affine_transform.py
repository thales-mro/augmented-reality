import cv2
import math
import numpy as np
import random

def point_distance(x, x_next, y, y_next):
    return math.sqrt(((x_next - x)**2) + ((y_next - y)**2))

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
        n = math.log(1 - self._success_rate)/math.log(1 - (self._inliers_rate)**2)
        self._n_ransac_iterations = int(math.ceil(n))

    def set_success_rate(self, sr):
        self._success_rate = sr
        self._calculate_ransac_iterations()

    def set_inliers_rate(self, ir):
        self._inliers_rate = ir
        self._calculate_ransac_iterations()

    def set_threshold(self, threshold):
        self._threshold = threshold

    def _matches_based_on_affine_matrix(self, previous_points, current_points, a):
        #print(n_points)
        n_match_points = 0
        #print(a)
        for idx, point in enumerate(previous_points):
            #print(point)
            tf_x = a[0][0]*point[0] + a[1][0]*point[1] + a[2][0]
            # print(tf_x)
            tf_y = a[3][0]*point[0] + a[4][0]*point[1] + a[5][0]
            # print(tf_y)
            d = point_distance(current_points[idx][0], tf_x, current_points[idx][1], tf_y)
            #print("Calculated Distance:", d)
            if d <= self._threshold:
                n_match_points += 1
        #print("Number of match points: ", n_match_points)
        return n_match_points

    def _least_squares(self, points_idxs, prev_points, curr_points):
        
        x = np.array([
            [prev_points[points_idxs[0]][0], prev_points[points_idxs[0]][1], 1, 0, 0, 0],
            [0, 0, 0, prev_points[points_idxs[0]][0], prev_points[points_idxs[0]][1], 1],
            [prev_points[points_idxs[1]][0], prev_points[points_idxs[1]][1], 1, 0, 0, 0],
            [0, 0, 0, prev_points[points_idxs[1]][0], prev_points[points_idxs[1]][1], 1],
            [prev_points[points_idxs[2]][0], prev_points[points_idxs[2]][1], 1, 0, 0, 0],
            [0, 0, 0, prev_points[points_idxs[2]][0], prev_points[points_idxs[2]][1], 1]
        ])

        y = np.array([
            [curr_points[points_idxs[0]][0]],
            [curr_points[points_idxs[0]][1]],
            [curr_points[points_idxs[1]][0]],
            [curr_points[points_idxs[1]][1]],
            [curr_points[points_idxs[2]][0]],
            [curr_points[points_idxs[2]][1]]
        ])

        # print(x)
        # print(y)
        b = (x.transpose()).dot(y)
        # print("b shape:", b)
        # print("x transp x", (x.transpose()).dot(x))
        a = (np.linalg.inv((x.transpose()).dot(x))).dot(((x.transpose()).dot(y)))

        # print("a:")
        # print(a)

        return a

    def ransac(self, previous_f_points, current_f_points):

        n_points = len(current_f_points)
        point_matches = np.zeros((self._n_ransac_iterations, 3), dtype=int)
        best_match_points = 0
        best_a = []
        for i in range(self._n_ransac_iterations):
            # find 3-points combination
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

            point_matches[i] = guess_array

            a = self._least_squares(guess_array, previous_f_points, current_f_points)

            n_matches = self._matches_based_on_affine_matrix(previous_f_points, current_f_points, a)
            if n_matches > best_match_points:
                best_match_points = n_matches
                best_a = a
        # print("Best number of matches:", best_match_points)
        # print("Ratio:", 100*best_match_points/n_points)
        # print(a)

        return best_a
