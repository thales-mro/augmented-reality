import cv2
import numpy as np
import random

class AffineTransform:
    """
    A class to calculate the Affine transform parameters

    """

    def __init__(self):
        pass

    def _calculate_ransac_iterations(self, n_points):
        return 1

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

        print(x)
        print(y)

        return []

    def ransac(self, previous_f_points, current_f_points):

        n = self._calculate_ransac_iterations(len(previous_f_points))
        n_points = len(current_f_points)
        print(n_points)

        point_matches = np.zeros((n, 3), dtype=int)
        point_matches[0] = [1, 2, 3]

        print()

        for i in range(n):
            print("RANSAC iteration ", i)

            # find 3-points combination
            valid_eq_points = False
            while not valid_eq_points:
                
                valid_guess = False
                guess_array = np.full((1, 3), -1)
                while not valid_guess:
                    guess = random.randint(0, n_points - 1)
                guess = np.array([random.randint(0, n_points - 1), random.randint(0, n_points - 1), random.randint(0, n_points - 1)])
                valid_eq_points = not (point_matches == guess).all(1).any()
            point_matches[i] = guess
            self._least_squares(guess, previous_f_points, current_f_points)

        print(point_matches)

        return 0