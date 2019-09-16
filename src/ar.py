import cv2
import numpy as np
from matching import Matching


class AR:
    """
       A class used to execute the Augmented Reality operations

       Methods
       -------
        execute_video(input_path, output_path, operation)
            Execeute the ar operation on a video
       """

    def __init__(self, targetPath, sourcePath):

        # Create SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()

        # Create the Matching Algorithm
        self.matcher = Matching()

        # Read the target image
        self.target = cv2.imread(targetPath)

        # Get the target dimensions
        self.target_w, self.target_h, _ = self.target.shape

        # Read the source image
        self.source = cv2.imread(sourcePath)

        # Get the source dimensions
        self.source_w, self.source_h, _ = self.source.shape

        # Resize the source image to the target's size
        self.source = cv2.resize(self.source, (
            self.target_h, self.target_w), interpolation = cv2.INTER_AREA)

        # Find the rectangle around the target image
        self.initial_target_edges = np.float32(
            [[0, 0], [0, self.target_w-1], [self.target_h-1, self.target_w-1],
             [self.target_h-1, 0]]).reshape(-1, 1, 2)

    def execute_video(self, input_path, output_path, operation, max_frames=-1, save_frame=False, min_matches=5):
        """
        It executes the ar for a video file

        Keyword arguments:
        input_path -- the input video path
        output_path -- the output video path
        operation -- operation to apply o nthe frame
        max_frames -- maximum number of frames to process
        save_frame -- Flag to save the frame as an image
        min_matches -- minimum number os matches to find the homography matrix
        """

        processed_frames = []

        # The accumulative homography matrix
        h_all = None

        # Open the video
        video_capture = cv2.VideoCapture(input_path)

        # Define the frame 0 as the target image
        previous_frame = self.target

        # Read the frame 1
        success, current_frame = video_capture.read()

        # Compute keypoints and descriptors of the previous frame
        keypoints_previous_frame, descriptors_previous_frame = self.sift.detectAndCompute(
            previous_frame, None)

        index = 0

        # For each frame
        while success:

            print(f"Processing Frame {index}")

            # Compute keypoints and descriptors of the current frame
            keypoints_current_frame, descriptors_current_frame = self.sift.detectAndCompute(
                current_frame, None)

            # Find the matches between the previous frame and the current frame
            matches = self.matcher.match(descriptors_previous_frame, descriptors_current_frame, k=2)

            if len(matches) > min_matches:

                # Get the keypoints for each match
                previous_frame_points = np.float32(
                    [keypoints_previous_frame[m.queryIdx].pt for m in matches])
                current_frame_points = np.float32(
                    [keypoints_current_frame[m.trainIdx].pt for m in matches])

                # Find the homography matrix
                h, _ = cv2.findHomography(
                    previous_frame_points.reshape(-1, 1, 2),
                    current_frame_points.reshape(-1, 1, 2), cv2.RANSAC, 5.0)

                # Set the accumulative transformations
                h_all = h if h_all is None else np.matmul(h, h_all)

                # Project the edges on the image using the homography matrix
                target_edges = cv2.perspectiveTransform(self.initial_target_edges, h_all)

                if operation == 0:
                    
                    # Convert to opencv match class
                    matches_opencv = [cv2.DMatch(i.queryIdx, i.trainIdx, i.distance) for i in matches]
                    
                    # Draw the 20 matches
                    output_frame = cv2.drawMatchesKnn(
                        previous_frame.copy(), keypoints_previous_frame,
                        current_frame.copy(), keypoints_current_frame,
                        [matches_opencv], None, flags=2)
                elif operation == 1:
                    # Draw the rectangle around the target image
                    output_frame = cv2.polylines(
                        current_frame.copy(), [np.int32(target_edges)], True, 255, 3, cv2.LINE_AA)

                elif operation == 2 and h is not None:

                    # Warp the source image
                    source = cv2.warpPerspective(
                        self.source.copy(), h_all, (current_frame.shape[1], current_frame.shape[0]))

                    # Create a convex cover
                    filler = cv2.convexHull(np.int32(target_edges))

                    # Fill it with white color
                    filled_source = cv2.fillConvexPoly(source.copy(), filler, [255, 255, 255])

                    # Convert it to gray scale
                    filled_source_gray = cv2.cvtColor(filled_source, cv2.COLOR_BGR2GRAY)

                    # Define the threshold to separate foreground from background
                    _, mask = cv2.threshold(filled_source_gray, 150, 255, cv2.THRESH_BINARY_INV)

                    # Get the inverted mask
                    mask_inv = cv2.bitwise_not(mask)

                    # Get the background image
                    background = cv2.bitwise_and(current_frame, current_frame, mask=mask)

                    # Get the foreground image
                    foregound = cv2.bitwise_and(source, source, mask=mask_inv)

                    # Add both images
                    output_frame = cv2.add(background, foregound)


                processed_frames.append(output_frame)
                
                # Save the frame as an image
                if save_frame:
                    cv2.imwrite(f"output/frame-{index}.jpg", output_frame)

                index += 1

                if max_frames > 0 and index > max_frames:
                    break

            # Set the previous frame
            previous_frame = current_frame

            # Set the previous keypoints and descriptors
            keypoints_previous_frame = keypoints_current_frame
            descriptors_previous_frame = descriptors_current_frame

            # Read the next frame
            success, current_frame = video_capture.read()

        # Get the proper size
        size = (int(
            video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Get the proper fps
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        # Write the dvix header
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G') # DVIX

        # Create the video out writter
        video_out = cv2.VideoWriter(output_path, fourcc, fps, size)

        # Write each frame to a new video
        for i in processed_frames:
            video_out.write(i)

        # Release the video file
        video_out.release()
