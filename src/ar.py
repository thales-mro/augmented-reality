import cv2
import numpy as np
import affine_transform as at
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
            
        # Create the target mask for extracting the image
        self.initial_target_mask = np.ones_like(self.target)*255


    def _warpAffine(self, source, a, shape):
        """
        It warps the image using an affine matrix
        
        Keyword arguments:
        source -- source image
        a -- affine matrix
        shape -- new image shape
        """
        
        # Define the new shape
        new_shape = (shape[1], shape[0], 3)
        
        # Create the empty new image
        new_image = np.zeros(new_shape, dtype=np.uint8)
        
        for x in range(source.shape[1]):
            for y in range(source.shape[0]):
                
                # Build the point
                p = np.array([x, y, 1])

                # Apply the affine transformation
                x_1, y_1, _ = np.matmul(a, p)
                
                # Check if the transform is inside the frame
                if x_1 < 0:
                    x_1 = 0
                    
                if y_1 < 0:
                    y_1 = 0
                
                if x_1 >= new_image.shape[1]:
                    x_1 = new_image.shape[1]-2
                    
                if y_1 >= new_image.shape[0]:
                    y_1 = new_image.shape[0]-2
                
                # Index the new image
                new_image[int(y_1), int(x_1), :] = source[y, x, :]
        
        return new_image
    

    def execute(self, input_path, output_path, operation=2, compare_frame_by_frame=True, start_frame=-1, max_frames=-1, print_frames=False, min_matches=5):
        """
        It executes the ar for a video file

        Keyword arguments:
        input_path -- the input video path
        output_path -- the output video path
        operation -- operation to apply on the frame
        compare_frame_by_frame -- compare fram by frame
        start_frame -- starting video frame
        max_frames -- maximum number of frames to process
        print_frames -- flag to print the current frame
        min_matches -- minimum number os matches to find the affine matrix
        """

        # The accumulative affine matrix
        a_all = None

        # Open the video
        video_capture = cv2.VideoCapture(input_path)
        
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

        # Define frame 0 as the target image
        previous_frame = self.target

        # Read frame 1
        success, current_frame = video_capture.read()

        # Compute keypoints and descriptors of the previous frame
        keypoints_previous_frame, descriptors_previous_frame = self.sift.detectAndCompute(
            previous_frame, None)

        # Create the affine transform
        transform = at.AffineTransform(0.99, 0.125, 0.1)
        
        index_2 = 0
        while success and index_2 < start_frame:
            success, current_frame = video_capture.read()
            index_2 += 1 
        
        index = 0
        
        # For each frame
        while success:
            
            try:

                print(f"Processing Frame {index}")

                # Compute keypoints and descriptors of the current frame
                keypoints_current_frame, descriptors_current_frame = self.sift.detectAndCompute(
                    current_frame, None)

                # Find the matches between the previous frame and the current frame
                matches = self.matcher.match(descriptors_previous_frame, descriptors_current_frame, k=2)
                
                # Sort the matches based on the distance
                matches.sort(key=lambda x: x.distance)

                # If the matches are greate than the threshold
                if len(matches) > min_matches:

                    # Get the keypoints for each match
                    previous_frame_points = np.float32(
                        [keypoints_previous_frame[m.queryIdx].pt for m in matches])
                    current_frame_points = np.float32(
                        [keypoints_current_frame[m.trainIdx].pt for m in matches])

                    # Find the affine matrix                     
                    a = transform.get_affine_transform_matrix(previous_frame_points, current_frame_points)

                    # Set the accumulative transformations
                    if a_all is None or compare_frame_by_frame == False:
                        a_all = np.vstack((a, [0,0,1]))
                    elif np.sum(a) > 0:
                        
                        # Convert to homogeneous coordinates
                        i = np.vstack((a, [0,0,1]))
                        a_all = np.matmul(a_all, i)

                    if operation == 0:
                        
                        # Convert to opencv match class
                        matches_opencv = [cv2.DMatch(i.queryIdx, i.trainIdx, i.distance) for i in matches]
                        
                        # Draw the 20 matches
                        output_frame = cv2.drawMatchesKnn(
                            previous_frame.copy(), keypoints_previous_frame,
                            current_frame.copy(), keypoints_current_frame,
                            [matches_opencv], None, flags=2)

                    elif operation == 1:
                        img_aux = np.zeros_like(current_frame)
                        cv2.drawKeypoints(current_frame, keypoints_current_frame, img_aux, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        current_frame = img_aux
                    elif operation == 2:
                                            
                        # Warp the source image
                        source = self._warpAffine(self.source, a_all, (current_frame.shape[1], current_frame.shape[0]))
                        
                        # Warp the mask
                        target_mask = self._warpAffine(self.initial_target_mask, a_all, (current_frame.shape[1], current_frame.shape[0]))

                        # Convert it to gray scale
                        target_mask_gray = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)

                        # Define the threshold to separate foreground from background
                        _, mask = cv2.threshold(target_mask_gray, 150, 255, cv2.THRESH_BINARY_INV)

                        # Get the inverted mask
                        mask_inv = cv2.bitwise_not(mask)

                        # Get the background image
                        background = cv2.bitwise_and(current_frame, current_frame, mask=mask)

                        # Get the foreground image
                        foregound = cv2.bitwise_and(source, source, mask=mask_inv)

                        # Add both images
                        output_frame = cv2.add(background, foregound)

                    
                    # Write each frame to a new video
                    video_out.write(output_frame)
                    
                    # Save the current frame as an image
                    if print_frames:
                        cv2.imwrite(f"output/frame-{index}.jpg", output_frame)

                    index += 1

                    if max_frames > 0 and index > max_frames:
                        break

                # Set the previous frame
                previous_frame = current_frame

                if compare_frame_by_frame:
                    # Set the previous keypoints and descriptors
                    keypoints_previous_frame = keypoints_current_frame
                    descriptors_previous_frame = descriptors_current_frame

                # Read the next frame
                success, current_frame = video_capture.read()

                #transform.set_inliers_rate(0.75)
                
            except:
                continue
            

        # Release the video file
        video_out.release()
        
        
