import cv2
import numpy as np


class AR:
    """
       A class used to execute the Augmented Reality operations

       Methods
       -------
        execute_video(inputPath, outputPath, operation)
            Execeute the ar operation on a video
       """

    def __init__(self, targetPath, sourcePath):
        
        # Create ORB detector
        self.orb = cv2.ORB_create()
        
        # Create BFMatcher
        self.bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Read the target image
        self.target = cv2.imread(targetPath)
        
        # Get the target dimensions
        self.target_w, self.target_h, _ = self.target.shape
        
        # Read the source image
        self.source =  cv2.imread(sourcePath)
        
        # Get the source dimensions
        self.source_w, self.source_h, _ = self.source.shape
                
        # Resize the source image to the target's size
        self.source = cv2.resize(self.source, (self.target_h, self.target_w), interpolation = cv2.INTER_AREA)
        
        # Find the rectangle around the target image
        self.initial_target_edges = np.float32([[0, 0], [0, self.target_w-1], [self.target_h-1, self.target_w-1], [self.target_h-1, 0]]).reshape(-1, 1, 2)

      
    def execute_video(self, inputPath, outputPath, operation, min_matches=5):
        """
        It executes the ar for a video file

        Keyword arguments:
        inputPath -- the input video path
        outputPath -- the output video path
        operation -- operation to apply o nthe frame
        min_matches -- minimum number os matches to find the homography matrix
        """
        
        processedFrames = []
        
        # The accumulative homography matrix
        H_all = None
                
        # Open the video
        video_capture = cv2.VideoCapture(inputPath)
        
        # Define the frame 0 as the target image
        previous_frame = self.target
        
        # Read the frame 1
        success, current_frame = video_capture.read()
        
        # Compute keypoints and descriptors of the previous frame
        keypoints_previous_frame, descriptors_previous_frame = self.orb.detectAndCompute(previous_frame, None)
        
        # For each frame
        while success:

            # Compute keypoints and descriptors of the current frame
            keypoints_current_frame, descriptors_current_frame = self.orb.detectAndCompute(current_frame, None)

            # Find the matches between the previous frame and the current frame
            matches = self.bfMatcher.match(descriptors_previous_frame, descriptors_current_frame)

            # Sort based on distance
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > min_matches:
                
                # Get the keypoints for each match
                previous_frame_points = np.float32([keypoints_previous_frame[m.queryIdx].pt for m in matches])
                current_frame_points = np.float32([keypoints_current_frame[m.trainIdx].pt for m in matches])
                
                # Find the homography matrix
                H, _ = cv2.findHomography(previous_frame_points.reshape(-1, 1, 2), current_frame_points.reshape(-1, 1, 2), cv2.RANSAC, 3.0)
                
                # Set the accumulative transformations
                H_all = H if H_all is None else np.matmul(H, H_all)
                
                # Project the edges on the image using the homography matrix
                target_edges = cv2.perspectiveTransform(self.initial_target_edges, H_all)
        
                if operation == 0:
                    # Draw the 20 matches
                    output_frame = cv2.drawMatches(previous_frame.copy(), keypoints_previous_frame, current_frame.copy(), keypoints_current_frame, matches[:20], 0, flags=2)
                elif operation == 1: 
                    # Draw the rectangle around the target image
                    output_frame = cv2.polylines(current_frame.copy(), [np.int32(target_edges)], True, 255, 3, cv2.LINE_AA) 
                
                elif operation == 2 and H is not None:

                    # Warp the source image
                    source = cv2.warpPerspective(self.source.copy(), H_all, (current_frame.shape[1], current_frame.shape[0]))

                    # Create a convex cover
                    filler = cv2.convexHull(np.int32(target_edges))

                    # Fill it with white color
                    filledSource = cv2.fillConvexPoly(source.copy(), filler, [255, 255, 255])
                    
                    # Convert it to gray scale
                    filledSourceGray = cv2.cvtColor(filledSource, cv2.COLOR_BGR2GRAY)
                    
                    # Define the threshold to separate foreground from background
                    ret, mask = cv2.threshold(filledSourceGray, 150, 255, cv2.THRESH_BINARY_INV)
                    
                    # Get the inverted mask
                    mask_inv = cv2.bitwise_not(mask)
                    
                    # Get the background image
                    background = cv2.bitwise_and(current_frame, current_frame, mask = mask)
                    
                    # Get the foreground image
                    foregound = cv2.bitwise_and(source, source, mask = mask_inv)
                    
                    # Add both images
                    output_frame = cv2.add(background, foregound)

                    processedFrames.append(output_frame)
            
                
            # Set the previous frame
            previous_frame = current_frame
            
            # Set the previous keypoints and descriptors
            keypoints_previous_frame = keypoints_current_frame
            descriptors_previous_frame = descriptors_current_frame
            
            # Read the next frame
            success, current_frame = video_capture.read()
            
        
        # Get the proper size
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # Get the proper fps
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        # Write the dvix header
        fourcc = cv2.VideoWriter_fourcc('M','P','E','G') # DVIX
                
        # Create the video out writter
        video_out = cv2.VideoWriter("output/o-0.mp4",  fourcc, fps, size)
        
        # Write each frame to a new video
        for i in processedFrames:
            video_out.write(i)
            
        # Release the video file
        video_out.release() 


