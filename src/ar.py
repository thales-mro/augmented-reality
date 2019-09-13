import cv2
import numpy as np


class AR:
    """
       A class used to execute the Augmented Reality operations

       Methods
       -------

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
        
        # Compute keypoints and descriptors of the target image
        self.keypoints_target, self.descriptors_target = self.orb.detectAndCompute(self.target, None)
        
        # Resize the source image to the target's size
        self.source = cv2.resize(self.source, (self.target_h, self.target_w), interpolation = cv2.INTER_AREA)

      
    def execute_video(self, videoPath):
        """
        It execute the ar for a video file

        Keyword arguments:
        videoPath -- the video path
        """
        
        processedFrames = []
                
        # Open the video
        video_capture = cv2.VideoCapture(videoPath)
        
        # Read the first frame
        success, image = video_capture.read()
        
        # For each frame
        while success:
            
            frame = self._generate_frame(image, operation=2)
                        
            # Add the processed frame to the list
            if frame is not None:
        
                processedFrames.append(frame)
                
                cv2.imshow('current_frame', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            # Read the next frame
            success, image = video_capture.read()

        # Get the proper size
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # Get the proper fps
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        # Write the dvix header
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # DVIX
                
        # Create the video out writter
        video_out = cv2.VideoWriter("output/o-0.mp4",  fourcc, fps, size)
        
        # Write each frame to a new video
        for i in processedFrames:
            video_out.write(i)
            
        # Release the video file
        video_out.release() 
        
        
    def _generate_frame(self, frame, operation):
        """
        It generates a newframe based on source and target images

        Keyword arguments:
        """
        
        # Compute keypoints and descriptors of the target frame
        keypoints_frame, descriptors_frame = self.orb.detectAndCompute(frame, None)

        # Find the matches between the target and the current frame
        matches = self.bfMatcher.match(self.descriptors_target, descriptors_frame)

        # Sort based on distance
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:
            
            # Get the keypoints for each match
            target_points = np.float32([self.keypoints_target[m.queryIdx].pt for m in matches])
            frame_points = np.float32([keypoints_frame[m.trainIdx].pt for m in matches])
            
            # Find the homography matrix
            H, _ = cv2.findHomography(target_points.reshape(-1, 1, 2), frame_points.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
            
             # Find the rectangle around the target image
            target_edges = np.float32([[0, 0], [0, self.target_w-1], [self.target_h-1, self.target_w-1], [self.target_h-1, 0]]).reshape(-1, 1, 2)
            
            # Project the edges on the image using the homography matrix
            target_edges = np.int32(cv2.perspectiveTransform(target_edges, H))
        
            if operation == 0:
                # Draw the 20 matches
                return cv2.drawMatches(self.target, self.keypoints_target, frame, keypoints_frame, matches[:20], 0, flags=2)
            elif operation == 1: 
                # Draw the rectangle around the target image
                return cv2.polylines(frame, [target_edges], True, 255, 3, cv2.LINE_AA)  
            
            elif operation == 2 and H is not None:

                # Warp the source image
                s = cv2.warpPerspective(self.source, H, (frame.shape[1], frame.shape[0]))

                # Create a convex cover
                filler = cv2.convexHull(target_edges)

                # Fill it with white color
                filledSource = cv2.fillConvexPoly(s.copy(), filler, [255, 255, 255])
                
                # Convert it to gray scale
                filledSourceGray = cv2.cvtColor(filledSource, cv2.COLOR_BGR2GRAY)
                
                # Define the threshold to separate foreground from background
                ret, mask = cv2.threshold(filledSourceGray, 150, 255, cv2.THRESH_BINARY_INV)
                
                 # Get the inverted mask
                mask_inv = cv2.bitwise_not(mask)
                
                # Get the background image
                background = cv2.bitwise_and(frame, frame, mask = mask)
                
                return background
        
        else:
            return frame

