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
        
        # Read the target image
        self.target = cv2.imread(targetPath)
        
        # Get the target dimensions
        self.target_w, self.target_h, _ = self.target.shape
        
        # Read the source image
        self.source =  cv2.imread(sourcePath)
        
        # Get the source dimensions
        self.source_w, self.source_h, _ = self.source.shape
        

      
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
        
            processedFrames.append(image)
                
            cv2.imshow('current_frame', image)
            
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
        
        

