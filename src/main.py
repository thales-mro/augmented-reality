import numpy as np
import cv2
import imutils


def generate_panoramic(image_2, image_1, output, ratio=0.6, method='sift', threshold=4):

    # Read the images
    image_1 = cv2.imread(image_1)
    image_2 = cv2.imread(image_2)

    # Resize the image
    image_1 = imutils.resize(image_1, width=600)
    image_2 = imutils.resize(image_2, width=600)

    # Convert the images to gray scale
    gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    if method == 'sift':

        print("SIFT Method")

        descriptor = cv2.xfeatures2d.SIFT_create()

        # Find the interesting points
        kps_1, features_1 = descriptor.detectAndCompute(gray_1, None)
        kps_2, features_2 = descriptor.detectAndCompute(gray_2, None)

    elif method == 'surf':

        print("SURF Method")

        descriptor = cv2.xfeatures2d.SURF_create()

        kps_1, features_1 = descriptor.detectAndCompute(gray_1, None)
        kps_2, features_2 = descriptor.detectAndCompute(gray_2, None)

    elif method == 'brief':

        print("BRIEF Method")

        descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # Initiate STAR detector
        star = cv2.xfeatures2d.StarDetector_create()

        kp_1 = star.detect(gray_1, None)
        kp_2 = star.detect(gray_2, None)

        # compute the descriptors with ORB
        kps_1, features_1 = descriptor.compute(gray_1, kp_1)
        kps_2, features_2 = descriptor.compute(gray_2, kp_2)

    elif method == 'orb':

        print("ORB Method")

        descriptor = cv2.ORB_create()

        # compute the descriptors with ORB
        kps_1, features_1 = descriptor.detectAndCompute(gray_1, None)
        kps_2, features_2 = descriptor.detectAndCompute(gray_2, None)


    # Convert to float
    kps_1 = np.float32([kp.pt for kp in kps_1])
    kps_2 = np.float32([kp.pt for kp in kps_2])

    # Find the matches and the homography matrix
    matches, H, status = findMatchPoints(kps_1, kps_2, features_1, features_2, ratio, threshold)

    print(H)

    # Generate the result image
    image_result = cv2.warpPerspective(image_1, H, (image_1.shape[1] + image_2.shape[1], image_1.shape[0] + image_2.shape[0]))

    # Concat the two images
    image_result[0:image_2.shape[0], 0:image_2.shape[1]] = image_2

    # Draw the lines for the match
    image_match = drawLines(image_1, image_2, kps_1, kps_2, matches, status)

    # Save jpg
    cv2.imwrite(output, image_result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # Show the matches and result
    cv2.imshow("Image 1", image_1)
    cv2.imshow("Image 2", image_2)
    cv2.imshow("Image Match", image_match)
    cv2.imshow("Image Result", image_result)
    cv2.waitKey(0)


def findMatchPoints(kps_1, kps_2, features_1, features_2, ratio, threshold):

    # Create the brute force descriptor matcher
    matcher = cv2.BFMatcher()

    # Finds the k best matches for each descriptor
    rawM = matcher.knnMatch(features_1, features_2, 2)

    matches = []

    # For each match
    for m in rawM:

        # Add it to the array if the distance is below the ratio
        if m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:

        # Construct the set of points
        pts_1 = np.float32([kps_1[i] for (_, i) in matches])
        pts_2 = np.float32([kps_2[i] for (i, _) in matches])

        # Compute the homography matrix using RANSAC method
        (homograpy, status) = cv2.findHomography(pts_1, pts_2, cv2.RANSAC, threshold)

    return matches, homograpy, status


def drawLines(image_1, image_2, kps_1, kps_2, matches, status):

    # Initialize the output image
    (hA, wA) = image_1.shape[:2]
    (hB, wB) = image_2.shape[:2]

    image_match = np.zeros((max(hA, hB), wA + wB, 3), dtype=np.uint8)

    # Concat the two images
    image_match[0:hA, 0:wA] = image_1
    image_match[0:hB, wA:] = image_2

    # For each match
    for ((trainIdx, queryIdx), s) in zip(matches, status):

        # If the match is successful
        if s == 1:

            # Get the points coordinates
            pt_1 = (int(kps_1[queryIdx][0]), int(kps_1[queryIdx][1]))
            pt_2 = (int(kps_2[trainIdx][0]) + wA, int(kps_2[trainIdx][1]))

            # Draw the line
            cv2.line(image_match, pt_1, pt_2, (0, 255, 0), 1)

    return image_match


# Generate panoramic image with SIFT
#generate_panoramic('input/foto1A.jpg', 'input/foto1B.jpg', 'output/sift/foto1.jpg', 0.6, 'sift')
#generate_panoramic('input/foto2A.jpg', 'input/foto2B.jpg', 'output/sift/foto2.jpg', 0.6, 'sift')
#generate_panoramic('input/foto3A.jpg', 'input/foto3B.jpg', 'output/sift/foto3.jpg', 0.6, 'sift')
#generate_panoramic('input/foto4A.jpg', 'input/foto4B.jpg', 'output/sift/foto4.jpg', 0.6, 'sift')

# # Generate panoramic image with SURF
#generate_panoramic('input/foto1A.jpg', 'input/foto1B.jpg', 'output/surf/foto1.jpg', 0.6, 'surf')
#generate_panoramic('input/foto2A.jpg', 'input/foto2B.jpg', 'output/surf/foto2.jpg', 0.6, 'surf')
#generate_panoramic('input/foto3A.jpg', 'input/foto3B.jpg', 'output/surf/foto3.jpg', 0.6, 'surf')
#generate_panoramic('input/foto4A.jpg', 'input/foto4B.jpg', 'output/surf/foto4.jpg', 0.6, 'surf')


# Generate panoramic image with BRIEF
#generate_panoramic('input/foto1A.jpg', 'input/foto1B.jpg', 'output/brief/foto1.jpg', 0.6, 'brief')
#generate_panoramic('input/foto2A.jpg', 'input/foto2B.jpg', 'output/brief/foto2.jpg', 0.6, 'brief')
#generate_panoramic('input/foto3A.jpg', 'input/foto3B.jpg', 'output/brief/foto3.jpg', 0.6, 'brief')
#generate_panoramic('input/foto4A.jpg', 'input/foto4B.jpg', 'output/brief/foto4.jpg', 0.9, 'brief')


# Generate panoramic image with ORB
generate_panoramic('input/foto1A.jpg', 'input/foto1B.jpg', 'output/orb/foto1.jpg', 0.6, 'orb')
generate_panoramic('input/foto2A.jpg', 'input/foto2B.jpg', 'output/orb/foto2.jpg', 0.6, 'orb')
generate_panoramic('input/foto3A.jpg', 'input/foto3B.jpg', 'output/orb/foto3.jpg', 0.6, 'orb')
generate_panoramic('input/foto4A.jpg', 'input/foto4B.jpg', 'output/orb/foto4.jpg', 0.6, 'orb')
