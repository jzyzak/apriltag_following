import cv2
import numpy as np
from dt_apriltags import Detector
import numpy
import matplotlib.pyplot as plt
from pid import *

def write_video(video):
    """
    Function: Takes a video and separates it into a list of frames
    
    Parameters:
        - video: a video file that you want to process
        
    Return: A list of all frames from the video
    """

    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #output_file = 'output_apriltag_video.avi'
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #output_video = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

    ret, frame = video.read()
    
    frames = []
    frames.append(frame)
    while ret:
        ret, frame = video.read()
        frames.append(frame)
    return frames

def detect_tag(frame, at_detector, cameraMatrix = numpy.array([ 353.571428571, 0, 320, 0, 353.571428571, 180, 0, 0, 1]).reshape((3,3))):
    """
    Function: Detects all the apriltags in a frame and returns their centers and the z-coordinate of the last apriltag detected
    
    Parameters:
        - frame: the current frame of the AUV being analyzed
        - at_detector: the apriltag detector to detect the specific apriltags
        - cameraMatrix: the cameraMatrix parameters for the AUV's camera
        
    Return: A list of the center positions of the detected apriltags and the z-coordinate of the last apriltag detected
    """

    # Specific camera parameters
    camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )

    # Creates the grayscale image and colored image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    shape = img.shape

    # Detects all tags available in the frame
    tags = at_detector.detect(img, True, camera_params, tag_size = 0.1)
    
    pos = []
    # Returns all tag center positions and the last tag's z-coordinate if there are any tags available in the frame
    if len(tags) > 0: 
        for tag in tags:
            pos.append(tag.center)
            # print(tag)
            tag_z = tag.pose_t[2]

    return pos, tag_z

def PID_tags(frameShape, horizontal_distance, vertical_distance, horizontal_pid, vertical_pid):
    """
    Function: Calculates the PID outputs required to reposition the apriltag in the center of the AUV's camera

    Parameters:
        - frameShape (tuple): the shape of the current frame
        - horizontal_distance (float): the horizontal distance between the center of the camera and the center of the apriltag
        - vertical_distance (float): the vertical distance between the center of the camera and the center of the apriltag 
        - horizontal_pid (PID): the horizontal PID controller
        - vertical_pid (PID): the vertical PID Controller

    Return: Returns the horizontal and vertical outputs for the PID controllers (used for apriltag tracking)
    """

    # Calculates the horizontal and vertical errors and outputs
    horizontal_error = ((frameShape[1]/2)-horizontal_distance)/frameShape[1]
    vertical_error = ((frameShape[0]/2)-vertical_distance)/frameShape[0]
    horizontal_output = (np.clip(horizontal_pid.update(horizontal_error), -100, 100))*100
    vertical_output = (np.clip(vertical_pid.update(vertical_error), -100, 100))*100

    return horizontal_output, vertical_output
    

def drawOnImage(img, tagPositions, horizontalPidOutput, verticalPidOutput):
    """
    Function: Draws the arrowed lines for the apriltag's position and the circles that represent the center of the camera
              and where the center of the tag lies on the x and y axes

    Parameters:
        - img: the frame of the AUV being drawn on
        - tagPositions (array): the coordinates of the center of the apriltag found in the frame
        - horizontalPidOutput (float): the output from the horizontal PID controller
        - verticalPidOutput (float): the output from the vertical PID controller

    Return: The image with the arrowed lines and circles drawn on it
    """
    # Draws the center of the tag
    imgWidth =  img.shape[1]
    imgHeight =  img.shape[0]
    widthCenter = int(imgWidth/2)
    heightCenter = int(imgHeight/2)
    xCord = tagPositions[0]
    yCord = tagPositions[1]
    # Draws a circle at the center of the camera
    cv2.circle(img, (int(xCord),int(yCord)), 50, (255, 0, 0), 5)
    # Draw where the center of the tag lies on the x and y axes
    cv2.circle(img, (int(widthCenter),int(yCord)), 50, (255, 0, 0), 5)
    cv2.circle(img, (int(xCord),int(heightCenter)), 50, (255, 0, 0), 5)
    # Draws the arrowed lines aimed towards the apriltag's center on the x and y axes
    cv2.arrowedLine(img, (widthCenter,heightCenter), (widthCenter, int(yCord)), 
            (0, 100, 255), 5)
    cv2.arrowedLine(img, (widthCenter,heightCenter), (int(xCord),heightCenter), 
            (0, 100, 255), 5)
    
    cv2.putText(img, str(horizontalPidOutput), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, str(verticalPidOutput), (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #img_array.append(frame)
    print(f"Horizontal PID Output: {horizontalPidOutput}%")
    print(f"Vertical PID Output: {verticalPidOutput}%")
    return img