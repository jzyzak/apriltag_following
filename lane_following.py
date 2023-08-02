# Imports
import cv2
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
#from dt_apriltags import Detector
from lane_detection import *
import matplotlib.cm as cm
from pid import *

def get_lane_center(frameHeight, lanes):
    """
    Function: Gets the x-intercept and slope of the line running through the center of a given lane

    Parameters:
        - frameHeight (int): represents the height of the AUV's camera
        - lanes (list of line objects): contains a list of the two lines that act as the boundaries for the lane
    """
    # Calculates the x-intercept of the line running through the center of the given lane
    # center_intercept = (lanes[0][0]+lanes[1][0])/2

    # Calculates the slopes of the two boundary lines
    x1, y1, x2, y2 = lanes[0]
    slope1 = (y1-y2)/(x1-x2)
    x1, y1, x2, y2 = lanes[1]
    slope2 = (y1-y2)/(x1-x2)

    # Calculates the slope of the line running through the center of the lane
    center_slope = 1/(((1/slope1)+(1/slope2))/2)

    # Calculates the x-intercept of the line running through the center of the given lane
    center_intercept = ((((frameHeight - y2)/center_slope)  )+ x2)
    return center_intercept, center_slope

def draw_center_lane(img, center_intercept, center_slope, xPoint = 0, yPoint = 0):
    """
    Function: Draws the line running through the center of a given lane
    
    Parameters:
        - img (numpy array of image information): the actual image/frame from the AUV that needs to be processed
        - center_intercept (float): the x-intercept of the line running through the center of the lane
        - center_slope (float): the slope of the line running through the center of the lane
        - xPoint (float): the x-coordinate of the endpoint of the line running through the center of the lane
        - yPoint (float): the y-coordinate of the endpoint of the line running through the center of the lane
    """

    global imgPixelHeight 
    imgPixelHeight  = img.shape[0]
    cv2.line(img, (int(center_intercept), imgPixelHeight), (int(xPoint), int(yPoint)), (0,0,255), 6)
    return img

def recommend_direction(frameWidth, center, slope):
    """
    Function: Returns the difference in horizontal position between the AUV's x-position and the x-coordinate of
              the center of the lane as well as the difference in heading for the AUV
              
    Parameters:
        - frameWidth (int): the width of the frame/image being processed
        - center (float): the x-coordinate of the line running through the center of the lane
        - slope (float): the slope of the line running through the line running the center of the lane
    """
    
    # Calculates the difference between the center of the camera and the x-coordinate of the center of the lane
    halfOfRes = frameWidth/2
    HorizontalDiff = halfOfRes-center

    # Calculates the approximate difference in angle heading
    AproxAUVAngle = 90 - angle_between_lines(slope, 0)  

    # get the approx angle of the auv with the line by calculating the slope of the 
    #center line with a horizontal line relative to the rov
    if slope > 0:
        AproxAUVAngle = -AproxAUVAngle

    return HorizontalDiff, AproxAUVAngle 

def lane_PID(heading_error, strafe_error, heading_pid, strafe_pid):
    """
    Function: Calculates the PID outputs for the differences in angle and horizontal position between the AUV
              and the center of the lane
    
    Parameters:
        - heading_error (float): the error for the angle between the AUV and the center of the lane
        - strafe_error (float): the error for the distance between the AUV and the x-coordinate of the center of the lane
        - heading_pid (PID object): the PID controller for the AUV's heading
        - strafe_pid (PID object): the PID controller for the AUV's horizontal position
    """
    
    heading_output = np.clip(heading_pid.update(heading_error),-100,100) * 100
    strafe_output = np.clip(strafe_pid.update(strafe_error),-100,100) * 100
    return heading_output, strafe_output