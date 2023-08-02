import cv2
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
#from dt_apriltags import Detector
from lane_detection import *
import matplotlib.cm as cm
from pid import *

def get_lane_center(frameHeight, lanes):
    center_intercept = (lanes[0][0]+lanes[1][0])/2
    x1, y1, x2, y2 = lanes[0]
    slope1 = (y1-y2)/(x1-x2)
    x1, y1, x2, y2 = lanes[1]
    slope2 = (y1-y2)/(x1-x2)
    center_slope = 1/(((1/slope1)+(1/slope2))/2)
    center_intercept = ((((frameHeight - y2)/center_slope)  )+ x2)
    return center_intercept, center_slope

def draw_center_lane(img, center_intercept, center_slope, xPoint = 0, yPoint = 0):
    global imgPixelHeight 
    imgPixelHeight  = img.shape[0]
    cv2.line(img, (int(center_intercept), imgPixelHeight), (int(xPoint), int(yPoint)), (0,0,255), 6)
    return img

def recommend_direction(center, slope):
    halfOfRes = 1920/2
    HorizontalDiff = halfOfRes-center

    #horizontal_magnitudes = set_horizontal_control(HorizontalDiff, PIDoutputPosition(HorizontalDiff, horizontal_pid))

    AproxAUVAngle = 90 - angle_between_lines(slope, 0)  
    # get the approx angle of the auv with the line by calculating the slope of the 
    #center line with a horizontal line relative to the rov
    if slope > 0:
        AproxAUVAngle = -AproxAUVAngle

    #heading_magnitudes = set_heading_control(AproxAUVAngle, heading_pid)

    #thruster_magnitudes = do_both(horizontal_magnitudes, heading_magnitudes)

    return HorizontalDiff, AproxAUVAngle 

def lane_PID(heading_error, strafe_error, heading_pid, strafe_pid):
    heading_output = np.clip(heading_pid.update(heading_error),-100,100) * 100
    strafe_output = np.clip(strafe_pid.update(strafe_error),-100,100) * 100
    return heading_output, strafe_output
