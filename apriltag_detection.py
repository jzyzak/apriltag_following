import cv2
import numpy as np
from dt_apriltags import Detector
import numpy
import matplotlib.pyplot as plt
from pid import *

def write_video(video):
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = 'output_apriltag_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

    ret, frame = video.read()
    i = 0
    cameraMatrix = numpy.array([ 1060.71, 0, 960, 0, 1060.71, 540, 0, 0, 1]).reshape((3,3))
    camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )
    at_detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)
    
    frames = []
    frames.append(frame)
    while ret:
        ret, frame = video.read()
        frames.append(frame)

def detect_tag(frame, at_detector, cameraMatrix = numpy.array([ 1060.71, 0, 960, 0, 1060.71, 540, 0, 0, 1]).reshape((3,3))):

    camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    shape = img.shape()
    tags = at_detector.detect(img, True, camera_params, tag_size = 0.1)
    for tag in tags:
        pos = tag.center
    return pos

def PID_tags(frameShape, horizontal_distance, vertical_distance, horizontal_pid, vertical_pid):
    '''
    function that takes in the positions of the april tags and returns the output for the AUV motors
    '''

    horizontal_error = ((frameShape[1]/2)-horizontal_distance)/frameShape[1]
    vertical_error = ((frameShape[0]/2)-vertical_distance)/frameShape[0]
    horizontal_output = (np.clip(horizontal_pid.update(horizontal_error), -100, 100))*100
    vertical_output = (np.clip(vertical_pid.update(vertical_error), -100, 100))*100

    return horizontal_output, vertical_output
    

def DrawOnImage(img, tagPositions, horizontalPid, verticalPid):
    """
    Takes in the img, tagPositions as a 2d array, with each tag having a vertical, horizizontal cord, and a depth. 
    Also takes in the horizontalPid and verticalPid outputs
    """
    for tag in tagPositions:
        #draw the center of the tag
        imgWidth =  img.shape[0]
        imgHeight =  img.shape[1]
        widthCenter = int(imgWidth/2)
        heightCenter = int(imgHeight/2)
        xCord = tag[0]
        yCord = tag[1]
        cv2.circle(img, (xCord,yCord), 10, (0, 0, 255), 5)
        #draw where the center of the tag lies on the x an y axis
        cv2.circle(img, (xCord,int(imgWidth/2)), 10, (0, 0, 255), 5)
        cv2.circle(img, (yCord,int(imgHeight/2)), 10, (0, 0, 255), 5)
    #draw the PID vectors as arrows
    cv2.arrowedLine(img, (widthCenter,heightCenter), (widthCenter, int(verticalPid) + heightCenter), 
            (0, 100, 255), 5, tipLength = 0.5)
    cv2.arrowedLine(img, (widthCenter,heightCenter), (int(horizontalPid) + widthCenter,heightCenter), 
            (0, 100, 255), 5, tipLength = 0.5)
    return img