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

    camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    shape = img.shape
    tags = at_detector.detect(img, True, camera_params, tag_size = 0.1)
    # tags = at_detector.detect(img, True, )
    
    pos = []
    if len(tags) > 0: 
        for tag in tags:
            pos.append(tag.center)

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
    

def drawOnImage(img, tagPositions, horizontalPidOutput, verticalPidOutput):
    """
    Takes in the img, tagPositions as a 2d array, with each tag having a vertical, horizizontal cord, and a depth. 
    Also takes in the horizontalPid and verticalPid outputs
    """
    #draw the center of the tag
    imgWidth =  img.shape[1]
    imgHeight =  img.shape[0]
    widthCenter = int(imgWidth/2)
    heightCenter = int(imgHeight/2)
    xCord = tagPositions[0]
    yCord = tagPositions[1]
    cv2.circle(img, (int(xCord),int(yCord)), 50, (255, 0, 0), 5)
    #draw where the center of the tag lies on the x an y axis
    cv2.circle(img, (int(widthCenter),int(yCord)), 50, (255, 0, 0), 5)
    cv2.circle(img, (int(xCord),int(heightCenter)), 50, (255, 0, 0), 5)
    #draw the PID vectors as arrows
    #cv2.arrowedLine(img, (widthCenter,heightCenter), (widthCenter, -int((verticalPidOutput/100)*15) - heightCenter), 
            #(0, 100, 255), 5)
    #cv2.arrowedLine(img, (widthCenter,heightCenter), (-int((horizontalPidOutput/100)*5) + widthCenter,heightCenter), 
            #(0, 100, 255), 5)

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
