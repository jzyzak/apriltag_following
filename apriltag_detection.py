import cv2
import numpy as np
from dt_apriltags import Detector
import numpy
import matplotlib.pyplot as plt
from pid import *

def PID_tags(frameShape, horizontal_distance, vertical_distance, horizontal_pid, vertical_pid):
    horizontal_error = ((frameShape[1]/2)-horizontal_distance)/frameShape[1]
    vertical_error = ((frameShape[0]/2)-vertical_distance)/frameShape[0]

    # Add low pass filter/gaussian blur stuff

    horizontal_output = (np.clip(horizontal_pid.update(horizontal_error), -100, 100))*100
    vertical_output = (np.clip(vertical_pid.update(vertical_error), -100, 100))*100

    return horizontal_output, vertical_output
    

def detect_tag(video, cameraMatrix = numpy.array([ 1060.71, 0, 960, 0, 1060.71, 540, 0, 0, 1]).reshape((3,3))):

    '''fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = 'april_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_file, fourcc, 30, (width, height))'''

    ret, frame = video.read()
    camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )
    at_detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.line(color_img,(960,0),(960,1080),(0,255,0),5)
    cv2.line(color_img,(0,540),(2000,540),(0,255,0),5)
    #plt.imshow(frame)
    tags = at_detector.detect(img, True, camera_params, tag_size = 0.1)
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))
        plt.imshow(color_img)
        #print(f'{tag.pose_t} \n\n\n\n {tag.pose_R}')
        pos = tag.center
        #heading = tag.center
    #output_video.write(processed_frame)
    return pos
    #video.release()
    #output_video.release()