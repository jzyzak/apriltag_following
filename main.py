# Imports
from threading import Thread, Event
from time import sleep
from pid import PID
from video import Video
from bluerov_interface import BlueROV
from pymavlink import mavutil
from dt_apriltags import Detector
import cv2

# TODO: import your processing functions
from apriltag_detection import *
from lane_following import *
from lane_detection import *

# Create the video object
video = Video()
# Create the PID object
pid_vertical = PID(K_p=0.1, K_i=0.0, K_d=0.01, integral_limit=1)
pid_horizontal_at = PID(K_p=0.1, K_i=0.0, K_d=0.01, integral_limit=1)
pid_horizontal_lf = PID(K_p=0.1, K_i=0.0, K_d=0.01, integral_limit=1)
pid_heading_lf = PID(K_p=30, K_i=0, K_d=-10, integral_limit=100)
# Create the mavlink connection
mav_comn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
# Create the BlueROV object
bluerov = BlueROV(mav_connection=mav_comn)

frame = None
frame_available = Event()
frame_available.set()

vertical_power = 0
lateral_power = 0
yaw_power = 0
followRobot = False

def _get_frame():
    global frame, vertical_power, lateral_power, yaw_power, followRobot
    #vertical_pid = PID(1, 0, 0, 100)
    #horizontal_pid = PID(1, 0, 0, 100)
    at_detector = Detector(families='tag36h11',
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

    while not video.frame_available():
        print("Waiting for frame...")
        sleep(0.01)
    try:
        while True:
            if video.frame_available():
                frame = video.frame()
                center_tags = detect_tag(frame, at_detector)
                print("Got frame")
                if len(center_tags) > 0:
                    followRobot = True
                    center_tags = center_tags[-1]
                    print("Got tag")
                    horizontal_output, vertical_output = PID_tags(frame.shape, center_tags[0], center_tags[1], pid_horizontal_at, pid_vertical)
                    img = drawOnImage(frame, center_tags, horizontal_output, vertical_output)
                    #TODO: set vertical_power and lateral_power here
                    vertical_power = vertical_output
                    lateral_power = horizontal_output
                    cv2.imwrite("ROV_frame.jpg", img)
                else:
                    followRobot = False

                if(followRobot == False):
                    # Run lane following directions
                    lines = lines(frame, 49, 50, 3, 500, 40)
                    lanes = detect_lanes(lines)
                    if(len(lanes) > 0):
                        center_intercept, center_slope = get_lane_center(frame.shape[1], lanes)
                        horizontal_diff, heading_diff = recommend_direction(center_intercept, center_slope)
                        yaw_power, lateral_power = lane_PID(heading_diff, horizontal_diff, pid_heading_lf, pid_horizontal_lf) 
                    

    except KeyboardInterrupt:
        return


def _send_rc():
    while True:
        #bluerov.disarm()
        #bluerov.arm()
        bluerov.set_vertical_power(int(vertical_power))
        bluerov.set_lateral_power(int(lateral_power))
        bluerov.set_yaw_rate_power(int(yaw_power))


# Start the video thread
video_thread = Thread(target=_get_frame)
video_thread.start()

# Start the RC thread
rc_thread = Thread(target=_send_rc)
rc_thread.start()

# Main loop
try:
    while True:
        mav_comn.wait_heartbeat()
except KeyboardInterrupt:
    video_thread.join()
    rc_thread.join()
    bluerov.disarm()
    print("Exiting...")