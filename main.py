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
pid_horizontal = PID(K_p=0.1, K_i=0.0, K_d=0.01, integral_limit=1)
pid_horizontal_lf = PID(K_p=0.1, K_i=0.0, K_d=0.01, integral_limit=1)
pid_heading_lf = PID(K_p=30, K_i=0, K_d=-10, integral_limit=100)
# Create the mavlink connection
mav_comn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
# Create the BlueROV object
bluerov = BlueROV(mav_connection=mav_comn)

frame = None
frame_available = Event()
frame_available.set()

# Power values and followRobot boolean
vertical_power = 0
lateral_power = 0
yaw_power = 0
longitudinal_power = 0
followRobot = False


def _get_frame():
    """
    Function: Gets the frame from the AUV and determines how to move based on if it detects a lane or apriltags
    """

    # Global variables for frame, power, and robot following
    global frame, vertical_power, lateral_power, yaw_power, longitudinal_power, followRobot

    # Sets the values for the PID controllers for the vertical and horizontal directions
    vertical_pid = PID(2, 0, 0, 100)
    horizontal_pid = PID(1, 0, 0, 100)

    # Creates a detector that can be used to detect the apriltags on the robot and on the walls
    at_detector = Detector(families='tag36h11',
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)
    
    at_wall_detector = Detector(families='tag36h11', # should change tag
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)
    
    # Keeps waiting until a video frame is available
    while not video.frame_available():
        print("Waiting for frame...")
        sleep(0.01)

    # Tries to process a frame and adjust the robot's movement accordingly
    try:
        while True:
            if video.frame_available():
                frame = video.frame() # Initializes the frame
                
                tags = detect_tag(frame, at_detector) # detects and returns all apriltags in the given frame
                print("Got frame")
                # Checks if any apriltags have been detected
                if len(center_tags) > 0:
                    # Sets followRobot variable to True so the robot will track the robot's apriltag and move towards it
                    followRobot = True
                    # Utilizies the last tag found
                    center_tags = tags[-1]
                    print("Got tag")
                    # Gets the horizontal and vertical outputs for the PID to make thruster adjustments and move towards the other AUV's apriltag
                    horizontal_output, vertical_output = PID_tags(frame.shape, center_tags[0], center_tags[1], horizontal_pid, vertical_pid)
                    # Draws the arrows in terms of where the apriltag is in relation to the center of the AUV's camera
                    img = drawOnImage(frame, center_tags, horizontal_output, vertical_output)
                    
                    # Sets vertical and lateral/horizontal power for the thrusters
                    vertical_power = vertical_output
                    lateral_power = horizontal_output

                    # Goes straight forward if the apriltag is already in the center of the AUV's camera
                    if(vertical_power.almostEquals(0) and lateral_power.almostEquals(0)):
                        longitudinal_power = 20
                    # Sets longitudinal/forward thruster magnitudes to 0 so it doesn't go forward
                    else:
                        longitudinal_power = 0
                    
                    # Writes image so we can see the apriltag the AUV detected in VS Code
                    cv2.imwrite("ROV_frame.jpg", img)

                    # Turns off thrusters and flashes lights on and off if the AUV is in "shooting" range of the other AUV
                    if (center_tags.pose_t[2]<1):
                        for i in range(10):
                            bluerov.set_rc_channel(9,1100)
                            time.sleep(0.2)
                            bluerov.set_rc_channel(9,1500)

                        vertical_power = 0
                        lateral_power = 0
                        longitudinal_power = 0

                # If no AUV's apriltag is detected, the AUV shuts off all thrusters and prepares to follow the lanes
                else:
                    followRobot = False
                    vertical_power = 0
                    lateral_power = 0
                    longitudinal_power = 0
                
                # Checks if the AUV isn't following another robot and performs lane detection/following
                if(followRobot == False):
                    # Creates a list of the lines that have been detected
                    line_list = detect_lines(frame, 49, 50, 3, 500, 40)

                    # Tries to detect the lanes from any lines that have been found
                    try:
                        print("got line")
                        lanes = detect_lanes(line_list)
                        # Tries to analyze the lanes and align the AUV with the center of the lane
                        try:
                            print("got lane")
                            # Gets the x-intercept and the slope of the line running through the center of the lane
                            center_intercept, center_slope = get_lane_center(frame.shape[1], lanes)
                            # Calculates the difference in horizontal position and heading/angle alignment with the center of the lane
                            horizontal_diff, heading_diff = recommend_direction(frame.shape[1], center_intercept, center_slope)

                            # Calculates the yaw and lateral thruster magnitudes if the AUV is not aligned with the center of the lane
                            if(horizontal_diff != 0 and heading_diff != 0):
                                # Sets longitudinal power to 0 to make sure the AUV isn't moving straight at the same time the robot is realigning itself
                                longitudinal_power = 0
                                # Calculates the yaw and lateral thruster powers
                                yaw_power, lateral_power = lane_PID(heading_diff, horizontal_diff, pid_heading_lf, pid_horizontal_lf)
                            
                                # Draws the lanes on the frame
                                img = draw_lanes(frame, lanes)
                                
                                print(yaw_power)
                                print(lateral_power)
                            else:
                                # Sets longitudinal power to 0 so the robot moves forward
                                longitudinal_power = 20

                            # Writes the image so we can see the lanes the AUV detected in VS code
                            cv2.imwrite("ROV_frame.jpg", img)
                        # Turns off thrusters and sets followRobot to False in case no lanes are detected
                        except:
                            followRobot = False
                            vertical_power = 0
                            lateral_power = 0
                            longitudinal_power = 0
                    # Turns off thrusters and sets followRobot to False in case no lines/lanes are detected
                    except:
                        followRobot = False
                        vertical_power = 0
                        lateral_power = 0
                        longitudinal_power = 0                  

    except KeyboardInterrupt:
        return


def _send_rc():
    global vertical_power, lateral_power, yaw_power, longitudinal_power
    bluerov.set_rc_channels_to_neutral()
    mav_comn.set_mode(19)
    while True:
        # bluerov.disarm()
        bluerov.arm()

        # Sets the powers of the thrusters based on outputs for the PID controllers
        bluerov.set_vertical_power(int(vertical_power))
        bluerov.set_lateral_power(-int(lateral_power))
        bluerov.set_yaw_rate_power(int(yaw_power))
        bluerov.set_longitudinal_power(int(longitudinal_power))

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