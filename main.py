from threading import Thread, Event
from time import sleep

from pid import PID
from video import Video
from bluerov_interface import BlueROV
from pymavlink import mavutil
from apriltag_detection import *
from dt_apriltags import Detector

# TODO: import your processing functions


# Create the video object
video = Video()
# Create the PID object
pid_vertical = PID(K_p=0.1, K_i=0.0, K_d=0.01, integral_limit=1)
pid_horizontal = PID(K_p=0.1, K_i=0.0, K_d=0.01, integral_limit=1)
# Create the mavlink connection
mav_comn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
# Create the BlueROV object
bluerov = BlueROV(mav_connection=mav_comn)

frame = None
frame_available = Event()
frame_available.set()

vertical_power = 0
lateral_power = 0


def _get_frame():
    global frame
    while not video.frame_available():
        print("Waiting for frame...")
        sleep(0.01)

    vertical_pid = PID(1, 0, 0, 100)
    horizontal_pid = PID(1, 0, 0, 100)
    try:
        while True:
            if video.frame_available():
                frame = video.frame()
                # TODO: Add frame processing here
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                tags = at_detector.detect(img, False, camera_params, tag_size = 0.1)
                for tag in tags:
                    for idx in range(len(tag.corners)):
                        cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))
                        cv2.circle(color_img, (int(tag.center[0].item()),int(tag.center[1].item())), 50, (0, 0, 255), 2)
                    
                    pos = tag.center
                    #print(type(tag.center[0]))
                    #cv2.line(color_img,,(960,540),(0,255,0),5)
                    cv2.arrowedLine(color_img,(960,540),(-1*(int(960+pos[0])),540),(255,0,0),5)
                    cv2.arrowedLine(color_img,(960,540),(960,-1*(int(540+pos[1]))),(255,0,0),5)
                    horizontal_output, vertical_output = PID_tags(frame.shape, pos[0], pos[1], horizontal_pid, vertical_pid)
                    #print(frame.shape)
                    print(horizontal_output, vertical_output)
                    #cv2.arrowedLine(color_img,(960,540),(960,int(vertical_output)+540),(255,0,0),5)
                    #cv2.arrowedLine(color_img,(960,540),(int(horizontal_output)+960,540),(255,0,0),5)
                    cv2.putText(color_img, str(horizontal_output), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(color_img, str(vertical_output), (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    #img_array.append(frame)
                    print(f"Horizontal PID Output: {horizontal_output}%")
                    print(f"Vertical PID Output: {vertical_output}%")
                # TODO: set vertical_power and lateral_power here
                lateral_power = horizontal_output
                vertical_power = vertical_output
                print(frame.shape)
    except KeyboardInterrupt:
        return


def _send_rc():
    bluerov.set_vertical_power(vertical_power)
    bluerov.set_lateral_power(lateral_power)


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
