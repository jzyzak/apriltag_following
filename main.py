from threading import Thread, Event
from time import sleep

from pid import PID
from video import Video
from bluerov_interface import BlueROV
from pymavlink import mavutil
from dt_apriltags import Detector

# TODO: import your processing functions
from apriltag_detection import *

def write_video(video):
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = 'output_apriltag_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_file, fourcc, 30, (width, height))



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
        # TODO: set vertical_power and lateral_power here
        pass
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
