# ------------------------------------------------------------------------------------------------ #
# video_recorder.py
# Author: Alexandre Rodrigues Emidio [alexandre dot rodrigues dot emidio at gmail dot com]
#
# Python version: 2.7
#
# Software description:
# Simple python script used to record a video stream from a webcam (opencv library required).
#
# Usage:
# python video_recorder.py --help
# python video_recorder.py -f <output_filename> [-i <camera_index>]
#
# During execution:
#   - press <q> to quit the program
#   - press <r> to start recording
#   - press <s> to stop recording
# Every time the <r> key is pressed the file <output_filename>_X.avi is created. X is a incremental
# number used to differentiate the filenames of the videos recorded since the start of the program.
# ------------------------------------------------------------------------------------------------ #

import cv2
import argparse


# check command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='''
Record a video stream from a webcam.

User guide:
  - press <q> to quit the program
  - press <r> to start recording
  - press <s> to stop recording
                                 ''')
parser.add_argument('-f', '--filename', required=True, help='filename of the output video file')
parser.add_argument('-i', '--cam_index', type=int, default=0,
                    help='index of the webcam [default value = 0]')
args = parser.parse_args()

# try to open the camera indexed by cam_index
cap = cv2.VideoCapture(args.cam_index)
if not cap.isOpened():
    print 'Error: cannot open the selected camera.\n'

# get camera resolution
width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

# set FPS (frame per second)
fps = 10.0  # TODO: get the FPS from the camera (cap.get(cv2.cv.CV_CAP_PROP_FPS) is not working)
cap.set(cv2.cv.CV_CAP_PROP_FPS, fps)

# main loop
fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')  # motion-jpeg codec
recording = False
i = 0
rec_video = None
while(True):
    ret, frame = cap.read()
    if ret:
        # check for user input
        key = cv2.waitKey(1)
        if key == ord('q'):  # quit the program
            if recording:
                rec_video.release()
            break
        elif key == ord('r') and not recording:  # start recording
            i += 1
            filename = '%s_%d.avi' % (args.filename, i)
            rec_video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            recording = True
        elif key == ord('s') and recording:  # stop recording
            rec_video.release()
            recording = False

        if recording:  # display an indication that the camera is recording
            rec_video.write(frame)
            cv2.putText(frame, 'Recording ...', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, .7,
                        (0, 255, 0))
        # display capture
        cv2.imshow('Video Stream', frame)
    else:
        break

# close camera
cap.release()
