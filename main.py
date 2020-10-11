import numpy as np
import cv2

# Loading the classifier for frontal face:
FaceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start video capturing using the embedded camera (built-in webcam):
capture = cv2.VideoCapture(0)

# Main program: (press 'q' or 'Q' to quit)
while(True):
    # input video:
    _, video_input = capture.read()

    # Change the original video into a gray-scaled video:
    video_gray = cv2.cvtColor(video_input, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Video', video_gray)

    # to QUIT the program: >>> press (q) OR (Q) button
    key = cv2.waitKey(1) & 0xFF
    if (key == ord('q')) | (key == ord('Q')):
        break

# Close any open windows:
cv2.destroyAllWindows()
