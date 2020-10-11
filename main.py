import numpy as np
import cv2

# Loading the classifier for frontal face:
FaceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
