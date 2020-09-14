import cv2
import numpy as np
from matplotlib import pyplot as plt 
from modules.video_stabilizer import Video_Stabilizer

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

previous_frame = np.zeros((720, 1280, 3))
stabilizer = Video_Stabilizer()

while True:
    _, current_frame = cap.read()
    #current_frame = cv2.resize(current_frame, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    ### Stabilize shit here ###
    stabilizer.add_frames(previous_frame, current_frame)
    stabilized_frame = stabilizer.stabilize()
    
    # Subplots of prev and curr frame
    # f, axarr = plt.subplots(1,2)
    # axarr[0].imshow(previous_frame)
    # axarr[1].imshow(current_frame)
    # plt.show()
    ###

    cv2.imshow('Input', stabilized_frame)

    previous_frame = current_frame

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

