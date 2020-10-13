import cv2
import sys, os
import getopt
import numpy as np
from matplotlib import pyplot as plt
from modules.video_stabilizer import Video_Stabilizer

if __name__ == "__main__":
    filename = ""
    cap = None

    if len(sys.argv) == 2:  # From input video file
        filename = sys.argv[1]
        if os.path.exists(filename):
            print("Reading from file:", filename)
            cap = cv2.VideoCapture(filename)
        else:
            raise IOError("File does not exist")

    else:  # From webcam
        print("Reading from webcam")
        cap = cv2.VideoCapture(0)
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    previous_frame = np.zeros((HEIGHT, WIDTH, 3))
    stabilizer = Video_Stabilizer(HEIGHT, WIDTH)
    counter = 0

    # For saving video
    fourcc = cv2.VideoWriter_fourcc(*"FMP4")
    out = cv2.VideoWriter("video_out.mp4", fourcc, 24.0, (WIDTH, HEIGHT))

while True:
    _, current_frame = cap.read()
    if current_frame is None:
        break

    ### Stabilize here ###
    stabilizer.add_frames(previous_frame, current_frame)
    stabilized_frame = stabilizer.stabilize()

    if stabilized_frame is not None:
        current_frame = current_frame.astype("uint8")
        stabilized_frame = stabilized_frame.astype("uint8")
        frame_out = cv2.hconcat([current_frame, stabilized_frame])

        if frame_out.shape[1] is not 1920:
            frame_out = cv2.resize(
                frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2)
            )
            cv2.imshow("Before and After", frame_out)
            out.write(stabilized_frame)
            cv2.waitKey(10)

    previous_frame = current_frame

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

