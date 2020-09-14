import numpy as np
import cv2

class Video_Stabilizer():
    def __init__(self):
        self.current_frame_rgb = np.zeros((720, 1280, 3)) # Change to camera dependent size?
        self.previous_frame_rgb = np.zeros((720, 1280, 3))
        self.current_frame = np.zeros((720, 1280))
        self.previous_frame = np.zeros((720, 1280))

    def add_frames(self, previous_frame, current_frame):
        self.current_frame_rgb = current_frame
        self.previous_frame_rgb = previous_frame
        self.current_frame = cv2.cvtColor(current_frame.astype('float32'), cv2.COLOR_BGR2GRAY)
        self.previous_frame = cv2.cvtColor(previous_frame.astype('float32'), cv2.COLOR_BGR2GRAY)

    def stabilize(self):
        H = self.motion_estimation(self.previous_frame, self.current_frame)
        return self.previous_frame_rgb

    def motion_estimation(self, previous_frame, current_frame):
        coords = cv2.goodFeaturesToTrack(previous_frame, maxCorners=30, qualityLevel=0.2, minDistance=4)
        self.draw_coords(coords)
        self.homography_estimation()
        self.frame_orbit_generating()

    def draw_coords(self, coords):
        if coords is not None:
            for coord in coords:
                self.previous_frame_rgb = cv2.circle(self.previous_frame_rgb, (coord[0][0], coord[0][1]), radius=4, color=(0, 0, 255), thickness=-1) 

    def homography_estimation(self):
        pass

    def frame_orbit_generating(self):
        pass