import numpy as np
import cv2

# params for ShiTomasi corner detection
FEATURE_PARAMS = dict( maxCorners=100,
                       qualityLevel=0.3,
                       minDistance=4,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
LK_PARAMS = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class Video_Stabilizer():
    def __init__(self, height, width):
        self.current_frame_rgb = np.zeros((height, width, 3)) # Change to camera dependent size?
        self.previous_frame_rgb = np.zeros((height, width, 3))
        self.current_frame = np.zeros((height, width))
        self.previous_frame = np.zeros((height, width))
        self.H_cummulative = np.ones((2, 3))

    def add_frames(self, previous_frame, current_frame):
        self.current_frame_rgb = current_frame
        self.previous_frame_rgb = previous_frame
        self.current_frame = cv2.cvtColor(current_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        self.previous_frame = cv2.cvtColor(previous_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    def stabilize(self):
        H = self.motion_estimation(self.previous_frame, self.current_frame)
        if H is None: 
            return self.previous_frame_rgb

        self.H_cummulative = H

        # Motion filtering
        #H_smoothed = self.get_motion_filter()

        # Inverse of H
        H = np.concatenate((self.H_cummulative, np.array([[0, 0, 1]])))
        H = np.linalg.inv(H)
        H = np.delete(H, 2, 0)

        # Smoothed transformation
        # A_smoothed = M_smoothed . (M_cummulative)^-1 . A_current from article
        H_res = H
    
        # Warp through affine matrix
        stabilized_image = cv2.warpAffine(self.previous_frame_rgb, H_res, (self.height, self.width))
        return stabilized_image

    def motion_estimation(self, previous_frame, current_frame):
        coords = cv2.goodFeaturesToTrack(previous_frame, mask = None, **FEATURE_PARAMS)

        if coords is None:
            return None

        good_coords, good_next_coords = self.get_optical_flow(coords)
        self.draw_tracks(good_coords, good_next_coords)
        if good_coords.shape[0] < 3:
            return None

        # Get Affine Transformation
        H, _ = cv2.estimateAffine2D(good_coords, good_next_coords)

        #good_coords = np.float32(good_coords)
        #good_next_coords = np.float32(good_next_coords)

        # Get Homography (4-points)
        #good_coords = np.float32(good_coords)
        #good_next_coords = np.float32(good_next_coords)
        #H = cv2.getPerspectiveTransform(good_coords, good_next_coords)

        #warped_image = np.dot(np.linalg.inv(H), self.previous_frame)

        # Warp through homography matrix
        #warped_image = cv2.warpPerspective(self.previous_frame_rgb, H, (self.previous_frame.shape[1], self.previous_frame.shape[0]))

        # Return transformation matrix
        return H

    # FOR DEBUG PURPOSES
    def draw_tracks(self, coords, next_coords, drawing_type=True):
        mask = np.zeros_like(self.previous_frame)
        
        # FOR DEBUG PURPOSES
        color = np.random.randint(0,255,(100,3))
        
        for i,(new,old) in enumerate(zip(next_coords, coords)):
            a, b = new.ravel() 
            c, d = old.ravel()
            cv2.circle(self.previous_frame_rgb,(a,b),5,color[i].tolist(),-1)
            if drawing_type:
                cv2.line(self.previous_frame_rgb, (a,b),(c,d), color[i].tolist(), 2)

    def get_optical_flow(self, coords):
        # calculate optical flow
        next_coords, status, err = cv2.calcOpticalFlowPyrLK(self.previous_frame, self.current_frame, coords, None, **LK_PARAMS)
    
        # Select good points
        good_next_coords = next_coords[status==1]
        good_old_coords = coords[status==1]
        return good_old_coords, good_next_coords
    
    def get_motion_filter(self):
        # KALMAN
        return None

    