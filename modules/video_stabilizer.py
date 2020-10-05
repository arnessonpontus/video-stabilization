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
        self.height = height
        self.width = width
        self.H_last = np.ones((2, 3)) # Previous transform in case current frame does not have one
        self.frame_counter = 0
        # -------------- Accumulated frame-to-frame transforms ----------
        self.x = 0  
        self.y = 0  
        self.a = 0
        #  ------------- Kalman parameters ---------------
        self.X = np.zeros((1,3)) # Initialize estimate to 0
        self.X_ = np.zeros((1,3))
        self.P = np.ones((1,3)) # Initialize error variance to 1
        self.P_ = np.zeros((1,3))
        self.K = np.zeros((1,3))
        self.pstd = 4e-3
        self.cstd = 0.25
        self.z = 0
        self.Q = np.array((self.pstd, self.pstd, self.pstd))
        self.R = np.array((self.cstd, self.cstd, self.cstd))

    def add_frames(self, previous_frame, current_frame):
        self.current_frame_rgb = current_frame
        self.previous_frame_rgb = previous_frame
        self.current_frame = cv2.cvtColor(current_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        self.previous_frame = cv2.cvtColor(previous_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    def stabilize(self):
        H = self.motion_estimation(self.previous_frame, self.current_frame)
        
        if H is None: 
            H = self.H_last
        
        self.H_last = H
        
        # Motion filtering
        H = self.motion_filter(H)
        
        # Inverse of H
        # H = np.concatenate((self.H_cummulative, np.array([[0, 0, 1]])))
        # H = np.linalg.inv(H)
        # H = np.delete(H, 2, 0)
    
        # Warp through affine matrix
        stabilized_image = cv2.warpAffine(self.previous_frame_rgb, H, (self.width, self.height))
        
        self.frame_counter += 1
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
        H, _ = cv2.estimateAffinePartial2D(good_coords, good_next_coords)

        return H

    # FOR DEBUG PURPOSES
    def draw_tracks(self, coords, next_coords, draw_lines=True):
        mask = np.zeros_like(self.previous_frame)
        
        # FOR DEBUG PURPOSES
        color = np.random.randint(0,255,(100,3))

        for i,(new,old) in enumerate(zip(next_coords, coords)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.circle(self.previous_frame_rgb,(a,b),5,color[i].tolist(),-1)
            if draw_lines:
                cv2.line(self.previous_frame_rgb, (a,b),(c,d), color[i].tolist(), 2)

    def get_optical_flow(self, coords):
        # calculate optical flow
        next_coords, status, err = cv2.calcOpticalFlowPyrLK(self.previous_frame, self.current_frame, coords, None, **LK_PARAMS)
    
        # Select good points
        good_next_coords = next_coords[status==1]
        good_old_coords = coords[status==1]
        return good_old_coords, good_next_coords
    
    def motion_filter(self, H):
        dx = H[0, 2]
        dy = H[1 ,2]
        da = np.arctan2(H[1,0], H[0,0])

        # Accumulate frame to frame transformation
        self.x += dx
        self.y += dy
        self.a += da

        self.z = np.array((self.x, self.y, self.a))

        if self.frame_counter > 0:

            #  -------------  KALMAN  -----------------
            # Prediction
            self.X_ = self.X
            self.P_ = self.P + self.Q

            # Correction 
            self.K = self.P_ / (self.P_ + self.R)
            self.X = self.X_ + self.K * (self.z - self.X_)
            self.P = (np.ones((1,3)) - self.K) * self.P_
            # ------------------------------------------
        
        # corrected tranform - accumulated transform
        diff_x = self.X[0,0] - self.x
        diff_y = self.X[0,1] - self.y
        diff_a = self.X[0,2] - self.a

        dx += diff_x
        dy += diff_y
        da += diff_a

        # Construct new transform
        H[0,0] = np.cos(da)
        H[0,1] = -np.sin(da)
        H[1,0] = np.sin(da)
        H[1,1] = np.cos(da)

        H[0,2] = dx
        H[1,2] = dy
        
        return H
    
    