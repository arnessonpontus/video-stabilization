import numpy as np
import cv2
import random

# params for ShiTomasi corner detection
FEATURE_PARAMS = dict( maxCorners=200,
                       qualityLevel=0.1,
                       minDistance=8,
                       blockSize=11 )

# Parameters for lucas kanade optical flow
LK_PARAMS = dict( winSize  = (21,21),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class Video_Stabilizer:
    def __init__(self, height, width):
        self.current_frame_rgb = np.zeros(
            (height, width, 3)
        )  # Change to camera dependent size?
        self.neighbouring_frames = np.zeros((height, width, 3, 3))
        self.previous_frame_rgb = np.zeros((height, width, 3))
        self.current_frame = np.zeros((height, width))
        self.previous_frame = np.zeros((height, width))
        self.output_frame = self.previous_frame_rgb
        self.height = height
        self.width = width
        self.H_last = np.ones(
            (2, 3)
        )  # Previous transform in case current frame does not have one
        self.frame_counter = 0
        # -------------- Accumulated frame-to-frame transforms ----------
        self.x = 0
        self.y = 0
        self.a = 0
        #  ------------- Kalman parameters ---------------
        self.X = np.zeros((1, 3))  # Initialize estimate to 0
        self.X_ = np.zeros((1, 3))
        self.P = np.ones((1, 3))  # Initialize error variance to 1
        self.P_ = np.zeros((1, 3))
        self.K = np.zeros((1, 3))
        self.pstd = 4e-3
        self.cstd = 0.25
        self.z = 0
        self.Q = np.array((self.pstd, self.pstd, self.pstd))
        self.R = np.array((self.cstd, self.cstd, self.cstd))

    def add_frames(self, previous_frame, current_frame):
        self.current_frame_rgb = current_frame
        self.previous_frame_rgb = previous_frame
        self.current_frame = cv2.cvtColor(
            current_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY
        )
        self.previous_frame = cv2.cvtColor(
            previous_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY
        )

    def stabilize(self):
        H = self.motion_estimation(self.previous_frame, self.current_frame)
        
        if H is None:
            H = self.H_last

        # Discard the extreme transforms
        if np.abs(H[0, 2]) > 100 or np.abs(H[1, 2]) > 100:
            H = self.H_last

        """
        dx = H[0, 2]
        dy = H[1 ,2]
        da = np.arctan2(H[1,0], H[0,0])

        if np.abs(da) > 0.5:
            H = self.H_last

        print(dx, dy, da)
        """

        self.H_last = H

        # Motion filtering
        H = self.motion_filter(H)

        # Inverse of H
        # H = np.concatenate((self.H_cummulative, np.array([[0, 0, 1]])))
        # H = np.linalg.inv(H)
        # H = np.delete(H, 2, 0)

        # Warp through affine matrix
        stabilized_frame = cv2.warpAffine(self.previous_frame_rgb, H, (self.width, self.height))


        # Crop and resize
        stabilized_frame = self.crop_and_resize(stabilized_frame)
        
        
        self.neighbouring_frames = np.concatenate(
            (
                stabilized_frame[:, :, :, np.newaxis],
                self.neighbouring_frames[:, :, :, 0:-1],
            ),
            axis=3,
        )
        

        # Testing inpainting
        if self.frame_counter > 2:
            image_mask = np.ones((stabilized_frame.shape[0], stabilized_frame.shape[1])).astype("uint8") * 255
            image_mask = cv2.warpAffine(image_mask, H, (self.width, self.height))
            image_mask = self.crop_and_resize(image_mask)
            image_mask = cv2.bitwise_not(image_mask)

            stabilized_frame = self.inpainting(stabilized_frame, image_mask, H)

        # old inpaint test
        # stabilized_image_inpainted = cv2.inpaint(self.previous_frame_rgb.astype('uint8')*255, stabilized_image_mask, 5, cv2.INPAINT_TELEA)
        # stabilized_image_inpainted = cv2.bitwise_not(stabilized_image_inpainted)
        """
        test = np.zeros((self.height, self.width, 3))
        test[:,:,0] = stabilized_image_mask
        test[:,:,1] = stabilized_image_mask
        test[:,:,2] = stabilized_image_mask
        """
        
        self.frame_counter += 1
        return stabilized_frame

    def motion_estimation(self, previous_frame, current_frame):
        coords = cv2.goodFeaturesToTrack(previous_frame, mask=None, **FEATURE_PARAMS)

        if coords is None:
            return None

        good_coords, good_next_coords = self.get_optical_flow(coords)
        # self.draw_tracks(good_coords, good_next_coords)
        if good_coords.shape[0] < 3:
            return None

        # Get Affine Transformation
        H, _ = cv2.estimateAffinePartial2D(good_coords, good_next_coords)

        return H

    # FOR DEBUG PURPOSES
    def draw_tracks(self, coords, next_coords, draw_lines=True):
        mask = np.zeros_like(self.previous_frame)

        # FOR DEBUG PURPOSES
        color = np.random.randint(0, 255, (200, 3))

        for i, (new, old) in enumerate(zip(next_coords, coords)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.circle(self.previous_frame_rgb, (a, b), 5, color[i].tolist(), -1)
            if draw_lines:
                cv2.line(self.previous_frame_rgb, (a, b), (c, d), color[i].tolist(), 2)

    def get_optical_flow(self, coords):
        # calculate optical flow
        next_coords, status, err = cv2.calcOpticalFlowPyrLK(
            self.previous_frame, self.current_frame, coords, None, **LK_PARAMS
        )

        # Select good points
        good_next_coords = next_coords[status == 1]
        good_old_coords = coords[status == 1]
        return good_old_coords, good_next_coords

    def motion_filter(self, H):
        dx = H[0, 2]
        dy = H[1, 2]
        da = np.arctan2(H[1, 0], H[0, 0])

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
            self.P = (np.ones((1, 3)) - self.K) * self.P_
            # ------------------------------------------

        # corrected tranform - accumulated transform
        diff_x = self.X[0, 0] - self.x
        diff_y = self.X[0, 1] - self.y
        diff_a = self.X[0, 2] - self.a

        dx += diff_x
        dy += diff_y
        da += diff_a

        # Construct new transform
        H[0, 0] = np.cos(da)
        H[0, 1] = -np.sin(da)
        H[1, 0] = np.sin(da)
        H[1, 1] = np.cos(da)

        H[0, 2] = dx
        H[1, 2] = dy

        return H

    def inpainting(self, frame_cur, frame_mask, H):
        """
        Following steps by 
        https://github.com/jahaniam/Real-time-Video-Mosaic
        """
        transformed_corners = self.transformed_corners(self.previous_frame_rgb, H)

        #frame_cur = self.draw_border(frame_cur, transformed_corners)

        w = np.flip(np.arange(self.neighbouring_frames.shape[3]+1))
        print(w)
        w = np.exp(w*3)
            
        
        self.output_frame[frame_mask == 255] *= w[0]
        
        for i in range(self.neighbouring_frames.shape[3]):
            self.output_frame[frame_mask == 255] += self.neighbouring_frames[:,:,:,i][frame_mask == 255] * w[i+1]
        
        self.output_frame[frame_mask == 255] = self.output_frame[frame_mask == 255] // np.sum(w)

        #self.output_frame[frame_mask == 255] = (self.output_frame[frame_mask == 255]*w2 + frame_cur[frame_mask == 255]*w1)//(w1+w2)
        self.output_frame[frame_mask == 0] = frame_cur[frame_mask == 0]

        output_temp = np.copy(self.output_frame)
        output_temp = self.draw_border(
            output_temp, transformed_corners, color=(0, 0, 255)
        )

        return self.output_frame

    def transformed_corners(self, frame_cur, H):
        corner_0 = np.array([0, 0])
        corner_1 = np.array([frame_cur.shape[1], 0])
        corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
        corner_3 = np.array([0, frame_cur.shape[0]])

        corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        H = np.concatenate((H, [[0, 0, 1]]))

        # transformed_corners = cv2.warpAffine(corners, H, (self.width, self.height))
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = np.array(transformed_corners, dtype=np.int32)

        return transformed_corners

    def draw_border(self, image, corners, color=(0, 0, 0)):
        """This functions draw rectancle border

        Args:
            image ([type]): current mosaiced output
            corners (np array): list of corner points
            color (tuple, optional): color of the border lines. Defaults to (0, 0, 0).

        Returns:
            np array: the output image with border
        """
        for i in range(corners.shape[1] - 1, -1, -1):
            cv2.line(
                image,
                tuple(corners[0, i, :]),
                tuple(corners[0, i - 1, :]),
                thickness=5,
                color=color,
            )
        return image

    def crop_and_resize(self, stabilized_frame):
        #Croping and scaling
        y = 80
        x = 80
        h = int(stabilized_frame.shape[0] * 0.8)
        w = int(stabilized_frame.shape[1] * 0.8)
        crop_img = stabilized_frame[y:y+h, x:x+w]

        scale_percent = 220 # percent of original size
        width = int(stabilized_frame.shape[1])
        height = int(stabilized_frame.shape[0])
        dim = (width, height)
        
        return cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

    
