import cv2
import numpy as np

class BallDetect:
    def __init__(self):
        """
        Initialize the ball tracker
        Set hsv range for tennis ball
        """
        #HSV range for ball masking
        self.hsv_vals = {'hmin': 0, 'smin': 47, 'vmin': 0, 'hmax': 89, 'smax': 255, 'vmax': 255}
        #Ball Bounding Box
        self.ball_bb = None
        #Ball Contour
        self.ball_ct = None
        #Ball Center
        self.ball_center = None
        #Found ball
        self.found_ball = False
        #TODO Track ball over n frames
        # self.ball_locations = []

    def detectBall(self, img):
        #Hsv thresholding
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hv = self.hsv_vals
        lower = np.array([hv['hmin'], hv['smin'], hv['vmin']])
        upper = np.array([hv['hmax'], hv['smax'], hv['vmax']])
        mask = cv2.inRange(img_hsv, lower, upper)

        #Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #Find largest contour
        contours, _= cv2.findContours(mask, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            # ball_kp = self.detector.detect(mask)
            cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            self.ball_ct = cntsSorted[-1]
            M = cv2.moments(self.ball_ct)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.ball_center = (cX, cY) 
            #Min enclose rectangle
            x,y,w,h = cv2.boundingRect(cntsSorted[-1])
            self.ball_bb = (x,y,w,h)
            self.found_ball = True
        else:
            self.ball_ct = None
            self.ball_bb = None
            self.ball_center = None
            self.found_ball = False

    def foundBall(self):
        return self.found_ball