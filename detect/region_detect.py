import cv2
import numpy as np

class RegionDetect():
    def __init__(self):
        self.regions = None
        self.ball_region = None
        self.hsv_vals = {'hmin': 113, 'smin': 95, 'vmin': 0,
                           'hmax': 179, 'smax': 255, 'vmax': 255}

    def discernRegions(self, cnts):
        """
        Given a list of contours, specify the regions of the image
        """
        if len(cnts) == 4:
            rgs = {}
            for i, cnt in enumerate(cnts): 
                #convert contour to convex hull for robustness when ball moves over lines
                cnt = cv2.convexHull(cnt)
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if i == 0:
                    for j in range(1,5):
                        rgs[j] = {"contour": cnt, "center": (cX, cY)}
                    continue
                if cX < rgs[1]["center"][0]:
                    rgs[1] = {"contour": cnt, "center": (cX, cY)}
                if cY < rgs[2]["center"][1]:
                    rgs[2] = {"contour": cnt, "center": (cX, cY)}
                if cX > rgs[3]["center"][0]:
                    rgs[3] = {"contour": cnt, "center": (cX, cY)}
                if cY > rgs[4]["center"][1]:
                    rgs[4] = {"contour": cnt, "center": (cX, cY)}
            self.regions = rgs
        else:
            #Use previous regions
            pass

    def outlineRegion(self, img):
        """
        Use the HSV values to get contour of regions and identify regions
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hv = self.hsv_vals
        lower = np.array([hv['hmin'], hv['smin'], hv['vmin']])
        upper = np.array([hv['hmax'], hv['smax'], hv['vmax']])
        mask = cv2.inRange(img_hsv, lower, upper)

        contours, hierarchy = cv2.findContours(mask, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            self.discernRegions(contours)
        else:
            self.regions = None

    def ballRegion(self, bt):
        """
        Determine the region of the ball
        """
        if not bt.foundBall():
            self.ball_region = None
            return
        for key, region in self.regions.items():
            result = cv2.pointPolygonTest(region["contour"], bt.ball_center, False) 
            if result == 1 or result == 0:
                self.ball_region = key
                return
        self.ball_region = None