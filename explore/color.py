import cv2
import numpy as np
import logging

# Reconfigured from code in https://github.com/cvzone/cvzone/blob/c4b68aaf9d83756ce721f54c4a94dd38774f22ea/cvzone/ColorModule.py
class ColorFinder:
    def __init__(self, hsvVals =None, trackBar=False):
        self.trackBar = trackBar
        if self.trackBar:
            self.initTrackbars(hsvVals)

    def empty(self, a):
        pass

    def initTrackbars(self, hsvVals):
        """
        To intialize Trackbars . Need to run only once
        """
        if hsvVals is None:
            hsvVals = {'hmin': 0, 'smin': 0, 'vmin':0,
                       'hmax': 179, 'smax': 255, 'vmax': 255}
            
            
        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars", 640, 240)
        cv2.createTrackbar("Hue Min", "TrackBars", hsvVals["hmin"], 179, self.empty)
        cv2.createTrackbar("Hue Max", "TrackBars", hsvVals["hmax"], 179, self.empty)
        cv2.createTrackbar("Sat Min", "TrackBars", hsvVals["smin"], 255, self.empty)
        cv2.createTrackbar("Sat Max", "TrackBars", hsvVals["smax"], 255, self.empty)
        cv2.createTrackbar("Val Min", "TrackBars", hsvVals["vmin"], 255, self.empty)
        cv2.createTrackbar("Val Max", "TrackBars", hsvVals["vmax"], 255, self.empty)

    def getTrackbarValues(self):
        """
        Gets the trackbar values in runtime
        :return: hsv values from the trackbar window
        """
        hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
        smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
        vmin = cv2.getTrackbarPos("Val Min", "TrackBars")
        hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
        smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
        vmax = cv2.getTrackbarPos("Val Max", "TrackBars")

        hsvVals = {"hmin": hmin, "smin": smin, "vmin": vmin,
                   "hmax": hmax, "smax": smax, "vmax": vmax}
        return hsvVals

    def update(self, img, myColor=None):
        """
        :param img: Image in which color needs to be found
        :param hsvVals: List of lower and upper hsv range
        :return: (mask) bw image with white regions where color is detected
                 (imgColor) colored image only showing regions detected
        """
        imgColor = [],
        mask = []

        if self.trackBar:
            myColor = self.getTrackbarValues()

        if isinstance(myColor, str):
            myColor = self.getColorHSV(myColor)

        if myColor is not None:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([myColor['hmin'], myColor['smin'], myColor['vmin']])
            upper = np.array([myColor['hmax'], myColor['smax'], myColor['vmax']])
            mask = cv2.inRange(imgHSV, lower, upper)
            imgColor = cv2.bitwise_and(img, img, mask=mask)
        return imgColor, mask

    def getColorHSV(self, myColor):

        if myColor == 'red':
            output = {'hmin': 146, 'smin': 141, 'vmin': 77, 'hmax': 179, 'smax': 255, 'vmax': 255}
        elif myColor == 'green':
            output = {'hmin': 44, 'smin': 79, 'vmin': 111, 'hmax': 79, 'smax': 255, 'vmax': 255}
        elif myColor == 'blue':
            output = {'hmin': 103, 'smin': 68, 'vmin': 130, 'hmax': 128, 'smax': 255, 'vmax': 255}
        else:
            output = None
            logging.warning("Color Not Defined")
            logging.warning("Available colors: red, green, blue ")

        return output

def findBall(mask):
    contours, hierarchy = cv2.findContours(mask, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    return cntsSorted

def main():
    # Ball Color
    # hsvVals = {'hmin': 0, 'smin': 47, 'vmin': 0,
    #            'hmax': 89, 'smax': 255, 'vmax': 255}
    #Region Color
    hsvVals = {'hmin': 113, 'smin': 95, 'vmin': 0,
               'hmax': 179, 'smax': 255, 'vmax': 255}
    myColorFinder = ColorFinder(hsvVals=hsvVals, trackBar=True)
    cap = cv2.VideoCapture("../videos/RollingTennisBall.mov")

    # Custom Orange Color
    success, image = cap.read()
    while True:
        img = image.copy()
        img = cv2.GaussianBlur(img,(11,11),cv2.BORDER_DEFAULT)
        # hsvVals = myColorFinder.getTrackbarValues()
        img = cv2.resize(img, (640, 480))
        imgRed, _ = myColorFinder.update(img, "red")
        imgGreen, _ = myColorFinder.update(img, "green")
        imgBlue, _ = myColorFinder.update(img, "blue")
        imgOrange, mask = myColorFinder.update(img, hsvVals)
        cntsSorted = findBall(mask)
        print(len(cntsSorted))
        img = cv2.drawContours(img, cntsSorted, -1, (0, 0, 255), 3)
        # cv2.imshow("Blue", imgBlue)
        # cv2.imshow("Green", imgGreen)
        cv2.imshow("Orange", imgOrange)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        if k == ord('n'):
            success, image = cap.read()
            if not success:
                break
            continue
        if k == ord('q'):
            break


if __name__ == "__main__":
    main()