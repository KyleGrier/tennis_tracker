import cv2
import numpy as np

class Visualize:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        stripes = np.zeros((self.height, self.width, 1), np.float32) 
        stripes[10, 10, :] = 1.0
        stripes = cv2.idft(stripes)
        stripes = cv2.inRange(stripes, -2, 0)
        self.stripes = stripes

    def outlineBallBB(self, c_img,  bt):
        if bt.ball_bb is None:
            return
        bb = bt.ball_bb
        # Draw a rectangle around the ball
        c_img = cv2.rectangle(c_img, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (255, 0, 255), 5)
        cX, cY = bt.ball_center
        cv2.circle(c_img, (cX, cY), 20, (255, 0, 255), -1)
        return c_img

    def outlineBall(self, c_img, bt):
        #Draw contour around ball
        if bt.ball_ct is not None:
            c_img = cv2.drawContours(c_img, [bt.ball_ct], -1, (0, 255, 255), 3)
        return c_img

    def outlineRegion(self, c_img, rt):
        if rt.regions is None:
            return
        colors = [(0,95,255), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        for key, region in rt.regions.items():
            cnt = region["contour"]
            c = colors[key-1] 
            if key == rt.ball_region:
                c_fill = cv2.drawContours(np.zeros_like(c_img), [cnt], -1, c, -1)
                c_fill = cv2.bitwise_and(c_fill, c_fill, mask=self.stripes)
                c_img = cv2.addWeighted(c_img,1.0,c_fill,0.6,0)
            else:
                c_img = cv2.drawContours(c_img, [cnt], -1, c, 10)
            # compute the center of the contour
            cX = region["center"][0] 
            cY = region["center"][1] 
            # draw the contour and center of the shape on the image
            cv2.circle(c_img, (cX, cY), 20, c, -1)
            cv2.putText(c_img, f"{key}", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
        return c_img

class Writer:
    def __init__(self, video_path="out.mov", width=1920, height=1080, fps=20.0):
        """
        Create video writer
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(video_path, fourcc, fps, (int(width), int(height)))

    def write(self, img):
        self.writer.write(img)

class VideoProcess():
    def __init__(self, video_path):
        """
        Load the video file and initialize the video stream
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #get video width and height
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #c_img = Current Image
        self.c_img = None
        #f_img = Filtered Image
        self.f_img = None

    def getImage(self):
        """
        Get the current frame and filtered frame
        """
        ret, self.c_img = self.cap.read()
        if ret:
            self.f_img = cv2.GaussianBlur(self.c_img,(11,11),cv2.BORDER_DEFAULT)
        else:
            self.f_img = None
        return ret, self.c_img, self.f_img