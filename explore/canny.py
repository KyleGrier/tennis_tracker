import cv2
import numpy as np
import math

cap = cv2.VideoCapture("../videos/RollingTennisBall.mov")
#Create a trackbar for canny edge detection cv2
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H","Trackbars",128,255,lambda x: None)
cv2.createTrackbar("L-S","Trackbars",255,255,lambda x: None)
cv2.createTrackbar("L-V","Trackbars",0,255,lambda x: None)
cv2.createTrackbar("U-H","Trackbars",255,255,lambda x: None)
cv2.createTrackbar("U-S","Trackbars",255,255,lambda x: None)
cv2.createTrackbar("U-V","Trackbars",255,255,lambda x: None)

ret, src = cap.read()
#Perform clahe on src
src = cv2.GaussianBlur(src,(11,11),cv2.BORDER_DEFAULT)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
# lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
# lab[:, :, 0] = clahe.apply(lab[:, :, 0])
# src = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
kernel = np.ones((5,5),np.uint8)
while True:
    #Get trackbar values
    l_h = cv2.getTrackbarPos("L-H","Trackbars")
    l_s = cv2.getTrackbarPos("L-S","Trackbars")
    print(l_h)
    #Perform edge detection
    # edges = cv2.Canny(src, 50, 200, None, 3)
    dst = cv2.Canny(src.copy(), l_h, l_s)
    dst = cv2.dilate(dst,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(dst, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    
    print("Number of Contours found = " + str(len(contours)))
    
    # Draw all contours
    # -1 signifies drawing all contours
    src_contour = cv2.drawContours(np.zeros_like(src), cntsSorted, -1, (0, 255, 0), 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    #Perform Hough Transform
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    # lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    # edges = cv2.resize(edges, (0,0), fx=0.5, fy=0.5)
    src_out = cv2.resize(src, (0, 0), fx=0.25, fy=0.25)
    cdst = cv2.resize(cdst, (0, 0), fx=0.25, fy=0.25)
    cdstP = cv2.resize(cdstP, (0, 0), fx=0.25, fy=0.25)
    src_contour = cv2.resize(src_contour, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Source", src_out)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    cv2.imshow("Contours", src_contour)
    # cv2.imshow("great", edges)
    # cv2.imshow("great", img)
    k = cv2.waitKey(1)
    if k == ord("n"):
        ret, src = cap.read()
        src = cv2.GaussianBlur(src,(11,11),cv2.BORDER_DEFAULT)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
        # lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        # lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # src = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if k == 27:
        break