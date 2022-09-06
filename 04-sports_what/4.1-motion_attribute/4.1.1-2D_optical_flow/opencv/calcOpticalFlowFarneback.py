import cv2 as cv
import numpy as np

cap = cv.VideoCapture('slow.mp4')

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
i = 0
while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

    # cv.imshow('frame2',bgr)
    if i % 50 == 0:
        cv.imwrite("slow_flow_calcOpticalFlowFarneback_result/flow%i.png" % i, bgr)
        # cv.imwrite('opticalfb.png',frame2)
        # cv.imwrite('opticalhsv.png',bgr)
    i += 1
    prvs = next

cap.release()
cv.destroyAllWindows()
