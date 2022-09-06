import numpy as np
import cv2 as cv

cap = cv.VideoCapture('slow.mp4')

# ShiTomasi 角点检测的参数
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Lucas-Kanade 光流算法的参数
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建一组随机颜色数
color = np.random.randint(0,255,(100,3))

# 取第一帧并寻找角点
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# 创建绘制轨迹用的遮罩图层
mask = np.zeros_like(old_frame)
i = 0
while(1):
    ret,frame = cap.read()
    
    try:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    except:
        break
    # 计算光流
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 选取最佳始末点
    good_new = p1[st==1]
    good_old = p0[st==1]

    # 绘制轨迹
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    
    cv.imwrite('slow_flow_calcOpticalFlowPyrLK_result/frame%i.jpg' % i,img)
    i += 1

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # 更新选取帧与特征点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv.destroyAllWindows()
cap.release()