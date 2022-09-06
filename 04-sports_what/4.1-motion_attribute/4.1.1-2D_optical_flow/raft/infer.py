import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddle.vision.transforms import functional as F

import sys
sys.path.append('core')

import paddle
from frame_utils import *
from PIL import Image
from core.raft import RAFT
from core import flow_viz


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = paddle.to_tensor(img)
    img = paddle.cast(img, dtype='float32')
    img = paddle.transpose(x=img, perm=[2,0,1])
    return img[None]


def viz(flo, isPic):  # img,
    flo = paddle.squeeze(flo)
    flo = paddle.transpose(x=flo, perm=[1,2,0])
    flo = np.array(flo)
    flo = flow_viz.flow_to_image(flo)

    if isPic:
        cv2.imwrite('test/infer_flow.png', flo)
    else:
        return flo


def infer_img():
    place = paddle.set_device('gpu')
    # ----------------------------------------------Hyperparameter begin--------------------------------------------------------
    model_path = 'output/97500.pdparams'           # 模型的地址
    imfile1 = 'test/13724_img1.ppm'                     # 第一张图片
    imfile2 = 'test/13724_img2.ppm'                     # 第二张图片
    flow_gt_path = 'test/13724_flow.flo'                # 真实的光流文件，如果没有光流文件，可以注释53、54、55、56行
    gt_flo = read_gen(flow_gt_path)                
    gt_flo = flow_viz.flow_to_image(gt_flo)
    cv2.imwrite('test/gt_flo.png', gt_flo)
    # ----------------------------------------------Hyperparameter begin--------------------------------------------------------

    model = RAFT()
    model.load_dict(paddle.load(model_path))
    image1 = load_image(imfile1)
    image2 = load_image(imfile2)
    flow = model(image1, image2)[-1]
    viz(flow, isPic=True)


def infer_video():
    place = paddle.set_device('gpu')
    # ----------------------------------------------Hyperparameter begin--------------------------------------------------------
    model_path = 'output/97500.pdparams'           # 模型的地址
    videofile = 'test/test.mp4'                    # 视频地址
    outputfile = 'test/output.mp4'                 # 导出地址
    # ----------------------------------------------Hyperparameter begin--------------------------------------------------------
    print('开始创建模型')
    model = RAFT()
    model.load_dict(paddle.load(model_path))
    print("创建模型成功")
    capture = cv2.VideoCapture('test/test.mp4')
    frames = []
    print('开始读取视频帧')
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    l = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    while True:
        ret, frame = capture.read()
        if ret:
            frame = np.array(frame).astype(np.uint8)
            frame = F.to_tensor(frame)
            frame = paddle.cast(frame, dtype='float32')
            # frame = paddle.transpose(x=frame, perm=[2,0,1])
            frame = paddle.unsqueeze(x=frame, axis=0)
            frames.append(frame.cuda())
        else:
            break
    
    print('总帧数：', l)

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(filename=outputfile, fourcc=fourcc, fps=fps ,frameSize=(video_width, video_height))
    for i in range(0, l - 1):
        flow_p = model(frames[i], frames[i + 1 ])
        flow = viz(flow_p[-1], isPic=False)
        video.write(flow)
    print('完成！')
    video.release()
        

if __name__ == '__main__':
    infer_img()


















