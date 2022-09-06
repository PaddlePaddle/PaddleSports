import numpy as np
from loadmodel import *
import paddle, cv2
from skimage import transform as trans


class MTCNN:
    def __init__(self):
        self.pnet_predictor, self.pnet_input_handles, self.pnet_output_handles = create_predictor('./model/PNet')
        self.rnet_predictor, self.rnet_input_handles, self.rnet_output_handles = create_predictor('./model/RNet')
        self.onet_predictor, self.onet_input_handles, self.onet_output_handles = create_predictor('./model/ONet')

    def predict_pnet(self, img):
        # set input
        input_img_size = img.shape
        self.pnet_input_handles[0].reshape([1, 3, input_img_size[2], input_img_size[3]])
        self.pnet_input_handles[0].copy_from_cpu(img)
        # run predictor
        self.pnet_predictor.run()
        # get outputs
        cls_prob = self.pnet_output_handles[0]
        bbox_pred = self.pnet_output_handles[1]
        cls_prob = cls_prob.copy_to_cpu()  # type = numpy.ndarray
        bbox_pred = bbox_pred.copy_to_cpu()
        cls_prob = cls_prob[0]
        bbox_pred = bbox_pred[0]
        cls_prob = softmax_p(cls_prob)
        return cls_prob, bbox_pred

    def predict_rnet(self, img):
        input_img_size = img.shape
        self.rnet_input_handles[0].reshape([input_img_size[0], 3, input_img_size[2], input_img_size[3]])
        self.rnet_input_handles[0].copy_from_cpu(img)
        # run predictor
        self.rnet_predictor.run()
        # get outputs
        cls_prob = self.rnet_output_handles[0]
        bbox_pred = self.rnet_output_handles[1]
        cls_prob = cls_prob.copy_to_cpu()
        bbox_pred = bbox_pred.copy_to_cpu()
        cls_prob = paddle.squeeze(paddle.to_tensor(cls_prob))
        bbox_pred = paddle.squeeze(paddle.to_tensor(bbox_pred))
        return cls_prob.numpy(), bbox_pred.numpy()

    def predict_onet(self, img):

        input_img_size = img.shape
        self.onet_input_handles[0].reshape([input_img_size[0], 3, input_img_size[2], input_img_size[3]])
        self.onet_input_handles[0].copy_from_cpu(img)

        self.onet_predictor.run()

        cls_prob = self.onet_output_handles[0]
        bbox_pred = self.onet_output_handles[1]
        landmark_pred = self.onet_output_handles[2]
        cls_prob = cls_prob.copy_to_cpu()
        bbox_pred = bbox_pred.copy_to_cpu()
        landmark_pred = landmark_pred.copy_to_cpu()
        cls_prob = softmax_o(cls_prob)
        return cls_prob, bbox_pred, landmark_pred

    @staticmethod
    def processed_image(img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
        image = np.array(img_resized).astype(np.float32)
        # trans to format CHW
        image = image.transpose((2, 0, 1))
        image = np.reshape(image, (1, 3, new_height, new_width))
        image = (image - 127.5) / 128
        return image

    def detect_pnet(self, im, min_face_size, scale_factor, thresh):
        net_size = 12
        current_scale = float(net_size) / min_face_size
        im_resized = self.processed_image(im, current_scale)
        _, c, current_height, current_width = im_resized.shape
        all_boxes = list()
        # 图像金字塔
        while min(current_height, current_width) > net_size:
            # 类别和box
            cls_cls_map, reg = self.predict_pnet(im_resized)
            boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
            current_scale *= scale_factor  # 继续缩小图像做金字塔
            im_resized = self.processed_image(im, current_scale)
            _, c, current_height, current_width = im_resized.shape
            if boxes.size == 0:
                continue
            # 非极大值抑制留下重复低的box
            keep = py_nms(boxes[:, :5], 0.5, mode='Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes) == 0:
            return None
        all_boxes = np.vstack(all_boxes)
        # 将金字塔之后的box也进行非极大值抑制
        keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
        all_boxes = all_boxes[keep]
        # box的长宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        # 对应原图的box坐标和分数
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T
        return boxes_c

    def detect_rnet(self, im, dets, thresh):
        """通过rent选择box
            参数：
              im：输入图像
              dets:pnet选择的box，是相对原图的绝对坐标
            返回值：
              box绝对坐标
        """
        h, w, c = im.shape
        # 将pnet的box变成包含它的正方形，可以避免信息损失
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        # 调整超出图像的box
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        delete_size = np.ones_like(tmpw) * 20
        ones = np.ones_like(tmpw)
        zeros = np.zeros_like(tmpw)
        num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
        cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        for i in range(int(num_boxes)):
            # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
            if tmph[i] < 20 or tmpw[i] < 20:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            try:
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
                img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
                img = img.transpose((2, 0, 1))
                img = (img - 127.5) / 128
                cropped_ims[i, :, :, :] = img
            except:
                continue
        cls_scores, reg = self.predict_rnet(cropped_ims)
        if len(cls_scores.shape) > 1:
            cls_scores = softmax_r(cls_scores)
            cls_scores = cls_scores[:, 1]
            keep_inds = np.where(cls_scores > thresh)[0]
            if len(keep_inds) > 0:
                boxes = dets[keep_inds]
                boxes[:, 4] = cls_scores[keep_inds]
                reg = reg[keep_inds]
            else:
                return None
        else:
            return None

        keep = py_nms(boxes, 0.4, mode='Union')
        boxes = boxes[keep]
        # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
        boxes_c = calibrate_box(boxes, reg[keep])
        return boxes_c

    def detect_onet(self, im, dets, thresh):
        """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        cls_scores, reg, landmark = self.predict_onet(cropped_ims)

        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > thresh)[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        w = boxes[:, 2] - boxes[:, 0] + 1

        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = calibrate_box(boxes, reg)

        keep = py_nms(boxes_c, 0.6, mode='Minimum')
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes_c, landmark

    # 对齐
    @staticmethod
    def estimate_norm(lmk):
        tform = trans.SimilarityTransform()
        src = np.array([[30.29459953, 51.69630051],
                        [65.53179932, 51.50139999],
                        [48.02519989, 71.73660278],
                        [33.54930115, 92.3655014],
                        [62.72990036, 92.20410156]], dtype=np.float32)
        tform.estimate(lmk, src)
        M = tform.params[0:2, :]
        return M

    def norm_crop(self, img, landmark, image_size=112):
        M = self.estimate_norm(landmark)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

    def inference(self, img):
        boxes_c = self.detect_pnet(img, 20, 0.79, 0.9)
        if boxes_c is None:
            return None, None, None
        boxes_c = self.detect_rnet(img, boxes_c, 0.6)
        if boxes_c is None:
            return None, None, None
        boxes_c, landmarks = self.detect_onet(img, boxes_c, 0.7)
        if boxes_c is None:
            return None, None, None
        imgs = []
        for landmark in landmarks:
            landmark = [[float(landmark[i]), float(landmark[i + 1])] for i in range(0, len(landmark), 2)]
            landmark = np.array(landmark, dtype='float32')
            img = self.norm_crop(img, landmark)
            imgs.append(img)
        return imgs, boxes_c, landmarks

    def det_face(self, img):
        orimg_shape = img.shape
        img = cv2.resize(img, (int(orimg_shape[1] * 0.2), int(orimg_shape[0] * 0.2)))
        boxes_c = self.detect_pnet(img, 20, 0.79, 0.9)
        if boxes_c is None:
            return False
        boxes_c = self.detect_rnet(img, boxes_c, 0.6)
        if boxes_c is None:
            return False
        boxes_c, landmarks = self.detect_onet(img, boxes_c, 0.7)
        if boxes_c is None:
            return False
        return True


def softmax_p(x):
    tmp = np.max(x, axis=0)
    x -= tmp
    x = np.exp(x)
    tmp = np.sum(x, axis=0)
    x /= tmp
    return x


def softmax_r(x):
    dim = x.shape[0]
    tmp = np.reshape((np.max(x, axis=1)), (dim, -1))
    x -= tmp
    x = np.exp(x)
    tmp = np.sum(x, axis=1)
    x /= np.reshape(tmp, (dim, -1))
    return x


def softmax_o(x):
    dim = x.shape[0]
    tmp = np.reshape((np.max(x, axis=1)), (dim, -1))
    x -= tmp
    x = np.exp(x)
    tmp = np.sum(x, axis=1)
    x /= np.reshape(tmp, (dim, -1))
    return x


def generate_bbox(cls_map, reg, scale, threshold):
    """
     得到对应原图的box坐标，分类分数，box偏移量
    """
    # pnet大致将图像size缩小2倍
    stride = 2

    cellsize = 12

    # 将置信度高的留下
    t_index = np.where(cls_map > threshold)

    # 没有人脸
    if t_index[0].size == 0:
        return np.array([])
    # 偏移量
    dx1, dy1, dx2, dy2 = [reg[i, t_index[0], t_index[1]] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    # 对应原图的box坐标，分类分数，box偏移量
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    # shape[n,9]
    return boundingbox.T


def py_nms(dets, thresh, mode="Union"):
    """
    贪婪策略选择人脸框
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def convert_to_square(box):
    """将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    """
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找寻正方形最大边长
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box


def pad(bboxes, w, h):
    """将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:图像长宽
    返回值：
      dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
      edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
      y, x : 调整后的box在原图上左上角的坐标
      ex, ex : 调整后的box在原图上右下角的坐标
      tmph, tmpw: 原始box的长宽
    """
    # box的长宽
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def calibrate_box(bbox, reg):
    """校准box
    参数：
      bbox:pnet生成的box

      reg:rnet生成的box偏移值
    返回值：
      调整后的box是针对原图的绝对坐标
    """

    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c
