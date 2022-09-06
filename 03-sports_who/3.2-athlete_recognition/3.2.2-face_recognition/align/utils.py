from paddle import inference
import paddle
import cv2, math
import numpy as np


def create_predictor(model_dir):
    # refer   https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_doc/Config/GPUConfig.html
    model_file = model_dir + '.pdmodel'
    params_file = model_dir + '.pdiparams'
    config = inference.Config()
    config.set_prog_file(model_file)
    config.set_params_file(params_file)
    # 启用 GPU 进行预测 - 初始化 GPU 显存 50M, Deivce_ID 为 0
    config.enable_use_gpu(50, 0)
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    input_handles = []
    for input_name in input_names:
        input_handles.append(predictor.get_input_handle(input_name))
    output_names = predictor.get_output_names()
    output_handles = []
    for output_name in output_names:
        output_handles.append(predictor.get_input_handle(output_name))
    print('model {} has {} inputs tensor {} outputs tensor'.format(model_dir, len(input_names), len(output_names)))
    return predictor, input_handles, output_handles


def create_predictor_un_combined(model_dir):
    # 加载非Combined 模型
    config = inference.Config()
    config.set_model(model_dir)
    config.enable_use_gpu(50, 0)
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    input_handles = []
    for input_name in input_names:
        input_handles.append(predictor.get_input_handle(input_name))
    output_names = predictor.get_output_names()
    output_handles = []
    for output_name in output_names:
        output_handles.append(predictor.get_input_handle(output_name))
    print('model {} has {} inputs tensor {} outputs tensor'.format(model_dir, len(input_names), len(output_names)))
    return predictor, input_handles, output_handles


class FaceDetector640:
    def __init__(self):
        self.face_detector, self.face_detector_input_handles, self.face_detector_output_handles = create_predictor_un_combined(
            'models/ultra_light_fast_generic_face_detector_1mb_640')
        self.confs_threshold = 0.96
        self.iou_threshold = 0.5

    @staticmethod
    def input_preprocess(inputs):
        # 处理图片
        image = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).astype('float32')
        return image

    @staticmethod
    def area_of(left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        # _, indexes = scores.sort(descending=True)
        indexes = np.argsort(scores)
        # indexes = indexes[:candidate_size]
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            # current = indexes[0]
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            # indexes = indexes[1:]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]

    def inference(self, inputs):
        orig_im_shape = inputs.shape
        # 数据处理
        inputs = self.input_preprocess(inputs)
        # 模型推理
        self.face_detector_input_handles[0].reshape([1, 3, 480, 640])
        self.face_detector_input_handles[0].copy_from_cpu(inputs)
        self.face_detector.run()
        confidences = self.face_detector_output_handles[0].copy_to_cpu()
        boxes = self.face_detector_output_handles[1].copy_to_cpu()
        # 后处理
        picked_box_probs = []
        picked_labels = []
        for i in range(confidences.shape[0]):
            # confidences shape is [N,num,2]
            for class_index in range(1, confidences[i].shape[1]):
                probs = np.array(confidences[i][:, class_index])
                mask = probs > self.confs_threshold
                mask = np.array(mask)
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = boxes[i][mask, :]
                box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
                box_probs = self.hard_nms(box_probs, iou_threshold=self.iou_threshold, top_k=-1)
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

        if len(picked_box_probs) == 0:
            return None

        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= orig_im_shape[1]
        picked_box_probs[:, 1] *= orig_im_shape[0]
        picked_box_probs[:, 2] *= orig_im_shape[1]
        picked_box_probs[:, 3] *= orig_im_shape[0]

        return picked_box_probs


class FaceLandmarkLocalization:
    def __init__(self):
        self.face_detector = FaceDetector640()
        self.face_mkp_predictor, self.face_mkp_predictor_input_handles, self.face_mkp_predictor_output_handles = create_predictor_un_combined(
            'models/face_landmark_localization')

    def inference_face_landmark(self, or_img, face_boxes):
        # 关键点检测
        face_list, location_list = self.face_crop(or_img, face_boxes)
        batch_size = len(face_list)
        self.face_mkp_predictor_input_handles[0].reshape([batch_size, 1, 60, 60])
        input_batch_img = np.concatenate((face_list[:]), axis=0).astype('float32')
        self.face_mkp_predictor_input_handles[0].copy_from_cpu(input_batch_img)
        self.face_mkp_predictor.run()
        output = self.face_mkp_predictor_output_handles[0].copy_to_cpu()
        point68_arr = np.array(output.reshape([-1, 68, 2]))
        for i in range(len(location_list)):
            point68_arr[i][:, 0] *= (location_list[i]['x2'] - location_list[i]['x1'])
            point68_arr[i][:, 0] += location_list[i]['x1']
            point68_arr[i][:, 1] *= (location_list[i]['y2'] - location_list[i]['y1'])
            point68_arr[i][:, 1] += location_list[i]['y1']
        return point68_arr

    def face_crop(self, or_img, face_boxes):
        # 将人脸裁剪出来
        # or_img: 原始图片
        or_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2RGB)
        face_list = []
        location_list = []
        for face in face_boxes:
            width = or_img.shape[1]
            height = or_img.shape[0]
            x1 = 0 if int(face[0]) < 0 else int(face[0])
            x2 = width if int(face[2]) > width else int(face[2])
            y1 = 0 if int(face[1]) < 0 else int(face[1])
            y2 = height if int(face[3]) > height else int(face[3])
            roi = or_img[y1:y2 + 1, x1:x2 + 1, :]
            gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            gray_img = cv2.resize(
                gray_img, (60, 60), interpolation=cv2.INTER_LINEAR)
            mean, std_dev = cv2.meanStdDev(gray_img)
            gray_img = (gray_img - mean[0][0]) / (0.000001 + std_dev[0][0])
            gray_img = np.expand_dims(gray_img, axis=0)
            gray_img = np.expand_dims(gray_img, axis=0)
            face_list.append(gray_img)
            location_list.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
        return face_list, location_list

    def inference(self, inputs):
        picked_box_probs = self.face_detector.inference(inputs)
        # print(picked_box_probs)
        if picked_box_probs is None:
            return None, None

        point68_list = self.inference_face_landmark(inputs, picked_box_probs)
        crops = []
        for landmarks in point68_list:
            landmarks = np.array(landmarks)
            # rotation angle
            left_eye_corner = landmarks[36]
            right_eye_corner = landmarks[45]
            radian = np.arctan(
                (left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

            # image size after rotating
            height, width, _ = inputs.shape
            cos = math.cos(radian)
            sin = math.sin(radian)
            new_w = int(width * abs(cos) + height * abs(sin))
            new_h = int(width * abs(sin) + height * abs(cos))

            # translation
            Tx = new_w // 2 - width // 2
            Ty = new_h // 2 - height // 2

            # affine matrix
            M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                          [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])
            image = cv2.warpAffine(inputs, M, (new_w, new_h), borderValue=(255, 255, 255))
            landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
            landmarks = np.dot(M, landmarks.T).T
            landmarks_top = np.min(landmarks[:, 1])
            landmarks_bottom = np.max(landmarks[:, 1])
            landmarks_left = np.min(landmarks[:, 0])
            landmarks_right = np.max(landmarks[:, 0])

            # expand bbox
            # top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
            # bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
            # left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
            # right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

            top = int(landmarks_top - 0.2 * (landmarks_bottom - landmarks_top))
            bottom = int(landmarks_bottom)
            left = int(landmarks_left)
            right = int(landmarks_right)

            # crop
            if bottom - top > right - left:
                left -= ((bottom - top) - (right - left)) // 2
                right = left + (bottom - top)
            else:
                top -= ((right - left) - (bottom - top)) // 2
                bottom = top + (right - left)

            image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

            h, w = image.shape[:2]
            left_white = max(0, -left)
            left = max(0, left)
            right = min(right, w - 1)
            right_white = left_white + (right - left)
            top_white = max(0, -top)
            top = max(0, top)
            bottom = min(bottom, h - 1)
            bottom_white = top_white + (bottom - top)
            image_crop[top_white:bottom_white + 1, left_white:right_white + 1] = image[top:bottom + 1,
                                                                                 left:right + 1].copy()

            crops.append(image_crop)
        # print('get crop cost :{}'.format(time.time() - st))
        return crops, picked_box_probs
