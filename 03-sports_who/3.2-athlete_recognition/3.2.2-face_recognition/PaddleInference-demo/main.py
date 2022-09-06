from tqdm import tqdm
import os
from PIL import ImageDraw, ImageFont, Image
import pickle
from mtcnn import *
import paddle.vision.transforms as transforms


def l2_norm(input, axis=0):
    norm = np.linalg.norm(input, 2, axis, True)
    output = np.divide(input, norm)
    return output


class FaceRecognizer:
    def __init__(self,model_path='./model/Backbone'):
        self.threshold = 0.9

        # image preprocess
        self.mean = [127.5, 127.5, 127.5]
        self.std = [127.5, 127.5, 127.5]
        self.trans = transforms.Compose([
            transforms.Resize([128, 128]),  # smaller side resized
            transforms.CenterCrop([112, 112]),
            transforms.Transpose(order=(2, 0, 1)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # face detect and align
        self.face_align = MTCNN()
        # self.face_align = FaceLandmarkLocalization()

        # create predictor
        self.face_eval, self.face_eval_input_handles, self.face_eval_output_handles = self.create_predictor(model_path)

        # folders of face images
        self.folder_path = 'demo/FaceDatabase'

        # a file saved extracted
        self.extracted_feature = 'face_data.fdb'

        # get extracted_feature or extracted feature from face database
        self.face_database = self.add_face2database()

    def add_face2database(self):
        # extract face feature for per identity and save
        face_db = {}
        assert os.path.exists(self.folder_path), 'folder_path {} not exist'.format(self.folder_path)

        for path in tqdm(os.listdir(self.folder_path)):
            name = os.path.basename(path).split('.')[0]
            features = []
            images_folder_path = os.path.join(self.folder_path,path)
            face_list = os.listdir(images_folder_path)

            for _path in face_list:
                image_path = os.path.join(images_folder_path, _path)
                img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
                imgs, _, _ = self.face_align.inference(img)

                if imgs is None or len(imgs) > 1:
                    print('**** image %s does not meet requirements, skip ****' % image_path)
                    continue

                for img in imgs:
                    hflip_img = img.copy()
                    img = self.process(img)
                    # -->[1,3,input size,input size]
                    hflip_img = self.process(hflip_img, hflip=True)
                    features.append(l2_norm(self.infer(hflip_img)+self.infer(img)))

            if len(features) > 0:
                feature_ = np.vstack(features)
                feature_ = np.sum(feature_,axis=0)
                face_db[name] = feature_/len(features)

        # save features and identity
        with open(self.extracted_feature, "wb") as f:
            pickle.dump(face_db, f)

        print('extract face feature for per identity and save!')
        return face_db

    def load_face_data(self):
        # load face features and identity from data
        if not os.path.exists(self.extracted_feature):
            print('extracted_feature not exist!,try to Extract from database ')
            face_db = self.update_face_data()
            return face_db
        with open(self.extracted_feature, "rb") as f:
            face_db = pickle.load(f)
        print('finished loaded extracted_feature!')
        return face_db

    def process(self, img, hflip=False):
        # image preprocess

        if hflip:
            img = transforms.hflip(img)
        img = self.trans(img)
        return img[np.newaxis, :].astype('float32')

    @staticmethod
    def create_predictor(model_dir):
        # create predictor
        model_file = os.path.join(model_dir, '.pdmodel')
        params_file = os.path.join(model_dir, '.pdiparams')
        config = inference.Config()
        config.set_prog_file(model_file)
        config.set_params_file(params_file)
        config.use_gpu()
        config.enable_use_gpu(500, 0)
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

    def infer(self, imgs):
        # input
        input_img_size = imgs.shape
        self.face_eval_input_handles[0].reshape([input_img_size[0], 3, input_img_size[2], input_img_size[3]])
        self.face_eval_input_handles[0].copy_from_cpu(imgs)
        # run predictor
        self.face_eval.run()
        # get outputs
        features = self.face_eval_output_handles[0].copy_to_cpu()  # type numpy.ndarray
        # [n,emb]-->[emb]
        return features[0]

    def recognition(self, img, use_cos=False):
        imgs, boxes, landmarks = self.face_align.inference(img)
        if imgs is None:
            return None, None, None, None
        features = []
        for img in imgs:
            h_img = img.copy()
            img = self.process(img)
            h_img = self.process(h_img, hflip=True)
            features.append(self.infer(img) + self.infer(h_img))
        names = []
        probs = []
        for i in range(len(features)):
            feature = l2_norm(features[i])
            results_dict = {}
            for name in self.face_database.keys():
                feature1 = self.face_database[name]
                if use_cos:
                    prob = np.dot(feature, feature1) / (np.linalg.norm(feature) * np.linalg.norm(feature1))
                    results_dict[name] = prob
                else:
                    diff = np.subtract(feature, feature1)
                    dist = np.sum(np.square(diff), axis=0)
                    results_dict[name] = dist

            results = sorted(results_dict.items(), key=lambda d: d[1], reverse=use_cos)
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if use_cos:
                if prob > self.threshold:
                    name = result[0]
                    names.append(name)
                else:
                    names.append('?')
            else:
                if prob < self.threshold:
                    name = result[0]
                    names.append(name)
                else:
                    names.append('?')
        return boxes, names, probs, landmarks

    def add_text(self, img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('simsun.ttc', size)
        draw.text((left, top), text, color, font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # visualization
    def draw_face(self, img, boxes_c, names, probs, landmarks):
        if boxes_c is not None:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                name = names[i]
                prob = probs[i]
                landmark = landmarks[i]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                # landmark on face
                # cv2.circle(img,(landmark[0],landmark[5]),radius=2,color=(255,0,0),thickness=2)
                # cv2.circle(img, (landmark[6], landmark[1]), radius=2, color=(255, 0, 0), thickness=2)
                # cv2.circle(img, (landmark[2], landmark[7]), radius=2, color=(255, 0, 0), thickness=2)
                # cv2.circle(img, (landmark[3], landmark[8]), radius=2, color=(255, 0, 0), thickness=2)
                # cv2.circle(img, (landmark[4], landmark[9]), radius=2, color=(255, 0, 0), thickness=2)

                # face drawing rectangle
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 2)

                # font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
                # img = cv2.putText(img, name, (corpbbox[0], corpbbox[1]), font, 0.5, (0, 255, 0), 1)
                text = name + ':' + str(prob)[:4]
                img = self.add_text(img, text, corpbbox[0], corpbbox[1] + 25, color=(255, 255, 0), size=30)
        cv2.imshow("visualization", img)
        cv2.waitKey(1)
        return img


if __name__ == '__main__':
    work_path = os.getcwd()
    save_video = False
    print(work_path)
    test = FaceRecognizer(model_path='./model/Backbone')
    test.load_face_data()
    cap = cv2.VideoCapture('./demo/test.avi')
    if save_video:
        # video = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 20, (1280, 720))
        video = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (1280, 720))
    ret = True
    name_map = {}

    while ret:
        ret, img = cap.read()
        if ret:
            boxes, names, probs, landmarks = test.recognition(img)
            # print(names, probs)
            if boxes is not None:
                for i in range(len(names)):
                    if name_map.get(names[i]) is None:
                        name_map[names[i]] = [probs[i],1]
                    else:
                        name_map[names[i]][0] += probs[i]
                        name_map[names[i]][1] += 1
            img = test.draw_face(img, boxes, names, probs, landmarks)
            if save_video:
                video.write(img)

    for key,value in name_map.items():
        print('name {} avg prob {} ,times {}'.format(key,value[0]/value[1],value[1]))
    if save_video:
        video.release()
