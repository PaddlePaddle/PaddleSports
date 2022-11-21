import numpy as np
from loadmodel import *
import paddle, cv2
from paddle.vision import transforms
import skimage.transform as trans


class BlazeFace:
    def __init__(self):
        self.blaze_predictor, self.blaze_input_handles, self.blaze_output_handles = create_predictor(
            './model/blazeface_1000e/model')
        self.mean = [123, 117, 104]
        self.std = [127.502231, 127.502231, 127.502231]
        self.trans = transforms.Compose([
            transforms.Transpose(order=(2, 0, 1)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

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


    def infer(self, inputs):
        # input
        input_img_size = inputs.shape
        # image shape
        input_size = np.array([input_img_size[2],input_img_size[3]]).reshape([1,2]).astype('float32')
        print(input_size.shape)
        self.blaze_input_handles[0].reshape([1,2])
        self.blaze_input_handles[0].copy_from_cpu(input_size)
        # inputs
        self.blaze_input_handles[1].reshape([input_img_size[0], 3, input_img_size[2], input_img_size[3]])
        self.blaze_input_handles[1].copy_from_cpu(inputs)
        # scale factor
        self.blaze_input_handles[2].reshape([1,2])
        self.blaze_input_handles[2].copy_from_cpu(np.array([1.,1.]).astype('float32'))
        # run predictor
        self.blaze_predictor.run()
        # get outputs
        op1 = self.blaze_output_handles[0].copy_to_cpu()  # type numpy.ndarray
        op2 = self.blaze_output_handles[1].copy_to_cpu()
        op3 = self.blaze_output_handles[2].copy_to_cpu()
        # [n,emb]-->[emb]
        # print(op1)
        # print(op2)
        # print(op3)

    def det_face(self, img):
        inputs = self.trans(img)[np.newaxis, :].astype('float32')
        self.infer(inputs)
        pass


if __name__ == '__main__':
    blaze = BlazeFace()
    img = cv2.imread('test.jpg')
    print(img.shape)
    blaze.det_face(img)
