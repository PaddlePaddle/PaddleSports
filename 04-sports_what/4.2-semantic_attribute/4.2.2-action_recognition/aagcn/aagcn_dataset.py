import numpy as np
import random
import paddle


class AutoPadding(object):
    """
    为了处理帧数问题，window_size为最后你想要的帧数，如果现有帧数小于window_size，就在现有帧前后补0,如果是现有帧数大于window_size,就在现有帧上抽取
    Sample or Padding frame skeleton feature.
    Args:
        window_size: int, temporal size of skeleton feature.
        random_pad: bool, whether do random padding when frame length < window size. Default: False.
    """

    def __init__(self, window_size, random_pad=False):
        self.window_size = window_size
        self.random_pad = random_pad

    def get_frame_num(self, data):
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if tmp > 0:
                T = i + 1
                break
        return T

    def __call__(self, results):
        # data = results['data']
        data = results

        C, T, V, M = data.shape
        T = self.get_frame_num(data)
        if T == self.window_size:
            data_pad = data[:, :self.window_size, :, :]
        elif T < self.window_size:
            begin = random.randint(0, self.window_size -
                                   T) if self.random_pad else 0
            data_pad = np.zeros((C, self.window_size, V, M))
            data_pad[:, begin:begin + T, :, :] = data[:, :T, :, :]
        else:
            if self.random_pad:
                index = np.random.choice(T, self.window_size,
                                         replace=False).astype('int64')
            else:
                index = np.linspace(0, T-1, self.window_size).astype("int64")
            data_pad = data[:, index, :, :]

        # results['data'] = data_pad
        # return results
        return data_pad

'''
来自：https://github.com/PaddlePaddle/PaddleVideo/blob/2f10ee26e232b83809ac4f11ac76b30fa262a86e/paddlevideo/loader/pipelines/skeleton_pipeline.py
默认是使用joint模式就是使用关节坐标，如果只设置bone = True,那么就是使用骨骼长度与方向
'''
class SketeonModalityTransform(object):
    """
    Sketeon Crop Sampler.
    Args:
        crop_model: str, crop model, support: ['center'].
        p_interval: list, crop len
        window_size: int, sample windows size.
    """

    def __init__(self, bone = False, motion = False, joint=True, graph='fsd10'):

        self.joint = joint
        self.bone = bone
        self.motion = motion
        self.graph = graph
        if self.graph == "ntu_rgb_d":
            self.bone_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                               (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                               (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                               (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                               (22, 23), (21, 21), (23, 8), (24, 25), (25, 12))
        elif self.graph == 'fsd10':
            self.num_node = 25
            self.bone_pairs  = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                             (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                             (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                             (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                             (21, 14), (19, 14), (20, 19)]

        
        else:
            raise NotImplementedError

    def __call__(self, results):
        if self.joint:
            return results
        data_numpy = results
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.bone_pairs:
                bone_data_numpy[:, :, v1 -
                                1] = data_numpy[:, :, v1 -
                                                1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.motion:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        results = data_numpy
        return results


class Dataset(paddle.io.Dataset):
    def __init__(self,args,is_train = None):
        data_file = args.data_file
        label_file= args.label_file
        data = np.load(data_file).astype("float32") #[2922, 3, 2500, 25, 1]
        label = np.load(label_file)
                #每7个取一个当验证集
        
        train_index = []
        valid_index= []
        for i in range(label.shape[0]):
            if i%7 !=1:
                train_index.append(i)
            else:
                valid_index.append(i)
        train_index =np.array(train_index).astype("int64")
        valid_index = np.array(valid_index).astype("int64")
        self.autopad = AutoPadding(window_size= args.window_size)
        self.train_data = data[train_index,:,:,:,:]
        self.valid_data = data[valid_index,:,:,:,:]
        self.train_label = label[train_index]
        self.valid_label = label[valid_index]
        self.is_train = args.is_train
        if is_train is not None:
            self.is_train = is_train
        if self.is_train == True:
            self.size = len(self.train_data)
        else:
            self.size = len(self.valid_data)
        if args.data_mode == "joint":
            self.sketeon = SketeonModalityTransform(bone = False, motion = False, joint=True)
        elif args.data_mode == "bone":
            self.sketeon = SketeonModalityTransform(bone = True, motion = False, joint=False)

    def __getitem__(self, index):
        if self.is_train == True:
            one_row = self.train_data[index]
            one_label = self.train_label[index]
        else:
            one_row = self.valid_data[index]
            one_label = self.valid_label[index]
        one_row = one_row[:2, :, :, :]#舍弃置信度
        one_row = self.autopad(one_row).astype("float32")
        one_row = self.sketeon(one_row)
        return one_row,one_label

    def __len__(self):
        return self.size