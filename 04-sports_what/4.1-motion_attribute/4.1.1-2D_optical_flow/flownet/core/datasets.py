import numpy as np
import paddle
from paddle.vision.transforms import functional as F
import cv2
from paddle.io import Dataset
import os
import random
from glob import glob
import os.path as osp
from frame_utils import *
from augmentor import *


class FlowDataset(Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        # 这个sparse好像是专门针对KITTI数据集的一个指标
        if aug_params is not None:
            self.augmentor = FlowAugmentor(**aug_params)
            # if sparse:
            #     self.augmentor = SparseFlowAugmentor(**aug_params)
            # else:
                # self.augmentor = FlowAugmentor(**aug_params)
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
    
    def __getitem__(self, index):
        flow = read_gen(self.image_list[index][2])
        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])
        
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor != None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)
        flow = F.to_tensor(flow)
        
        flow = paddle.cast(flow, dtype='float32')


        # if valid is not None:
        #     valid = torch.from_numpy(valid)
        # else:
        valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        valid = paddle.cast(valid, dtype='float32')

        return img1, img2, flow, valid
    
    
    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v* self.flow_list
        return self


    def __add__(self, x):
        self.flow_list += x.flow_list
        self.image_list += x.flow_list
        return self


    def __len__(self):
        return len(self.image_list)


def changeI2S(i):
    if i<10: return '000' + str(i)
    elif i<100: return '00' + str(i)
    elif i<1000: return '0' + str(i)
    else: return str(i)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/aistudio/work/datasets/Sintel/', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)
        if split == 'test':
            self.is_test = True
        
        # 原版的读取方式看不明白，而且报错，重新按照paddle规则写了一下，列表元素是列表，包括img1 img2 flow 的地址 
        for scene in os.listdir(image_root):
            img_list = os.listdir(osp.join(image_root, scene))
            for i in range(len(img_list)-1):
                self.image_list.append(['%s%s/%s/%s/frame_%s.png' % (root,split,dstype,scene, changeI2S(i+1)), '%s%s/%s/%s/frame_%s.png' % (root,split,dstype,scene, changeI2S(i+2)), '%s%s/flow/%s/frame_%s.flo' % (root,split,scene, changeI2S(i+1))])


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/home/aistudio/work/datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)
        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('/home/aistudio/work/chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='test' and xid==2):
                # self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1], flows[i]] ]


def fetch_dataloader(stage, batch_size, image_size, TRAIN_DS, num_workers, split='training', shuffle=True):
    if stage == 'sintel':
        aug_params = {'crop_size': image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split=split, dstype='clean')
        sintel_final = MpiSintel(aug_params, split=split, dstype='final')      
        train_dataset = sintel_clean + sintel_final
    
    elif stage == 'chairs':
        if split == 'training':
            aug_params = {'crop_size': image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        else:
            aug_params = None
        train_dataset = FlyingChairs(aug_params, split=split)
    
    # 要注意一下下面的num_workers,cpu训练的话最好使用1，至尊版的话应该可以上4
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    
    if 'train' in split:
        print('Training with %d image pairs' % len(train_dataset))
    else:
        print('Evaluate with %d image pairs' % len(train_dataset))
    return train_loader












