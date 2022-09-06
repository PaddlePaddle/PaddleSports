from paddle.vision.transforms import Resize
import paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from visualdl import LogWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2


class Dataset(paddle.io.Dataset):
    def __init__(self,args,is_train = True):
        self.data_dir = args.data_dir
        train_csv = pd.read_csv(self.data_dir+'train_split.csv', dtype='a')
        val_csv = pd.read_csv(self.data_dir+'val_split.csv', dtype='a')

        self.train_data = train_csv['filename']
        self.valid_data = val_csv['filename']
        self.train_label = train_csv['class']
        self.valid_label = val_csv["class"]
        self.class2id = {
            "AmericanFootball":0,
            "Basketball":1,
            "BikeRacing":2,
            "CarRacing":3,
            "Fighting":4,
            "Hockey":5,
            "Soccer":6,
            "TableTennis":7,
            "Tennis":8,
            "Volleyball":9
        } 
        if args.is_train == 1:
            self.is_train = True
        else:
            self.is_train = False
        if is_train == False:
            self.is_train = False
        self.transform = Resize(size = (360,640)) 
        if self.is_train == True:
            self.size = len(self.train_data)
        else:
            self.size = len(self.valid_data)
        
    @staticmethod
    def loader(path):
        return cv2.cvtColor(cv2.imread(path, flags=cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    def __getitem__(self, index):
        if self.is_train == True:
            one_img = self.loader (self.data_dir+self.train_data[index])
            one_label = self.class2id[self.train_label[index]]
        else:
            one_img = self.loader(self.data_dir+self.valid_data[index])
            one_label = self.class2id[self.valid_label[index]]     
        one_img = self.transform(one_img)
        return one_img,one_label

    def __len__(self):
        return self.size