import paddle
import os, tqdm,sys
import numpy as np
from paddle.vision import transforms
import paddle.vision as vision
from paddle.io import WeightedRandomSampler, Sampler


class DataSetVal(paddle.io.Dataset):
    def __init__(self,data_root, input_size, mean, std):
        super(DataSetVal, self).__init__()
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.images_path,self.label_list ,self.num_classes,self.class_count = self.read_data(data_root)
        # val transforms
        self.trans = transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
            transforms.CenterCrop([input_size[0],input_size[1]]),
            transforms.Transpose(order=(2, 0, 1)),
            transforms.Normalize(mean=self.mean,
                                 std=self.std),
        ])
        num_classes = len(self.class_count)
        weight_per_class = [0.] * num_classes
        total_sample = sum(self.class_count)
        for i in range(num_classes):
            weight_per_class[i] = self.class_count[i] / total_sample
        self.random_weight = weight_per_class
        self.class_mask = np.arange(num_classes)
        self.is_same = 3 # keep positive samples and negative samples form a team  1:1
        self.target_index = 0
        print('val dataset {} class num :{}  total sample:{}'.format(data_root,num_classes,total_sample))

    def __getitem__(self, index):
        if self.is_same % 4 == 0:
            class_index = np.random.randint(low=0,high=self.class_count[self.target_index]) # [0,n)
            _index = sum(self.class_count[:self.target_index])+class_index
            image = vision.image_load(self.images_path[_index], backend='cv2')
            input_data = paddle.to_tensor(self.trans(image),dtype='float32')
            label = paddle.to_tensor(self.label_list[_index],dtype='int64')
        else:
            self.target_index = np.random.choice(self.class_mask,p=self.random_weight)
            class_index = np.random.randint(low=0, high=self.class_count[self.target_index])  # [0,n)
            _index = sum(self.class_count[:self.target_index]) + class_index
            image = vision.image_load(self.images_path[_index], backend='cv2')
            input_data = paddle.to_tensor(self.trans(image))
            label = paddle.to_tensor(self.label_list[_index],dtype='int64')

        self.is_same += 1
        return input_data, label

    def read_data(self,data_root):
        count = []
        images_data = []
        images_label = []
        data_root = data_root
        class_file_list = os.listdir(data_root)
        num_classes = len(class_file_list)
        total_count = 0
        for i in tqdm.tqdm(range(len(class_file_list)), ncols=80):
            images_path = os.listdir(os.path.join(data_root, class_file_list[i]))
            count_value = 0
            for image_path in images_path:
                if image_path.endswith('.jpg'):
                    images_data.append(data_root + '/' + class_file_list[i] + '/' + image_path)
                    # vision.image_load(data_root + '/' + class_file_list[i] + '/' + image_path, backend='cv2')
                    images_label.append(i)
                    count_value += 1
                    total_count += 1
                    # sample n images
                    # if count_value>5:
                    #     break
            if count_value == 0:
                empty_path = os.path.join(data_root, class_file_list[i])
                print('\n an empty class folder: {} raise a error ,please check!'.format(empty_path))
                sys.exit()
            count.append(count_value)
        return images_data,images_label,num_classes,count


    def __len__(self):
        return len(self.images_path)


class DataSetNormal(paddle.io.Dataset):
    def __init__(self,data_root, input_size, mean, std,label_map=None):
        super(DataSetNormal, self).__init__()
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.images_path,self.label_list ,self.num_classes= self.read_data(data_root,label_map)
        print('{}  total_data:{}'.format(data_root,len(self.images_path)))
        self.trans = transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
            transforms.RandomCrop([input_size[0], input_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.Transpose(order=(2, 0, 1)),
            transforms.Normalize(mean=self.mean,
                                 std=self.std),
        ])
        print('Normal dataset {} class num :{}  total sample:{}'.format(data_root, self.num_classes,len(self.images_path)))

    def __getitem__(self, index):
        image = vision.image_load(self.images_path[index], backend='cv2')
        input_data = paddle.to_tensor(self.trans(image),dtype='float32')
        label = paddle.to_tensor(self.label_list[index],dtype='int64')
        # print(self.images_path[index][25:33]+'=='+str(label.numpy()))
        return input_data, label

    def read_data(self,data_root,label_map):
        count = []
        images_data = []
        images_label = []
        data_root = data_root
        class_file_list = os.listdir(data_root)
        num_classes = len(class_file_list)
        total_count = 0
        for i in tqdm.tqdm(range(len(class_file_list)), ncols=80):
            images_path = os.listdir(os.path.join(data_root, class_file_list[i]))
            count_value = 0
            for image_path in images_path:
                if image_path.endswith('.jpg'):
                    images_data.append(data_root + '/' + class_file_list[i] + '/' + image_path)
                    # vision.image_load(data_root + '/' + class_file_list[i] + '/' + image_path, backend='cv2')
                    if label_map is not None:
                        images_label.append(label_map[class_file_list[i]])
                    else:
                        images_label.append(i)
                    count_value += 1
                    total_count += 1
                    # sample n images
                    # if count_value>5:
                    #     break
            if count_value == 0:
                empty_path = os.path.join(data_root, class_file_list[i])
                print('\n an empty class folder: {} raise a error ,please check!'.format(empty_path))
                sys.exit()
            count.append(count_value)
        return images_data,images_label,num_classes


    def __len__(self):
        return len(self.images_path)


class BalancingSampler(Sampler):
    def __init__(self, data_source,batch_size,num_samples = 128000):
        super(Sampler,self).__init__()
        self.data_source = data_source
        self.weight_random_sampler = WeightedRandomSampler(
            weights=data_source.weight,
            num_samples=num_samples
        )
        print('BalancingSampler sample {} image per epoch'.format(num_samples))

    def __iter__(self):
        return iter(self.weight_random_sampler)

    def __len__(self):
        return self.data_source.len


class BalancingClassDataset(paddle.io.Dataset):
    def __init__(self, data_root, input_size, mean, std):
        super(BalancingClassDataset, self).__init__()
        self.input_size = input_size
        self.data_root = data_root
        self.mean = mean
        self.std = std
        self.trans = transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
            transforms.RandomCrop([input_size[0],input_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.Transpose(order=(2, 0, 1)),
            transforms.Normalize(mean=self.mean,
                                 std=self.std),
        ])
        self.weight, self.image_data, self.image_label, self.num_classes,self.label_map = self.data_prepare()
        self.sampler_num = int(1/min(self.weight))
        print('BalancingSample dataset {} class num :{}  total sample:{}'.format(data_root, self.num_classes,len(self.image_data)))
        self.len = len(self.image_data)

    def __getitem__(self, index):
        image = vision.image_load(self.image_data[index], backend='cv2')
        input_data = paddle.to_tensor(self.trans(image),dtype='float32')
        label = paddle.to_tensor(self.image_label[index],dtype='int64')
        return input_data, label

    def __len__(self):
        return self.len

    def data_prepare(self):
        '''
            Make a vector of weights for each image in the dataset, based
            on class frequency. The returned vector of weights can be used
            to create a WeightedRandomSampler for a DataLoader to have
            class balancing when sampling for a training batch.
                images - torchvisionDataset.imgs
                nclasses - len(torchvisionDataset.classes)
            https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
        '''
        train_dataset = os.path.join(os.path.dirname(self.data_root),os.path.basename(self.data_root) + '.txt')
        label_map = {}
        # save a label map for resuming model weights
        write_label_map = False
        if os.path.exists(train_dataset):
            with open(train_dataset,'r') as f:
                _data = f.readlines()
            for line in _data:
                folder_name,label = line.split(' ')
                label_map[folder_name] = int(label.replace('\n',''))
            print('finished loaded label map')
        else:
            write_label_map = True
            print('not available lable map ,generate it')
        count = []
        images_data = []
        images_label = []
        data_root = self.data_root
        class_file_list = os.listdir(data_root)
        for i in tqdm.tqdm(range(len(class_file_list)), ncols=80):
            images_path = os.listdir(os.path.join(data_root, class_file_list[i]))
            if write_label_map:
                assert label_map.get(class_file_list[i]) is None,'error ,label Repetitive!'
                label_map[class_file_list[i]] = i
            count_value = 0
            for image_path in images_path:
                images_data.append(data_root + '/' + class_file_list[i] + '/' + image_path)
                # vision.image_load(data_root + '/' + class_file_list[i] + '/' + image_path, backend='cv2')
                images_label.append(label_map[class_file_list[i]])
                count_value += 1
                # sample n images
                # if count_value>1:
                #     break
            if count_value ==0:
                empty_path = os.path.join(data_root, class_file_list[i])
                assert count_value ==0,'\n an empty class folder: {} raise a error ,please check!'.format(empty_path)
            count.append(count_value)
        num_classes = len(class_file_list)
        num_sample = len(images_label)
        weight_per_class = [0.] * num_classes
        for i in range(num_classes):
            weight_per_class[i] = 1 / (num_classes * count[i])
        weight = [0] * num_sample
        for idx, val in enumerate(images_label):
            weight[idx] = weight_per_class[images_label[idx]]
        if write_label_map:
            with open(train_dataset,'w') as f:
                for key,value in label_map.items():
                    f.write('{} {}\n'.format(key,value))
            print('saved label map in {}'.format(train_dataset))
        return weight, images_data, images_label, num_classes,label_map


if __name__ == '__main__':
    data = DataSetNormal('data/casia', [112, 112], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data.__getitem__(1)
