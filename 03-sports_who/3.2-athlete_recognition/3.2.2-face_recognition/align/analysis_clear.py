import os
import cv2
from multiprocessing import Process
import argparse


def rm_nonetype_img(image_path_list):
    count = 0
    for img in image_path_list:
        if count % 10000 == 0:
            print('check image {}'.format(count))
        image = cv2.imread(img)
        try:
            _shape = image.shape
        except:
            os.system('rm {}'.format(img))
            print('rm {}'.format(img))
        count += 1

    print('a cpu task finished checked total  {}images'.format(count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="clera nonotype image")
    parser.add_argument("-source_root", "--source_root", help="specify your source dir", default="./casia", type=str)
    parser.add_argument("-cpu_num", "--cpu_num", help="cpu_num", default=4, type=int)
    args = parser.parse_args()
    data_path = args.source_root
    class_folder_list = os.listdir(data_path)
    image_path_list = []
    sample_of_classes = []
    print('分类文件夹总数{}'.format(len(class_folder_list)))
    for _path in class_folder_list:
        _classes_folder_path = os.path.join(data_path, _path)
        if not os.path.isdir(_classes_folder_path):
            continue
        image_list = os.listdir(_classes_folder_path)
        count = 0
        for __path in image_list:
            image_path_list.append(os.path.join(_classes_folder_path, __path))
            count += 1
        if count == 0:
            print('warning: folder {} image num is 0'.format(_classes_folder_path))
        sample_of_classes.append(count)

    print('per folder images num:')
    print(sample_of_classes)
    total_num = sum(sample_of_classes)
    mean_sample_of_classes = total_num / len(sample_of_classes)
    print('总共的照片数{}  最大样本数:{},最小样本数:{} 平均样本数:{}'.format(total_num, max(sample_of_classes), min(sample_of_classes),mean_sample_of_classes))

    cpu_num = args.cpu_num
    part_size = total_num // cpu_num
    index = 0
    for i in range(cpu_num):
        if i == cpu_num - 1:
            head = index
            clear_task = Process(target=rm_nonetype_img, args=([image_path_list[head:]]))
            clear_task.start()
            print('clear_task run in cpu{} data[{}:]'.format(i, head))
        else:
            head = index
            index = index + part_size
            clear_task = Process(target=rm_nonetype_img, args=([image_path_list[head:index]]))
            clear_task.start()
            print('clear_task run in cpu{} data[{}:{}]'.format(i, head, index))