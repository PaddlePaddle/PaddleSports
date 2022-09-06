import json
import glob
import os
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter

def main(args):
    labels_folder = args.clips_folder
    output_folder = args.output_folder
    label_file_dir = os.path.join(output_folder, 'label_list.txt')
    annotation_file_dir = os.path.join(output_folder, 'annotation.txt')

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # files = sorted(glob.glob(os.path.join(labels_folder, '*.json'), recursive= True))
    with open(os.path.join(labels_folder, 'json_list.txt'), 'r') as f:
        lines = f.readlines()
    files = [line.strip() for line in lines]

    labels = []
    num_labels_list = []
    for filename in tqdm(files):
        # print(filename)
        with open(filename, 'r') as f:
            data = json.load(f)
            num_labels = len(data['annotations'])
            num_labels_list.append(num_labels)
            for annotation in data["annotations"]:
                labels.append(annotation["label"])

    print('len(num_labels_list)', len(num_labels_list))
    counter = Counter(num_labels_list)
    print('num_labels_list counter', counter)

    labels = list(dict.fromkeys(labels))
    labels.append('background')
    print('number of labels', len(labels))

    dict_labels = {labels[i] : i for i in range(len(labels))}

    # this file contains the text to label index mapping
    with open(label_file_dir, "w") as label_file:
        for i, label in enumerate(labels):
            label_file.write('{} {}\n'.format(i, label))

    print('writing annotations file...')
    with open(annotation_file_dir, "w") as annotation_file:
        for filename in tqdm(files):
            with open(filename, "r") as f:
                data = json.load(f)
                if len(data['annotations']) == 1:
                    annotation_file.write('{} {}\n'.format(data['path'], dict_labels[data["annotations"][0]["label"]]))
                elif len(data['annotations']) == 0:
                    annotation_file.write('{} {}\n'.format(data['path'], dict_labels['background']))
                elif len(data['annotations']) >= 2:
                    for i in range(len(data['annotations'])):
                        annotation_file.write('{} {}\n'.format(data['path'], dict_labels[data["annotations"][i]["label"]]))
                    # annotation_file.write('{} {}\n'.format(data['path'], dict_labels[data["annotations"][1]["label"]]))
                # else:
                #     print('warning: {} has more than 2 labels'.format(data['path']), data['annotations'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips_folder', type=str, required = True, help = 'Where json annotation files are')
    parser.add_argument('--output_folder', type=str, required = True, help = 'Where label_list.txt and annotation.txt are saved')

    args = parser.parse_args()
    main(args)