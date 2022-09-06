import glob
import numpy as np
import argparse
import os

def main(args):
    train_val_test_path = args.splits_folder
    train_val_test_files = glob.glob(train_val_test_path + '/*.npy')
    
    train_set = set()
    val_set = set()
    test_set = set()

    def match_relative_path_to_video_prefix(relative_path):
        video_prefix = '.'.join(relative_path.split('/'))
        video_prefix = video_prefix.replace(' ', '_')
        return video_prefix
    
    def video_name_to_key(video_name):
        return video_name.split('/')[-1].split('_HQ')[0][:-2]

    for filename in train_val_test_files:
        game_names = np.load(filename)

        # example game_name
        # 'england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal'

        if 'test' in filename.lower():
            test_set = set(game_names)
        elif 'train' in filename.lower():
            train_set = set(game_names)
        elif 'valid' in filename.lower():
            val_set = set(game_names)
        else:
            print("Unrecognized list type", filename)

    train_set = [match_relative_path_to_video_prefix(item) for item in list(train_set)]
    val_set = [match_relative_path_to_video_prefix(item) for item in list(val_set)]
    test_set = [match_relative_path_to_video_prefix(item) for item in list(test_set)]

    train_list = open(os.path.join(args.clips_folder, 'train.list'), 'w')
    val_list = open(os.path.join(args.clips_folder, 'val.list'), 'w')
    test_list = open(os.path.join(args.clips_folder, 'test.list'), 'w')

    with open(args.annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            video_name = line.split(' ')[0]
            video_name = video_name_to_key(video_name)
            if video_name in train_set:
                train_list.write(line)
            elif video_name in val_set:
                val_list.write(line)
            elif video_name in test_set:
                test_list.write(line)
            else:
                print('video_name not found', video_name)

    train_list.close()
    val_list.close()
    test_list.close()

    # import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', required=True)
    parser.add_argument('--clips_folder', type=str, required = True)
    parser.add_argument('--splits_folder', type=str, required = True, help = 'Folder containing the Soccernet match splits.')

    args = parser.parse_args()
    main(args)