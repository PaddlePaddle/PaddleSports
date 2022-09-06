import glob
import subprocess
import datetime
import os
import argparse
import json
import copy
import cv2
# import moviepy
# from moviepy.editor import *
from tqdm import tqdm

# def get_video_duration(video_file_path):
#     """
#     Get video duration in secs at video_file_path.

#     :param video_file_path: path to the file, e.g. ./abc/v_123.mp4.
#     :return: a float number for the duration.
#     """
#     get_duration_cmd = ('ffprobe -i "%s" -show_entries format=duration ' +
#                         '-v quiet -of csv="p=0"')
#     output = subprocess.check_output(
#         get_duration_cmd % video_file_path,
#         shell=True,  # Let this run in the shell
#         stderr=subprocess.STDOUT)
#     return float(output)

def video_duration(video_file_path):
    # print(video_file_path)
    cap = cv2.VideoCapture(video_file_path)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = total / fps
    return float(duration)

def parse_gamestart_secs_line(line):
    return int(line.split('=')[-1])

def main(args):
    # sample filename /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/spain_laliga/2016-2017/2016-08-20 - 19-15 Barcelona 6 - 2 Betis/1_HQ.mkv
    files = sorted(glob.glob(os.path.join(args.labels_root, '**/Labels-v2.json'), recursive= True))
    print('found', len(files), 'label files.')
    # if not os.path.isdir(args.output_folder):
    #     os.mkdir(args.output_folder)

    # /mnt/big/multimodal_sports/soccer/SoccerNetv2/spain_laliga/2014-2015/2015-02-14 - 20-00 Real Madrid 2 - 0 Dep. La Coruna/Labels-v2.json
    json_list = []
    for filename in tqdm(files):
        with open(filename, 'r') as f:
            label_v2 = json.load(f)
        
        # "UrlLocal": "spain_laliga/2014-2015/2015-02-14 - 20-00 Real Madrid 2 - 0 Dep. La Coruna/",

        # video_filenames = [
        #     os.path.join(os.path.join(args.video_root, label_v2['UrlLocal']), '1_HQ.mkv'),
        #     os.path.join(os.path.join(args.video_root, label_v2['UrlLocal']), '2_HQ.mkv')]

        # print(filename)
        if label_v2['UrlLocal'] is None:
            continue
        video_filenames = [
            os.path.join(os.path.join(args.raw_videos_root, label_v2['UrlLocal']), '1_HQ.mkv'),
            os.path.join(os.path.join(args.raw_videos_root, label_v2['UrlLocal']), '2_HQ.mkv')]

        durations = [video_duration(video_filenames[0]),video_duration(video_filenames[1])]

        json_files = {}

        for video_index in [0, 1]:
            if video_index not in json_files:
                json_files[video_index] = {}

            parts = video_filenames[video_index].split('/')
            new_shortname_root = '.'.join(parts[-4:])

            duration = durations[video_index]
            for start in range(0, int(duration), args.clip_length):
                if start + args.clip_length > duration:
                    break
                if start == 0:
                    effective_start = 0
                else:
                    effective_start = start - 10

                start_time_str = str(datetime.timedelta(seconds=effective_start))
                start_time_str_filename = start_time_str.replace(':','-')
                # new_filename = os.path.join(args.clips_folder, new_shortname_root).replace('.mkv', '.{}.{}.mp4'.format(start,args.clip_length))

                new_filename = os.path.join(args.clips_folder, new_shortname_root).replace(
                    args.extension, f'{start_time_str_filename}.{start}.{args.clip_length}.{args.extension}')
                new_filename = new_filename.replace(" ", "_")

                single_json_data = {'path': new_filename, 'full_half_path': video_filenames[video_index],
                    'clip_length': args.clip_length, 'clip_start': start, 'clip_end': start + args.clip_length,
                    'label_v2_filename': filename, 
                    "annotations":[]}
                
                # print(new_filename)
                # assert '..' not in new_filename

                json_files[video_index][start // args.clip_length] = single_json_data

                # import ipdb; ipdb.set_trace()

            # effective_start = int(duration) - 80
            # start_time_str = str(datetime.timedelta(seconds=effective_start))
            # new_filename = os.path.join(args.output_folder, new_shortname_root).replace('.mkv', '.{}.{}.{}.mkv'.format(start_time_str.replace(':','-'),effective_start,args.clip_length))
            # single_json_data = {'path': new_filename, 'full_half_path': video_filenames[video_index],
            #     'clip_length': args.clip_length, 'clip_start': start, 'clip_end': start + args.clip_length,
            #     'label_v2_filename': filename, 
            #     "annotations":[]}

            # json_files[video_index][-1] = single_json_data

        # loop over annotations
        for annotation in label_v2['annotations']:
            # {
            #     "gameTime": "1 - 00:00",
            #     "label": "Kick-off",
            #     "position": "94",
            #     "team": "home",
            #     "visibility": "not shown"
            # }

            # {
            #     "gameTime": "2 - 16:39",
            #     "label": "Indirect free-kick",
            #     "position": "999583",
            #     "team": "away",
            #     "visibility": "visible"
            # },            
            game_half_index = int(annotation['gameTime'].split('-')[0]) - 1 # convert 1, 2 to 0, 1
            game_time_str = annotation['gameTime'].split('-')[1]
            game_time_secs = int(game_time_str.split(':')[0]) * 60 + int(game_time_str.split(':')[1])
            event_time_game_half_video = game_time_secs
            
            game_time_secs_index = event_time_game_half_video // args.clip_length
            for json_index in [-1, game_time_secs_index - 1, game_time_secs_index, game_time_secs_index + 1]:
                if json_index not in json_files[game_half_index]:
                    continue
                candidate_json = json_files[game_half_index][json_index]
                if event_time_game_half_video >= candidate_json['clip_start'] and event_time_game_half_video <= candidate_json['clip_end']:
                    annotation_for_clip = copy.deepcopy(annotation)
                    annotation_for_clip['event_time'] = event_time_game_half_video - candidate_json['clip_start']
                    # annotation['game_start_secs_in_video'] = game_start_secs_in_videos[game_half_index]
                    candidate_json['annotations'].append(annotation_for_clip)

        # write the json files
        for video_index in [0, 1]:
            json_files_half = json_files[video_index]
            for key in json_files_half:
                json_data = json_files_half[key]
                json_filename = json_data['path'].replace(args.extension, 'json')
                # json_filename = json_filename.replace('video_clips', 'annotations')
                with open(json_filename, 'w') as f:
                    json.dump(json_data, f, indent=4)
                # print('Wrote', json_filename)
                json_list.append(json_filename)
    
    with open(os.path.join(args.clips_folder, 'json_list.txt'), "w") as g:
        for json_filename in json_list:
            g.write(json_filename + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_videos_root', type=str, required = True)
    parser.add_argument('--labels_root', type=str, required = True)
    parser.add_argument('--clips_folder', type=str, required = True)
    parser.add_argument('--clip_length', type=int, default = 10)
    parser.add_argument('--extension', type=str, default = 'mp4')

    args = parser.parse_args()
    main(args)

