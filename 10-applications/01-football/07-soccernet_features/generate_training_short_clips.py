import glob
import subprocess
import datetime
import os
import argparse

def get_video_duration(video_file_path):
    """
    Get video duration in secs at video_file_path.
    :param video_file_path: path to the file, e.g. ./abc/v_123.mp4.
    :return: a float number for the duration.
    """
    get_duration_cmd = ('ffprobe -i "%s" -show_entries format=duration ' +
                        '-v quiet -of csv="p=0"')
    output = subprocess.check_output(
        get_duration_cmd % video_file_path,
        shell=True,  # Let this run in the shell
        stderr=subprocess.STDOUT)
    return float(output)

def main(args):
    # sample filename /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data/spain_laliga/2016-2017/2016-08-20 - 19-15 Barcelona 6 - 2 Betis/1_HQ.mkv
    files = sorted(glob.glob(os.path.join(args.input_folder, '**/*_HQ.mkv'), recursive= True))
    if not os.path.isdir(args.clips_folder):
        os.mkdir(args.output_folder)

    # import ipdb;ipdb.set_trace()

    for filename in files:
        # make necessary folders
        parts = filename.split('/')
        new_shortname_root = '.'.join(parts[-4:])

        duration = get_video_duration(filename)
        # for start in range(0, int(duration), args.clip_length):
        start = 0
        while start < duration:
            if start + args.clip_length > duration:
                break
            if start == 0:
                effective_start = 0
                # 80 s
            else:
                effective_start = start - 10

            start_time_str = str(datetime.timedelta(seconds=effective_start))

            start_time_str_filename = start_time_str.replace(':','-')
            new_filename = os.path.join(args.clips_folder, new_shortname_root).replace('.mkv', f'.{start_time_str_filename}.{start}.{args.clip_length}.{args.extension}')
            new_filename = new_filename.replace(" ", "_")

            # current_filename = os.path.join(args.clips_folder, new_shortname_root).replace('.mkv', 
            #     '.{}.{}.{}.{}'.format(start_time_str.replace(':','-'), effective_start, args.clip_length, 'mkv'))

            # command = f"mv \'{current_filename}\' {new_filename}"
            # print(command)

            command = f'ffmpeg -ss {start_time_str} -i "{filename}" \
                -vf scale=456x256 -map 0:v -map 0:a -c:v libx264 -c:a aac -strict experimental -b:a 98k \
                -t {str(datetime.timedelta(seconds=args.clip_length))} "{new_filename}"'

            print(command)

            start += args.clip_length

        # last clip
        # effective_start = int(duration) - 80
        # start_time_str = str(datetime.timedelta(seconds=effective_start))
        # new_filename = os.path.join(args.output_folder, new_shortname_root).replace('.mkv', '.{}.{}.{}.mkv'.format(start_time_str.replace(':','-'),effective_start,args.clip_length))

        # command = f'ffmpeg -ss {start_time_str} -i "{filename}" \
        #     -vf scale=456x256 -c:a aac -strict experimental -b:a 98k \
        #     -t {str(datetime.timedelta(seconds=args.clip_length))} "{new_filename}" -y'
        # print(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required = True)
    parser.add_argument('--clips_folder', type=str, required = True)
    parser.add_argument('--clip_length', type=int, default = 10)
    parser.add_argument('--extension', type=str, default = 'mkv')


    args = parser.parse_args()
    main(args)
