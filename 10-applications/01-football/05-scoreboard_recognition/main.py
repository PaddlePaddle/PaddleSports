# output raw json
# per frame ocr result

import argparse
from SoccerBoard import SoccerBoard

import json
def str2sec(x):
    m, s = x.strip().split(':')  # .split()函数将其通过':'分隔开，.strip()函数用来除去空格
    return int(m) * 60 + int(s)  # int()函数转换成整数运算
def sec2str(sec:int):
    minute_x = int(sec//60)
    sec_x = int(sec - minute_x*60)
    return str(minute_x)+":"+str(sec_x)
def get_video_time_from_game_time(json_filename, game_time):

    if isinstance(game_time,str):
        game_time = str2sec(game_time)
    # with open(json_filename,"r+") as f:
    f = open(json_filename,"r+") 
    txt = f.read()
    # print(txt)
    video_info = json.loads(txt)['video_processed_infomation']
    # print(video_info)
    time_info = [{'game_teams': [], 'game_time': '00:00', 'game_scores': '0-0', 'video_time(s)': "0"}]
    for i in video_info:
        if i["processed_event"] == "game_begin":
            time_info.append(i)
        if i["processed_event"] == "time_jump":
            time_info.append(i)
    print(time_info)
    for step,i in enumerate(time_info):
        a = str2sec(i["game_time"])
        print(a)
        if game_time < str2sec(time_info[1]["game_time"]):
            b = float(time_info[1]["video_time(s)"]) - ( str2sec(time_info[1]["game_time"]) - game_time)
            break
        if a>game_time:
            if (game_time-str2sec(time_info[step-1]["game_time"])) <(float(time_info[step]["video_time(s)"]) - float(time_info[step-1]["video_time(s)"])):
                b = float(time_info[step-1]["video_time(s)"])+game_time - str2sec(time_info[step-1]["game_time"])
                # print(b)
                break
            else:
                b = float(time_info[step]["video_time(s)"]) - ( str2sec(time_info[step]["game_time"]) - game_time)
                break
            
    f.close()
    c = sec2str(b)

    return c 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str,default="myvideo.mp4")
    parser.add_argument('--team_list_file', type=str, help = "List of team names in text",default="team_name.txt")
    parser.add_argument('--output_dir', type=str,default="")
    parser.add_argument("--extract_frames_fps",type=int,default=2)
    parser.add_argument("--seq_second",type=int,default=10,help="粗粒度间隔seq_second秒判别视频中记分牌出现时间")
    # main()

    soccer_board = SoccerBoard(parser.parse_args())
    soccer_board.run()
