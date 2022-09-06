import os
from PIL import Image
import numpy as np
import cv2
from PaddleOCR import PaddleOCR, draw_ocr
import os
import paddle
import paddle.nn as nn
import json
import math
import re
import paddle.nn.functional as F

class   SoccerBoard(object) :
    def __init__(self, args):
        self.video_file = args.video_file
        self.team_list_file = args.team_list_file
        if args.output_dir:
            self.extract_frames_file = args.output_dir
        else:
            self.extract_frames_file = self.video_file.split(".")[0]

        self.get_team_list()
        self.extract_frames_fps = args.extract_frames_fps
        self.seq_second = args.seq_second
        self.time_pattern = r"\d\d[:,：,.]\d\d"
        self.point_pattern = r"[\d,O][ ,:,：,-]+[\d,O]"

        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", cls_image_shape='3, 96, 384', det_db_thresh=0.2,
                        det_db_box_thresh=0.3)
    def get_one_image_board_mask(self,infile):
        def get_dominant_colors(infile):
            image = Image.open(infile)
            # 缩小图片，否则计算机压力太大
            small_image = image.resize((80, 80))
            result = small_image.convert(
                "P", palette=Image.ADAPTIVE, colors=10
            )
            # # 找到主要的颜色
            palette = result.getpalette()
            # print(len(palette))
            color_counts = sorted(result.getcolors(), reverse=True)
            colors = list()

            for i in range(4):
                palette_index = color_counts[i][1]
                dominant_color = palette[palette_index * 3: palette_index * 3 + 3]
                colors.append(tuple(dominant_color))

            return colors

        img = cv2.cvtColor(cv2.imread(infile, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        # img = cv2.imread(image_path, cv2.COLOR_BGR2HSV)
        # print(img.shape)
        main_color = get_dominant_colors(infile)
        # print(main_color)
        mask0 = np.zeros_like(img[:, :, 0])
        for i in range(len(main_color)):
            color_Low = np.array(main_color[i]) - 20
            color_high = np.array(main_color[i]) + 20
            # print("color_low",color_Low)
            # print("color_high",color_high)
            mask = cv2.inRange(img, color_Low, color_high)  # 该像素在范围内为255，不在则为0
            mask0 += mask
            # break
        img_mask = 1 - mask0 / 255
        return img_mask

    def get_video_board_mask(self):
        folder_name = self.extract_frames_file
        img_mask = np.ones_like(cv2.imread(os.path.join(folder_name, self.frames_dirs[6]))[:, :, 0]).astype("float32")
        h, w = img_mask.shape
        # print(begin_video_time_fps)
        for k in range(4):
            index = self.begin_video_time_fps + 40 * k + 5
            img_mask0 = self.get_one_image_board_mask(os.path.join(folder_name, self.frames_dirs[index]))
            img_mask *= img_mask0
        img_mask = np.stack([img_mask, img_mask, img_mask], axis=-1)  # 得到mask
        img_mask = nn.MaxPool2D(kernel_size=5, stride=1, padding=2, data_format="NHWC")( \
            paddle.to_tensor(img_mask).unsqueeze(0)).squeeze(0).numpy()
        self.video_board_mask = img_mask
    def get_team_list(self):
        f = open(self.team_list_file, mode="r+",encoding="utf-8")
        self.team_list = [i.replace("\n", "") for i in f.readlines()]
        f.close()

    def extract_frames(self):
        '''
        out_folder name is default as the video_name
        '''
        fps = self.extract_frames_fps
        out_folder = self.extract_frames_file
        if os.path.exists(out_folder):
            os.system('rm -rf ' + out_folder + '/*')
            os.system('rm -rf ' + out_folder)
        os.makedirs(out_folder)
        cmd = 'ffmpeg -v 0 -i %s -r %d -q 0 %s/%s.jpg' % (self.video_file, fps,
                                                          out_folder, '%08d')
        os.system(cmd)
    def get_begin_time(self):
        self.frames_dirs = sorted(os.listdir(self.extract_frames_file))
        seq_second =self.seq_second
        fps = self.extract_frames_fps
        dirs1 = self.frames_dirs[::fps * seq_second]  # 每隔十秒选1张，毕竟1秒5帧
        ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory
        folder_name = self.extract_frames_file
        begin_video_time_fps = 0
        temp = 0

        # 粗粒度判断记分牌时间出现的视频时间
        for i in range(len(dirs1)):
            if dirs1[i].split(".")[-1] != "jpg":
                continue
            one_path = os.path.join(folder_name, dirs1[i])

            output = cv2.imread(one_path)

            result = ocr.ocr(output, cls=True)
            allwords_list = []
            # if result is not None:
            words_point_list = []
            for line in result:
                if line[1][1] > 0.8:  # 自己手工设置一个置信度阈值
                    # print(line)
                    allwords_list.append(line[1][0])
            one_image_str = " ".join(allwords_list)
            # if "：" in one_image_str or ":" in one_image_str:
            #     pattern = r"\d\d[:,：]\d\d"
            #     image_time = re.search(pattern, one_image_str, flags=0)
            #     if image_time is not None:
            #         temp +=1
            #         if temp == 2: #连续两次粗粒度都有时间元素
            #             begin_video_time_fps = i-1
            #             break
            # else:
            #     temp = 0
            if "：" in one_image_str or ":" in one_image_str:
                pattern = r"\d\d[:,：,.]\d\d"
                image_time = re.search(pattern, one_image_str, flags=0)
                if image_time is not None:
                    if begin_video_time_fps == 0:
                        begin_video_time_fps = i
                        break
        # print(begin_video_time_fps)
        # 细粒度判断记分牌时间出现的视频时间
        for j in range(begin_video_time_fps * fps * seq_second - fps * seq_second,
                       begin_video_time_fps * fps * seq_second):
            one_path = os.path.join(folder_name, self.frames_dirs[j])
            output = cv2.imread(one_path)
            result = ocr.ocr(output, cls=True)
            allwords_list = []
            # if result is not None:
            words_point_list = []
            for line in result:
                if line[1][1] > 0.8:  # 自己手工设置一个置信度阈值
                    # print(line)
                    allwords_list.append(line[1][0])
            one_image_str = " ".join(allwords_list)
            if "：" in one_image_str or ":" in one_image_str:
                pattern = r"\d\d[:,：]\d\d"
                image_time = re.search(pattern, one_image_str, flags=0)
                if image_time is not None:
                    begin_video_time_fps = j
                    print(j, one_image_str, "此刻为比赛时间" + image_time[0], sep='------')
                    break

        self.begin_video_time_fps = begin_video_time_fps
        # begin_video_time = begin_video_time_fps / fps  # 单位秒
        # begin_match_time = image_time[0]
        # print("这个比赛开始于该视频的", begin_video_time // 60, "分", begin_video_time - begin_video_time // 60 * 60, "秒",
        #       "此刻比赛时间为", begin_match_time)
        one_path = os.path.join(folder_name, self.frames_dirs[begin_video_time_fps])
        output = cv2.imread(one_path)
        result = ocr.ocr(output, cls=True)        
        for line in result:
            image_time = re.search(self.time_pattern, line[1][0], flags=0)
            if image_time is not None:
                time_location = line[0]
                x = sum([one_local[0] for one_local in time_location]) / 4
                y = sum([one_local[1] for one_local in time_location]) / 4
                self.time_center_local = (x, y)
                # print("time_center_local",time_center_local)
                break

    def text_match(self,word_list, sentence):
        '''
        # 自己写的文本匹配算法，需要保证足球队名按顺序提取就行
        :param
        soccer_team = ["巴塞罗那", "皇马"]
        sentence = "巴塞罗那0：0皇马"
        :return:
        ["巴塞罗那", "皇马"]
        '''
        return_words = []
        len_sen = len(sentence)
        begin_index = 0
        end_index = 0
        while True:
            temp = 0
            # print(sentence[begin_index:end_index])
            for one_word in word_list:
                if one_word in sentence[begin_index:end_index]:
                    return_words.append(one_word)
                    begin_index = end_index
                    temp += 1
                    break
            if temp == 0:
                end_index += 1
            if end_index == len_sen + 1:
                break
        return return_words

    def str2sec(self,x):
        '''
        字符串时分秒转换成秒
        '''
        m, s = x.strip().split(':')  # .split()函数将其通过':'分隔开，.strip()函数用来除去空格
        return int(m) * 60 + int(s)  # int()函数转换成整数运算

    def get_one_frame_information(self,image_path):
        '''

        :param image_path: i r"soccer_image2/00001330.jpg"
        :return:
        '''
        img_mask = self.video_board_mask
        h, w = img_mask.shape[:-1]

        ### image_path为要进行记分牌检测的图片
        img = cv2.imread(image_path)  # , cv2.IMREAD_GRAYSCALE
        # output = img
        output = img_mask * img
        # print(img.shape)
        output = output.astype(np.float32)
        # # output =
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("mask_img.jpg", output)
        output = output.astype(np.float32)
        ocr = self.ocr

        result = ocr.ocr(output, cls=True)
        print(result)

        # for line in result:
        #     image_time = re.search(self.time_pattern, line[1][0], flags=0)
        #     if image_time is not None:
        #         time_location = line[0]
        #         x = sum([one_local[0] for one_local in time_location]) / 4
        #         y = sum([one_local[1] for one_local in time_location]) / 4
        #         time_center_local = (x, y)
        #         print("time_center_local",time_center_local)
        #         break
        jifenpai_list = []
        jifenpai_dict = {}
        jifenpai_dict["game_teams"] = []
        jifenpai_dict["game_time"] = ""
        jifenpai_dict["game_scores"] = ""
        soccer_team = self.team_list

        for line in result:
            key_local = line[0]
            x = sum([one_local[0] for one_local in key_local]) / 4
            y = sum([one_local[1] for one_local in key_local]) / 4
            key_center_local = (x, y)
            distance_with_time = math.sqrt(
                sum([math.pow((self.time_center_local[i] - key_center_local[i]), 2) for i in range(2)]))
            if line[1][1] > 0.5 and distance_with_time < 0.5 * w:  # 自己手工设置一个置信度阈值
                print(line)
                jifenpai_list.append(line[1][0])
                if self.text_match(soccer_team, line[1][0]):
                    jifenpai_dict["game_teams"] += self.text_match(soccer_team, line[1][0])
                if re.search(self.time_pattern, line[1][0], flags=0) is not None:
                    jifenpai_dict["game_time"] = re.search(self.time_pattern, line[1][0], flags=0)[0]
                if re.search(self.time_pattern, line[1][0], flags=0) is None and re.search(self.point_pattern, line[1][0],
                                                                                      flags=0) is not None:
                    score = re.search(self.point_pattern, line[1][0], flags=0)[0].replace("O", "0")
                    jifenpai_dict["game_scores"] = score
        soccer_team = jifenpai_dict["game_teams"]
        new_soccer_team = list(set(soccer_team))
        new_soccer_team.sort(key = soccer_team.index)
        jifenpai_dict["game_teams"] =new_soccer_team
        # frame_index = int(image_path.split("/")[-1].split(".")[0])
        # return_list = [frame_index,jifenpai_dict]
        # print(return_list)
        # return return_list
        return_dict = jifenpai_dict
        return_dict["image_filename"] = image_path
        return return_dict

    def sec2str(self,sec:int):
        minute_x = sec//60
        sec_x = int(sec - minute_x*60)
        return str(minute_x)+":"+str(sec_x)

    def get_video_information(self):
        def is_number(s):
            try:
                return float(s)
            except ValueError:
                return False
        video_raw_infomations = []
        video_processed_infomation = []
        soccer_teams = []
        for step,one_frame in enumerate(self.frames_dirs):
            if one_frame.split(".")[-1] not in ["jpg"]:
                continue
            one_frame1 = os.path.join(self.extract_frames_file,one_frame)
            one_frame_info = self.get_one_frame_information(one_frame1)
            # print("one_frame_info",one_frame_info)
            if step<self.begin_video_time_fps:
                one_frame_info["game_time"] = str(-1)
            if len(one_frame_info["game_teams"]) == 2:
                if one_frame_info["game_teams"][0] != one_frame_info["game_teams"][1]:
                    soccer_team = one_frame_info["game_teams"]
                    soccer_teams.append(soccer_team)
            video_raw_infomations.append(one_frame_info)

        for step,one_info in enumerate(video_raw_infomations):
            # [video_time_fps,{"game_time":game_time,"game_teams":game_teams,"game_scores":game_scores}]= one_info
            #  [{"image_filename":image_path,"game_time":game_time,"game_teams":game_teams,"game_scores":game_scores}]= one_info
            image_path = one_info["image_filename"]
            video_time_fps = int(image_path.split("/")[-1].split(".")[0]) 
            game_time = one_info["game_time"]
            game_scores = one_info["game_scores"]


            # video_all_infomation[step]
            if step == self.begin_video_time_fps:
                before_sec = self.str2sec(one_info["game_time"])
                before_score = one_info["game_scores"]
                one_info["processed_event"] = "game_begin"
                video_processed_infomation.append(one_info)
            if step > self.begin_video_time_fps:
                before_sec += 1/self.extract_frames_fps
                print("before_sec",before_sec,video_time_fps)
                if game_time:
                    new_sec = self.str2sec(game_time)
                    if abs(new_sec-before_sec) >3:
                        try:
                            temp_sec = 3*self.extract_frames_fps
                            x = 0
                            flag = 0
                            # flag1 = 0
                            while True:
                                x += 1
                                if video_raw_infomations[step+x]["game_time"]:
                                    step_sec = self.str2sec(video_raw_infomations[step+x]["game_time"])
                                    if abs(step_sec - new_sec) >= x/self.extract_frames_fps+4:
                                        flag = 1
                                        break
                                else:
                                    temp_sec +=1

                                if x == temp_sec:
                                    break
                            # while True:
                            #     if video_raw_infomations[step+x][1]["game_time"]:
                            #         step_sec = self.str2sec(video_raw_infomations[step+x][1]["game_time"])
                            #         if abs(step_sec - before_sec) <= x/self.extract_frames_fps+4:
                            #             flag1 = 1
                            #             break
                            #     else:
                            #         temp_sec +=1
                            #     x += 1
                            #     if x == temp_sec:
                            #         break
                            if flag ==0:
                                # one_info.append(["time reason",before_sec,new_sec])
                                one_info["processed_event"] = "time_jump"
                                one_info["time_jump"] = [str(before_sec),str(new_sec)]
                                video_processed_infomation.append(one_info)
                                before_sec = new_sec
                            else:
                                # before_sec += 1/self.extract_frames_fps
                                pass
                        except (IndexError):
                            # before_sec += 1/self.extract_frames_fps
                            pass

                else:
                    # before_sec += 1/self.extract_frames_fps
                    pass
                
                if game_scores:
                    if game_scores != before_score:
                        game_score0 = 0
                        for i in game_scores:
                            if is_number(i):
                                game_score0 += is_number(i)

                        before_score0 = 0
                        for i in before_score:
                            if is_number(i):
                                before_score0 += is_number(i)

                        if abs(game_score0-before_score0) == 1:
                            temp_sec = 2*self.extract_frames_fps
                            x = 0
                            score_flag = 0
                            try:
                                while True:
                                    x += 1
                                    if video_raw_infomations[step+x]["game_scores"]:
                                        step_score = video_raw_infomations[step+x]["game_scores"]
                                        if step_score != game_scores:
                                            score_flag = 1
                                            break
                                    else:
                                        temp_sec +=1

                                    if x == temp_sec:
                                        break
                                if score_flag ==0:
                                    # one_info.append("score reason")
                                    one_info["processed_event"] = "score_change" 
                                    one_info["score_change"] = [before_score,game_scores]
                                    video_processed_infomation.append(one_info)
                                    before_score = game_scores
                            except IndexError:
                                pass
        
        for step,i in enumerate(video_processed_infomation):
            i["game_teams"] = soccer_teams[0]
            image_path = i["image_filename"]
            video_time_fps = int(image_path.split("/")[-1].split(".")[0]) 
            i["video_time(s)"] = str(video_time_fps/self.extract_frames_fps)
            if i["game_scores"] =="":
                i["game_scores"] = video_processed_infomation[step-1]["game_scores"]
            if i["game_time"] == "":
                before_game_time = self.str2sec(video_processed_infomation[step-1]["game_time"])
                x = float(i["video_time(s)"]) - float(video_processed_infomation[step-1]["video_time(s)"])
                i["game_time"] = self.sec2str(int(before_game_time+x))

            
        for i in  video_raw_infomations:
            i["game_teams"] = soccer_teams[0]


    
        need_video_dict = json.dumps({"video_processed_infomation":video_processed_infomation},indent = 4) #
        all_video_dict = json.dumps({"video_raw_infomation":video_raw_infomations},indent = 4) #raw
        with open(self.extract_frames_file+".json","w+") as f:
            f.write(need_video_dict)
        with open(self.extract_frames_file+".raw.json","w+") as f:
            f.write(all_video_dict)
                    
                
    def run(self):
        self.extract_frames()
        self.get_begin_time()
        self.get_video_board_mask()
        self.get_video_information()