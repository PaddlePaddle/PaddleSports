# PPSIG：PaddleSports 足球记分牌识别任务

> 详细算法主要原理核心请见:[足球记分牌提取项目(解释版) - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/projectdetail/4378771)
>
> 一键运行Aistudio位置：[PPSIG：PaddleSports 足球记分牌识别（封装版） - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/projectdetail/4401558)



>任务简要：输入一段足球比赛视频和预设的队伍的txt，识别记分牌的信息，得到视频每一秒的比赛时间，比分，比赛队伍名称。实现视频时间与比赛时间相对应（因为视频刚开始往往会有与比赛无关的信息，并且视频中间可能存在比赛的跳跃，比如在视频2分30秒时比赛时间为1分40秒，但是视频2分31秒时比赛时间为3分11秒），输出的txt文件包含比赛跳跃的时刻和比分变动的时刻，这样就可以输入比赛时间得到对应视频时间。

## 简单算法思想介绍：

1. 每隔几秒通过PaddleOCR识别足球比赛图片，定位记分牌刚出现比赛时间的视频时间区间。然后再具体定位第一次视频出现比赛时间信息是哪一秒。
2. 然后通过制作足球记分牌Mask，更好的排除其他信息干扰。
3. 然后接下来逐帧识别视频足球比赛记分牌信息并记录，将视频时间与比赛时间进行对应。

>足球记分牌Mask基本效果展示（电视台台标等与记分牌无关信息在后续算法中计算与时间位置的距离会进行处理）
>
>![img](https://ai-studio-static-online.cdn.bcebos.com/ca6fe63da00c47d9898353b60d9cf6a05fb263c05ff946c9acba15fab7f21106)
>
>![img](https://ai-studio-static-online.cdn.bcebos.com/c660ff555b524c3fb3e233546299bfadb75e5dd0657549a1b93ad85f534a9eb4)
>
>![img](https://ai-studio-static-online.cdn.bcebos.com/85611b318f2a4fe8b9478ae8e829837a5b8eca97f5e246c9b7de1f70e68918d1)



> PaddleOCR的识别效果：（使用记分牌Mask，这里简单的把台标遮掉进行展示）
>
> ![img](https://ai-studio-static-online.cdn.bcebos.com/e172a764c6904a36918d016896e4b3c4adfe55b8968b4687a1c9f99ebc868bf4)

## 1.测试视频介绍
该视频是我自己制作的，就是人工把完整的一个视频中一些片段抽取出来，从而让视频出现跳跃，然后我一共制作了两个视频进行测试放在了[test_soccervideo](https://aistudio.baidu.com/aistudio/datasetdetail/162746)，分别为myvideo.mp4和myvideo1.mp4。

>game_begin的意思为第一次记分牌上出现比赛时间

### 1.1 myvideo.mp4标注介绍

| video_time      | game_time | reason-label |
| --------------- | --------- | ------------ |
| 12秒            | 12s       | game_begin   |
| 4分25秒(265s)   | 7分10秒   | 视频跳跃     |
| 7分7秒(427s)    | 11分49    | 视频跳跃     |
| 10分25(625s)    | 15分34    | 视频跳跃     |
| 8分32秒（512s） | 13分15    | 比分变动     |

### 1.2 myvideo1.mp4标注介绍

| video_time        | game_time | reason-label |
| ----------------- | --------- | ------------ |
| 9秒               | 9s        | game_begin   |
| 2分12秒(132s)     | 4分6秒    | 视频跳跃     |
| 7分41秒(461s)     | 14分38    | 视频跳跃     |
| 11分20(680s)      | 21分40    | 视频跳跃     |
| 18分4秒（1084s）  | 29分27    | 视频跳跃     |
| 17分21秒（1041s） | 27分41    | 比分变动     |
| 18分12秒（1092s） | 29分36    | 比分变动     |

## 2. 输出文件
>（假设output_dir为默认，video_file为myvideo.mp4）

1. 首先将视频使用"ffmpeg"进行抽帧(默认2帧每秒)，输出中间文件夹 ./myvideo

2. 2个json文件，分别为 ./myvideo.json 和 ./myvideo.raw.json ,./myvideo.json 输出的就是第一次识别到记分牌比赛时间信息和视频跳跃的信息和比分变动的信息，./myvideo.raw.json 输出的是每一帧的信息（方便进行中间检查）。

> ./myvideo.json 内容 
> ![](https://ai-studio-static-online.cdn.bcebos.com/76242a4a11df4454a1b952a9067e2da7140c446ac0a043fbac98a7b27e9c0635)


```python
'''
整个json就一个key-value {"video_processed_infomation":list}
然后这个list有几个元素，每个元素为一个dict，就代表几个事件。然后这个每个元素的dict的key ，game_teams代表比赛双方队伍，game_time代表比赛时间，game_scores代表比赛比分，image_filename代表图片的文件路径名，processed_event有三个状态game_begin,time_jump和score_change 分别代表比赛开始，时间跳跃和比分变动，然后如果状态为time_jump，那么dict就会多一个key-value，key为time_jump,value为List（此刻如果不时间跳跃应该的比赛时间和此刻的比赛时间,都是按秒算的），如果状态为score_change，那么dict就会多一个key-value，key为score_change,value为List（刚才的比分和此刻的比分）
'''


```

## 3. main.py中argparse参数介绍

1. parser.add_argument('--video_file', type=str,default="myvideo.mp4")
>video_file参数代表需要操作的比赛视频的文件路径

2. parser.add_argument('--team_list_file', type=str, help = "List of team names in text",default="team_name.txt")
>team_list_file为预设全部足球比赛的队伍名称信息

3.  parser.add_argument('--output_dir', type=str,default="")
> 如果不填参数则2个json和抽帧文件夹为video_file同路径。 如果out_dir为./12/video1 那么2个txt文件路径为./12/video1.json 和 ./12/video1.raw.json 那么抽帧图片文件夹为./12/video1。

4. parser.add_argument("--extract_frames_fps帧",type=int,default=2)
> ffmpeg一秒extract_frames_fps帧

5. parser.add_argument("--seq_second",type=int,default=10,help="粗粒度间隔seq_second秒判别视频中记分牌出现时间")

## 4. 算法注意事项：

1. 因为PaddleOCR提取记分牌文字并不能每帧都很好提取，所以我进行了很多的处理

2. 比如比赛时间从25秒突然变成1分钟，有可能是OCR识别错了，也有可能是出现比赛跳跃现象，那么我会判断后面3秒（默认及6帧）内是否比赛时间为正常进行。如果后3秒内的图片有没有game_time的，那么就往后依次顺延，直到判断有game_time的6帧图片比赛时间都正常进行。


```python
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
```

3. 如果比分变动，为了防止是OCR识别错误，我会判断接下来2秒内比分是否一致，若有帧无比分，则顺延。同时判断总比分是否每次相较于上次加1

```python
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
```

4. 如果比分变动时候，无法识别比赛时间，则根据视频时间算差，计算得到。

```python
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
```

## 5. 一键运行示例

```python
!git clone -b v2.5.0 https://gitee.com/paddlepaddle/PaddleOCR.git
!pip install -r ./PaddleOCR/requirements.txt
#请先创建空文件夹./output 和下载测试的mp4数据集
!python main.py --video_file ./myvideo1.mp4 --output_dir output/video1 --seq_second 5

from main import get_video_time_from_game_time
'''
输入比赛时间得到对应的视频时间
'''
get_video_time_from_game_time("output/video1.json","00:02")


```





## 6. 小贴士

由于我判断比赛跳跃和比分变动的条件比较苛刻，所以识别出来的比分变动往往比实际的比分变动后几秒，视频跳跃倒是前后相差往往小于2s.

