# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import time
import traceback

import yaml
import glob
import cv2
import numpy as np
import math
import paddle
import sys
import copy
from collections import Sequence, defaultdict
from datacollector import DataCollector, Result
from queue import Queue

# add deploy path of PadleDetection to sys.path

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from pipe_utils import argsparser, print_arguments, merge_cfg, PipeTimer
from pipe_utils import get_test_images, crop_image_with_det, crop_image_with_mot, parse_mot_res, parse_mot_keypoint

from python.infer import Detector, DetectorPicoDet
from python.keypoint_infer import KeyPointDetector
from python.keypoint_postprocess import translate_to_ori_images
from python.preprocess import decode_image, ShortSizeScale
from python.visualize import visualize_box_mask, visualize_attr, visualize_pose, visualize_action, visualize_speed, visualize_team, visualize_singleplayer, visualize_boating, visualize_ball, visualize_link_player, visualize_golf, visualize_player_rec, visualize_ball_control

from pptracking.python.mot_sde_infer import SDE_Detector
from pptracking.python.mot.visualize import plot_tracking_dict
from pptracking.python.mot.utils import flow_statistic

from pphuman.attr_infer import AttrDetector
from pphuman.video_action_infer import VideoActionRecognizer
from pphuman.action_infer import SkeletonActionRecognizer, DetActionRecognizer, ClsActionRecognizer
from pphuman.action_utils import KeyPointBuff, ActionVisualHelper
from pphuman.reid import ReID
from pphuman.mtmct import mtmct_process

from numOCR.number_OCR import num_predictor

from download import auto_download_model


class Pipeline(object):
    """
    Pipeline

    Args:
        cfg (dict): config of models in pipeline
        image_file (string|None): the path of image file, default as None
        image_dir (string|None): the path of image directory, if not None, 
            then all the images in directory will be predicted, default as None
        video_file (string|None): the path of video file, default as None
        camera_id (int): the device id of camera to predict, default as -1
        device (string): the device to predict, options are: CPU/GPU/XPU, 
            default as CPU
        run_mode (string): the mode of prediction, options are: 
            paddle/trt_fp32/trt_fp16, default as paddle
        trt_min_shape (int): min shape for dynamic shape in trt, default as 1
        trt_max_shape (int): max shape for dynamic shape in trt, default as 1280
        trt_opt_shape (int): opt shape for dynamic shape in trt, default as 640
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True. default as False
        cpu_threads (int): cpu threads, default as 1
        enable_mkldnn (bool): whether to open MKLDNN, default as False
        output_dir (string): The path of output, default as 'output'
        draw_center_traj (bool): Whether drawing the trajectory of center, default as False
        secs_interval (int): The seconds interval to count after tracking, default as 10
        do_entrance_counting(bool): Whether counting the numbers of identifiers entering 
            or getting out from the entrance, default as False, only support single class
            counting in MOT.
    """

    def __init__(self, args, cfg):
        self.multi_camera = False
        reid_cfg = cfg.get('REID', False)
        self.enable_mtmct = reid_cfg['enable'] if reid_cfg else False
        self.is_video = False
        self.output_dir = args.output_dir
        self.vis_result = cfg['visual']
        self.input = self._parse_input(args.image_file, args.image_dir,
                                       args.video_file, args.video_dir,
                                       args.camera_id)
        if self.multi_camera:
            self.predictor = []
            for name in self.input:
                predictor_item = PipePredictor(
                    args, cfg, is_video=True, multi_camera=True)
                predictor_item.set_file_name(name)
                self.predictor.append(predictor_item)

        else:
            self.predictor = PipePredictor(args, cfg, self.is_video)
            if self.is_video:
                self.predictor.set_file_name(args.video_file)

        self.output_dir = args.output_dir
        self.draw_center_traj = args.draw_center_traj
        self.secs_interval = args.secs_interval
        self.do_entrance_counting = args.do_entrance_counting
        self.do_break_in_counting = args.do_break_in_counting
        self.region_type = args.region_type
        self.region_polygon = args.region_polygon
        self.speed_predict = args.speed_predict
        self.mapping_ratio = args.mapping_ratio
        self.x_ratio = args.x_ratio
        self.y_ratio = args.y_ratio
        self.team_clas = args.team_clas
        if self.region_type == 'custom':
            assert len(
                self.region_polygon
            ) > 6, 'region_type is custom, region_polygon should be at least 3 pairs of point coords.'

    def _parse_input(self, image_file, image_dir, video_file, video_dir,
                     camera_id):

        # parse input as is_video and multi_camera

        if image_file is not None or image_dir is not None:
            input = get_test_images(image_dir, image_file)
            self.is_video = False
            self.multi_camera = False

        elif video_file is not None:
            assert os.path.exists(video_file), "video_file not exists."
            self.multi_camera = False
            input = video_file
            self.is_video = True

        elif video_dir is not None:
            videof = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
            if len(videof) > 1:
                self.multi_camera = True
                videof.sort()
                input = videof
            else:
                input = videof[0]
            self.is_video = True

        elif camera_id != -1:
            self.multi_camera = False
            input = camera_id
            self.is_video = True

        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file', 'camera_id', 'image_file', 'image_dir']"
            )

        return input

    def run(self):
        if self.multi_camera:
            multi_res = []
            for predictor, input in zip(self.predictor, self.input):
                predictor.run(input)
                collector_data = predictor.get_result()
                multi_res.append(collector_data)
            if self.enable_mtmct:
                mtmct_process(
                    multi_res,
                    self.input,
                    mtmct_vis=self.vis_result,
                    output_dir=self.output_dir)

        else:
            self.predictor.run(self.input)


def get_model_dir(cfg):
    # auto download inference model
    model_dir_dict = {}
    for key in cfg.keys():
        if type(cfg[key]) ==  dict and \
            ("enable" in cfg[key].keys() and cfg[key]['enable']
                or "enable" not in cfg[key].keys()):

            if "model_dir" in cfg[key].keys():
                model_dir = cfg[key]["model_dir"]
                downloaded_model_dir = auto_download_model(model_dir)
                if downloaded_model_dir:
                    model_dir = downloaded_model_dir
                model_dir_dict[key] = model_dir
                print(key, " model dir:", model_dir)
            elif key == "VEHICLE_PLATE":
                det_model_dir = cfg[key]["det_model_dir"]
                downloaded_det_model_dir = auto_download_model(det_model_dir)
                if downloaded_det_model_dir:
                    det_model_dir = downloaded_det_model_dir
                model_dir_dict["det_model_dir"] = det_model_dir
                print("det_model_dir model dir:", det_model_dir)

                rec_model_dir = cfg[key]["rec_model_dir"]
                downloaded_rec_model_dir = auto_download_model(rec_model_dir)
                if downloaded_rec_model_dir:
                    rec_model_dir = downloaded_rec_model_dir
                model_dir_dict["rec_model_dir"] = rec_model_dir
                print("rec_model_dir model dir:", rec_model_dir)
        elif key == "MOT":  # for idbased and skeletonbased actions
            model_dir = cfg[key]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                model_dir = downloaded_model_dir
            model_dir_dict[key] = model_dir

    return model_dir_dict


class PipePredictor(object):
    """
    Predictor in single camera
    
    The pipeline for image input: 

        1. Detection
        2. Detection -> Attribute

    The pipeline for video input: 

        1. Tracking
        2. Tracking -> Attribute
        3. Tracking -> KeyPoint -> SkeletonAction Recognition
        4. VideoAction Recognition

    Args:
        cfg (dict): config of models in pipeline
        is_video (bool): whether the input is video, default as False
        multi_camera (bool): whether to use multi camera in pipeline, 
            default as False
        camera_id (int): the device id of camera to predict, default as -1
        device (string): the device to predict, options are: CPU/GPU/XPU, 
            default as CPU
        run_mode (string): the mode of prediction, options are: 
            paddle/trt_fp32/trt_fp16, default as paddle
        trt_min_shape (int): min shape for dynamic shape in trt, default as 1
        trt_max_shape (int): max shape for dynamic shape in trt, default as 1280
        trt_opt_shape (int): opt shape for dynamic shape in trt, default as 640
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True. default as False
        cpu_threads (int): cpu threads, default as 1
        enable_mkldnn (bool): whether to open MKLDNN, default as False
        output_dir (string): The path of output, default as 'output'
        draw_center_traj (bool): Whether drawing the trajectory of center, default as False
        secs_interval (int): The seconds interval to count after tracking, default as 10
        do_entrance_counting(bool): Whether counting the numbers of identifiers entering 
            or getting out from the entrance, default as False, only support single class
            counting in MOT.
    """

    def __init__(self, args, cfg, is_video=True, multi_camera=False):
        device = args.device
        run_mode = args.run_mode
        trt_min_shape = args.trt_min_shape
        trt_max_shape = args.trt_max_shape
        trt_opt_shape = args.trt_opt_shape
        trt_calib_mode = args.trt_calib_mode
        cpu_threads = args.cpu_threads
        enable_mkldnn = args.enable_mkldnn
        output_dir = args.output_dir
        draw_center_traj = args.draw_center_traj
        secs_interval = args.secs_interval
        do_entrance_counting = args.do_entrance_counting
        do_break_in_counting = args.do_break_in_counting
        region_type = args.region_type
        region_polygon = args.region_polygon
        speed_predict = args.speed_predict
        self.player_recognize = args.player_recognize
        if self.player_recognize:
            self.id2num = {}

        # general module for pphuman and ppvehicle
        self.with_mot = cfg.get('MOT', False)['enable'] if cfg.get(
            'MOT', False) else False
        self.with_human_attr = cfg.get('ATTR', False)['enable'] if cfg.get(
            'ATTR', False) else False
        if self.with_mot:
            print('Multi-Object Tracking enabled')
        if self.with_human_attr:
            print('Human Attribute Recognition enabled')

        # only for pphuman
        self.with_skeleton_action = cfg.get(
            'SKELETON_ACTION', False)['enable'] if cfg.get('SKELETON_ACTION',
                                                           False) else False
        self.with_video_action = cfg.get(
            'VIDEO_ACTION', False)['enable'] if cfg.get('VIDEO_ACTION',
                                                        False) else False
        self.with_idbased_detaction = cfg.get(
            'ID_BASED_DETACTION', False)['enable'] if cfg.get(
                'ID_BASED_DETACTION', False) else False
        self.with_idbased_clsaction = cfg.get(
            'ID_BASED_CLSACTION', False)['enable'] if cfg.get(
                'ID_BASED_CLSACTION', False) else False
        self.with_mtmct = cfg.get('REID', False)['enable'] if cfg.get(
            'REID', False) else False
        if args.singleplayer or args.player_recognize:
            self.with_skeleton_action = True

        if self.with_skeleton_action:
            print('SkeletonAction Recognition enabled')
        if self.with_video_action:
            print('VideoAction Recognition enabled')
        if self.with_idbased_detaction:
            print('IDBASED Detection Action Recognition enabled')
        if self.with_idbased_clsaction:
            print('IDBASED Classification Action Recognition enabled')
        if self.with_mtmct:
            print("MTMCT enabled")
        if self.player_recognize:
            print("player number OCR enabled")
            self.player_OCR = num_predictor()
        # only for ppvehicle
        self.with_vehicleplate = cfg.get(
            'VEHICLE_PLATE', False)['enable'] if cfg.get('VEHICLE_PLATE',
                                                         False) else False
        if self.with_vehicleplate:
            print('Vehicle Plate Recognition enabled')

        self.with_vehicle_attr = cfg.get(
            'VEHICLE_ATTR', False)['enable'] if cfg.get('VEHICLE_ATTR',
                                                        False) else False
        if self.with_vehicle_attr:
            print('Vehicle Attribute Recognition enabled')

        self.modebase = {
            "framebased": False,
            "videobased": False,
            "idbased": False,
            "skeletonbased": False
        }

        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg
        self.output_dir = output_dir
        self.draw_center_traj = draw_center_traj
        self.secs_interval = secs_interval
        self.do_entrance_counting = do_entrance_counting
        self.do_break_in_counting = do_break_in_counting
        self.region_type = region_type
        self.region_polygon = region_polygon
        self.speed_predict = speed_predict
        self.mapping_ratio = args.mapping_ratio
        self.x_ratio = args.x_ratio
        self.y_ratio = args.y_ratio
        self.team_clas = args.team_clas
        self.singleplayer = args.singleplayer
        self.boating = args.boating
        self.ball_drawing = args.ball_drawing
        self.link_player = args.link_player
        self.ball_control = args.ball_control
        if self.ball_control:
            self.present_ball_id = []
        self.loc_dir = args.loc_dir
        self.show = args.show
        self.golf = args.golf
        if self.link_player or self.singleplayer or self.team_clas or self.golf:
            self.no_box_visual = True
        else:
            self.no_box_visual = False
        self.save_loc = args.save_loc
        if self.save_loc:
            self.loc_list = []
        self.warmup_frame = self.cfg['warmup_frame']
        self.pipeline_res = Result()
        self.pipe_timer = PipeTimer()
        self.file_name = None
        self.collector = DataCollector()

        # auto download inference model
        model_dir_dict = get_model_dir(self.cfg)
        # self.no_box_visual = True
        if not is_video:
            det_cfg = self.cfg['DET']
            model_dir = model_dir_dict['DET']
            batch_size = det_cfg['batch_size']
            self.det_predictor = Detector(
                model_dir, device, run_mode, batch_size, trt_min_shape,
                trt_max_shape, trt_opt_shape, trt_calib_mode, cpu_threads,
                enable_mkldnn)
            if self.with_human_attr:
                attr_cfg = self.cfg['ATTR']
                model_dir = model_dir_dict['ATTR']
                batch_size = attr_cfg['batch_size']
                basemode = attr_cfg['basemode']
                self.modebase[basemode] = True
                self.attr_predictor = AttrDetector(
                    model_dir, device, run_mode, batch_size, trt_min_shape,
                    trt_max_shape, trt_opt_shape, trt_calib_mode, cpu_threads,
                    enable_mkldnn)

        else:
            if self.with_human_attr:
                attr_cfg = self.cfg['ATTR']
                model_dir = model_dir_dict['ATTR']
                batch_size = attr_cfg['batch_size']
                basemode = attr_cfg['basemode']
                self.modebase[basemode] = True
                self.attr_predictor = AttrDetector(
                    model_dir, device, run_mode, batch_size, trt_min_shape,
                    trt_max_shape, trt_opt_shape, trt_calib_mode, cpu_threads,
                    enable_mkldnn)
            if self.with_idbased_detaction:
                idbased_detaction_cfg = self.cfg['ID_BASED_DETACTION']
                model_dir = model_dir_dict['ID_BASED_DETACTION']
                batch_size = idbased_detaction_cfg['batch_size']
                basemode = idbased_detaction_cfg['basemode']
                threshold = idbased_detaction_cfg['threshold']
                display_frames = idbased_detaction_cfg['display_frames']
                skip_frame_num = idbased_detaction_cfg['skip_frame_num']
                self.modebase[basemode] = True

                self.det_action_predictor = DetActionRecognizer(
                    model_dir,
                    device,
                    run_mode,
                    batch_size,
                    trt_min_shape,
                    trt_max_shape,
                    trt_opt_shape,
                    trt_calib_mode,
                    cpu_threads,
                    enable_mkldnn,
                    threshold=threshold,
                    display_frames=display_frames,
                    skip_frame_num=skip_frame_num)
                self.det_action_visual_helper = ActionVisualHelper(1)

            if self.with_idbased_clsaction:
                idbased_clsaction_cfg = self.cfg['ID_BASED_CLSACTION']
                model_dir = model_dir_dict['ID_BASED_CLSACTION']
                batch_size = idbased_clsaction_cfg['batch_size']
                basemode = idbased_clsaction_cfg['basemode']
                threshold = idbased_clsaction_cfg['threshold']
                self.modebase[basemode] = True
                display_frames = idbased_clsaction_cfg['display_frames']
                skip_frame_num = idbased_clsaction_cfg['skip_frame_num']

                self.cls_action_predictor = ClsActionRecognizer(
                    model_dir,
                    device,
                    run_mode,
                    batch_size,
                    trt_min_shape,
                    trt_max_shape,
                    trt_opt_shape,
                    trt_calib_mode,
                    cpu_threads,
                    enable_mkldnn,
                    threshold=threshold,
                    display_frames=display_frames,
                    skip_frame_num=skip_frame_num)
                self.cls_action_visual_helper = ActionVisualHelper(1)

            if self.with_skeleton_action:
                skeleton_action_cfg = self.cfg['SKELETON_ACTION']
                skeleton_action_model_dir = model_dir_dict['SKELETON_ACTION']
                skeleton_action_batch_size = skeleton_action_cfg['batch_size']
                skeleton_action_frames = skeleton_action_cfg['max_frames']
                display_frames = skeleton_action_cfg['display_frames']
                self.coord_size = skeleton_action_cfg['coord_size']
                basemode = skeleton_action_cfg['basemode']
                self.modebase[basemode] = True

                self.skeleton_action_predictor = SkeletonActionRecognizer(
                    skeleton_action_model_dir,
                    device,
                    run_mode,
                    skeleton_action_batch_size,
                    trt_min_shape,
                    trt_max_shape,
                    trt_opt_shape,
                    trt_calib_mode,
                    cpu_threads,
                    enable_mkldnn,
                    window_size=skeleton_action_frames)
                self.skeleton_action_visual_helper = ActionVisualHelper(
                    display_frames)

                if self.modebase["skeletonbased"]:
                    kpt_cfg = self.cfg['KPT']
                    kpt_model_dir = model_dir_dict['KPT']
                    kpt_batch_size = kpt_cfg['batch_size']
                    self.kpt_predictor = KeyPointDetector(
                        kpt_model_dir,
                        device,
                        run_mode,
                        kpt_batch_size,
                        trt_min_shape,
                        trt_max_shape,
                        trt_opt_shape,
                        trt_calib_mode,
                        cpu_threads,
                        enable_mkldnn,
                        use_dark=False)
                    self.kpt_buff = KeyPointBuff(skeleton_action_frames)


            if self.with_mtmct:
                reid_cfg = self.cfg['REID']
                model_dir = model_dir_dict['REID']
                batch_size = reid_cfg['batch_size']
                basemode = reid_cfg['basemode']
                self.modebase[basemode] = True
                self.reid_predictor = ReID(
                    model_dir, device, run_mode, batch_size, trt_min_shape,
                    trt_max_shape, trt_opt_shape, trt_calib_mode, cpu_threads,
                    enable_mkldnn)

            if self.with_mot or self.modebase["idbased"] or self.modebase[
                    "skeletonbased"]:
                mot_cfg = self.cfg['MOT']
                model_dir = model_dir_dict['MOT']
                tracker_config = mot_cfg['tracker_config']
                batch_size = mot_cfg['batch_size']
                basemode = mot_cfg['basemode']
                self.modebase[basemode] = True
                self.mot_predictor = SDE_Detector(
                    model_dir,
                    tracker_config,
                    device,
                    run_mode,
                    batch_size,
                    trt_min_shape,
                    trt_max_shape,
                    trt_opt_shape,
                    trt_calib_mode,
                    cpu_threads,
                    enable_mkldnn,
                    draw_center_traj=draw_center_traj,
                    secs_interval=secs_interval,
                    do_entrance_counting=do_entrance_counting,
                    do_break_in_counting=do_break_in_counting,
                    region_type=region_type,
                    region_polygon=region_polygon)

            if self.with_video_action:
                video_action_cfg = self.cfg['VIDEO_ACTION']

                basemode = video_action_cfg['basemode']
                self.modebase[basemode] = True

                video_action_model_dir = model_dir_dict['VIDEO_ACTION']
                video_action_batch_size = video_action_cfg['batch_size']
                short_size = video_action_cfg["short_size"]
                target_size = video_action_cfg["target_size"]

                self.video_action_predictor = VideoActionRecognizer(
                    model_dir=video_action_model_dir,
                    short_size=short_size,
                    target_size=target_size,
                    device=device,
                    run_mode=run_mode,
                    batch_size=video_action_batch_size,
                    trt_min_shape=trt_min_shape,
                    trt_max_shape=trt_max_shape,
                    trt_opt_shape=trt_opt_shape,
                    trt_calib_mode=trt_calib_mode,
                    cpu_threads=cpu_threads,
                    enable_mkldnn=enable_mkldnn)

    def set_file_name(self, path):
        if path is not None:
            self.file_name = os.path.split(path)[-1]
        else:
            # use camera id
            self.file_name = None

    def get_result(self):
        return self.collector.get_res()

    def run(self, input):
        if self.is_video:
            self.predict_video(input)
        else:
            self.predict_image(input)
        self.pipe_timer.info()

    def predict_image(self, input):
        # det
        # det -> attr
        batch_loop_cnt = math.ceil(
            float(len(input)) / self.det_predictor.batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * self.det_predictor.batch_size
            end_index = min((i + 1) * self.det_predictor.batch_size, len(input))
            batch_file = input[start_index:end_index]
            batch_input = [decode_image(f, {})[0] for f in batch_file]

            if i > self.warmup_frame:
                self.pipe_timer.total_time.start()
                self.pipe_timer.module_time['det'].start()
            # det output format: class, score, xmin, ymin, xmax, ymax
            det_res = self.det_predictor.predict_image(
                batch_input, visual=False)
            det_res = self.det_predictor.filter_box(det_res,
                                                    self.cfg['crop_thresh'])
            if i > self.warmup_frame:
                self.pipe_timer.module_time['det'].end()
            self.pipeline_res.update(det_res, 'det')

            if self.with_human_attr:
                crop_inputs = crop_image_with_det(batch_input, det_res)
                attr_res_list = []

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].start()

                for crop_input in crop_inputs:
                    attr_res = self.attr_predictor.predict_image(
                        crop_input, visual=False)
                    attr_res_list.extend(attr_res['output'])

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].end()

                attr_res = {'output': attr_res_list}
                self.pipeline_res.update(attr_res, 'attr')

            if self.with_vehicle_attr:
                crop_inputs = crop_image_with_det(batch_input, det_res)
                vehicle_attr_res_list = []

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_attr'].start()

                for crop_input in crop_inputs:
                    attr_res = self.vehicle_attr_predictor.predict_image(
                        crop_input, visual=False)
                    vehicle_attr_res_list.extend(attr_res['output'])

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_attr'].end()

                attr_res = {'output': vehicle_attr_res_list}
                self.pipeline_res.update(attr_res, 'vehicle_attr')

            self.pipe_timer.img_num += len(batch_input)
            if i > self.warmup_frame:
                self.pipe_timer.total_time.end()

            if self.cfg['visual']:
                self.visualize_image(batch_file, batch_input, self.pipeline_res)

    def predict_video(self, video_file):
        # mot
        # mot -> attr
        # mot -> pose -> action
        capture = cv2.VideoCapture(video_file)
        video_out_name = 'output.mp4' if self.file_name is None else self.file_name

        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("video fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name)
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        frame_id = 0

        entrance, records, center_traj = None, None, None
        if self.draw_center_traj or self.speed_predict or self.ball_drawing:
            center_traj = [{}]
            speed_dict = {}

        if self.team_clas:
            id_team = {}
        id_set = set()
        interval_id_set = set()
        in_id_list = list()
        out_id_list = list()
        prev_center = dict()
        records = list()
        if self.do_entrance_counting or self.do_break_in_counting:
            if self.region_type == 'horizontal':
                entrance = [0, height / 2., width, height / 2.]
            elif self.region_type == 'vertical':
                entrance = [width / 2, 0., width / 2, height]
            elif self.region_type == 'custom':
                entrance = []
                assert len(
                    self.region_polygon
                ) % 2 == 0, "region_polygon should be pairs of coords points when do break_in counting."
                for i in range(0, len(self.region_polygon), 2):
                    entrance.append(
                        [self.region_polygon[i], self.region_polygon[i + 1]])
                entrance.append([width, height])
            else:
                raise ValueError("region_type:{} unsupported.".format(
                    self.region_type))

        video_fps = fps

        video_action_imgs = []

        if self.with_video_action:
            short_size = self.cfg["VIDEO_ACTION"]["short_size"]
            scale = ShortSizeScale(short_size)

        while (1):
            if frame_id % 10 == 0:
                print('frame id: ', frame_id)

            ret, frame = capture.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.modebase["idbased"] or self.modebase["skeletonbased"]:
                if frame_id > self.warmup_frame:
                    self.pipe_timer.total_time.start()
                    self.pipe_timer.module_time['mot'].start()
                res = self.mot_predictor.predict_image(
                    [copy.deepcopy(frame_rgb)], visual=False)

                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].end()

                # mot output format: id, class, score, xmin, ymin, xmax, ymax
                mot_res = parse_mot_res(res)

                # flow_statistic only support single class MOT
                boxes, scores, ids = res[0]  # batch size = 1 in MOT
                mot_result = (frame_id + 1, boxes[0], scores[0],
                              ids[0])  # single class
                # json.dump(mot_result, open(str(frame_id+1)+".json", "w"))
                statistic = flow_statistic(
                    mot_result, self.secs_interval, self.do_entrance_counting,
                    self.do_break_in_counting, self.region_type, video_fps,
                    entrance, id_set, interval_id_set, in_id_list, out_id_list,
                    prev_center, records)
                records = statistic['records']

                # nothing detected
                if len(mot_res['boxes']) == 0:
                    if self.save_loc:
                        self.loc_list.append([])
                    frame_id += 1
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.img_num += 1
                        self.pipe_timer.total_time.end()
                    if self.cfg['visual']:
                        _, _, fps = self.pipe_timer.get_total_time()

                        im = self.visualize_video(frame, mot_res, frame_id, fps,
                                                      entrance, records,
                                                      center_traj)  # visualize
                        if self.ball_drawing:
                            his_location = set()
                            for i in center_traj[0].values():
                                if len(i) <= 3:
                                    continue
                                else:
                                    for loc in i:
                                        his_location.add(loc)
                            his_location = list(his_location)
                            x = []
                            y = []
                            for i in his_location:
                                x.append(i[0])
                                y.append(i[1])
                            his_location = {'res': [x, y]}
                            if x:
                                im = visualize_ball(im, his_location)
                        writer.write(im)
                        if self.file_name is None or self.show:  # use camera_id
                            cv2.imshow('Paddle-Pipeline', im)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                    continue
                if self.save_loc:
                    # print(scores)
                    temp_index = list(scores).index(max(scores))
                    self.loc_list.append(boxes[0][temp_index])
                    # print(self.loc_list)
                self.pipeline_res.update(mot_res, 'mot')
                crop_input, new_bboxes, ori_bboxes = crop_image_with_mot(
                    frame_rgb, mot_res)

                if self.link_player:
                    link_boxes = []
                    link_crop = []
                    temp, boxes, scores, ids = mot_result
                    crop_input, _, _ = crop_image_with_mot(
                        frame_rgb, mot_res, expand=False)
                    for i in range(len(ids)):
                        if ids[i] in self.link_player:
                            link_boxes.append(boxes[i])
                            link_crop.append(crop_input[i])
                    self.pipeline_res.update({"result": link_boxes, "crop_input": link_crop}, "link_player")

                if self.ball_drawing:
                    his_location = set()
                    for i in center_traj[0].values():
                        if len(i) <= 3:
                            continue
                        else:
                            for loc in i:
                                his_location.add(loc)
                    his_location = list(his_location)
                    x = []
                    y = []
                    for i in his_location:
                        x.append(i[0])
                        y.append(i[1])
                    his_location = {'res': [x, y]}
                    if x:
                        self.pipeline_res.update(his_location, 'ball_drawing')

                if self.speed_predict:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['speed_predict'].start()
                    speed_result = []
                    index2id = {}
                    temp, boxes, scores, ids = mot_result
                    for i in range(len(ids)):
                        temp_id = ids[i]
                        index2id[i] = temp_id
                        try:
                            value = center_traj[0].get(temp_id)
                            location = value[-1]
                            re_location = value[-2]
                            x_ratio, y_ratio = None, None
                            if self.x_ratio or self.y_ratio:
                                if len(self.x_ratio) % 3 != 0 or len(self.y_ratio) % 3 != 0:
                                    raise "'x_ratio' or 'y_ratio' format error!"
                                for j in range(len(self.x_ratio) // 3):
                                    if (self.x_ratio[j * 3] >= location[0] >= self.x_ratio[j * 3 + 1]) or (
                                            self.x_ratio[j * 3] <= location[0] <= self.x_ratio[j * 3 + 1]):
                                        x_ratio = self.x_ratio[j * 3 + 2]/abs(self.x_ratio[j * 3]-self.x_ratio[j * 3 + 1])
                                        break
                                for j in range(len(self.y_ratio) // 3):
                                    if (self.y_ratio[j * 3] >= location[1] >= self.y_ratio[j * 3 + 1]) or (
                                            self.y_ratio[j * 3] <= location[1] <= self.y_ratio[j * 3 + 1]):
                                        y_ratio = self.y_ratio[j * 3 + 2]/abs(self.y_ratio[j * 3]-self.y_ratio[j * 3 + 1])
                                        break

                                if (not x_ratio) and y_ratio:
                                    x_ratio = y_ratio
                                elif (not y_ratio) and x_ratio:
                                    y_ratio = x_ratio
                                else:
                                    x_ratio, y_ratio = 1, 1
                            elif self.mapping_ratio:
                                x_ratio = self.mapping_ratio[0] / width
                                y_ratio = self.mapping_ratio[1] / height
                            else:
                                x_ratio = 1
                                y_ratio = 1
                            speed = pow(pow(location[0]*x_ratio - re_location[0]*x_ratio, 2) +
                                        pow(location[1]*y_ratio - re_location[1]*y_ratio, 2), 0.5)
                            if not speed_dict.get(temp_id):
                                speed_dict[temp_id] = [speed * fps]
                            else:
                                speed_dict[temp_id].append(speed * fps)
                            if x_ratio == y_ratio == 1:
                                speed = "speed:{:.2f}pixel/s".format(speed * fps)
                            else:
                                speed = "speed:{:.2f}km/h".format(speed * 3.6 * fps)
                        except:
                            speed = "speed:unknown"
                        speed_result.append([speed])
                    attr_res = {'output': speed_result, 'speed_dict': speed_dict, 'index2id': index2id}
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['speed_predict'].end()
                    self.pipeline_res.update(attr_res, 'speed_predict')

                if self.team_clas:
                    color = {
                        "black": {"color_lower": np.array([0, 0, 0]), "color_upper": np.array([180, 255, 46])},
                        "white": {"color_lower": np.array([0, 0, 221]), "color_upper": np.array([180, 30, 255])},
                        "blue": {"color_lower": np.array([100, 43, 46]), "color_upper": np.array([124, 255, 255])},
                        "red": {"color_lower": np.array([156, 43, 46]), "color_upper": np.array([180, 255, 255])},
                        "yellow": {"color_lower": np.array([26, 43, 46]), "color_upper": np.array([34, 255, 255])},
                        "green": {"color_lower": np.array([35, 43, 46]), "color_upper": np.array([77, 255, 255])},
                        "purple": {"color_lower": np.array([125, 43, 46]), "color_upper": np.array([155, 255, 255])},
                        "orange": {"color_lower": np.array([11, 43, 46]), "color_upper": np.array([25, 255, 255])}
                    }
                    if len(self.team_clas) != 4:
                        raise "team_clas input error"
                    team_list = [[self.team_clas[0], self.team_clas[1]], [self.team_clas[2], self.team_clas[3]]]
                    team_result = []
                    crop_input_team, temp0, temp1 = crop_image_with_mot(
                        frame_rgb, mot_res, expand=False)
                    index2id = {}
                    temp, boxes, scores, ids = mot_result
                    for i in range(len(ids)):
                        temp_id = ids[i]
                        index2id[i] = temp_id
                    for i in range(len(crop_input_team)):
                        ori_img = crop_input_team[i]
                        img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2HSV)  # 转成HSV
                        color_img_0 = cv2.inRange(img, color[team_list[0][0]]["color_lower"],
                                                  color[team_list[0][0]]["color_upper"])  # 筛选出符合的颜色
                        color_img_1 = cv2.inRange(img, color[team_list[1][0]]["color_lower"],
                                                  color[team_list[1][0]]["color_upper"])  # 筛选出符合的颜色
                        img_class_0 = color_img_0.flatten().tolist()
                        img_class_1 = color_img_1.flatten().tolist()
                        ratio_0 = img_class_0.count(255) / len(img_class_0)
                        ratio_1 = img_class_1.count(255) / len(img_class_1)
                        if ratio_0 > ratio_1 and ratio_0 > 0:
                            count = 0
                        elif ratio_1 > 0:
                            count = 1
                        else:
                            count = -1
                        if id_team.get(index2id[i]):
                            re_team = id_team.get(index2id[i])
                            if count == 1:
                                id_team[index2id[i]] = [re_team[0], re_team[1] + 1, re_team[2]]
                            elif count == 0:
                                id_team[index2id[i]] = [re_team[0] + 1, re_team[1], re_team[2]]
                            else:
                                id_team[index2id[i]] = [re_team[0], re_team[1], re_team[2] + 1]
                            team_index = id_team[index2id[i]].index(max(id_team[index2id[i]]))
                            if team_index == 1:
                                team_result.append([team_list[1][1]])
                            elif team_index == 0:
                                team_result.append([team_list[0][1]])
                            else:
                                team_result.append(['unknown'])
                        else:
                            if count == 1:
                                id_team[index2id[i]] = [0, 1, 0]
                                team_result.append([team_list[1][1]])
                            elif count == 0:
                                id_team[index2id[i]] = [1, 0, 0]
                                team_result.append([team_list[0][1]])
                            else:
                                id_team[index2id[i]] = [0, 0, 1]
                                team_result.append(['unknown'])
                    attr_res = {'output': team_result, 'color': {team_list[0][1]: team_list[0][0], team_list[1][1]: team_list[1][0]}}
                    self.pipeline_res.update(attr_res, 'team_clas')

                if self.singleplayer:
                    index2id = {}
                    temp, boxes, scores, ids = mot_result
                    for i in range(len(ids)):
                        temp_id = ids[i]
                        index2id[i] = temp_id
                    singleplayer_res = {'index2id': index2id}
                    self.pipeline_res.update(singleplayer_res, 'singleplayer')

                if self.with_human_attr:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].start()
                    attr_res = self.attr_predictor.predict_image(
                        crop_input, visual=False)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].end()
                    self.pipeline_res.update(attr_res, 'attr')

                if self.with_idbased_detaction:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['det_action'].start()
                    det_action_res = self.det_action_predictor.predict(
                        crop_input, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['det_action'].end()
                    self.pipeline_res.update(det_action_res, 'det_action')

                    if self.cfg['visual']:
                        self.det_action_visual_helper.update(det_action_res)

                if self.with_idbased_clsaction:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['cls_action'].start()
                    cls_action_res = self.cls_action_predictor.predict_with_mot(
                        crop_input, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['cls_action'].end()
                    self.pipeline_res.update(cls_action_res, 'cls_action')

                    if self.cfg['visual']:
                        self.cls_action_visual_helper.update(cls_action_res)

                if self.with_skeleton_action:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].start()
                    kpt_pred = self.kpt_predictor.predict_image(
                        crop_input, visual=False)
                    keypoint_vector, score_vector = translate_to_ori_images(
                        kpt_pred, np.array(new_bboxes))
                    kpt_res = {}
                    kpt_res['keypoint'] = [
                        keypoint_vector.tolist(), score_vector.tolist()
                    ] if len(keypoint_vector) > 0 else [[], []]
                    kpt_res['bbox'] = ori_bboxes
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].end()

                    self.pipeline_res.update(kpt_res, 'kpt')

                    self.kpt_buff.update(kpt_res, mot_res)  # collect kpt output
                    state = self.kpt_buff.get_state(
                    )  # whether frame num is enough or lost tracker

                    skeleton_action_res = {}
                    if state:
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time[
                                'skeleton_action'].start()
                        collected_keypoint = self.kpt_buff.get_collected_keypoint(
                        )  # reoragnize kpt output with ID
                        skeleton_action_input = parse_mot_keypoint(
                            collected_keypoint, self.coord_size)
                        skeleton_action_res = self.skeleton_action_predictor.predict_skeleton_with_mot(
                            skeleton_action_input)
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time['skeleton_action'].end()
                        self.pipeline_res.update(skeleton_action_res,
                                                 'skeleton_action')

                    if self.cfg['visual']:
                        self.skeleton_action_visual_helper.update(
                            skeleton_action_res)

                if self.player_recognize:
                    index2id = {}
                    temp, boxes, scores, ids = mot_result
                    for i in range(len(ids)):
                        temp_id = ids[i]
                        index2id[i] = temp_id
                    if frame_id % 5 == 0:
                        rec_input = []
                        for i, kpt in enumerate(kpt_res["keypoint"][0]):
                            x_min = int(min(kpt[5][0], kpt[6][0], kpt[11][0], kpt[12][0]))
                            x_max = int(max(kpt[5][0], kpt[6][0], kpt[11][0], kpt[12][0]))
                            y_min = int(min(kpt[5][1], kpt[6][1], kpt[11][1], kpt[12][1]))
                            y_max = int(max(kpt[5][1], kpt[6][1], kpt[11][1], kpt[12][1]))
                            box = [x_min, x_max, y_min, y_max]
                            for j in range(2):
                                if box[j] < 0:
                                    box[j] = 0
                                if box[j] >= frame.shape[0]:
                                    box[j] = frame.shape[0] - 1
                            for j in range(2, 4):
                                if box[j] < 0:
                                    box[j] = 0
                                if box[j] >= frame.shape[1]:
                                    box[j] = frame.shape[1] - 1
                            if box[0] == box[1]:
                                box[1] += 1
                            if box[2] == box[3]:
                                box[3] += 1
                            rec_input.append(frame[box[2]:box[3], box[0]:box[1], :])
                        OCR_res = self.player_OCR.predict(rec_input)
                        update_res = []
                        for i, res in enumerate(OCR_res):
                            res = res[0]
                            text = ""
                            if res.isdigit() and int(res) != 1:
                                text = int(res)
                                # cv2.imwrite(res+".jpg", rec_input[i])
                            player_mot_id = index2id[i]
                            if text != '':
                                if self.id2num.get(player_mot_id):
                                    if text not in self.id2num[player_mot_id]:
                                        self.id2num[player_mot_id][text] = 1
                                    else:
                                        self.id2num[player_mot_id][text] += 1
                                else:
                                    self.id2num[player_mot_id] = {text: 1}
                            num_dict = self.id2num.get(player_mot_id)
                            if num_dict:
                                text = list(num_dict.keys())[list(num_dict.values()).index(max(num_dict.values()))]
                            else:
                                text = ""
                            update_res.append(str(text))

                            # cv2.imwrite("%d-%d.png" % (frame_id, i), crop_input[i])
                            self.pipeline_res.update({"result": update_res, "mot_res": mot_res}, "player_rec")
                        # print(OCR_res)
                    else:
                        update_res = []
                        for i, kpt in enumerate(kpt_res["keypoint"][0]):
                            player_mot_id = index2id[i]
                            num_dict = self.id2num.get(player_mot_id)
                            if num_dict:
                                text = list(num_dict.keys())[list(num_dict.values()).index(max(num_dict.values()))]
                            else:
                                text = ""
                            update_res.append(str(text))
                            self.pipeline_res.update({"result": update_res, "mot_res": mot_res}, "player_rec")

                if self.ball_control:
                    if self.loc_dir:
                        loc_info = np.load(self.loc_dir, allow_pickle=True)
                        loc_info = loc_info[frame_id]

                    ball_id = -1
                    ball_score = -1
                    for info in det_action_res:
                        if info[1]["class"] == 1:
                            continue
                        if ball_id == -1:
                            ball_id = info[0]
                            ball_score = info[1]["score"]
                        else:
                            if ball_score <= info[1]["score"]:
                                ball_id = info[0]
                                ball_score = info[1]["score"]
                    if ball_id != -1:
                        self.present_ball_id.append([ball_id, ball_score])
                    ball_score = -1
                    for i in self.present_ball_id[-5:]:
                        if ball_score <= i[1]:
                            ball_id = i[0]
                    team_index = id_team[ball_id].index(max(id_team[ball_id]))
                    if team_index == 1:
                        team_name = team_list[1][1]
                    elif team_index == 0:
                        team_name = team_list[0][1]
                    else:
                        team_name = 'unknown'
                    # for i in ids:
                    #     if ball_id == ids[i]:
                    #         ball_index = i
                    #         break
                    # ball_control_res = {"id": ball_index, "team": team_name, "box":()}
                    ball_control_res = {"team": team_name}
                    self.pipeline_res.update(ball_control_res, "ball_control")

                if self.with_mtmct and frame_id % 10 == 0:
                    crop_input, img_qualities, rects = self.reid_predictor.crop_image_with_mot(
                        frame_rgb, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['reid'].start()
                    reid_res = self.reid_predictor.predict_batch(crop_input)

                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['reid'].end()

                    reid_res_dict = {
                        'features': reid_res,
                        "qualities": img_qualities,
                        "rects": rects
                    }
                    self.pipeline_res.update(reid_res_dict, 'reid')
                else:
                    self.pipeline_res.clear('reid')

            if self.with_video_action:
                # get the params
                frame_len = self.cfg["VIDEO_ACTION"]["frame_len"]
                sample_freq = self.cfg["VIDEO_ACTION"]["sample_freq"]

                if sample_freq * frame_len > frame_count:  # video is too short
                    sample_freq = int(frame_count / frame_len)

                # filter the warmup frames
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['video_action'].start()

                # collect frames
                if frame_id % sample_freq == 0:
                    # Scale image
                    scaled_img = scale(frame_rgb)
                    video_action_imgs.append(scaled_img)

                # the number of collected frames is enough to predict video action
                if len(video_action_imgs) == frame_len:
                    classes, scores = self.video_action_predictor.predict(
                        video_action_imgs)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['video_action'].end()

                    video_action_res = {"class": classes[0], "score": scores[0]}
                    self.pipeline_res.update(video_action_res, 'video_action')

                    print("video_action_res:", video_action_res)

                    video_action_imgs.clear()  # next clip

            self.collector.append(frame_id, self.pipeline_res)

            if frame_id > self.warmup_frame:
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
            frame_id += 1
            if self.cfg['visual']:
                _, _, fps = self.pipe_timer.get_total_time()
                im = self.visualize_video(frame, self.pipeline_res, frame_id,
                                              fps, entrance, records,
                                              center_traj)  # visualize
                writer.write(im)
                if self.file_name is None:  # use camera_id
                    cv2.imshow('Paddle-Pipeline', im)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        if self.save_loc:
            np.array(self.loc_list).dump("loc_info.npy")

        writer.release()
        print('save result to {}'.format(out_path))

    def visualize_video(self,
                        image,
                        result,
                        frame_id,
                        fps,
                        entrance=None,
                        records=None,
                        center_traj=None):
        mot_res = copy.deepcopy(result.get('mot'))
        if mot_res is not None:
            ids = mot_res['boxes'][:, 0]
            scores = mot_res['boxes'][:, 2]
            boxes = mot_res['boxes'][:, 3:]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
            scores = np.zeros([0])

        # single class, still need to be defaultdict type for ploting
        num_classes = 1
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        online_tlwhs[0] = boxes
        online_scores[0] = scores
        online_ids[0] = ids

        if mot_res is not None:
            image = plot_tracking_dict(
                image,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=self.mot_predictor.pred_config.labels,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                entrance=entrance,
                records=records,
                center_traj=center_traj,
                draw_center_traj=self.draw_center_traj,
                singleplayer=self.no_box_visual
            )

        index_list = None
        singleplayer_res = result.get('kpt')
        if singleplayer_res and self.singleplayer:
            boxes = mot_res['boxes'][:, 1:]
            image, index_list = visualize_singleplayer(
                image,
                singleplayer_res,
                self.singleplayer,
                boxes,
                result.get('singleplayer')
            )

        player_res = result.get("player_rec")
        if self.player_recognize and player_res:
            image = visualize_player_rec(image, player_res)

        ball_res = result.get('ball_drawing')
        if self.ball_drawing and ball_res:
            image = visualize_ball(image, ball_res)

        link_res = result.get('link_player')
        if self.link_player and link_res:
            image = visualize_link_player(image, link_res)

        kpt = result.get('kpt')
        if self.golf:
            image = visualize_golf(image, kpt)

        boating_res = result.get('kpt')
        if boating_res and self.boating:
            boxes = mot_res['boxes'][:, 1:]
            image = visualize_boating(
                image,
                boating_res,
                boxes,
                index_list,
                self.singleplayer
            )

        ball_control_res = result.get("ball_control")
        if self.ball_control and ball_control_res:
            image = visualize_ball_control(image, ball_control_res)

        human_attr_res = result.get('attr')
        if human_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            human_attr_res = human_attr_res['output']
            image = visualize_attr(image, human_attr_res, boxes)
            image = np.array(image)

        kpt_res = result.get('kpt')
        if (kpt_res is not None) and (not self.singleplayer) and (not self.golf) and (not self.player_recognize):
            image = visualize_pose(
                image,
                kpt_res,
                visual_thresh=self.cfg['kpt_thresh'],
                returnimg=True)

        speed_predict_res = result.get('speed_predict')
        if speed_predict_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            image = visualize_speed(image, speed_predict_res, boxes, index_list)
            image = np.array(image)

        team_clas_res = result.get('team_clas')
        if team_clas_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            image = visualize_team(image, team_clas_res, boxes)
            image = np.array(image)

        video_action_res = result.get('video_action')
        if video_action_res is not None:
            video_action_score = None
            if video_action_res and video_action_res["class"] == 1:
                video_action_score = video_action_res["score"]
            mot_boxes = None
            if mot_res:
                mot_boxes = mot_res['boxes']
            image = visualize_action(
                image,
                mot_boxes,
                action_visual_collector=None,
                action_text="SkeletonAction",
                video_action_score=video_action_score,
                video_action_text="Fight")


        visual_helper_for_display = []
        action_to_display = []

        skeleton_action_res = result.get('skeleton_action')
        if skeleton_action_res is not None:
            visual_helper_for_display.append(self.skeleton_action_visual_helper)
            action_to_display.append(" ")

        det_action_res = result.get('det_action')
        if det_action_res is not None:
            visual_helper_for_display.append(self.det_action_visual_helper)
            action_to_display.append("Soccer")

        cls_action_res = result.get('cls_action')
        if cls_action_res is not None:
            visual_helper_for_display.append(self.cls_action_visual_helper)
            action_to_display.append("Calling")

        if len(visual_helper_for_display) > 0:
            image = visualize_action(image, mot_res['boxes'],
                                     visual_helper_for_display,
                                     action_to_display)
        if self.show:
            cv2.imshow('Paddle-Pipeline', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
        return image

    def visualize_image(self, im_files, images, result):
        start_idx, boxes_num_i = 0, 0
        det_res = result.get('det')
        human_attr_res = result.get('attr')
        vehicle_attr_res = result.get('vehicle_attr')

        for i, (im_file, im) in enumerate(zip(im_files, images)):
            if det_res is not None:
                det_res_i = {}
                boxes_num_i = det_res['boxes_num'][i]
                det_res_i['boxes'] = det_res['boxes'][start_idx:start_idx +
                                                      boxes_num_i, :]
                im = visualize_box_mask(
                    im,
                    det_res_i,
                    labels=['person'],
                    threshold=self.cfg['crop_thresh'])
                im = np.ascontiguousarray(np.copy(im))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if human_attr_res is not None:
                human_attr_res_i = human_attr_res['output'][start_idx:start_idx
                                                            + boxes_num_i]
                im = visualize_attr(im, human_attr_res_i, det_res_i['boxes'])
            if vehicle_attr_res is not None:
                vehicle_attr_res_i = vehicle_attr_res['output'][
                    start_idx:start_idx + boxes_num_i]
                im = visualize_attr(im, vehicle_attr_res_i, det_res_i['boxes'])

            img_name = os.path.split(im_file)[-1]
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(out_path, im)
            print("save result to: " + out_path)
            start_idx += boxes_num_i


def main():
    cfg = merge_cfg(FLAGS)
    print_arguments(cfg)

    pipeline = Pipeline(FLAGS, cfg)
    pipeline.run()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
