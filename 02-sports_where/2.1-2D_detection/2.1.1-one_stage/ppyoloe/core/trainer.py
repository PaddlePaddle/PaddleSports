from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import time
import typing

from PIL import ImageFile
from PIL import Image,ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

import paddle
from tqdm import tqdm
from paddle import amp
from paddle.static import InputSpec
from data.dataset import COCODataSet,SniperCOCODataSet
from data.reader import *
from data.source.category import get_categories
from core.optimizer import LearningRate,OptimizerBuilder
from utils.checkpoint import load_weight, load_pretrain_weight
from utils import stats as stats
from modeling import architectures
from core.optimizer import *
from metrics import Metric, COCOMetric, VOCMetric, WiderFaceMetric, get_infer_results
from utils import profiler
from core.callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer, WiferFaceEval, VisualDLWriter,SniperProposalsGenerator
from core.export_utils import _dump_infer_config, _prune_input_spec
from utils.visualizer import visualize_results, save_result
from utils.logger import setup_logger

logger = setup_logger('ppdet.engine')

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        assert mode.lower() in ['train', 'eval', 'test'], \
            "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()
        print('trainer mode : {}'.format(self.mode))
        self.optimizer = None
        self.is_loaded_weights = False

        # build datasets loader

        if self.mode == 'train':
            # use coco
            self.dataset = COCODataSet(
                dataset_dir=cfg['TrainDataset']['COCODataSet']['dataset_dir'],
                image_dir=cfg['TrainDataset']['COCODataSet']['image_dir'],
                anno_path=cfg['TrainDataset']['COCODataSet']['anno_path'],
                data_fields=cfg['TrainDataset']['COCODataSet']['data_fields'],
            )

            loader_ = TrainReader(
                sample_transforms=cfg['TrainReader']['sample_transforms'],
                batch_transforms=cfg['TrainReader']['batch_transforms'],
                batch_size=cfg['TrainReader']['batch_size'],
                shuffle=cfg['TrainReader']['shuffle'],
                drop_last=cfg['TrainReader']['drop_last']
            )
            self.loader = loader_(
                dataset=self.dataset,
                worker_num=cfg['worker_num']
            )

        # build model
        assert cfg['architecture'] in ['YOLOv3'],'architecture {} not available!'.format(cfg['architecture'])
        if cfg['architecture'] == 'YOLOv3':
            _architectures = getattr(architectures, 'YOLOv3', None)
            assert _architectures is not None, 'set architect failed'
            self.model = _architectures(
                cfg=cfg,
                backbone=cfg['YOLOv3']['backbone'],
                neck=cfg['YOLOv3']['neck'],
                yolo_head=cfg['YOLOv3']['yolo_head'],
                post_process=cfg['YOLOv3']['post_process']
            )

        self.model.load_meanstd(cfg['TestReader']['sample_transforms'])
        self.use_ema = ('use_ema' in cfg and cfg['use_ema'])
        if self.use_ema:
            ema_decay = self.cfg.get('ema_decay', 0.9998)
            cycle_epoch = self.cfg.get('cycle_epoch', -1)
            ema_decay_type = self.cfg.get('ema_decay_type', 'threshold')
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch)

        # EvalDataset build with BatchSampler to evaluate in single device
        # TODO: multi-device evaluate
        if self.mode == 'eval':
            # use coco
            self.dataset = COCODataSet(
                dataset_dir=cfg['EvalDataset']['COCODataSet']['dataset_dir'],
                image_dir=cfg['EvalDataset']['COCODataSet']['image_dir'],
                anno_path=cfg['EvalDataset']['COCODataSet']['anno_path']
            )
            self._eval_batch_sampler = paddle.io.BatchSampler(
                self.dataset, batch_size=self.cfg['EvalReader']['batch_size'])
            reader_name = '{}Reader'.format(self.mode.capitalize())
            # If metric is VOC, need to be set collate_batch=False.
            if cfg['metric'] == 'VOC':
                cfg[reader_name]['collate_batch'] = False
            loader_ = EvalReader(
                    sample_transforms=cfg['EvalReader']['sample_transforms'],
                    batch_size=cfg['EvalReader']['batch_size']
                )
            self.loader = loader_(
                dataset=self.dataset,
                worker_num=cfg['worker_num'],
                batch_sampler=self._eval_batch_sampler
            )
        # TestDataset build after user set images, skip loader creation here

        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            _lr = LearningRate(
                base_lr=cfg['LearningRate']['base_lr'],
                schedulers=cfg['LearningRate']['schedulers']
            )
            self.lr = _lr(steps_per_epoch)

            _optimizer = OptimizerBuilder(
                regularizer=cfg['OptimizerBuilder']['regularizer'],
                optimizer=cfg['OptimizerBuilder']['optimizer']
            )
            self.optimizer = _optimizer(self.lr, self.model)

        # set epoch
        self.start_epoch = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg['epoch']

        self.status = {}

        # initial default callbacks
        self._init_callbacks()

    def resume_weights(self, weights):
        # support Distill resume weights
        if hasattr(self.model, 'student_model'):
            self.start_epoch = load_weight(self.model.student_model, weights,
                                           self.optimizer)
        else:
            self.start_epoch = load_weight(self.model, weights, self.optimizer,
                                           self.ema if self.use_ema else None)
        logger.debug("Resume weights of epoch {}".format(self.start_epoch))

    def load_weights(self, weights):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model, weights)
        logger.debug("Load weights {} to start training".format(weights))

    def _init_callbacks(self):
        if self.mode == 'train':
            self._callbacks = [LogPrinter(self), Checkpointer(self)]
            if self.cfg.get('use_vdl', False):
                self._callbacks.append(VisualDLWriter(self))
            if self.cfg.get('save_proposals', False):
                self._callbacks.append(SniperProposalsGenerator(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'eval':
            self._callbacks = [LogPrinter(self)]
            if self.cfg['metric'] == 'WiderFace':
                self._callbacks.append(WiferFaceEval(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'test' and self.cfg.get('use_vdl', False):
            self._callbacks = [VisualDLWriter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"
        Init_mark = False
        if validate:
            self.eval_dataset = COCODataSet(
                dataset_dir=self.cfg['EvalDataset']['COCODataSet']['dataset_dir'],
                image_dir=self.cfg['EvalDataset']['COCODataSet']['image_dir'],
                anno_path=self.cfg['EvalDataset']['COCODataSet']['anno_path'],
            )

        model = self.model

        # enabel auto mixed precision mode
        if self.cfg.get('amp', False):
            scaler = amp.GradScaler(
                enable=self.cfg.use_gpu or self.cfg.use_npu,
                init_loss_scaling=1024)

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg['log_iter'], fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg['log_iter'], fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg['log_iter'])

        profiler_options = self.cfg.get('profiler_options', None)

        self._compose_callback.on_train_begin(self.status)
        for epoch_id in range(self.start_epoch, self.cfg['epoch']):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            model.train()
            iter_tic = time.time()
            for step_id, data in enumerate(self.loader):
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                profiler.add_profiler_step(profiler_options)
                self._compose_callback.on_step_begin(self.status)
                data['epoch_id'] = epoch_id

                if self.cfg.get('amp', False):
                    with amp.auto_cast(enable=self.cfg['use_gpu']):
                        # model forward
                        outputs = model(data)
                        loss = outputs['loss']

                    # model backward
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    # in dygraph mode, optimizer.minimize is equal to optimizer.step
                    scaler.minimize(self.optimizer, scaled_loss)
                else:
                    # model forward
                    outputs = model(data)
                    loss = outputs['loss']
                    # model backward
                    loss.backward()
                    self.optimizer.step()
                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                # if self.cfg.get('unstructured_prune'):
                #     self.pruner.step()
                self.optimizer.clear_grad()
                self.status['learning_rate'] = curr_lr

                # if self._nranks < 2 or self._local_rank == 0:
                self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_callback.on_step_end(self.status)
                if self.use_ema:
                    self.ema.update()
                iter_tic = time.time()

            is_snapshot = ((epoch_id + 1) % self.cfg['snapshot_epoch'] == 0 or epoch_id == self.end_epoch - 1)
            if is_snapshot and self.use_ema:
                # apply ema weight on model
                weight = copy.deepcopy(self.model.state_dict())
                self.model.set_dict(self.ema.apply())
                self.status['weight'] = weight

            self._compose_callback.on_epoch_end(self.status)
            if validate and is_snapshot:
                if not hasattr(self, '_eval_loader'):
                    # build evaluation dataset and loader
                    self._eval_batch_sampler = paddle.io.BatchSampler(
                        self.eval_dataset, batch_size=self.cfg['EvalReader']['batch_size'])
                    reader_name = '{}Reader'.format(self.mode.capitalize())
                    # If metric is VOC, need to be set collate_batch=False.
                    if self.cfg['metric'] == 'VOC':
                        self.cfg[reader_name]['collate_batch'] = False
                    loader_ = EvalReader(
                        sample_transforms=self.cfg['EvalReader']['sample_transforms'],
                        batch_size=self.cfg['EvalReader']['batch_size']
                    )
                    self._eval_loader = loader_(
                        dataset=self.eval_dataset,
                        worker_num=self.cfg['worker_num'],
                        batch_sampler=self._eval_batch_sampler
                    )
                # if validation in training is enabled, metrics should be re-init
                # Init_mark makes sure this code will only execute once
                if validate and Init_mark == False:
                    Init_mark = True
                    self._init_metrics(validate=validate)
                    self._reset_metrics()

                with paddle.no_grad():
                    self.status['save_best_model'] = True
                    self._eval_with_loader(self._eval_loader)

            if is_snapshot and self.use_ema:
                # reset original weight
                self.model.set_dict(weight)
                self.status.pop('weight')

        self._compose_callback.on_train_end(self.status)

    def _init_metrics(self, validate=False):
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return
        classwise = self.cfg['classwise'] if 'classwise' in self.cfg else False
        if self.cfg['metric'] == 'COCO' or self.cfg['metric'] == "SNIPERCOCO":
            # TODO: bias should be unified
            bias = self.cfg['bias'] if 'bias' in self.cfg else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            # pass clsid2catid info to metric instance to avoid multiple loading
            # annotation file
            clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()} \
                                if self.mode == 'eval' else None

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            anno_file = self.dataset.get_anno()

            IouType = self.cfg['IouType'] if 'IouType' in self.cfg else 'bbox'
            if self.cfg['metric'] == "COCO":
                self._metrics = [
                    COCOMetric(
                        anno_file=anno_file,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only)
                ]
            elif self.cfg['metric'] == "SNIPERCOCO":  # sniper
                print('metric not available')
        else:
            logger.warning("Metric not support for metric type {}".format(
                self.cfg['metric']))
            self._metrics = []

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def _eval_with_loader(self, loader):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            outs = self.model(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            # multi-scale inputs: all inputs have same im_id
            if isinstance(data, typing.Sequence):
                sample_num += data[0]['im_id'].numpy().shape[0]
            else:
                sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate(self):
        self._init_metrics(True)
        self._reset_metrics()
        with paddle.no_grad():
            self._eval_with_loader(self.loader)

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_results=False):
        self.dataset = SniperCOCODataSet(
            dataset_dir=self.cfg['TestDataset']['ImageFolder']['dataset_dir'],
            anno_path=self.cfg['TestDataset']['ImageFolder']['anno_path'],
        )

        self.dataset.set_images(images)

        loader_ = TestReader(
            sample_transforms = self.cfg['TestReader']['sample_transforms'],
            batch_size = self.cfg['TestReader']['batch_size']
        )
        loader = loader_(
            dataset=self.dataset,
            worker_num=0,
            infer=True
        )

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        imid2path = self.dataset.get_imid2path()

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg['metric'], anno_file=anno_file)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            for _m in metrics:
                _m.update(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        # sniper
        # if type(self.dataset) == SniperCOCODataSet:
        #     results = self.dataset.anno_cropper.aggregate_chips_detections(
        #         results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)
            bbox_num = outs['bbox_num']

            start = 0
            for i, im_id in enumerate(outs['im_id']):
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')
                image = ImageOps.exif_transpose(image)
                self.status['original_image'] = np.array(image.copy())

                end = start + bbox_num[i]
                bbox_res = batch_res['bbox'][start:end] \
                        if 'bbox' in batch_res else None
                mask_res = batch_res['mask'][start:end] \
                        if 'mask' in batch_res else None
                segm_res = batch_res['segm'][start:end] \
                        if 'segm' in batch_res else None
                keypoint_res = batch_res['keypoint'][start:end] \
                        if 'keypoint' in batch_res else None
                image = visualize_results(
                    image, bbox_res, mask_res, segm_res, keypoint_res,
                    int(im_id), catid2name, draw_threshold)
                self.status['result_image'] = np.array(image.copy())
                if self._compose_callback:
                    self._compose_callback.on_step_end(self.status)
                # save image with detection
                save_name = self._get_save_image_name(output_dir, image_path)
                logger.info("Detection bbox results save in {}".format(
                    save_name))
                image.save(save_name, quality=95)

                start = end

    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext