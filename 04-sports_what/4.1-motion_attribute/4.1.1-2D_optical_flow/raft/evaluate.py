import argparse
import paddle
import sys
import time
sys.path.append('core')
from raft import *
from datasets import *
from train import *


def eval_chair(model):
    step = 0
    gamma = 0.85
    max_flow = 400
    batch_size = 4
    num_workers = 1
    eval_metrics = {
        'epe': 0.0,
        'epe1': 0.0,
        'epe3': 0.0,
        'epe5': 0.0
    }
    sequence_loss = Sequence_loss(gamma=gamma, max_flow=max_flow)
    # train_loader = fetch_dataloader(stage='sintel', batch_size=batch_size, image_size=[368,768], TRAIN_DS=None, num_workers=num_workers)
    eval_loader = fetch_dataloader(stage='chairs', batch_size=batch_size, image_size=[368,496], 
                                    TRAIN_DS=None, num_workers=num_workers, split='test', shuffle=False)

    for batch_id, data in enumerate(eval_loader):
        img1 = data[0].cuda()
        img2 = data[1].cuda()
        flow = data[2].cuda()
        valid = data[3].cuda()

        flow_p = model(img1, img2)
        loss, metrics = sequence_loss(flow_p, flow, valid)
        step += 1
        for k in eval_metrics.keys():
            eval_metrics[k] += float(metrics[k])
    metrics_str = str()
    for i in eval_metrics.keys():
        metrics_str += ' %s:%.6f ' % (i, eval_metrics[i] / (step)) # self.batch_size * 
    t = time.localtime()
    time_str = "%i/%i %i:%i:%i " % (t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
    eval_str = 'Evaluate-----%s %s' % (time_str, metrics_str)
    return eval_str
        