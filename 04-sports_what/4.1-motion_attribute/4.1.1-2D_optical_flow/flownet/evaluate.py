import argparse
import paddle
import sys
import time
sys.path.append('core')
sys.path.append('models')
from datasets import *
from evaluate import *
from FlowNetS import *
from multiscaleloss import *


def eval_chair(model):
    step = 0
    gamma = 0.85
    max_flow = 400
    batch_size = 4
    num_workers = 1
    eval_metrics = {
        'epe': 0.0
    }
    
    multiscale_weights = [0.005,0.01,0.02,0.08,0.32]
    div_flow = 20
    milestones = [100, 150, 200]
    # train_loader = fetch_dataloader(stage='sintel', batch_size=batch_size, image_size=[368,768], TRAIN_DS=None, num_workers=num_workers)
    eval_loader = fetch_dataloader(stage='chairs', batch_size=batch_size, image_size=[384,512], 
                                    TRAIN_DS=None, num_workers=num_workers, split='test', shuffle=False)
    
    for batch_id, data in enumerate(eval_loader):
        img1 = data[0].cuda()
        img2 = data[1].cuda()
        img = paddle.concat(x=[img1, img2], axis=1)
        # print(img.shape)
        flow = data[2].cuda()
        output = model(img)
        loss = multiscaleEPE(output, flow, weights=multiscale_weights, sparse=False)
        flow2_EPE = div_flow * realEPE(output[0], flow, sparse=False)
        step += 1
        eval_metrics['epe'] += float(flow2_EPE)
    metrics_str = str()
    for i in eval_metrics.keys():
        metrics_str += ' %s:%.6f ' % (i, eval_metrics[i] / (step)) # self.batch_size * 
    t = time.localtime()
    time_str = "%i/%i %i:%i:%i " % (t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
    eval_str = 'Evaluate-----%s %s' % (time_str, metrics_str)
    return eval_str
        







